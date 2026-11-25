"""Taxonomy geometry of an encoder's frozen feature space (PLAN_ANALYSIS.md §4.1). CPU is enough.

    python tools/feature_geometry.py --config configs/h22clean_dinov2_b14.json \
        --c2_config results/configs/h22clean_dinov2_b14_tau_norm+c2f.json --out results/geometry

Train features only (view 0). Writes, per encoder:

<name>_geometry_class.csv   one row per class with train images:
    genus_margin     cos(centroid, nearest same-genus centroid) - cos(centroid, nearest other-genus centroid),
                     on L2-normalised features (> 0: the nearest relative is in the genus); empty for species
                     alone in their genus. family_margin: the same one level up (other genus, same family vs
                     other family). nearest_level: genus / family / other for the single nearest centroid.
    knn_genus_purity share of the k nearest train images (other species only) of each of the class's train
                     images that belong to its genus, averaged over its images (tail classes only; cosine).
                     knn_family_purity: the same for the family.
    donor_level      the level C2 takes donors from (lt.calibration.select_donors, the encoder's chosen c2_nmin);
                     donor_cos: cosine between the class mean and the count-weighted donor mean (raw features).
    transfer_rR      share of the class's own within-class variance inside the top-R eigen-subspace of its
                     donors' pooled covariance (weighted as C2 builds it), for R in --ranks. Computed for tail
                     classes with >= 3 images (noisy, the per-class predictor) and for --max_eval sampled
                     classes with >= --eval_min images (the effect size). own_rR: the class's own top-R share
                     (upper bound); global_rR: the subspace of all eligible classes pooled (no taxonomy).
<name>_geometry_shuf<seed>_class.csv   the same taxonomy measures with the A9 shuffled taxonomy (control).
<name>_geometry.json        summaries: medians and shares per group and donor level, real vs shuffled.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lt.calibration import select_donors  # noqa: E402
from lt.classes import ClassIndex, ShotGroups  # noqa: E402
from lt.features import load_split  # noqa: E402
from lt.taxonomy import Taxonomy  # noqa: E402


def log(*a):
    print('[{}]'.format(time.strftime('%H:%M:%S')), *a, flush=True)


def centroid_margins(cent, has, genus_of, family_of, chunk=1024):
    """Per class: genus margin, family margin, level of the nearest other centroid (cosine, normalised)."""
    C = cent.shape[0]
    g = torch.as_tensor(genus_of)
    f = torch.as_tensor(family_of)
    hs = torch.as_tensor(has)
    gm = np.full(C, np.nan)
    fm = np.full(C, np.nan)
    near = np.array([''] * C, dtype=object)
    for i in range(0, C, chunk):
        idx = torch.arange(i, min(C, i + chunk))
        s = cent[idx] @ cent.T                                    # [b, C]
        s[:, ~hs] = -np.inf
        s[torch.arange(len(idx)), idx] = -np.inf                  # not itself
        same_g = g[idx][:, None] == g[None, :]
        same_f = (f[idx][:, None] == f[None, :]) & (f[idx][:, None] >= 0)
        neg = torch.tensor(-np.inf)
        best_same_g = torch.where(same_g, s, neg).max(1).values
        best_other_g = torch.where(~same_g, s, neg).max(1).values
        best_fam_og = torch.where(same_f & ~same_g, s, neg).max(1).values
        best_other_f = torch.where(~same_f & ~same_g, s, neg).max(1).values
        nn = s.argmax(1)
        for j, c in enumerate(idx.tolist()):
            if not has[c]:
                continue
            if torch.isfinite(best_same_g[j]) and torch.isfinite(best_other_g[j]):
                gm[c] = float(best_same_g[j] - best_other_g[j])
            if family_of[c] >= 0 and torch.isfinite(best_fam_og[j]) and torch.isfinite(best_other_f[j]):
                fm[c] = float(best_fam_og[j] - best_other_f[j])
            n = int(nn[j])
            near[c] = 'genus' if genus_of[n] == genus_of[c] else (
                'family' if family_of[c] >= 0 and family_of[n] == family_of[c] else 'other')
    return gm, fm, near


def knn_purity(xn, y, query_classes, genus_of, family_of, k, qchunk=2048, dchunk=65536):
    """Mean share of the k nearest train images of another species that share the genus / family."""
    qmask = np.isin(y, query_classes)
    qi = np.nonzero(qmask)[0]
    yt = torch.as_tensor(y)
    g = torch.as_tensor(genus_of)
    f = torch.as_tensor(family_of)
    gp = np.zeros(len(qi))
    fp = np.zeros(len(qi))
    N = xn.shape[0]
    for a in range(0, len(qi), qchunk):
        q = torch.as_tensor(qi[a:a + qchunk])
        xq, yq = xn[q], yt[q]
        best_s = torch.full((len(q), k), -np.inf)
        best_i = torch.zeros((len(q), k), dtype=torch.long)
        for b in range(0, N, dchunk):
            s = xq @ xn[b:b + dchunk].T
            s[yq[:, None] == yt[b:b + dchunk][None, :]] = -np.inf  # other species only
            cs = torch.cat([best_s, s], 1)
            ci = torch.cat([best_i, torch.arange(b, b + s.shape[1]).expand(len(q), -1)], 1)
            best_s, pos = cs.topk(k, dim=1)
            best_i = ci.gather(1, pos)
        ny = yt[best_i]
        gp[a:a + len(q)] = (g[ny] == g[yq][:, None]).double().mean(1).numpy()
        fp[a:a + len(q)] = ((f[ny] == f[yq][:, None]) & (f[yq][:, None] >= 0)).double().mean(1).numpy()
        if (a // qchunk) % 5 == 0:
            log('  kNN {}/{} query images'.format(min(a + qchunk, len(qi)), len(qi)))
    C = len(genus_of)
    n = np.bincount(y[qi], minlength=C)
    with np.errstate(invalid='ignore'):
        return np.bincount(y[qi], weights=gp, minlength=C) / n, np.bincount(y[qi], weights=fp, minlength=C) / n


class Scatter(object):
    """Within-class scatter matrices from features sorted by class (raw space, float64)."""

    def __init__(self, x, y, C):
        self.order = np.argsort(y, kind='stable')
        self.n = np.bincount(y, minlength=C)
        self.start = np.concatenate([[0], np.cumsum(self.n)[:-1]])
        self.x = x
        self.D = x.shape[1]

    def mean(self, c):
        rows = self.order[self.start[c]:self.start[c] + self.n[c]]
        return self.x[torch.as_tensor(rows)].double().mean(0)

    def S(self, c):
        rows = self.order[self.start[c]:self.start[c] + self.n[c]]
        xc = self.x[torch.as_tensor(rows)].double()
        xc = xc - xc.mean(0, keepdim=True)
        return xc.T @ xc

    def T(self, c):
        """Donor contribution as lt.calibration._pooled_cov weights it: n_d / (n_d - 1) * S_d (up to scale)."""
        n = self.n[c]
        return self.S(c) * (n / max(n - 1, 1))


def shares(S, evecs, ranks):
    """Share of trace(S) inside the top-r eigenvectors (evecs sorted by decreasing eigenvalue)."""
    tr = float(torch.trace(S))
    if tr <= 0:
        return {r: np.nan for r in ranks}
    SU = S @ evecs[:, :max(ranks)]
    cum = torch.cumsum((evecs[:, :max(ranks)] * SU).sum(0), 0)
    return {r: float(cum[r - 1]) / tr for r in ranks}


def top_evecs(M):
    w, v = torch.linalg.eigh(M)
    return v.flip(1)


def own_shares(S, ranks):
    """The class's own top-r share (upper bound): eigenvalues only."""
    w = torch.linalg.eigvalsh(S).flip(0).clamp(min=0)
    tr = float(w.sum())
    cum = torch.cumsum(w, 0)
    return {r: (float(cum[r - 1]) / tr if tr > 0 else np.nan) for r in ranks}


def transfer(sc, counts, tax, few_end, nmin, eval_classes, tail_classes, ranks, global_evecs, class_cache):
    """Donor-subspace transfer (see module doc) for eval and tail classes, with the given taxonomy.
    class_cache keeps the taxonomy-free parts (own and global shares) across taxonomies."""
    C = len(counts)
    eligible = counts >= nmin
    genus_of, family_of = tax.genus_of, tax.family_of
    D = sc.D
    fam_sum = {}
    glob = torch.zeros(D, D, dtype=torch.float64)
    t0 = time.time()
    for c in np.nonzero(eligible)[0]:
        Tc = sc.T(c)
        glob += Tc
        if family_of[c] >= 0:
            fam_sum[family_of[c]] = fam_sum.get(family_of[c], 0) + Tc
    log('  pooled family/global scatter in {:.0f}s'.format(time.time() - t0))

    # donor rule for any class (tail classes use select_donors itself; eval classes the same rule)
    tail_sel = dict(zip(range(few_end), select_donors(counts, genus_of, family_of, few_end, nmin)))
    todo = sorted(set(eval_classes) | set(tail_classes))
    by_genus = {}
    for c in todo:
        by_genus.setdefault(int(genus_of[c]), []).append(c)
    out = {}
    done = 0
    shared = {}                       # donor matrices without the class's own term are shared: eigh once
    idx = np.arange(C)
    for gi, members in by_genus.items():
        gel = [d for d in np.nonzero((genus_of == gi) & eligible)[0]]
        Tg = {d: sc.T(d) for d in gel}
        gsum = sum(Tg.values()) if gel else None
        for c in members:
            if c < few_end:
                donors, level = tail_sel[c]
                if donors is None:
                    continue
            else:
                if any(d != c for d in gel):
                    level = 'genus'
                elif family_of[c] >= 0 and np.any(eligible & (family_of == family_of[c]) & (idx != c)):
                    level = 'family'
                else:
                    level = 'global'
            key = (level, gi if level == 'genus' else (int(family_of[c]) if level == 'family' else -1))
            base = gsum if level == 'genus' else (fam_sum[family_of[c]] if level == 'family' else glob)
            if eligible[c]:           # the class is in its own pool: take it out
                own = Tg[c] if c in Tg else sc.T(c)
                ev = top_evecs(base - own)
            else:
                if key not in shared:
                    shared[key] = top_evecs(base)
                ev = shared[key]
            Sc = sc.S(c)
            r = {'donor_level_eval': level}
            for k_, v_ in shares(Sc, ev, ranks).items():
                r['transfer_r{}'.format(k_)] = v_
            if c not in class_cache:
                cc = {}
                for k_, v_ in own_shares(Sc, ranks).items():
                    cc['own_r{}'.format(k_)] = v_
                for k_, v_ in shares(Sc, global_evecs, ranks).items():
                    cc['global_r{}'.format(k_)] = v_
                class_cache[c] = cc
            r.update(class_cache[c])
            out[c] = r
            done += 1
            if done % 500 == 0:
                log('  transfer {}/{} classes ({:.0f}s)'.format(done, len(todo), time.time() - t0))
    return out, glob


def donor_cos(sc, counts, tax, few_end, nmin):
    """Cosine between each tail class's raw mean and its count-weighted donor mean (C2's shrinkage target)."""
    sel = select_donors(counts, tax.genus_of, tax.family_of, few_end, nmin)
    mu = {}
    out = np.full(len(counts), np.nan)
    lv = np.array([''] * len(counts), dtype=object)
    for s, (donors, level) in enumerate(sel):
        lv[s] = level
        if donors is None:
            continue
        for d in donors:
            if d not in mu:
                mu[d] = sc.mean(d)
        w = torch.as_tensor(counts[donors], dtype=torch.float64)
        m = (torch.stack([mu[d] for d in donors]) * w[:, None]).sum(0) / w.sum()
        a = mu[s] if s in mu else sc.mean(s)
        out[s] = float(torch.nn.functional.cosine_similarity(a, m, dim=0))
    return out, lv


def summarise(rows, groups, prefix):
    def med(vals):
        v = np.asarray([x for x in vals if x is not None and not np.isnan(x)], dtype=float)
        return {'median': float(np.median(v)) if len(v) else None, 'mean': float(v.mean()) if len(v) else None,
                'n': int(len(v))}
    s = {}
    for name, sel in (('all', lambda r: True), ('few', lambda r: r['group'] == 'few'),
                      ('medium', lambda r: r['group'] == 'medium'), ('many', lambda r: r['group'] == 'many')):
        rr = [r for r in rows if sel(r)]
        s[name] = {k: med([r.get(k) for r in rr]) for k in ('genus_margin', 'family_margin', 'knn_genus_purity',
                                                              'knn_family_purity')}
        v = [r['genus_margin'] for r in rr if r.get('genus_margin') is not None and not np.isnan(r['genus_margin'])]
        s[name]['genus_margin_positive_share'] = float(np.mean(np.asarray(v) > 0)) if v else None
        near = [r['nearest_level'] for r in rr if r.get('nearest_level')]
        s[name]['nearest_in_genus_share'] = float(np.mean([n == 'genus' for n in near])) if near else None
    return {prefix: s}


def transfer_summary(tr, classes, ranks):
    out = {}
    for name, cl in classes.items():
        rows = [tr[c] for c in cl if c in tr]
        d = {'n': len(rows)}
        for lv in ('all', 'genus', 'family', 'global'):
            rr = rows if lv == 'all' else [r for r in rows if r['donor_level_eval'] == lv]
            if not rr:
                continue
            d[lv] = {'n': len(rr)}
            for r_ in ranks:
                for k in ('transfer', 'own', 'global'):
                    v = np.asarray([x['{}_r{}'.format(k, r_)] for x in rr], dtype=float)
                    d[lv]['{}_r{}_median'.format(k, r_)] = float(np.nanmedian(v))
        out[name] = d
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--c2_config', required=True, help='the chosen tau_norm+c2f config (gives c2_nmin, c2_rank)')
    p.add_argument('--feature_root', default=None)
    p.add_argument('--taxonomy', default=None)
    p.add_argument('--out', default='results/geometry')
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--ranks', default='8,32,64')
    p.add_argument('--eval_min', type=int, default=50, help='classes with this many images give the effect size')
    p.add_argument('--max_eval', type=int, default=2000)
    p.add_argument('--shuffle_seeds', default='0', help='A9 control taxonomies ("" = none)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=0)
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    conf = json.load(open(args.config, encoding='utf-8'))
    c2 = json.load(open(args.c2_config, encoding='utf-8'))
    nmin = int(c2.get('c2_nmin', 10))
    ranks = sorted({int(r) for r in args.ranks.split(',')} | ({int(c2['c2_rank'])} if str(c2.get('c2_rank')).isdigit()
                                                              else set()))
    root = args.feature_root or conf['feature_root']
    enc = conf['encoder']
    name = '{}_{}'.format(conf['dataset'], enc)
    os.makedirs(args.out, exist_ok=True)

    log('{}: loading train features from {}'.format(name, root))
    x, y0 = load_split(root, conf.get('splits', {}).get('train', 'train'), enc, views=1)
    x = x[0].float()
    index = ClassIndex(y0.numpy(), conf['num_classes'])
    y = index.remap(y0.numpy())
    counts = index.counts
    C = len(counts)
    groups = ShotGroups(counts, conf['few_thr'], conf['many_thr'])
    few_end = groups.few_end
    tax = Taxonomy.from_csv(args.taxonomy or conf['taxonomy'], index)
    log('{} images, {} dims, {} classes, few_end {}, c2_nmin {}, ranks {}'.format(len(y), x.shape[1], C, few_end,
                                                                                  nmin, ranks))
    has = counts > 0
    xn = torch.nn.functional.normalize(x, dim=1)
    cent = torch.zeros(C, x.shape[1])
    cent.index_add_(0, torch.as_tensor(y), xn)
    cent = torch.nn.functional.normalize(cent, dim=1)

    sc = Scatter(x, y, C)
    rng = np.random.default_rng(args.seed)
    big = np.nonzero(counts >= args.eval_min)[0]
    eval_classes = np.sort(rng.choice(big, min(args.max_eval, len(big)), replace=False)) if len(big) else big
    tail3 = [c for c in range(few_end) if counts[c] >= 3]
    tail_any = np.nonzero(has[:few_end])[0]

    group_name = np.array(['many'] * C, dtype=object)
    group_name[:few_end] = 'few'
    group_name[few_end:groups.many_start] = 'medium'

    taxes = [('real', tax)] + [('shuf{}'.format(s), tax.shuffled(int(s)))
                               for s in args.shuffle_seeds.split(',') if s.strip() != '']
    summary = {'encoder': enc, 'dataset': conf['dataset'], 'images': int(len(y)), 'dim': int(x.shape[1]),
               'classes': C, 'few_end': int(few_end), 'many_start': int(groups.many_start), 'c2_nmin': nmin,
               'c2_rank': c2.get('c2_rank'), 'k': args.k, 'ranks': ranks, 'eval_min': args.eval_min,
               'eval_classes': int(len(eval_classes)), 'tail_classes_transfer': len(tail3)}
    global_evecs = None
    class_cache = {}
    eval_set = set(eval_classes.tolist())
    for tag, tx in taxes:
        t0 = time.time()
        log('[{}] centroid margins'.format(tag))
        gm, fm, near = centroid_margins(cent, has, tx.genus_of, tx.family_of)
        log('[{}] kNN purity for {} tail classes (k={})'.format(tag, len(tail_any), args.k))
        kg, kf = knn_purity(xn, y, tail_any, tx.genus_of, tx.family_of, args.k)
        log('[{}] donor means'.format(tag))
        dc, dl = donor_cos(sc, counts, tx, few_end, nmin)
        if global_evecs is None:
            g = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float64)
            for c in np.nonzero(counts >= nmin)[0]:
                g += sc.T(c)
            global_evecs = top_evecs(g)
        log('[{}] donor-subspace transfer: {} eval + {} tail classes'.format(tag, len(eval_classes), len(tail3)))
        tr, _ = transfer(sc, counts, tx, few_end, nmin, eval_classes, tail3, ranks, global_evecs, class_cache)
        spg = np.bincount(tx.genus_of[has], minlength=tx.num_genera)
        rows = []
        for c in np.nonzero(has)[0]:
            r = {'class': int(c), 'orig_label': int(index.inverse[c]), 'train_n': int(counts[c]),
                 'group': group_name[c], 'genus': tx.genus_names[tx.genus_of[c]],
                 'family': tx.family_names[tx.family_of[c]] if tx.family_of[c] >= 0 else '',
                 'species_in_genus': int(spg[tx.genus_of[c]]),
                 'genus_margin': gm[c], 'family_margin': fm[c], 'nearest_level': near[c],
                 'knn_genus_purity': kg[c] if c < few_end else np.nan,
                 'knn_family_purity': kf[c] if c < few_end else np.nan,
                 'donor_level': dl[c] if c < few_end else '', 'donor_cos': dc[c] if c < few_end else np.nan,
                 'transfer_eval': int(c in eval_set)}
            r.update(tr.get(c, {}))
            rows.append(r)
        cols = ['class', 'orig_label', 'train_n', 'group', 'genus', 'family', 'species_in_genus', 'genus_margin',
                'family_margin', 'nearest_level', 'knn_genus_purity', 'knn_family_purity', 'donor_level',
                'donor_cos', 'transfer_eval', 'donor_level_eval'] + \
               ['{}_r{}'.format(k, r_) for k in ('transfer', 'own', 'global') for r_ in ranks]
        path = os.path.join(args.out, '{}_geometry{}_class.csv'.format(name, '' if tag == 'real' else '_' + tag))
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            w.writeheader()
            for r in rows:
                w.writerow({k: ('' if isinstance(v, float) and np.isnan(v) else
                                (round(v, 6) if isinstance(v, float) else v)) for k, v in r.items()})
        summary.update(summarise(rows, groups, tag))
        summary[tag]['transfer'] = transfer_summary(tr, {'eval': eval_classes.tolist(), 'tail': tail3}, ranks)
        summary[tag]['donor_levels_tail'] = {lv: int(sum(dl[c] == lv for c in range(few_end)))
                                             for lv in ('genus', 'family', 'global', 'none')}
        log('[{}] done in {:.0f}s -> {}'.format(tag, time.time() - t0, path))
    path = os.path.join(args.out, '{}_geometry.json'.format(name))
    json.dump(summary, open(path, 'w', encoding='utf-8'), indent=1)
    log('=>', path)
    for tag, _ in taxes:
        s = summary[tag]
        log('{:6s} few: genus margin median {} (>0: {}), kNN genus purity {}, nearest in genus {}'.format(
            tag, _fmt(s['few']['genus_margin']['median']), _fmt(s['few']['genus_margin_positive_share']),
            _fmt(s['few']['knn_genus_purity']['mean']), _fmt(s['few']['nearest_in_genus_share'])))
        t = s['transfer']['eval'].get('all', {})
        r_ = 32 if 32 in ranks else ranks[0]
        log('{:6s} eval transfer r{}: donors {} | global {} | own {} | random {:.3f}'.format(
            tag, r_, _fmt(t.get('transfer_r{}_median'.format(r_))), _fmt(t.get('global_r{}_median'.format(r_))),
            _fmt(t.get('own_r{}_median'.format(r_))), r_ / x.shape[1]))


def _fmt(v):
    return 'n/a' if v is None else '{:.3f}'.format(v)


if __name__ == '__main__':
    main()
