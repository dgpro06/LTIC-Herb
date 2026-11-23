"""Per-class link between an encoder's taxonomy geometry, its errors and C2's gain (PLAN_ANALYSIS.md §4.2).

    python tools/c2_class_analysis.py --geometry results/geometry/h22clean_dinov2_b14_geometry_class.csv \
        --preds results/preds --taxonomy taxonomy/h22.csv --out results/geometry

Runs on saved predictions only (CPU, seconds). For every tail class (fewer than few_thr train images):

    gain_recall / gain_f1   C2 (tau_norm+c2f) minus its base (tau_norm), per-class recall / F1, averaged over the
                            training seeds, on the pooled --splits (default test + vcal: ~1.5 test images per tail
                            class on H22, so pooling roughly doubles the images per class)
    tax_gain_recall         C2 minus the mean of its shuffled-taxonomy runs (A9; training seed 0): the part of the
                            gain that needs the real taxonomy
    base_within_genus       share of the base model's errors on the class that stay in the genus

and asks whether the geometry predicts them: Spearman correlation with a bootstrap 95% CI over classes, the same
with train_n partialled out, and the mean gain per quartile of each predictor. Also writes the mistake-severity
table (§4.2, first point) for every saved test prediction of the encoder: genus- and family-correct rates among
errors and the mean taxonomic distance (0 same genus, 1 same family, 2 other family or unknown).
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lt import cf  # noqa: E402
from lt.taxonomy import Taxonomy  # noqa: E402
from tools.cf_report import _Index, load_preds  # noqa: E402

PREDICTORS = ('genus_margin', 'knn_genus_purity', 'donor_cos', 'transfer_r32', 'transfer_rc2', 'train_n',
              'species_in_genus')
OUTCOMES = ('gain_recall', 'gain_f1', 'tax_gain_recall', 'base_within_genus')


def predictions(d):
    if 'pred' in d:
        return d['pred'].astype(np.int64)
    prior = d.get('prior_counts', d['counts'])
    return cf.predictions(d['logits'].astype(np.float32), np.log(np.maximum(prior, 1) / prior.sum()), float(d['tau']))


def per_class(y, pred, C):
    """Per-class recall and F1 (NaN where the class has no image in y)."""
    n = np.bincount(y, minlength=C).astype(float)
    tp = np.bincount(y[pred == y], minlength=C).astype(float)
    npred = np.bincount(pred, minlength=C).astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        rec = np.where(n > 0, tp / n, np.nan)
        prec = np.where(npred > 0, tp / npred, 0.0)
        f1 = np.where(n > 0, np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0), np.nan)
    return rec, f1


def load_run(pred_dir, name, splits, seed, class_map):
    """Pooled (y, pred) over splits for one saved run; None if any split is missing."""
    ys, ps = [], []
    for s in splits:
        path = os.path.join(pred_dir, '{}_{}_seed{}.npz'.format(name, s, seed))
        if not os.path.isfile(path):
            return None
        d = load_preds(path)
        if not np.array_equal(d['class_map'], class_map):
            raise SystemExit('{}: class order differs from the other runs'.format(path))
        ys.append(d['y'].astype(np.int64))
        ps.append(predictions(d))
    return np.concatenate(ys), np.concatenate(ps)


def spearman(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 10:
        return np.nan
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    return float(np.corrcoef(ra, rb)[0, 1])


def partial_spearman(a, b, z):
    ok = ~(np.isnan(a) | np.isnan(b) | np.isnan(z))
    if ok.sum() < 10:
        return np.nan
    ra, rb, rz = rankdata(a[ok]), rankdata(b[ok]), rankdata(z[ok])
    rab, raz, rbz = np.corrcoef(ra, rb)[0, 1], np.corrcoef(ra, rz)[0, 1], np.corrcoef(rb, rz)[0, 1]
    den = np.sqrt(max((1 - raz ** 2) * (1 - rbz ** 2), 1e-12))
    return float((rab - raz * rbz) / den)


def boot_ci(f, n, n_boot, rng):
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        v = f(i)
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return [np.nan, np.nan]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def severity(y, pred, genus_of, family_of, few_end):
    err = pred != y
    out = {}
    for name, m in (('all', np.ones_like(y, dtype=bool)), ('tail', y < few_end)):
        e = err & m
        ne = int(e.sum())
        sg = genus_of[pred[e]] == genus_of[y[e]]
        sf = (family_of[pred[e]] == family_of[y[e]]) & (family_of[y[e]] >= 0)
        dist = np.where(sg, 0, np.where(sf, 1, 2))
        out[name] = {'images': int(m.sum()), 'errors': ne,
                     'genus_correct': float(sg.mean()) if ne else np.nan,
                     'family_correct': float((sg | sf).mean()) if ne else np.nan,
                     'mean_tax_distance': float(dist.mean()) if ne else np.nan}
    return out


def fnum(v, nd=6):
    return '' if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), nd)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--geometry', required=True, help='<dataset>_<encoder>_geometry_class.csv (feature_geometry.py)')
    p.add_argument('--preds', default='results/preds')
    p.add_argument('--taxonomy', required=True)
    p.add_argument('--out', default='results/geometry')
    p.add_argument('--base', default='tau_norm')
    p.add_argument('--c2', default='tau_norm+c2f')
    p.add_argument('--shuffled', default='tau_norm+c2fshuf,tau_norm+c2fshuf-c2_shuffle_seed1,'
                                         'tau_norm+c2fshuf-c2_shuffle_seed2')
    p.add_argument('--splits', default='test,vcal')
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--n_boot', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    geo = list(csv.DictReader(open(args.geometry, encoding='utf-8')))
    m = re.match(r'(.+?)_(.+)_geometry_class\.csv$', os.path.basename(args.geometry))
    dataset, enc = m.group(1), m.group(2)
    prefix = '{}_{}_'.format(dataset, enc)
    splits = args.splits.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]
    gjson = os.path.join(os.path.dirname(args.geometry), '{}_{}_geometry.json'.format(dataset, enc))
    c2_rank = json.load(open(gjson, encoding='utf-8')).get('c2_rank') if os.path.isfile(gjson) else None

    ref = load_preds(os.path.join(args.preds, '{}{}_test_seed0.npz'.format(prefix, args.base)))
    class_map, counts, few_end = ref['class_map'], ref['counts'], int(ref['few_end'])
    C = len(counts)
    tax = Taxonomy.from_csv(args.taxonomy, _Index(class_map))
    for r in geo:                                       # same class order as the predictions
        if int(counts[int(r['class'])]) != int(r['train_n']):
            raise SystemExit('geometry and predictions disagree on the class order ({})'.format(r['class']))

    def recall_f1(method, seed_list):
        rs, fs = [], []
        for s in seed_list:
            run = load_run(args.preds, prefix + method, splits, s, class_map)
            if run is None:
                continue
            r, f = per_class(run[0], run[1], C)
            rs.append(r)
            fs.append(f)
        if not rs:
            return None, None, 0
        return np.nanmean(rs, 0), np.nanmean(fs, 0), len(rs)

    base_r, base_f, nb = recall_f1(args.base, seeds)
    c2_r, c2_f, nc = recall_f1(args.c2, seeds)
    c2_r0, _, _ = recall_f1(args.c2, [0])
    sh = [recall_f1(m_, [0])[0] for m_ in args.shuffled.split(',')]
    sh = [s for s in sh if s is not None]
    shuf_r = np.nanmean(sh, 0) if sh else np.full(C, np.nan)
    print('{} / {}: {} base seeds, {} C2 seeds, {} shuffled runs, splits {}'.format(dataset, enc, nb, nc, len(sh),
                                                                                       '+'.join(splits)))
    # base errors that stay in the genus (pooled over seeds)
    within = np.zeros(C)
    nerr = np.zeros(C)
    n_img = None
    for s in seeds:
        run = load_run(args.preds, prefix + args.base, splits, s, class_map)
        if run is None:
            continue
        y, pr = run
        n_img = np.bincount(y, minlength=C)
        e = pr != y
        nerr += np.bincount(y[e], minlength=C)
        within += np.bincount(y[e], weights=(tax.genus_of[pr[e]] == tax.genus_of[y[e]]).astype(float), minlength=C)
    with np.errstate(invalid='ignore'):
        base_within = np.where(nerr > 0, within / nerr, np.nan)

    g_by = {int(r['class']): r for r in geo}
    rows = []
    for c in range(few_end):
        if c not in g_by or n_img is None or n_img[c] == 0:
            continue
        g = g_by[c]
        row = {'class': c, 'train_n': int(counts[c]), 'eval_images': int(n_img[c]), 'genus': g['genus'],
               'species_in_genus': int(g['species_in_genus']), 'donor_level': g['donor_level']}
        for k in ('genus_margin', 'knn_genus_purity', 'donor_cos', 'transfer_r32'):
            row[k] = float(g[k]) if g.get(k, '') != '' else np.nan
        rc = 'transfer_r{}'.format(c2_rank)
        row['transfer_rc2'] = float(g[rc]) if c2_rank is not None and g.get(rc, '') != '' else np.nan
        row['base_recall'] = base_r[c]
        row['c2_recall'] = c2_r[c]
        row['gain_recall'] = c2_r[c] - base_r[c]
        row['gain_f1'] = c2_f[c] - base_f[c]
        row['tax_gain_recall'] = c2_r0[c] - shuf_r[c]
        row['base_within_genus'] = base_within[c]
        rows.append(row)
    print('tail classes with eval images: {}'.format(len(rows)))

    def col(k):
        return np.array([r[k] for r in rows], dtype=float)

    rng = np.random.default_rng(args.seed)
    n = len(rows)
    tn = col('train_n')
    corr = []
    for pk in PREDICTORS:
        a = col(pk)
        for ok in OUTCOMES:
            b = col(ok)
            ci = boot_ci(lambda i: spearman(a[i], b[i]), n, args.n_boot, rng)
            ci_p = boot_ci(lambda i: partial_spearman(a[i], b[i], tn[i]), n, args.n_boot, rng) \
                if pk != 'train_n' else [np.nan, np.nan]
            corr.append({'predictor': pk, 'outcome': ok, 'classes': int((~(np.isnan(a) | np.isnan(b))).sum()),
                         'spearman': fnum(spearman(a, b), 4), 'ci_low': fnum(ci[0], 4), 'ci_high': fnum(ci[1], 4),
                         'partial_spearman_train_n': fnum(partial_spearman(a, b, tn), 4) if pk != 'train_n' else '',
                         'partial_ci_low': fnum(ci_p[0], 4), 'partial_ci_high': fnum(ci_p[1], 4)})

    bins = []
    for pk in ('genus_margin', 'knn_genus_purity', 'donor_cos', 'transfer_rc2'):
        a = col(pk)
        ok = ~np.isnan(a)
        if ok.sum() < 20:
            continue
        q = np.nanpercentile(a, [25, 50, 75])
        qi = np.digitize(a, q)
        for b in range(4):
            sel = ok & (qi == b)
            for oc in ('gain_recall', 'gain_f1', 'tax_gain_recall', 'base_within_genus'):
                v = col(oc)[sel]
                v = v[~np.isnan(v)]
                ci = boot_ci(lambda i: float(v[i].mean()), len(v), args.n_boot, rng) if len(v) else [np.nan] * 2
                bins.append({'predictor': pk, 'quartile': b + 1,
                             'range': '{:.3f}..{:.3f}'.format(a[sel].min(), a[sel].max()) if sel.any() else '',
                             'outcome': oc, 'classes': int(len(v)), 'mean': fnum(v.mean() if len(v) else np.nan, 4),
                             'ci_low': fnum(ci[0], 4), 'ci_high': fnum(ci[1], 4)})
    for lv in ('genus', 'family', 'global'):
        sel = np.array([r['donor_level'] == lv for r in rows])
        for oc in ('gain_recall', 'gain_f1', 'tax_gain_recall'):
            v = col(oc)[sel]
            v = v[~np.isnan(v)]
            ci = boot_ci(lambda i: float(v[i].mean()), len(v), args.n_boot, rng) if len(v) else [np.nan] * 2
            bins.append({'predictor': 'donor_level', 'quartile': lv, 'range': '', 'outcome': oc,
                         'classes': int(len(v)), 'mean': fnum(v.mean() if len(v) else np.nan, 4),
                         'ci_low': fnum(ci[0], 4), 'ci_high': fnum(ci[1], 4)})

    sev = []
    for path in sorted(glob.glob(os.path.join(args.preds, '{}*_test_seed0.npz'.format(prefix)))):
        d = load_preds(path)
        if not np.array_equal(d['class_map'], class_map):
            continue
        s = severity(d['y'].astype(np.int64), predictions(d), tax.genus_of, tax.family_of, few_end)
        method = os.path.basename(path)[len(prefix):-len('_test_seed0.npz')]
        for part, v in s.items():
            sev.append(dict(method=method, images=part, **{k: fnum(x, 4) if isinstance(x, float) else x
                                                          for k, x in v.items() if k != 'images'}))

    os.makedirs(args.out, exist_ok=True)
    base = os.path.join(args.out, '{}_{}_'.format(dataset, enc))
    for name, data in (('c2_class.csv', rows), ('c2_corr.csv', corr), ('c2_bins.csv', bins),
                       ('severity.csv', sev)):
        if not data:
            continue
        with open(base + name, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            for r in data:
                w.writerow({k: fnum(v) if isinstance(v, float) else v for k, v in r.items()})
        print('=>', base + name)
    mean_gain = float(np.nanmean(col('gain_recall')))
    print('mean tail gain (recall) {:+.4f}, taxonomic part {:+.4f}'.format(mean_gain,
                                                                          float(np.nanmean(col('tax_gain_recall')))))
    for r in corr:
        if r['outcome'] in ('gain_recall', 'tax_gain_recall', 'base_within_genus') and r['predictor'] != 'species_in_genus':
            print('  {:17s} -> {:17s} rho {:>7} [{}, {}]  partial(n) {:>7}  ({} classes)'.format(
                r['predictor'], r['outcome'], r['spearman'], r['ci_low'], r['ci_high'],
                r['partial_spearman_train_n'], r['classes']))


if __name__ == '__main__':
    main()
