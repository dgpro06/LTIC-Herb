"""Tuning harness (plan §7, §8.2, §15).

Protocol, identical for every method:
  1. random search: `trials` configurations sampled from the method's space, each trained on train
     with the epoch chosen by macro-F1 on V-tune; the best trial by V-tune score wins;
  2. the chosen configuration is scored ONCE on V-select (gates and model selection);
  3. it is retrained with each seed and evaluated on test (and V-cal for conformal) -> rows in
     results/all_runs.csv; raw logits are saved for C-F (lt/cf.py) and C4 (lt/conformal.py).

A method name may carry fixed settings after '@', e.g. 'ltic3@gamma=0' (the gamma = 0 row, §3.2).
C2 is added to any base method with '+c2' (e.g. 'crt+c2', 'ltic3+c2@c2_rank="diag"'); '+c2shuf' is the A9
control (C2 with Taxonomy.shuffled(c2_shuffle_seed), all parameters re-tuned).
"""
import csv
import datetime
import hashlib
import json
import os

import numpy as np
import torch

from lt import losses as L
from lt import trainer as T
from lt.classes import ClassIndex, ShotGroups
from lt.features import load_split
from lt.metrics import evaluate, macro_f1
from lt.taxonomy import Taxonomy

# ---------------------------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------------------------
COMMON = {'lr': ('loguniform', 1e-3, 1.0), 'weight_decay': ('loguniform', 1e-6, 1e-3),
          'batch_size': ('choice', [256, 512, 1024])}
LTIC = {'alpha': ('uniform', 0.5, 0.95), 'gamma': ('uniform', 0.0, 1.0)}
SPACES = {
    'ce': {},
    'la': {'la_mode': ('choice', ['posthoc', 'loss']), 'tau': ('uniform', 0.0, 2.0), 'loss_tau': ('uniform', 0.5, 2.0)},
    'balanced_softmax': {},
    'crt': {},
    'tau_norm': {'test_tau': ('uniform', 0.0, 2.0)},
    'ldam_drw': {'ldam_max_m': ('uniform', 0.1, 0.8), 'ldam_s': ('choice', [10.0, 20.0, 30.0]),
                 'drw_frac': ('uniform', 0.5, 0.9)},
    'reslt': LTIC, 'ltic2': LTIC, 'ltic3': LTIC,
    'flat_genus': {'genus_weight': ('loguniform', 0.1, 3.0), 'genus_tau': ('uniform', 0.0, 2.0)},
    # s fixed at the paper's 32; m0 around the paper's 0.15; lam is not given in the paper
    'taxonet': {'taxonet_m0': ('uniform', 0.05, 0.35), 'taxonet_lam': ('loguniform', 0.01, 10.0),
                'taxonet_rho': ('choice', ['sum', 'max'])},
}
# C2 parameters, searched on top of the base method's own space (PLAN_NEXT.md §6)
C2_SPACE = {'c2_tau_mu': ('loguniform', 0.5, 50.0), 'c2_k': ('choice', [5, 10, 20, 40]),
            'c2_rank': ('choice', ['diag', 8, 32, 64]), 'c2_nmin': ('choice', [10, 20, 40])}
# +c2 / +c2shuf: C2 and the A9 control, base hyperparameters searched together with C2's (Pl@ntNet sessions)
# +c2f / +c2fshuf: the base method's tuned configuration is FROZEN (read from results/configs/); only C2's
#                  settings and the post-hoc knobs are searched, so base vs base+C2 differ only by C2
C2_SUFFIX = {'c2': {'c2': True}, 'c2shuf': {'c2': True, 'c2_shuffle': True},
             'c2f': {'c2': True, 'c2_frozen_base': True},
             'c2fshuf': {'c2': True, 'c2_frozen_base': True, 'c2_shuffle': True}}
# decision-time knobs: re-tuned on top of a frozen base, and grid-searched when the config sets posthoc_grid
POSTHOC_KEYS = ('tau', 'test_tau', 'gamma')
POSTHOC_GRID = [round(0.1 * i, 1) for i in range(21)]          # 0.0, 0.1, ..., 2.0


def search_seed_key(method_name):
    """The name the random search is seeded from. The A9 control shares it with its C2 method, so real and
    shuffled taxonomy are tried on the SAME candidate configurations; the shuffle seed is left out so every
    shuffle seed also gets those candidates.
    'crt+c2shuf@c2_shuffle_seed=2' -> 'crt+c2';  'crt+c2fshuf' -> 'crt+c2f';  'ltic3@gamma=0' unchanged.
    (Before 2026-11-18 the seed came from the full method name, so +c2 and +c2shuf drew different candidates.)"""
    base, _, rest = method_name.partition('@')
    base = base.replace('+c2fshuf', '+c2f').replace('+c2shuf', '+c2')
    kept = [kv for kv in filter(None, rest.split(',')) if not kv.startswith('c2_shuffle_seed=')]
    return base + ('@' + ','.join(kept) if kept else '')


def base_method_name(method_name):
    """The method C2 is added to: 'tau_norm+c2f' -> 'tau_norm'; 'ltic3+c2f@gamma=0,c2_rank=8' -> 'ltic3@gamma=0'."""
    base, _, rest = method_name.partition('@')
    base = base.partition('+')[0]
    kept = [kv for kv in filter(None, rest.split(',')) if not kv.startswith('c2')]
    return base + ('@' + ','.join(kept) if kept else '')


def frozen_settings(method, base_cfg):
    """Training settings taken from the base method's tuned config (everything except post-hoc knobs and C2)."""
    keys = set(COMMON) | set(SPACES[method]) | set(T.DEFAULTS)
    return {k: v for k, v in base_cfg.items()
            if k in keys and k not in POSTHOC_KEYS and k != 'method' and not k.startswith('c2')}


def parse_method(name):
    """'ltic3@gamma=0,alpha=0.85' -> ('ltic3', {'gamma': 0.0, 'alpha': 0.85});
    'crt+c2' -> ('crt', {'c2': True}); 'crt+c2shuf' -> ('crt', {'c2': True, 'c2_shuffle': True})."""
    base, _, rest = name.partition('@')
    fixed = {}
    base, plus, extra = base.partition('+')
    if plus:
        if extra not in C2_SUFFIX:
            raise ValueError('unknown suffix +{} in {} (choose from {})'.format(extra, name, sorted(C2_SUFFIX)))
        fixed.update(C2_SUFFIX[extra])
    for kv in filter(None, rest.split(',')):
        k, v = kv.split('=')
        try:
            fixed[k] = json.loads(v)
        except ValueError:
            fixed[k] = v
    if base not in SPACES:
        raise ValueError('unknown method {} (choose from {})'.format(base, sorted(SPACES)))
    return base, fixed


def sample_config(method, rng, fixed=None, base=None):
    space = dict(COMMON, **SPACES[method])
    if fixed and fixed.get('c2'):
        space.update(C2_SPACE)
    cfg = dict(base or {})
    for k, spec in space.items():
        if fixed and k in fixed:
            continue
        kind = spec[0]
        if kind == 'uniform':
            cfg[k] = float(rng.uniform(spec[1], spec[2]))
        elif kind == 'loguniform':
            cfg[k] = float(np.exp(rng.uniform(np.log(spec[1]), np.log(spec[2]))))
        elif kind == 'choice':
            cfg[k] = spec[1][int(rng.integers(len(spec[1])))]
    cfg.update(fixed or {})
    cfg['method'] = method
    if method == 'la' and cfg.get('la_mode') == 'loss':
        cfg['tau'] = 0.0                    # the loss form already adjusts; no second post-hoc shift
    return cfg


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------
class Data(object):
    """All splits of one (dataset, encoder) on one device, in output-index space."""

    def __init__(self, conf, device=None, splits=('train', 'vtune', 'vselect', 'vcal', 'test')):
        self.conf = conf
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        root, enc = conf['feature_root'], conf['encoder']
        names = conf.get('splits', {})
        train_x, train_y = load_split(root, names.get('train', 'train'), enc, views=conf.get('views'))
        self.index = ClassIndex(train_y.numpy(), conf['num_classes'])
        self.groups = ShotGroups(self.index.counts, conf['few_thr'], conf['many_thr'])
        self.fs = {'train': T.FeatureSet(train_x, self.index.remap(train_y.numpy()), self.device)}
        for s in splits:
            if s == 'train':
                continue
            try:
                x, y = load_split(root, names.get(s, s), enc, views=1)
            except FileNotFoundError:
                continue
            self.fs[s] = T.FeatureSet(x, self.index.remap(y.numpy()), self.device)
        self.taxonomy = Taxonomy.from_csv(conf['taxonomy'], self.index) if conf.get('taxonomy') else None
        self.log_pi = L.log_prior(self.index.counts, self.device)
        self.c2_cache = {}

    def prior_counts(self, cfg):
        """Counts behind the prior this configuration trains with (real counts unless c2_prior='augmented')."""
        stats = self.c2_stats(cfg) if cfg.get('c2') and cfg.get('c2_prior', 'real') != 'real' else None
        return T.prior_counts(cfg, self.index.counts, stats)

    def log_pi_for(self, cfg):
        if not cfg.get('c2') or cfg.get('c2_prior', 'real') == 'real':
            return self.log_pi
        return L.log_prior(self.prior_counts(cfg), self.device)

    def taxonomy_for(self, cfg):
        if self.taxonomy is None:
            raise ValueError('{} needs "taxonomy" in the config'.format(cfg['method']))
        if cfg.get('c2_shuffle'):
            return self.taxonomy.shuffled(cfg.get('c2_shuffle_seed', 0))
        return self.taxonomy

    def c2_stats(self, cfg):
        """Donor statistics depend only on (n_min, covariance form, taxonomy): built once, reused by trials."""
        cfg = dict(T.C2_DEFAULTS, **cfg)
        # low-rank statistics are built once at rank 64 and sliced for r = 8 / 32
        diag = cfg['c2_rank'] == 'diag'
        key = (cfg['c2_nmin'], 'diag' if diag else 'lowrank', bool(cfg['c2_shuffle']), cfg['c2_shuffle_seed'])
        if key not in self.c2_cache:
            if len(self.c2_cache) >= 6:                     # bound GPU memory
                self.c2_cache.pop(next(iter(self.c2_cache)))
            build = dict(cfg, c2_rank='diag' if diag else 64)
            self.c2_cache[key] = T.c2_stats_for(build, self.fs['train'], self.index.counts, self.groups.few_end,
                                                self.taxonomy_for(cfg))
        return self.c2_cache[key].at_rank(cfg['c2_rank'])

    def describe(self):
        parts = ['{} {}'.format(k, v.n) for k, v in self.fs.items()]
        return '{} / {}: {} classes ({} without train images) | {} | {}'.format(
            self.conf['dataset'], self.conf['encoder'], self.index.num_classes, self.index.empty_classes,
            self.groups.describe(), ', '.join(parts))


def train_one(cfg, data, seed):
    genus_of = data.taxonomy.genus_of if data.taxonomy is not None else None
    if cfg['method'] == 'flat_genus' and genus_of is None:
        raise ValueError('flat_genus needs "taxonomy" in the config')
    c2_stats = data.c2_stats(cfg) if cfg.get('c2') else None
    return T.train(cfg, data.fs['train'], data.fs['vtune'], data.index.counts, data.groups.few_end,
                   data.groups.many_start, seed=seed, genus_of=genus_of, log=lambda *_: None, c2_stats=c2_stats)


# ---------------------------------------------------------------------------------------------
# Results CSV
# ---------------------------------------------------------------------------------------------
METRIC_KEYS = sorted(['top1', 'top5', 'macro_f1', 'macro_acc', 'worst_class_acc', 'n',
                      'few_acc', 'few_acc_pooled', 'few_n_img', 'few_n_cls',
                      'medium_acc', 'medium_acc_pooled', 'medium_n_img', 'medium_n_cls',
                      'many_acc', 'many_acc_pooled', 'many_n_img', 'many_n_cls',
                      'weighted_precision_appendix', 'weighted_recall_appendix'])
BASE_KEYS = ['timestamp', 'dataset', 'encoder', 'method', 'config_id', 'stage', 'trial', 'seed', 'split',
             'best_epoch', 'vtune_score', 'seconds', 'hparams']


def append_rows(path, rows):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    header = BASE_KEYS + METRIC_KEYS
    new = not os.path.isfile(path)
    if not new:
        with open(path, newline='', encoding='utf-8') as f:
            existing = next(csv.reader(f), None)
        if existing != header:
            raise RuntimeError('{} has a different header; move it aside'.format(path))
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in header})


def config_id(cfg):
    keep = {k: v for k, v in sorted(cfg.items()) if k not in ('eval_every',)}
    return hashlib.sha1(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:10]


def _row(data, method_name, cfg, info, stage, trial, seed, split, metrics):
    return dict(timestamp=datetime.datetime.now().isoformat(timespec='seconds'), dataset=data.conf['dataset'],
                encoder=data.conf['encoder'], method=method_name, config_id=config_id(cfg), stage=stage,
                trial=trial, seed=seed, split=split, best_epoch=info['best_epoch'],
                vtune_score=round(info['vtune_score'], 6), seconds=round(info['seconds'], 1),
                hparams=json.dumps({k: cfg[k] for k in sorted(cfg)
                                    if k in T.DEFAULTS or k == 'method' or (cfg.get('c2') and k in T.C2_DEFAULTS)
                                    or (cfg['method'] == 'taxonet' and k in T.TAXONET_DEFAULTS)}),
                **metrics)


# ---------------------------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------------------------
def posthoc_knob(cfg, fixed=None):
    """The decision-time knob a grid can tune for this configuration, or None."""
    fixed = fixed or {}
    if cfg['method'] == 'tau_norm' and 'test_tau' not in fixed:
        return 'test_tau'
    if cfg['method'] == 'la' and cfg.get('la_mode') == 'posthoc' and 'tau' not in fixed:
        return 'tau'
    return None


def tune_posthoc(model, cfg, data, fixed=None):
    """Grid-search the post-hoc knob (tau-norm's test_tau, LA's tau) on V-tune for a trained model; no
    retraining. Returns (cfg with the best value, its V-tune score). Ties keep the smaller value."""
    knob = posthoc_knob(cfg, fixed)
    if knob is None:
        return cfg, None
    y = data.fs['vtune'].y.cpu().numpy()
    log_pi = data.log_pi_for(cfg)
    best_cfg, best_score = cfg, -np.inf
    for v in POSTHOC_GRID:
        c = dict(cfg, **{knob: v})
        logits, _ = T.predict(model, data.fs['vtune'], c, log_pi)
        pred = logits.argmax(1).numpy()
        score = macro_f1(y, pred, data.index.num_classes) if c.get('select_metric', 'macro_f1') == 'macro_f1' \
            else float((pred == y).mean())
        if score > best_score:
            best_cfg, best_score = c, score
    return best_cfg, best_score


def search(method_name, data, trials, results_csv, seed=0, base=None, log=print, frozen_base=None):
    """frozen_base: the base method's tuned config (required for +c2f / +c2fshuf)."""
    method, fixed = parse_method(method_name)
    if fixed.get('c2_frozen_base'):
        if frozen_base is None:
            raise ValueError('{} freezes the base config: pass the tuned config of {}'.format(
                method_name, base_method_name(method_name)))
        fixed = dict(frozen_settings(method, frozen_base), **fixed)
    # deterministic per method name (Python's hash() is salted per process); the A9 control shares the seed
    # of its C2 method (search_seed_key)
    key = search_seed_key(method_name)
    rng = np.random.default_rng(int(hashlib.sha1(key.encode()).hexdigest(), 16) % (2 ** 32) + seed)
    grid = bool(data.conf.get('posthoc_grid'))
    best = None
    rows = []
    for t in range(trials):
        cfg = sample_config(method, rng, fixed, base)
        model, info = train_one(cfg, data, seed)
        if grid:
            cfg_g, score_g = tune_posthoc(model, info['cfg'], data, fixed)
            if score_g is not None:
                info = dict(info, cfg=cfg_g, vtune_score=score_g)
        logits, _ = T.predict(model, data.fs['vtune'], info['cfg'], data.log_pi_for(info['cfg']))
        m = evaluate(logits, data.fs['vtune'].y.cpu().numpy(), data.groups)
        rows.append(_row(data, method_name, info['cfg'], info, 'trial', t, seed, 'vtune', m))
        log('  [{}] trial {:2d}/{}: V-tune macro-F1 {:.4f} (epoch {}, {:.0f}s){}'.format(
            method_name, t + 1, trials, info['vtune_score'], info['best_epoch'], info['seconds'],
            ' diverged' if info['diverged'] else ''))
        if best is None or info['vtune_score'] > best[1]['vtune_score']:
            best = (info['cfg'], info)
    append_rows(results_csv, rows)
    return best[0]


def score_vselect(method_name, cfg, data, results_csv, seed=0, save_dir=None):
    """The ONE V-select evaluation of a chosen configuration."""
    model, info = train_one(cfg, data, seed)
    logits, _ = T.predict(model, data.fs['vselect'], info['cfg'], data.log_pi_for(info['cfg']))
    m = evaluate(logits, data.fs['vselect'].y.cpu().numpy(), data.groups)
    append_rows(results_csv, [_row(data, method_name, info['cfg'], info, 'vselect', '', seed, 'vselect', m)])
    if save_dir:
        save_predictions(save_dir, method_name, seed, model, info['cfg'], data, ['vselect'])
    return m


def final_runs(method_name, cfg, data, seeds, results_csv, save_dir=None, splits=('test',)):
    out = []
    for seed in seeds:
        model, info = train_one(cfg, data, seed)
        rows = []
        for s in splits:
            if s not in data.fs:
                continue
            logits, _ = T.predict(model, data.fs[s], info['cfg'], data.log_pi_for(info['cfg']))
            m = evaluate(logits, data.fs[s].y.cpu().numpy(), data.groups)
            rows.append(_row(data, method_name, info['cfg'], info, 'final', '', seed, s, m))
            out.append((seed, s, m))
        append_rows(results_csv, rows)
        if save_dir:
            save_predictions(save_dir, method_name, seed, model, info['cfg'], data, splits)
    return out


def save_predictions(save_dir, method_name, seed, model, cfg, data, splits):
    """Raw species logits (tau = 0) + the chosen tau, so C-F can sweep tau and C4 can calibrate."""
    os.makedirs(save_dir, exist_ok=True)
    raw_cfg = dict(cfg, tau=0.0)
    for s in splits:
        if s not in data.fs:
            continue
        logits, genus = T.predict(model, data.fs[s], raw_cfg, data.log_pi_for(cfg))
        payload = dict(logits=logits.half().numpy(), y=data.fs[s].y.cpu().numpy(), tau=cfg.get('tau', 0.0), split=s,
                       counts=data.index.counts, prior_counts=data.prior_counts(cfg), class_map=data.index.class_map,
                       few_end=data.groups.few_end, many_start=data.groups.many_start,
                       method=method_name, encoder=data.conf['encoder'], dataset=data.conf['dataset'],
                       cfg=json.dumps(cfg))
        if genus is not None:
            payload['genus_logits'] = genus.half().numpy()
        name = '{}_{}_{}_{}_seed{}.npz'.format(data.conf['dataset'], data.conf['encoder'],
                                               method_name.replace('@', '-').replace('=', '').replace(',', '-'), s, seed)
        np.savez_compressed(os.path.join(save_dir, name), **payload)
