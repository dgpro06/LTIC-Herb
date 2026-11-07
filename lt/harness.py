"""Tuning harness (plan §7, §8.2, §15).

Protocol, identical for every method:
  1. random search: `trials` configurations sampled from the method's space, each trained on train
     with the epoch chosen by macro-F1 on V-tune; the best trial by V-tune score wins;
  2. the chosen configuration is scored ONCE on V-select (gates and model selection);
  3. it is retrained with each seed and evaluated on test (and V-cal for conformal) -> rows in
     results/all_runs.csv; raw logits are saved for C-F (lt/cf.py) and C4 (lt/conformal.py).

A method name may carry fixed settings after '@', e.g. 'ltic3@gamma=0' (the gamma = 0 row, §3.2).
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
from lt.metrics import evaluate
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
}


def parse_method(name):
    """'ltic3@gamma=0,alpha=0.85' -> ('ltic3', {'gamma': 0.0, 'alpha': 0.85})."""
    base, _, rest = name.partition('@')
    fixed = {}
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

    def describe(self):
        parts = ['{} {}'.format(k, v.n) for k, v in self.fs.items()]
        return '{} / {}: {} classes ({} without train images) | {} | {}'.format(
            self.conf['dataset'], self.conf['encoder'], self.index.num_classes, self.index.empty_classes,
            self.groups.describe(), ', '.join(parts))


def train_one(cfg, data, seed):
    genus_of = data.taxonomy.genus_of if data.taxonomy is not None else None
    if cfg['method'] == 'flat_genus' and genus_of is None:
        raise ValueError('flat_genus needs "taxonomy" in the config')
    return T.train(cfg, data.fs['train'], data.fs['vtune'], data.index.counts, data.groups.few_end,
                   data.groups.many_start, seed=seed, genus_of=genus_of, log=lambda *_: None)


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
        with open(path, newline='') as f:
            existing = next(csv.reader(f), None)
        if existing != header:
            raise RuntimeError('{} has a different header; move it aside'.format(path))
    with open(path, 'a', newline='') as f:
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
                hparams=json.dumps({k: cfg[k] for k in sorted(cfg) if k in T.DEFAULTS or k == 'method'}),
                **metrics)


# ---------------------------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------------------------
def search(method_name, data, trials, results_csv, seed=0, base=None, log=print):
    method, fixed = parse_method(method_name)
    # deterministic per method name (Python's hash() is salted per process)
    rng = np.random.default_rng(int(hashlib.sha1(method_name.encode()).hexdigest(), 16) % (2 ** 32) + seed)
    best = None
    rows = []
    for t in range(trials):
        cfg = sample_config(method, rng, fixed, base)
        model, info = train_one(cfg, data, seed)
        logits, _ = T.predict(model, data.fs['vtune'], info['cfg'], data.log_pi)
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
    logits, _ = T.predict(model, data.fs['vselect'], info['cfg'], data.log_pi)
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
            logits, _ = T.predict(model, data.fs[s], info['cfg'], data.log_pi)
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
        logits, genus = T.predict(model, data.fs[s], raw_cfg, data.log_pi)
        payload = dict(logits=logits.half().numpy(), y=data.fs[s].y.cpu().numpy(), tau=cfg.get('tau', 0.0), split=s,
                       counts=data.index.counts, class_map=data.index.class_map,
                       few_end=data.groups.few_end, many_start=data.groups.many_start,
                       method=method_name, encoder=data.conf['encoder'], dataset=data.conf['dataset'],
                       cfg=json.dumps(cfg))
        if genus is not None:
            payload['genus_logits'] = genus.half().numpy()
        name = '{}_{}_{}_{}_seed{}.npz'.format(data.conf['dataset'], data.conf['encoder'],
                                               method_name.replace('@', '-').replace('=', '').replace(',', '-'), s, seed)
        np.savez_compressed(os.path.join(save_dir, name), **payload)
