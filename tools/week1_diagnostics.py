"""Week-1 gate (plan §8.1) on one (dataset, encoder).

    python tools/week1_diagnostics.py --config configs/plantnet_clip_b32.json \
        --la_preds results/preds/plantnet_clip_b32_la_vselect_seed0.npz

Needs: features (train, vtune and the split of --la_preds), the taxonomy CSV, and LA-probe predictions
from tools/run_baselines.py (stage vselect writes V-select logits). Thresholds are read from
gates/week1_thresholds.json, which must be committed BEFORE this script is run (plan §8.1).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lt import cf, diagnostics, harness  # noqa: E402
from lt import losses as L  # noqa: E402
from lt import trainer as T  # noqa: E402
from lt.metrics import macro_f1  # noqa: E402


def train_genus_classifier(data, trials, seed=0, epochs=30, log=print):
    """Linear genus probe with the logit-adjusted (balanced softmax) loss, epoch and lr chosen on V-tune."""
    tax = data.taxonomy
    g_counts = np.bincount(tax.genus_of, weights=data.index.counts, minlength=tax.num_genera)
    tr, tu = data.fs['train'], data.fs['vtune']
    gtr = T.FeatureSet(tr.x, torch.as_tensor(tax.genus_of)[tr.y.cpu()], data.device)
    gtu = T.FeatureSet(tu.x, torch.as_tensor(tax.genus_of)[tu.y.cpu()], data.device)
    rng = np.random.default_rng(seed)
    best = None
    for t in range(trials):
        cfg = {'method': 'balanced_softmax', 'epochs': epochs, 'lr': float(np.exp(rng.uniform(np.log(1e-3), 0))),
               'weight_decay': float(np.exp(rng.uniform(np.log(1e-6), np.log(1e-3))))}
        model, info = T.train(cfg, gtr, gtu, g_counts, 0, 0, seed=seed, log=lambda *_: None)
        log('   genus probe trial {}/{}: V-tune genus macro-F1 {:.4f}'.format(t + 1, trials, info['vtune_score']))
        if best is None or info['vtune_score'] > best[1]['vtune_score']:
            best = (model, info)
    return best[0], best[1]['cfg'], L.log_prior(g_counts, data.device)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--la_preds', required=True, help='LA probe predictions (.npz) on the gate split')
    p.add_argument('--thresholds', default='gates/week1_thresholds.json')
    p.add_argument('--out', default=None, help='default: results/week1_gate_<dataset>_<encoder>.json')
    p.add_argument('--genus_trials', type=int, default=5)
    p.add_argument('--cov_k', type=int, default=10)
    p.add_argument('--n_perm', type=int, default=1000)
    p.add_argument('--n_boot', type=int, default=2000)
    args = p.parse_args()

    with open(args.config) as f:
        conf = json.load(f)
    with open(args.thresholds) as f:
        thr = json.load(f)
    preds = dict(np.load(args.la_preds))
    split = str(preds.get('split', '')) or next((s for s in ('vselect', 'vtune', 'test', 'vcal')
                                                if '_{}_'.format(s) in os.path.basename(args.la_preds)), None)
    data = harness.Data(conf, splits=('train', 'vtune', split))
    print('=> ' + data.describe())
    tax = data.taxonomy
    if tax is None:
        raise SystemExit('the config needs "taxonomy"')
    if not np.array_equal(preds['class_map'], data.index.class_map):
        raise SystemExit('class order of the predictions differs from the features (different train file?)')
    y = data.fs[split].y.cpu().numpy()
    if not np.array_equal(preds['y'], y):
        raise SystemExit('labels of --la_preds do not match the {} features'.format(split))
    few_end = data.groups.few_end
    counts = data.index.counts
    log_pi = np.log(np.maximum(counts, 1) / counts.sum())
    pred = cf.predictions(preds['logits'].astype(np.float32), log_pi, float(preds['tau']))

    print('=> diagnostic 1: donor pooling')
    d1 = diagnostics.donor_pooling(counts, tax.genus_of, few_end)
    print('=> diagnostic (a): cross-genus share of LA tail false positives on {}'.format(split))
    da = cf.shares_with_ci(y, pred, few_end, tax.genus_of, tax.family_of, n_boot=args.n_boot)
    print('=> diagnostic (b): genus classifier on cross-genus tail FPs')
    gmodel, gcfg, glog_pi = train_genus_classifier(data, args.genus_trials)
    glogits, _ = T.predict(gmodel, data.fs[split], gcfg, glog_pi)
    db = diagnostics.genus_rescue(y, pred, glogits.argmax(1).numpy(), few_end, tax.genus_of)
    db['genus_probe_split_macro_f1'] = macro_f1(tax.genus_of[y], glogits.argmax(1).numpy(), tax.num_genera)
    print('=> C2 check: covariance subspace similarity within genera')
    tr = data.fs['train']
    dc2 = diagnostics.covariance_check(tr.x[0].cpu(), tr.y.cpu(), counts, tax.genus_of, k=args.cov_k,
                                       n_perm=args.n_perm)

    checks = {
        'C1_donor_median_ge': (d1['median_ratio'], d1['median_ratio'] >= thr['donor_median_ratio_min']),
        'C1a_cross_genus_share_ge': (da['share']['cross_genus'], da['share']['cross_genus'] >= thr['cross_genus_share_min']),
        'C1b_genus_correct_ge': (db['genus_correct_share'], db['genus_correct_share'] >= thr['genus_correct_share_min']),
        'C2_p_value_lt': (dc2['p_value'], dc2['p_value'] < thr['c2_p_value_max']),
    }
    c1 = checks['C1_donor_median_ge'][1] and checks['C1a_cross_genus_share_ge'][1] and checks['C1b_genus_correct_ge'][1]
    c2 = checks['C2_p_value_lt'][1]
    verdict = ('C1 and C2 supported' if c1 and c2 else 'only C1 supported: keep C1' if c1 else
               'only C2 supported: keep C2' if c2 else 'both fail: fallback paper (C-F, corrected H22, baselines, C4)')

    d1_summary = {k: v for k, v in d1.items() if k != 'ratios'}
    result = {'dataset': conf['dataset'], 'encoder': conf['encoder'], 'split': split, 'thresholds': thr,
              'groups': data.groups.sizes(), 'donor_pooling': d1_summary, 'cross_genus': da, 'genus_rescue': db,
              'c2_covariance': dc2, 'checks': {k: {'value': v, 'pass': bool(ok)} for k, (v, ok) in checks.items()},
              'verdict': verdict}
    out = args.out or 'results/week1_gate_{}_{}.json'.format(conf['dataset'], conf['encoder'])
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w') as f:
        json.dump(result, f, indent=1, default=float)
    print()
    for k, (v, ok) in checks.items():
        print('  {:28s} {:8.4f}  {}'.format(k, v, 'PASS' if ok else 'FAIL'))
    print('  tail FPs: {}  cross-genus CI95: [{:.1%}, {:.1%}]'.format(
        da['n_tail_fp'], *da['ci95']['cross_genus']))
    print('=> verdict: {}\n=> {}'.format(verdict, out))


if __name__ == '__main__':
    main()
