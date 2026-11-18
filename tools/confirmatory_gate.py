"""Confirmatory gate: apply a PRE-REGISTERED rules file (e.g. gates/h22_confirmatory.json) to finished runs.

    python tools/confirmatory_gate.py --rules gates/h22_confirmatory.json --encoder dinov2_b14
    python tools/confirmatory_gate.py --rules gates/h22_confirmatory.json --summary

Everything that decides the outcome is read from the rules file, which must be committed BEFORE the runs it
judges (the git log is the record). Per encoder:
  selection   best baseline = highest V-select macro-F1 among rules["baselines"] (favours the baseline);
              C2 method     = highest V-tune score among rules["c2_methods"] (never V-select or the evaluation split)
  criterion 1 C2 vs best baseline on rules["evaluation"]["split"] (training seed rules["evaluation"]["seed"]):
              macro-F1 by paired randomization test (one-sided p < alpha) AND pooled few-shot accuracy by exact
              McNemar with C2 ahead (p < alpha). Class-stratified bootstrap CIs are reported, not decisive.
  criterion 2 A9: the shuffled-taxonomy controls of the chosen C2 method. Real C2 must beat EVERY shuffle seed
              (paired randomization, one-sided p < alpha), and the shuffled runs' mean macro-F1 must keep less
              than rules["a9"]["max_gain_retained"] of C2's gain over its own base.
--summary combines the per-encoder results into the verdict in rules["verdict"]. Results are written once to
results/confirmatory_<dataset>_<encoder>.json and never overwritten.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lt import metrics as M  # noqa: E402
from lt.harness import base_method_name  # noqa: E402
from tools.week2_gate import compare, load_pred  # noqa: E402


def read_rows(path, dataset, encoder):
    with open(path, newline='', encoding='utf-8') as f:
        return [r for r in csv.DictReader(f) if r['dataset'] == dataset and r['encoder'] == encoder]


def a9_names(rules, c2_method):
    return [c2_method.replace('+c2f', '+c2fshuf', 1) + ('' if s == 0 else '@c2_shuffle_seed={}'.format(s))
            for s in rules['a9']['shuffle_seeds']]


def judge(rules, encoder, results, preds, n_perm):
    ds, ev = rules['dataset'], rules['evaluation']
    alpha = rules['criteria']['alpha']
    rows = read_rows(results, ds, encoder)
    vsel, vtune, evalf1 = {}, defaultdict(lambda: -np.inf), {}
    for r in rows:
        if r['stage'] == 'vselect':
            vsel[r['method']] = float(r['macro_f1'])
        elif r['stage'] == 'trial':
            vtune[r['method']] = max(vtune[r['method']], float(r['vtune_score']))
        elif r['stage'] == 'final' and r['split'] == ev['split'] and int(r['seed']) == ev['seed']:
            evalf1[r['method']] = float(r['macro_f1'])
    missing = [m for m in rules['baselines'] if m not in vsel] + [m for m in rules['c2_methods'] if m not in vtune]
    if missing:
        raise SystemExit('{}: runs missing for {}'.format(encoder, missing))
    best_base = max(rules['baselines'], key=lambda m: vsel[m])
    c2 = max(rules['c2_methods'], key=lambda m: vtune[m])
    own_base = base_method_name(c2)
    shufs = a9_names(rules, c2)
    need = [m for m in [c2, best_base, own_base] + shufs if m not in evalf1]
    if need:
        raise SystemExit('{}: no {} seed-{} rows for {}'.format(encoder, ev['split'], ev['seed'], need))

    def pred(m):
        return load_pred(preds, ds, encoder, m, split=ev['split'], seed=ev['seed'])
    a = pred(c2)
    res = {'dataset': ds, 'encoder': encoder, 'rules_file': rules['_file'], 'evaluation': ev,
           'selection': {'best_baseline': best_base, 'best_baseline_vselect_macro_f1': vsel[best_base],
                         'baselines_vselect_macro_f1': {m: vsel[m] for m in rules['baselines']},
                         'c2_method': c2, 'c2_methods_vtune_score': {m: vtune[m] for m in rules['c2_methods']}}}
    c1 = compare(a, pred(best_base), c2, best_base, n_perm, stratified=True, f1_test='randomization')
    c1['macro_f1']['pass'] = bool(c1['macro_f1']['randomization_p_greater'] < alpha)
    fs = c1['few_shot_pooled_acc']
    fs['pass'] = bool(fs['mcnemar_p'] < alpha and fs['mcnemar_a_only_right'] > fs['mcnemar_b_only_right'])
    res['criterion1'] = c1
    res['criterion1']['pass'] = c1['macro_f1']['pass'] and fs['pass']
    res['same_base_reported'] = {'base': own_base, 'macro_f1_base': evalf1[own_base], 'macro_f1_c2': evalf1[c2],
                                 'delta': evalf1[c2] - evalf1[own_base]}

    gain = evalf1[c2] - evalf1[own_base]
    per_seed = {}
    for s in shufs:
        cmp = compare(a, pred(s), c2, s, n_perm, stratified=True, f1_test='randomization')
        per_seed[s] = {'macro_f1_shuffled': evalf1[s], 'randomization_p_greater': cmp['macro_f1']['randomization_p_greater'],
                       'ci95_stratified': cmp['macro_f1']['ci95'], 'real_beats_shuffled':
                       bool(cmp['macro_f1']['randomization_p_greater'] < alpha)}
    mean_shuf = float(np.mean([evalf1[s] for s in shufs]))
    retained = (mean_shuf - evalf1[own_base]) / gain if gain > 0 else float('inf')
    a9_ok = all(v['real_beats_shuffled'] for v in per_seed.values()) and retained < rules['a9']['max_gain_retained']
    res['criterion2_a9'] = {'base': own_base, 'gain_over_base': gain, 'mean_shuffled_macro_f1': mean_shuf,
                            'share_of_gain_retained': retained, 'per_shuffle_seed': per_seed, 'pass': bool(a9_ok)}
    res['pass'] = bool(res['criterion1']['pass'] and a9_ok)
    return res


def summarize(rules, results_dir):
    out = {}
    for enc in rules['encoders']['primary'] + rules['encoders'].get('reported', []):
        p = os.path.join(results_dir, 'confirmatory_{}_{}.json'.format(rules['dataset'], enc))
        if os.path.isfile(p):
            with open(p, encoding='utf-8') as f:
                out[enc] = json.load(f)['pass']
    primary = rules['encoders']['primary']
    if any(e not in out for e in primary):
        return {'per_encoder': out, 'verdict': 'INCOMPLETE: primary encoders missing {}'.format(
            [e for e in primary if e not in out])}
    n = sum(out[e] for e in primary)
    key = 'all_primary_pass' if n == len(primary) else 'some_primary_pass' if n else 'no_primary_pass'
    return {'per_encoder': out, 'verdict': rules['verdict'][key]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rules', required=True)
    p.add_argument('--encoder')
    p.add_argument('--summary', action='store_true')
    p.add_argument('--results', default='results/all_runs.csv')
    p.add_argument('--preds', default='results/preds')
    p.add_argument('--n_perm', type=int, default=None, help='default: rules["criteria"]["n_perm"]')
    args = p.parse_args()
    with open(args.rules, encoding='utf-8') as f:
        rules = json.load(f)
    rules['_file'] = args.rules
    out_dir = os.path.dirname(args.results) or '.'
    if args.summary:
        s = summarize(rules, out_dir)
        print(json.dumps(s, indent=1))
        return
    if not args.encoder:
        raise SystemExit('pass --encoder or --summary')
    out_path = os.path.join(out_dir, 'confirmatory_{}_{}.json'.format(rules['dataset'], args.encoder))
    if os.path.exists(out_path):
        raise SystemExit('{} exists: a confirmatory result is recorded once'.format(out_path))
    res = judge(rules, args.encoder, args.results, args.preds, args.n_perm or rules['criteria']['n_perm'])
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1)
    c1, a9 = res['criterion1'], res['criterion2_a9']
    print('Confirmatory gate, {} / {} ({} seed {})'.format(rules['dataset'], args.encoder, res['evaluation']['split'],
                                                            res['evaluation']['seed']))
    print('  best baseline {} | C2 {}'.format(res['selection']['best_baseline'], res['selection']['c2_method']))
    print('  1. macro-F1 delta {:+.4f}, randomization p {:.4f}  {}'.format(
        c1['macro_f1']['delta'], c1['macro_f1']['randomization_p_greater'], 'PASS' if c1['macro_f1']['pass'] else 'FAIL'))
    print('     few-shot pooled delta {:+.4f}, McNemar p {:.4f}  {}'.format(
        c1['few_shot_pooled_acc']['delta'], c1['few_shot_pooled_acc']['mcnemar_p'],
        'PASS' if c1['few_shot_pooled_acc']['pass'] else 'FAIL'))
    print('  2. A9: shuffled keeps {:.0%} of the gain; real beats every shuffle seed: {}  {}'.format(
        a9['share_of_gain_retained'], all(v['real_beats_shuffled'] for v in a9['per_shuffle_seed'].values()),
        'PASS' if a9['pass'] else 'FAIL'))
    print('  => {}  -> {}'.format('PASS' if res['pass'] else 'FAIL', out_path))


if __name__ == '__main__':
    main()
