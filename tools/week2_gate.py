"""Week-2 gate (PLAN_NEXT.md §5), on V-select only, from the saved predictions of tools/run_baselines.py.

    python tools/week2_gate.py --encoder dinov2_b14 [--dataset plantnet] [--results results/all_runs.csv]

Selection rules (fixed before any C2 result exists, so nothing is picked by hand):
  * best baseline = the method WITHOUT '+c2' / '+c2shuf' with the highest V-select macro-F1
  * C2            = the '<base>+c2' method with the highest V-select macro-F1
  * A9 control    = '<same base>+c2shuf'; A9 base = '<same base>' (without C2)
Criteria:
  1. C2 beats the best baseline on macro-F1 AND on few-shot accuracy (pooled image-level, plan §8.0):
     for each metric the paired-bootstrap 95% CI of (C2 - baseline) is above 0, or the exact McNemar test on the
     images concerned gives p < 0.05 with C2 ahead.
  2. A9: real C2 beats shuffled C2 on macro-F1 (same test), and the shuffled run keeps less than
     `a9_max_gain_retained` of the gain over the A9 base: (F1(shuf) - F1(base)) / (F1(c2) - F1(base)).
  3. Seed spread of every method on test (reported, not a criterion).
The result goes to results/week2_gate_<dataset>_<encoder>.json; an existing file is never overwritten.
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


def tag(method):
    return method.replace('@', '-').replace('=', '').replace(',', '-')


def load_pred(preds_dir, dataset, encoder, method, split='vselect', seed=0):
    path = os.path.join(preds_dir, '{}_{}_{}_{}_seed{}.npz'.format(dataset, encoder, tag(method), split, seed))
    d = np.load(path)
    logits = d['logits'].astype(np.float64)
    # post-hoc LA uses the prior the model was trained with (C2 with c2_prior='augmented' saves prior_counts)
    counts = np.maximum((d['prior_counts'] if 'prior_counts' in d.files else d['counts']).astype(np.float64), 1)
    tau = float(d['tau'])
    if tau:
        logits = logits - tau * np.log(counts / counts.sum())[None, :]   # the method's post-hoc LA
    return {'y': d['y'], 'pred': logits.argmax(1), 'few_end': int(d['few_end']), 'C': logits.shape[1], 'path': path}


def read_rows(path, dataset, encoder):
    with open(path, newline='', encoding='utf-8') as f:
        return [r for r in csv.DictReader(f) if r['dataset'] == dataset and r['encoder'] == encoder]


def is_c2(m):
    return '+c2' in m


def compare(a, b, label_a, label_b, n_boot, seed=0, stratified=False, f1_test='legacy'):
    """a vs b on the same images: macro-F1 and pooled few-shot accuracy.

    McNemar is two-sided on top-1 correctness, so it counts for `a` only when `a` is AHEAD on the images
    concerned (more images only `a` gets right). Before 2026-11-18 the macro-F1 criterion accepted any McNemar
    p < 0.05 with a positive macro-F1 delta, even when the other method was ahead on top-1 (DINOv2, session 5).

    stratified=True: class-stratified bootstrap for the reported CIs (lt.metrics.paired_bootstrap).
    f1_test: how the macro-F1 criterion is decided
      'legacy'        the rule fixed on Nov 16: bootstrap CI above 0, or McNemar p < 0.05 with `a` ahead
      'randomization' paired randomization test, one-sided p < 0.05 (McNemar tests top-1, not macro-F1; the
                      bootstrap CI is reported but does not decide)"""
    if f1_test not in ('legacy', 'randomization'):
        raise ValueError('f1_test must be legacy or randomization')
    y, C, few_end = a['y'], a['C'], a['few_end']
    if not np.array_equal(y, b['y']):
        raise SystemExit('{} and {} were evaluated on different images'.format(label_a, label_b))
    out = {}
    f1 = lambda t, p: M.macro_f1(t, p, C)                                            # noqa: E731
    d, ci, sig = M.paired_bootstrap(f1, y, a['pred'], b['pred'], n_boot=n_boot, seed=seed, stratified=stratified)
    n_a, n_b, p = M.mcnemar(y, a['pred'], b['pred'])
    out['macro_f1'] = {label_a: f1(y, a['pred']), label_b: f1(y, b['pred']), 'delta': d, 'ci95': ci,
                       'mcnemar_p_all_images': p, 'mcnemar_a_only_right': n_a, 'mcnemar_b_only_right': n_b,
                       'decided_by': f1_test}
    if f1_test == 'legacy':
        out['macro_f1']['pass'] = bool(ci[0] > 0 or (p < 0.05 and n_a > n_b and d > 0))
    else:
        _, p_gt, p_two = M.paired_randomization(f1, y, a['pred'], b['pred'], n_perm=n_boot, seed=seed)
        out['macro_f1'].update(randomization_p_greater=p_gt, randomization_p_two_sided=p_two,
                               **{'pass': bool(p_gt < 0.05)})
    few = y < few_end
    acc = lambda t, q: float((t == q).mean())                                        # noqa: E731
    d, ci, _ = M.paired_bootstrap(acc, y[few], a['pred'][few], b['pred'][few], n_boot=n_boot, seed=seed,
                                  stratified=stratified)
    n_a, n_b, p = M.mcnemar(y[few], a['pred'][few], b['pred'][few])
    out['few_shot_pooled_acc'] = {label_a: acc(y[few], a['pred'][few]), label_b: acc(y[few], b['pred'][few]),
                                  'n_images': int(few.sum()), 'delta': d, 'ci95': ci, 'mcnemar_p': p,
                                  'mcnemar_a_only_right': n_a, 'mcnemar_b_only_right': n_b,
                                  'pass': bool(ci[0] > 0 or (p < 0.05 and n_a > n_b))}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--encoder', required=True)
    p.add_argument('--dataset', default='plantnet')
    p.add_argument('--results', default='results/all_runs.csv')
    p.add_argument('--preds', default='results/preds')
    p.add_argument('--thresholds', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gates',
                                                        'week1_thresholds.json'))
    p.add_argument('--out', default=None)
    p.add_argument('--n_boot', type=int, default=5000)
    p.add_argument('--stratified', action='store_true',
                   help='class-stratified bootstrap (default off: the rule fixed on Nov 16 used image resampling)')
    args = p.parse_args()
    with open(args.thresholds, encoding='utf-8') as f:
        thr = json.load(f)
    out_path = args.out or os.path.join(os.path.dirname(args.results) or '.',
                                        'week2_gate_{}_{}.json'.format(args.dataset, args.encoder))
    if os.path.exists(out_path):
        raise SystemExit('{} exists: the gate is recorded once; pass --out for a re-run'.format(out_path))

    rows = read_rows(args.results, args.dataset, args.encoder)
    vsel = {}
    for r in rows:
        if r['stage'] == 'vselect':
            vsel[r['method']] = float(r['macro_f1'])                    # last row wins (a re-run after a fix)
    test = defaultdict(list)
    for r in rows:
        if r['stage'] == 'final' and r['split'] == 'test':
            test[r['method']].append(float(r['macro_f1']))
    baselines = {m: v for m, v in vsel.items() if not is_c2(m)}
    c2s = {m: v for m, v in vsel.items() if m.split('@')[0].endswith('+c2')}
    if not baselines or not c2s:
        raise SystemExit('need V-select rows for at least one baseline and one +c2 method (have: {})'.format(sorted(vsel)))
    best_base = max(baselines, key=baselines.get)
    best_c2 = max(c2s, key=c2s.get)
    base_of_c2 = best_c2.replace('+c2', '', 1)
    shuf = best_c2.replace('+c2', '+c2shuf', 1)

    def pred(m):
        return load_pred(args.preds, args.dataset, args.encoder, m)
    res = {'dataset': args.dataset, 'encoder': args.encoder, 'split': 'vselect', 'thresholds': thr,
           'rules': {'best_baseline': 'non-C2 method with the highest V-select macro-F1',
                     'c2_method': '+c2 method with the highest V-select macro-F1',
                     'a9': 'same base with +c2shuf; gain measured over the same base without C2',
                     'significance': 'paired bootstrap 95% CI > 0, or McNemar p < 0.05 with C2 ahead'},
           'vselect_macro_f1': dict(sorted(vsel.items(), key=lambda kv: -kv[1])),
           'best_baseline': best_base, 'c2_method': best_c2}
    a = pred(best_c2)
    res['bootstrap'] = 'class-stratified' if args.stratified else 'image'
    res['criterion1'] = compare(a, pred(best_base), best_c2, best_base, args.n_boot, stratified=args.stratified)
    c1 = res['criterion1']['macro_f1']['pass'] and res['criterion1']['few_shot_pooled_acc']['pass']

    c2_ok = None
    if shuf in vsel and base_of_c2 in vsel:
        cmp = compare(a, pred(shuf), best_c2, shuf, args.n_boot, stratified=args.stratified)
        f_real, f_shuf, f_base = vsel[best_c2], vsel[shuf], vsel[base_of_c2]
        gain = f_real - f_base
        retained = (f_shuf - f_base) / gain if gain > 0 else float('inf')
        beats = cmp['macro_f1']['pass']
        c2_ok = bool(beats and retained < thr['a9_max_gain_retained'])
        res['criterion2_a9'] = {'real_vs_shuffled': cmp, 'a9_base': base_of_c2, 'f1_base': f_base, 'f1_real': f_real,
                                'f1_shuffled': f_shuf, 'gain_real': gain, 'share_of_gain_retained_by_shuffled': retained,
                                'pass': c2_ok}
    else:
        res['criterion2_a9'] = {'missing': [m for m in (shuf, base_of_c2) if m not in vsel], 'pass': None}
    res['seed_spread_test_macro_f1'] = {m: {'mean': float(np.mean(v)), 'std': float(np.std(v, ddof=1)) if len(v) > 1
                                            else 0.0, 'n': len(v)} for m, v in sorted(test.items())}
    res['verdict'] = ('PASS: method paper (C2)' if c1 and c2_ok else
                      'INCOMPLETE: A9 runs missing' if c1 and c2_ok is None else
                      'FAIL: benchmark/analysis paper (PLAN_NEXT §5); C2 reported as a modest component')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1)

    print('Week-2 gate, {} / {} (V-select)'.format(args.dataset, args.encoder))
    print('  best baseline: {} ({:.4f})   C2: {} ({:.4f})'.format(best_base, vsel[best_base], best_c2, vsel[best_c2]))
    for k, v in res['criterion1'].items():
        print('  1. {:20s} delta {:+.4f} CI [{:+.4f}, {:+.4f}]  {}'.format(k, v['delta'], v['ci95'][0], v['ci95'][1],
                                                                       'PASS' if v['pass'] else 'FAIL'))
    a9 = res['criterion2_a9']
    if a9['pass'] is None:
        print('  2. A9: missing runs {}'.format(a9['missing']))
    else:
        print('  2. A9: shuffled keeps {:.0%} of the gain (max {:.0%}); real vs shuffled {}  {}'.format(
            a9['share_of_gain_retained_by_shuffled'], thr['a9_max_gain_retained'],
            'PASS' if a9['real_vs_shuffled']['macro_f1']['pass'] else 'FAIL', 'PASS' if a9['pass'] else 'FAIL'))
    print('  verdict: {}  -> {}'.format(res['verdict'], out_path))


if __name__ == '__main__':
    main()
