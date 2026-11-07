"""C4 prediction sets (plan §3.5): calibrate on V-cal, report on test.

    python tools/conformal_eval.py --cal results/preds/plantnet_clip_b32_la_vcal_seed0.npz \
        --test results/preds/plantnet_clip_b32_la_test_seed0.npz --taxonomy taxonomy/plantnet.csv

Logits are adjusted with the model's tuned tau before the softmax. Writes a CSV with one row per
(procedure, alpha, K): coverage, macro-coverage, classes below 50% coverage, set size in leaves,
fallback rate; plus the same metrics on V-cal as a sanity check (CP-3 must reach >= 1 - alpha there).
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lt.conformal import GenusFallbackCP  # noqa: E402
from lt.taxonomy import Taxonomy  # noqa: E402
from tools.cf_report import _Index, load_preds  # noqa: E402


def adjusted(d):
    counts = d['counts']
    log_pi = np.log(np.maximum(counts, 1) / counts.sum())
    return d['logits'].astype(np.float32) - float(d['tau']) * log_pi[None, :]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cal', required=True)
    p.add_argument('--test', required=True)
    p.add_argument('--taxonomy', required=True)
    p.add_argument('--alphas', default='0.05,0.1')
    p.add_argument('--K', default='3,5,10')
    p.add_argument('--score', default='pas', choices=['pas', 'softmax'])
    p.add_argument('--out', default=None)
    args = p.parse_args()

    cal, test = load_preds(args.cal), load_preds(args.test)
    if not np.array_equal(cal['class_map'], test['class_map']):
        raise SystemExit('V-cal and test predictions come from different class orders')
    tax = Taxonomy.from_csv(args.taxonomy, _Index(cal['class_map']))
    cp = GenusFallbackCP(cal['counts'], tax.genus_of, score=args.score)
    zc, zt = adjusted(cal), adjusted(test)
    rows = []
    for alpha in [float(a) for a in args.alphas.split(',')]:
        cp.fit(zc, cal['y'], alpha)
        for proc in ('CP-1', 'CP-2', 'CP-3', 'CP-4'):
            for K in ([int(k) for k in args.K.split(',')] if proc in ('CP-3', 'CP-4') else ['']):
                for split, z, y in (('vcal', zc, cal['y']), ('test', zt, test['y'])):
                    m = cp.metrics(cp.predict(z, proc, K or 5), y)
                    rows.append(dict(procedure=proc, alpha=alpha, K=K, split=split, score=args.score, **m))
                    if split == 'test':
                        print('{} alpha={:.2f} K={:>2}: coverage {:.3f} macro {:.3f} <50%: {:4d} size {:7.2f} fallback {:.2f}'.format(
                            proc, alpha, K, m['coverage'], m['macro_coverage'], m['classes_below_50pct'],
                            m['avg_set_size_leaves'], m['fallback_rate']))
    out = args.out or os.path.join('results', 'conformal', os.path.basename(args.test).replace('.npz', '_cp.csv'))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print('=> {}'.format(out))


if __name__ == '__main__':
    main()
