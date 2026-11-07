"""Finding C-F report (plan §4) from saved predictions (results/preds/*.npz, written by run_baselines).

    python tools/cf_report.py --preds results/preds/plantnet_clip_b32_la_test_seed0.npz \
        --taxonomy taxonomy/plantnet.csv --out results/cf

Writes <name>_cf.json (breakdown with bootstrap 95% CIs at the model's tuned tau), <name>_cf_tau.csv
(share vs tau from 0 to --tau_max), <name>_cf_count.csv (share vs the predicted tail class's train count)
and, if matplotlib is installed, <name>_cf.png.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lt import cf  # noqa: E402
from lt.taxonomy import Taxonomy  # noqa: E402


class _Index(object):
    """Minimal ClassIndex stand-in rebuilt from a predictions file."""

    def __init__(self, class_map):
        self.class_map = np.asarray(class_map)
        self.num_classes = len(self.class_map)


def load_preds(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def write_csv(rows, path):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def report(pred_path, taxonomy_csv, out_dir, tau=None, tau_max=None, n_boot=2000):
    d = load_preds(pred_path)
    tax = Taxonomy.from_csv(taxonomy_csv, _Index(d['class_map']))
    counts = d['counts']
    log_pi = np.log(np.maximum(counts, 1) / counts.sum())
    logits, y, few_end = d['logits'].astype(np.float32), d['y'], int(d['few_end'])
    tau = float(d['tau']) if tau is None else tau
    pred = cf.predictions(logits, log_pi, tau)
    res = cf.shares_with_ci(y, pred, few_end, tax.genus_of, tax.family_of, n_boot=n_boot)
    res.update(source=pred_path, tau=tau, few_end=few_end, top1=float((pred == y).mean()),
               method=str(d['method']), encoder=str(d['encoder']), dataset=str(d['dataset']))
    taus = np.linspace(0, tau_max if tau_max is not None else max(2.0, tau), 21)
    tau_rows = cf.tau_curve(logits, y, log_pi, few_end, tax.genus_of, tax.family_of, taus)
    count_rows = cf.count_curve(y, pred, few_end, tax.genus_of, tax.family_of, counts)

    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(pred_path))[0]
    with open(os.path.join(out_dir, name + '_cf.json'), 'w') as f:
        json.dump(res, f, indent=1)
    write_csv(tau_rows, os.path.join(out_dir, name + '_cf_tau.csv'))
    write_csv(count_rows, os.path.join(out_dir, name + '_cf_count.csv'))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        t = [r['tau'] for r in tau_rows]
        for k in ('same_genus', 'same_family', 'diff_family'):
            ax[0].plot(t, [r[k] for r in tau_rows], label=k.replace('_', ' '))
        ax[0].set_xlabel('tau'); ax[0].set_ylabel('share of tail false positives'); ax[0].legend()
        ax2 = ax[0].twinx()
        ax2.plot(t, [r['n_tail_fp'] for r in tau_rows], color='0.6', ls='--')
        ax2.set_ylabel('tail false positives (dashed)')
        labels = ['{}-{}'.format(r['count_lo'], r['count_hi'] - 1) for r in count_rows]
        ax[1].bar(labels, [r['cross_genus'] for r in count_rows])
        ax[1].set_xlabel('train images of the predicted tail class'); ax[1].set_ylabel('cross-genus share')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, name + '_cf.png'), dpi=150)
    except ImportError:
        pass
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--preds', required=True, nargs='+')
    p.add_argument('--taxonomy', required=True)
    p.add_argument('--out', default='results/cf')
    p.add_argument('--tau', type=float, default=None, help='default: the tau saved with the predictions')
    p.add_argument('--tau_max', type=float, default=None)
    p.add_argument('--n_boot', type=int, default=2000)
    args = p.parse_args()
    for path in args.preds:
        r = report(path, args.taxonomy, args.out, args.tau, args.tau_max, args.n_boot)
        s, ci = r['share'], r['ci95']
        print('{}: {} tail FPs of {} images (tau {:.2f})'.format(os.path.basename(path), r['n_tail_fp'], r['n_images'], r['tau']))
        for k in ('same_genus', 'same_family', 'diff_family', 'unknown_family', 'cross_genus'):
            print('   {:15s} {:6.1%}  [{:.1%}, {:.1%}]'.format(k, s[k], ci[k][0], ci[k][1]))


if __name__ == '__main__':
    main()
