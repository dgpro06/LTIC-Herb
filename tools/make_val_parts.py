"""Split a validation file into V-tune / V-select / V-cal, stratified by class (plan §3.5, §5.2 item 5),
and run the day-1 tail-count check (plan §8.0).

    python tools/make_val_parts.py --train splits/plantnet_train.txt --val splits/plantnet_val.txt \
        --out splits --prefix plantnet --few_thr 20 --auto_ratios

Writes <prefix>_vtune.txt, <prefix>_vselect.txt, <prefix>_vcal.txt and <prefix>_vparts.json
(chosen ratios, per-part image / class counts per shot group, the §8.0 decision).

--auto_ratios applies §8.0: keep 40/30/30 if V-select has >= 500 few-shot images, else switch to
35/40/25; if still < 500, keep 35/40/25 and record that the few-shot gate uses pooled accuracy.
Every class is spread over the parts as evenly as its image count allows (low-discrepancy assignment
with a random per-class offset); there is no per-class minimum (split conformal is marginal over images).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lt.classes import read_split_file  # noqa: E402

PARTS = ('vtune', 'vselect', 'vcal')
GOLDEN = (np.sqrt(5) - 1) / 2


def assign_parts(labels, ratios, seed):
    """Return part index (0..2) per image, stratified by class."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    cum = np.cumsum(ratios) / np.sum(ratios)
    part = np.empty(len(labels), dtype=np.int64)
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        idx = idx[rng.permutation(len(idx))]
        u = (rng.random() + GOLDEN * np.arange(len(idx))) % 1.0
        part[idx] = np.searchsorted(cum, u, side='right')
    return np.minimum(part, len(ratios) - 1)


def summarize(labels, part, few_classes):
    out = {}
    for k, name in enumerate(PARTS):
        sel = part == k
        lab = labels[sel]
        few = np.isin(lab, few_classes)
        out[name] = {'images': int(sel.sum()), 'classes': int(len(np.unique(lab))),
                     'few_images': int(few.sum()), 'few_classes': int(len(np.unique(lab[few])))}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--train', required=True, help='train split file (for shot groups)')
    p.add_argument('--val', required=True, help='validation split file to divide')
    p.add_argument('--out', required=True)
    p.add_argument('--prefix', required=True)
    p.add_argument('--few_thr', type=int, required=True)
    p.add_argument('--ratios', default='40,30,30', help='V-tune,V-select,V-cal shares')
    p.add_argument('--auto_ratios', action='store_true', help='apply the plan §8.0 rule')
    p.add_argument('--min_tail', type=int, default=500)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    _, train_labels = read_split_file(args.train)
    paths, labels = read_split_file(args.val)
    counts = np.bincount(train_labels, minlength=int(max(train_labels.max(), labels.max())) + 1)
    few_classes = np.flatnonzero((counts > 0) & (counts < args.few_thr))

    ratios = [float(r) for r in args.ratios.split(',')]
    part = assign_parts(labels, ratios, args.seed)
    summary = summarize(labels, part, few_classes)
    decision = 'fixed ratios'
    if args.auto_ratios:
        if summary['vselect']['few_images'] >= args.min_tail:
            decision = 'kept {} (V-select few-shot images >= {})'.format(args.ratios, args.min_tail)
        else:
            ratios = [35.0, 40.0, 25.0]
            part = assign_parts(labels, ratios, args.seed)
            summary = summarize(labels, part, few_classes)
            if summary['vselect']['few_images'] >= args.min_tail:
                decision = 'switched to 35/40/25'
            else:
                decision = ('switched to 35/40/25, still < {} few-shot images: the few-shot gate criterion uses '
                            'POOLED image-level tail accuracy; macro-F1 stays strict'.format(args.min_tail))

    os.makedirs(args.out, exist_ok=True)
    for k, name in enumerate(PARTS):
        out = os.path.join(args.out, '{}_{}.txt'.format(args.prefix, name))
        with open(out, 'w') as f:
            for i in np.flatnonzero(part == k):
                f.write('{} {}\n'.format(paths[i], labels[i]))
        print('{:8s} {:7d} images  {:6d} classes  few-shot: {:6d} images in {:5d} classes -> {}'.format(
            name, summary[name]['images'], summary[name]['classes'], summary[name]['few_images'],
            summary[name]['few_classes'], out))

    report = {'val_file': os.path.abspath(args.val), 'train_file': os.path.abspath(args.train),
              'seed': args.seed, 'few_thr': args.few_thr, 'ratios': ratios, 'decision_8_0': decision,
              'few_classes_in_train': int(len(few_classes)), 'parts': summary}
    with open(os.path.join(args.out, '{}_vparts.json'.format(args.prefix)), 'w') as f:
        json.dump(report, f, indent=1)
    print('plan 8.0 decision: ' + decision)
    print('Record these counts and ratios in the project notes before any gate is evaluated.')


if __name__ == '__main__':
    main()
