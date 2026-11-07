"""Leakage-free Herbarium 2022 split (plan §5.2) plus the H22 taxonomy table.

Official metadata (use this for every reported number):
    python tools/make_h22_clean.py --metadata /kaggle/input/herbarium-2022-fgvc9/train_metadata.json \
        --out splits --path_prefix train_images/

Dry run on the local IJCNN files (780,955 unique images, NOT the official 839,772):
    python tools/make_h22_clean.py --txt train_hbm.txt --txt val_hbm.txt --out splits_local

Rules (plan §5.2):
  * classes with n >= 3: test = val = max(1, floor(0.1 n + 0.5)), train = the rest
  * classes with n <= 2: all images to train (absent from val and test)
  * labels are renumbered to 0..K-1 over the categories that have images (K = 15,501 for H22)
Writes h22clean_{train,val,test}.txt, h22clean_label_map.json, taxonomy/h22.csv (with --metadata)
and h22clean_stats.json. Then split val with tools/make_val_parts.py --prefix h22clean.
"""
import argparse
import collections
import csv
import json
import math
import os

import numpy as np


def per_class_split(n):
    if n <= 2:
        return n, 0, 0
    k = max(1, int(math.floor(0.1 * n + 0.5)))
    return n - 2 * k, k, k


def load_metadata(path, prefix):
    with open(path) as f:
        meta = json.load(f)
    files = {img['image_id']: img['file_name'] for img in meta['images']}
    items = [(prefix + files[a['image_id']], int(a['category_id'])) for a in meta['annotations']]
    cats = {int(c['category_id']): c for c in meta.get('categories', [])}
    return items, cats


def load_txt(paths):
    seen = {}
    for txt in paths:
        with open(txt) as f:
            for line in f:
                parts = line.split()
                if parts:
                    seen[parts[0]] = int(parts[1])
    return sorted(seen.items()), {}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--metadata', help='official H22 train_metadata.json')
    src.add_argument('--txt', action='append', help='"path label" files; union, de-duplicated by path')
    p.add_argument('--path_prefix', default='train_images/', help='prepended to metadata file names')
    p.add_argument('--out', required=True)
    p.add_argument('--taxonomy_out', default='taxonomy/h22.csv')
    p.add_argument('--few_thr', type=int, default=20)
    p.add_argument('--many_thr', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    items, cats = load_metadata(args.metadata, args.path_prefix) if args.metadata else load_txt(args.txt)
    paths = [x[0] for x in items]
    if len(set(paths)) != len(paths):
        raise SystemExit('duplicate image paths in the source')
    by_cat = collections.defaultdict(list)
    for path, cat in items:
        by_cat[cat].append(path)
    cat_ids = sorted(by_cat)
    label_of = {c: i for i, c in enumerate(cat_ids)}
    if cats:
        no_images = sorted(set(cats) - set(cat_ids))
    else:
        no_images = sorted(set(range(max(cat_ids) + 1)) - set(cat_ids))
    print('{} images, {} categories with images; category ids without images: {}'.format(
        len(items), len(cat_ids), no_images[:20]))

    rng = np.random.default_rng(args.seed)
    splits = {'train': [], 'val': [], 'test': []}
    n_small, n_3to4 = 0, 0
    train_counts = np.zeros(len(cat_ids), dtype=np.int64)
    for cat in cat_ids:
        imgs = sorted(by_cat[cat])
        imgs = [imgs[i] for i in rng.permutation(len(imgs))]
        n_tr, n_va, n_te = per_class_split(len(imgs))
        n_small += len(imgs) <= 2
        n_3to4 += 3 <= len(imgs) <= 4
        lab = label_of[cat]
        splits['test'] += [(x, lab) for x in imgs[:n_te]]
        splits['val'] += [(x, lab) for x in imgs[n_te:n_te + n_va]]
        splits['train'] += [(x, lab) for x in imgs[n_te + n_va:]]
        train_counts[lab] = n_tr

    sets = {k: {x for x, _ in v} for k, v in splits.items()}
    assert not (sets['train'] & sets['val']) and not (sets['train'] & sets['test']) and not (sets['val'] & sets['test']), \
        'path overlap between splits'

    os.makedirs(args.out, exist_ok=True)
    for name, rows in splits.items():
        out = os.path.join(args.out, 'h22clean_{}.txt'.format(name))
        with open(out, 'w') as f:
            for x, lab in rows:
                f.write('{} {}\n'.format(x, lab))
        print('{:5s} {:7d} images -> {}'.format(name, len(rows), out))

    with open(os.path.join(args.out, 'h22clean_label_map.json'), 'w') as f:
        json.dump({str(i): {'category_id': c, 'name': cats.get(c, {}).get('scientificName')}
                   for i, c in enumerate(cat_ids)}, f, indent=1)

    if cats:
        os.makedirs(os.path.dirname(args.taxonomy_out) or '.', exist_ok=True)
        with open(args.taxonomy_out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['label', 'source_id', 'species', 'genus', 'family'])
            for i, c in enumerate(cat_ids):
                m = cats[c]
                species = m.get('scientificName') or '{} {}'.format(m.get('genus', ''), m.get('species', '')).strip()
                w.writerow([i, c, species, m.get('genus', ''), m.get('family', '')])
        print('taxonomy -> {}'.format(args.taxonomy_out))

    tc = train_counts
    groups = {'few': int((tc < args.few_thr).sum()),
              'medium': int(((tc >= args.few_thr) & (tc <= args.many_thr)).sum()),
              'many': int((tc > args.many_thr).sum())}
    full_counts = np.array([len(by_cat[c]) for c in cat_ids])
    stats = {'source': args.metadata or args.txt, 'seed': args.seed, 'images': len(items),
             'num_classes': len(cat_ids), 'category_ids_without_images': no_images,
             'largest_class_all_images': int(full_counts.max()), 'largest_class_train': int(tc.max()),
             'classes_n_le_2_train_only': int(n_small), 'classes_n_3_to_4_now_1_2_shot': int(n_3to4),
             'split_images': {k: len(v) for k, v in splits.items()},
             'thresholds': {'few_thr': args.few_thr, 'many_thr': args.many_thr},
             'shot_groups_clean_train': groups,
             'train_count_tertiles': [float(q) for q in np.quantile(tc, [1 / 3, 2 / 3])]}
    with open(os.path.join(args.out, 'h22clean_stats.json'), 'w') as f:
        json.dump(stats, f, indent=1)
    print(json.dumps({k: stats[k] for k in ('num_classes', 'largest_class_all_images', 'classes_n_le_2_train_only',
                                            'classes_n_3_to_4_now_1_2_shot', 'shot_groups_clean_train')}, indent=1))
    print('Train with --num_classes {} --few_thr {} --many_thr {}'.format(len(cat_ids), args.few_thr, args.many_thr))


if __name__ == '__main__':
    main()
