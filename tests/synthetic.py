"""Synthetic long-tailed dataset with a genus structure, written in the real on-disk formats
(feature-store chunks, taxonomy CSV, config JSON) so tests exercise the same code paths as real runs."""
import csv
import json
import os

import numpy as np
import torch

from lt.features import save_chunk, write_manifest


def make_counts(num_classes, max_count=300, min_count=2, seed=0):
    """Zipf-like train counts, shuffled so original labels are not already sorted."""
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, num_classes + 1)
    counts = np.maximum(min_count, (max_count / ranks ** 0.9).astype(int))
    return counts[rng.permutation(num_classes)]


def make_dataset(root, num_classes=60, num_genera=15, dim=16, seed=0, eval_per_class=6, chunk=500,
                 encoder='synth'):
    rng = np.random.default_rng(seed)
    counts = make_counts(num_classes, seed=seed)
    genus_of = rng.integers(0, num_genera, num_classes)
    genus_of[:num_genera] = np.arange(num_genera)            # every genus non-empty
    family_of_genus = rng.integers(0, 4, num_genera)
    genus_mu = rng.normal(0, 3.0, (num_genera, dim))
    species_mu = genus_mu[genus_of] + rng.normal(0, 1.0, (num_classes, dim))
    # same-genus species share a covariance direction (gives the C2 check something to find)
    genus_dir = rng.normal(0, 1, (num_genera, dim))

    def sample(c, n):
        z = rng.normal(0, 0.7, (n, dim)) + rng.normal(0, 1.5, (n, 1)) * genus_dir[genus_of[c]]
        return species_mu[c] + z

    splits = {'train': counts, 'vtune': np.full(num_classes, eval_per_class),
              'vselect': np.full(num_classes, eval_per_class), 'vcal': np.full(num_classes, eval_per_class),
              'test': np.full(num_classes, eval_per_class)}
    for split, n_per in splits.items():
        X = np.concatenate([sample(c, n) for c, n in enumerate(n_per)]).astype(np.float32)
        y = np.concatenate([np.full(n, c) for c, n in enumerate(n_per)])
        perm = rng.permutation(len(y))
        X, y = torch.tensor(X[perm]).half(), torch.tensor(y[perm]).long()
        n = len(y)
        nch = (n + chunk - 1) // chunk
        write_manifest(root, split, encoder, {'num_images': n, 'num_chunks': nch, 'chunk_size': chunk, 'views': 1,
                                              'txt_sha1': 'synthetic', 'feat_dim': dim})
        for i in range(nch):
            s, e = i * chunk, min(n, (i + 1) * chunk)
            save_chunk(root, split, encoder, i, {'feats': X[s:e].unsqueeze(0), 'labels': y[s:e], 'start': s, 'end': e})

    tax_path = os.path.join(root, 'taxonomy.csv')
    with open(tax_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['label', 'source_id', 'species', 'genus', 'family'])
        for c in range(num_classes):
            g = genus_of[c]
            w.writerow([c, c, 'G{} s{}'.format(g, c), 'G{}'.format(g), 'F{}'.format(family_of_genus[g])])
    conf = {'dataset': 'synth', 'encoder': encoder, 'feature_root': root, 'num_classes': num_classes,
            'few_thr': 20, 'many_thr': 100, 'taxonomy': tax_path, 'epochs': 6, 'eval_every': 2, 'trials': 2}
    conf_path = os.path.join(root, 'config.json')
    with open(conf_path, 'w') as f:
        json.dump(conf, f)
    return conf, conf_path, counts, genus_of
