"""Chunked fp16 feature store written by tools/extract_features.py (plan §11.2).

Layout:
    <root>/<split>/<encoder>/manifest.json
    <root>/<split>/<encoder>/chunk_00000.pt   {'feats': fp16 [views, n, d], 'labels': int64 [n], 'start', 'end'}

Labels are the ORIGINAL split-file labels. Remap them at load time with lt.classes.ClassIndex.
"""
import hashlib
import json
import os

import torch


def file_sha1(path, block=1 << 20):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(block), b''):
            h.update(chunk)
    return h.hexdigest()


def split_dir(root, split, encoder):
    return os.path.join(root, split, encoder)


def chunk_path(root, split, encoder, i):
    return os.path.join(split_dir(root, split, encoder), 'chunk_{:05d}.pt'.format(i))


def read_manifest(root, split, encoder):
    path = os.path.join(split_dir(root, split, encoder), 'manifest.json')
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def write_manifest(root, split, encoder, manifest):
    d = split_dir(root, split, encoder)
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, 'manifest.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(manifest, f, indent=1)
    os.replace(tmp, os.path.join(d, 'manifest.json'))


def save_chunk(root, split, encoder, i, payload):
    path = chunk_path(root, split, encoder, i)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, path)       # atomic: a killed run never leaves a half-written chunk


def load_split(root, split, encoder, views=None):
    """Return (feats fp16 [V, N, D], original labels int64 [N]); checks the manifest is complete."""
    m = read_manifest(root, split, encoder)
    if m is None:
        raise FileNotFoundError('no features for split={} encoder={} under {}'.format(split, encoder, root))
    n_chunks = m['num_chunks']
    missing = [i for i in range(n_chunks) if not os.path.isfile(chunk_path(root, split, encoder, i))]
    if missing:
        raise RuntimeError('{}/{}: {} of {} chunks missing (first: {}); resume the extractor'.format(
            split, encoder, len(missing), n_chunks, missing[0]))
    feats, labels, expected = [], [], 0
    for i in range(n_chunks):
        c = torch.load(chunk_path(root, split, encoder, i), map_location='cpu')
        if c['start'] != expected:
            raise RuntimeError('chunk {} starts at {}, expected {}'.format(i, c['start'], expected))
        expected = c['end']
        f = c['feats']
        feats.append(f if views is None else f[:views])
        labels.append(c['labels'])
    if expected != m['num_images']:
        raise RuntimeError('{}/{}: chunks cover {} of {} images'.format(split, encoder, expected, m['num_images']))
    return torch.cat(feats, dim=1), torch.cat(labels)


def load_cached_views(root, split, encoder, class_map, views=None):
    """Same dict format as utils/feature_cache.py ({'feats': [V, N, D], 'labels': remapped})."""
    feats, labels = load_split(root, split, encoder, views)
    return {'feats': feats, 'labels': torch.as_tensor(class_map)[labels]}
