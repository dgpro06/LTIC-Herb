"""Train the LTIC head on frozen-encoder features instead of images.

Two sources:
  * build_chunked_loaders  (--feature_dir): chunked fp16 features from tools/extract_features.py.
    Preferred: resumable, original labels stored, renumbered here with the run's class_map.
  * build_cached_loaders   (--cache_features): the older single-file cache, extracted in this process
    and kept in RAM. Fine for Pl@ntNet at --cache_views 1 (plan §16 fallback).
"""
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from lt.features import load_cached_views


class CachedViews(Dataset):
    """Train features stored as [views, N, D]; each sample draws one augmented view at random."""
    def __init__(self, feats, labels):
        self.feats = feats
        self.labels = labels

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, index):
        view = random.randrange(self.feats.size(0))
        return self.feats[view, index], self.labels[index]


def extract(encoder, loader, device, desc):
    net = nn.DataParallel(encoder) if torch.cuda.device_count() > 1 else encoder
    feats, labels = [], []
    with torch.no_grad(), torch.autocast(device.type, enabled=device.type == 'cuda'):
        for i, (images, target) in enumerate(loader):
            feats.append(net(images.to(device, non_blocking=True)).half().cpu())
            labels.append(target)
            if i % 100 == 0:
                print('=> extracting {} [{}/{}]'.format(desc, i, len(loader)))
    return torch.cat(feats), torch.cat(labels)


def _cache_path(args, txt, suffix=''):
    split = os.path.splitext(os.path.basename(txt))[0]
    norm = '_dsnorm' if args.dataset_norm else ''
    return os.path.join(args.cache_dir, '{}_{}{}_{}{}.pt'.format(args.dataset, args.backbone, norm, split, suffix))


def _load_or_extract(path, make, class_map):
    if os.path.isfile(path):
        print("=> loading cached features '{}'".format(path))
        cache = torch.load(path)
        # Labels in this cache are already remapped; refuse a cache built with another class order
        if 'class_map' not in cache or list(cache['class_map']) != list(class_map):
            raise SystemExit("=> '{}' was built with a different (or unrecorded) class order; delete it "
                             "or use --feature_dir".format(path))
        return cache
    cache = make()
    cache['class_map'] = list(class_map)
    torch.save(cache, path)
    print("=> saved cached features '{}'".format(path))
    return cache


def _loaders(train, val, batch_size):
    train_loader = DataLoader(CachedViews(train['feats'], train['labels']), batch_size=batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(TensorDataset(val['feats'], val['labels']), batch_size=batch_size,
                            shuffle=False, num_workers=0)
    return train_loader, val_loader


def build_cached_loaders(encoder, data, args):
    """Return (train_loader, val_loader) yielding (features, target) instead of (images, target)."""
    os.makedirs(args.cache_dir, exist_ok=True)
    device = next(encoder.parameters()).device
    class_map = [int(i) for i in data.trainset.class_map]

    def make_train():
        # Same augmentations as online training, one full pass per view
        loader = DataLoader(data.trainset, batch_size=data.batch_size, shuffle=False,
                            num_workers=data.num_works, pin_memory=True)
        views = [extract(encoder, loader, device, 'train view {}/{}'.format(v + 1, args.cache_views))
                 for v in range(args.cache_views)]
        return {'feats': torch.stack([f for f, _ in views]), 'labels': views[0][1]}

    def make_val():
        feats, labels = extract(encoder, data.test, device, 'val')
        return {'feats': feats, 'labels': labels}

    train = _load_or_extract(_cache_path(args, data.train_txt, '_x{}'.format(args.cache_views)), make_train, class_map)
    val = _load_or_extract(_cache_path(args, data.val_txt), make_val, class_map)
    return _loaders(train, val, data.batch_size)


def build_chunked_loaders(data, args):
    """Loaders over tools/extract_features.py output; labels remapped with this run's class_map."""
    class_map = data.trainset.class_map
    train = load_cached_views(args.feature_dir, args.train_split, args.backbone, class_map, views=args.cache_views)
    val = load_cached_views(args.feature_dir, args.val_split, args.backbone, class_map, views=1)
    val = {'feats': val['feats'][0], 'labels': val['labels']}
    return _loaders(train, val, data.batch_size)
