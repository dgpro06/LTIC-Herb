"""Chunked, resumable fp16 feature extraction for several frozen encoders at once (plan §11.2).

Each image is decoded once and fed to every encoder. Work is saved in chunks (default 50k images)
with an atomic write, so a Kaggle session that times out resumes from the last finished chunk.
Labels are stored as the ORIGINAL split-file labels; renumbering happens at load time.

    python tools/extract_features.py --root /data/plantnet_300K --out feats/plantnet \
        --encoders clip_b32,clip_b16,bioclip2,dinov2_b14 \
        --split train=splits/plantnet_train.txt --split vtune=splits/plantnet_vtune.txt \
        --split vselect=splits/plantnet_vselect.txt --split vcal=splits/plantnet_vcal.txt \
        --split test=splits/plantnet_test.txt

Throughput test (plan §11.2, Nov 1): add --limit 10000 --out /tmp/feat_bench.
View 0 is Resize(256) + CenterCrop(224); --views N adds N-1 RandomResizedCrop + flip views (A12 only).
"""
import argparse
import os
import sys
import time

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lt.classes import read_split_file                                   # noqa: E402
from lt.features import (chunk_path, file_sha1, read_manifest, save_chunk,  # noqa: E402
                         write_manifest)


class DecodeOnce(Dataset):
    """Returns uint8 [views, 3, S, S] crops; normalization is done per encoder on the GPU."""

    def __init__(self, root, paths, labels, size=224, resize=256, views=1, seed=0):
        self.root, self.paths, self.labels = root, paths, labels
        self.views, self.seed = views, seed
        self.center = transforms.Compose([transforms.Resize(resize), transforms.CenterCrop(size), transforms.PILToTensor()])
        self.aug = transforms.Compose([transforms.RandomResizedCrop(size), transforms.RandomHorizontalFlip(),
                                       transforms.PILToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with open(os.path.join(self.root, self.paths[i]), 'rb') as f:
            img = Image.open(f).convert('RGB')
        out = [self.center(img)]
        for v in range(1, self.views):
            torch.manual_seed(self.seed * 1000003 + i * 31 + v)   # reproducible augmented views
            out.append(self.aug(img))
        return torch.stack(out), int(self.labels[i]), i


def build_encoders(names, device):
    from networks.backbones import build_backbone
    encoders = {}
    for name in names:
        enc = build_backbone(name).to(device).eval()
        mean = torch.tensor(enc.mean, device=device).view(1, 3, 1, 1)
        std = torch.tensor(enc.std, device=device).view(1, 3, 1, 1)
        encoders[name] = (enc, mean, std)
        print('=> loaded {} ({}-d)'.format(name, enc.feat_dim), flush=True)
    return encoders


def chunk_done(out, split, names, i):
    return all(os.path.isfile(chunk_path(out, split, n, i)) for n in names)


def extract_split(args, split, txt, encoders, device):
    paths, labels = read_split_file(txt)
    if args.limit:
        paths, labels = paths[:args.limit], labels[:args.limit]
    n = len(paths)
    num_chunks = (n + args.chunk - 1) // args.chunk
    sha = file_sha1(txt)
    names = list(encoders)
    for name, (enc, _, _) in encoders.items():
        old = read_manifest(args.out, split, name)
        if old and (old['txt_sha1'] != sha or old['chunk_size'] != args.chunk or old['views'] != args.views
                    or old['num_images'] != n):
            raise SystemExit('{}/{}: existing features were made from a different split file or settings; '
                             'use a new --out or delete that folder'.format(split, name))
        write_manifest(args.out, split, name, {
            'split': split, 'txt': os.path.abspath(txt), 'txt_sha1': sha, 'num_images': n,
            'chunk_size': args.chunk, 'num_chunks': num_chunks, 'views': args.views,
            'encoder': name, 'feat_dim': enc.feat_dim, 'dtype': 'float16',
            'labels': 'original split-file labels', 'image_root': os.path.abspath(args.root),
            'resize': args.resize, 'size': args.size})

    dataset = DecodeOnce(args.root, paths, labels, args.size, args.resize, args.views, args.seed)
    t_start, done_imgs = time.time(), 0
    for i in range(num_chunks):
        if chunk_done(args.out, split, names, i):
            print('=> {} chunk {}/{} already done'.format(split, i + 1, num_chunks), flush=True)
            continue
        todo = [nm for nm in names if not os.path.isfile(chunk_path(args.out, split, nm, i))]
        start, end = i * args.chunk, min(n, (i + 1) * args.chunk)
        loader = DataLoader(torch.utils.data.Subset(dataset, range(start, end)), batch_size=args.batch,
                            shuffle=False, num_workers=args.workers, pin_memory=device.type == 'cuda',
                            persistent_workers=False)
        feats = {nm: [] for nm in todo}
        lab = []
        t0 = time.time()
        with torch.no_grad(), torch.autocast(device.type, enabled=device.type == 'cuda'):
            for b, (x, y, _) in enumerate(loader):
                V = x.size(1)
                x = x.to(device, non_blocking=True).flatten(0, 1).float().div_(255)
                for nm in todo:
                    enc, mean, std = encoders[nm]
                    f = enc((x - mean) / std).float().half()
                    feats[nm].append(f.view(-1, V, f.size(-1)).transpose(0, 1).cpu())
                lab.append(y)
                if b % 50 == 0:
                    print('   {} chunk {}/{} batch {}/{}'.format(split, i + 1, num_chunks, b, len(loader)), flush=True)
        lab = torch.cat(lab).long()
        for nm in todo:
            save_chunk(args.out, split, nm, i, {'feats': torch.cat(feats[nm], dim=1), 'labels': lab,
                                                'start': start, 'end': end})
        dt = time.time() - t0
        done_imgs += end - start
        print('=> {} chunk {}/{} saved ({} images, {:.1f} img/s for {} encoders)'.format(
            split, i + 1, num_chunks, end - start, (end - start) / max(dt, 1e-6), len(todo)), flush=True)
    if done_imgs:
        dt = time.time() - t_start
        print('=> {}: {} images in {:.1f} min ({:.1f} img/s)'.format(split, done_imgs, dt / 60, done_imgs / dt))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', required=True, help='image root; split-file paths are relative to it')
    p.add_argument('--out', required=True, help='feature store folder')
    p.add_argument('--encoders', required=True, help='comma-separated names from networks/backbones.py')
    p.add_argument('--split', action='append', required=True, metavar='NAME=TXT')
    p.add_argument('--chunk', type=int, default=50000)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--views', type=int, default=1, help='1 everywhere except A12 (plan §11.2)')
    p.add_argument('--size', type=int, default=224)
    p.add_argument('--resize', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--limit', type=int, default=0, help='first N images only (throughput test)')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoders = build_encoders(args.encoders.split(','), device)
    for spec in args.split:
        name, txt = spec.split('=', 1)
        extract_split(args, name, txt, encoders, device)
    print('=> done: {}'.format(args.out))


if __name__ == '__main__':
    main()
