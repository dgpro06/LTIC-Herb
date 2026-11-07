"""LTIC on a frozen encoder: train / evaluate the head on images, a feature cache, or chunked features.

C0 changes (plan v9.1 §3.1):
  * 3 branches by default: H (all classes), M (medium + tail), T (tail); prediction H + M + T.
    --branches 2 reproduces the IJCNN code (H + M; M on labels < --reslt_split).
  * the shared classifier has no bias (networks/ltic.py), so train and test logits match at gamma = 0.
  * --beta renamed --alpha (the paper's alpha; the gate's beta is something else).
  * shot groups from --few_thr / --many_thr (required) with R1 asserts on the sorted counts;
    explicit --few_end / --many_start only to reproduce the IJCNN configuration.
  * epoch selection by macro-F1 on the --val_txt file (pass V-tune, never V-select).
  * resume checks the class_map; --evaluate refuses to run without a trained checkpoint.
  * full seeding; no per-epoch confusion matrix; weighted P/R use zero_division=0 (appendix only).
  * the distributed path is removed (it was broken and unused).
For tuning many configurations use tools/run_baselines.py (GPU-resident, much faster).
"""
import argparse
import math
import os
import random
import shutil
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim

from datasets import dataset
from lt import losses as L
from lt.classes import ShotGroups
from lt.metrics import evaluate_predictions
from networks.backbones import BACKBONES
from networks.ltic import LTICNet
from utils import feature_cache

parser = argparse.ArgumentParser(description='LTIC head on a frozen encoder')
parser.add_argument('-dataset', required=True, type=str, help='dataset class in datasets/dataset.py')
parser.add_argument('-a', '--arch', default='clip_VIT', help='ignored; kept so old scripts still parse')
parser.add_argument('--epochs', default=90, type=int, metavar='N')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='checkpoint to resume from (default: <root_path>/<dataset>/<mark>/checkpoint.pth.tar '
                         'when training, model_best.pth.tar with --evaluate)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use')

# LTIC
parser.add_argument('--data_path', default=None, type=str)
parser.add_argument('--root_path', default="./data", type=str)
parser.add_argument('--mark', required=True, type=str, help='run name (folder under root_path/dataset)')
parser.add_argument('--num_works', default=8, type=int)
parser.add_argument('--alpha', default=0.85, type=float, help='weight of the branch loss (paper alpha)')
parser.add_argument('--branches', default=3, type=int, choices=[2, 3],
                    help='3 = C0 (H + M + T); 2 = the IJCNN code (H + M)')
parser.add_argument('--num_classes', default=None, type=int,
                    help='default: the dataset class count (herbarium 15505, plantnet 1081)')
parser.add_argument('--gamma', default=0.0, type=float, help='test-time weight normalization (0 = off)')
parser.add_argument('--select_metric', default='macro_f1', choices=['macro_f1', 'top1'])

# Feature extractor
parser.add_argument('--backbone', default='bioclip2', choices=sorted(BACKBONES),
                    help='frozen image encoder (clip_b32 = original LTIC-Herb)')
parser.add_argument('--dataset_norm', action='store_true',
                    help='normalize with the dataset mean/std instead of the encoder one (original setup)')
parser.add_argument('--train_txt', default=None, type=str, help='default: <data_path>/<dataset train file>')
parser.add_argument('--val_txt', default=None, type=str,
                    help='default: <data_path>/<dataset val file>; pass V-tune while training')

# Shot groups (labels are sorted by train count, rarest first)
parser.add_argument('--many_thr', default=None, type=int, help='many-shot: more train images than this')
parser.add_argument('--few_thr', default=None, type=int, help='few-shot: fewer train images than this')
parser.add_argument('--few_end', default=None, type=int, help='IJCNN reproduction only: first non-few label')
parser.add_argument('--many_start', default=None, type=int, help='IJCNN reproduction only: first many-shot label')
parser.add_argument('--reslt_split', default=None, type=int,
                    help='labels below this train the M branch (default: many_start)')

# Feature cache
parser.add_argument('--cache_features', action='store_true',
                    help='extract frozen encoder features once, then train the head on them')
parser.add_argument('--cache_dir', default='./feat_cache', type=str)
parser.add_argument('--cache_views', default=1, type=int, help='augmented views per train image (1 except A12)')
parser.add_argument('--feature_dir', default=None, type=str,
                    help='chunked features from tools/extract_features.py (uses --train_split/--val_split)')
parser.add_argument('--train_split', default='train', type=str)
parser.add_argument('--val_split', default='vtune', type=str)

best_score = -math.inf
args = parser.parse_args()
args.num_classes = args.num_classes or getattr(dataset, args.dataset).num_classes
args.root_model = f'{args.root_path}/{args.dataset}/{args.mark}'
os.makedirs(args.root_model, exist_ok=True)


def log(message):
    print(message)
    with open(args.root_model + "/train.log", "a+") as f:
        f.write(message + "\n")


def set_shot_splits(cls_num_list_sorted, args):
    counts = np.asarray(cls_num_list_sorted)
    if args.few_end is not None or args.many_start is not None:
        if args.few_end is None or args.many_start is None:
            raise SystemExit('pass both --few_end and --many_start, or neither')
        warnings.warn('explicit --few_end/--many_start: IJCNN reproduction only, no R1 checks')
        groups = ShotGroups.from_bounds(len(counts), args.few_end, args.many_start)
    else:
        if args.few_thr is None or args.many_thr is None:
            raise SystemExit('pass --few_thr and --many_thr explicitly (plan 3.1; H22: 20 / 50, '
                             'Pl@ntNet and iNat: 20 / 100)')
        groups = ShotGroups(counts, args.few_thr, args.many_thr)     # R1 asserts
        args.few_end, args.many_start = groups.few_end, groups.many_start
    if args.reslt_split is None:
        args.reslt_split = args.many_start
    log('=> backbone {} | {} classes | {} | {} branches, M branch on labels < {}'.format(
        args.backbone, len(counts), groups.describe() if groups.few_thr is not None else
        'few [0, {}) medium [{}, {}) many [{}, {})'.format(args.few_end, args.few_end, args.many_start,
                                                         args.many_start, len(counts)),
        args.branches, args.reslt_split))
    empty = int((counts == 0).sum())
    if empty:
        log('=> warning: {} of {} classes have no train images; they sit at labels 0..{} inside the tail '
            'branch (renumber the labels, plan §3.1)'.format(empty, len(counts), empty - 1))
    return groups


def head_state_dict(model):
    # The frozen encoder is rebuilt from its pretrained weights, so only the head is saved.
    return {k: v for k, v in model.state_dict().items() if not k.startswith('encoder.')}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    global best_score
    seed_everything(args.seed)
    args.device = torch.device('cuda', args.gpu) if args.gpu is not None else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log("=> creating LTIC head on backbone '{}'".format(args.backbone))
    model = LTICNet(num_classes=args.num_classes, backbone=args.backbone, gamma=args.gamma)
    encoder = model.encoder
    if not (args.cache_features or args.feature_dir) and torch.cuda.device_count() > 1 and args.gpu is None:
        # Online training only: DataParallel copies the frozen encoder to every GPU
        model.encoder = nn.DataParallel(model.encoder)
    model = model.to(args.device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Data first: the class order is needed to check a checkpoint
    mean, std = (None, None) if args.dataset_norm else (encoder.mean, encoder.std)
    data = getattr(dataset, args.dataset)(root=args.data_path, batch_size=args.batch_size, num_works=args.num_works,
                                          train_txt=args.train_txt, val_txt=args.val_txt,
                                          num_classes=args.num_classes, mean=mean, std=std)
    groups = set_shot_splits(data.cls_num_list_sorted, args)
    class_map = [int(i) for i in data.trainset.class_map]
    # Saved with every checkpoint: class_map[split-file label] = output index (rarest first)
    model_meta = {'num_classes': args.num_classes, 'gamma': args.gamma, 'dataset_norm': args.dataset_norm,
                  'mean': [float(m) for m in (mean or data.mean)], 'std': [float(s) for s in (std or data.std)],
                  'class_map': class_map, 'few_end': args.few_end, 'many_start': args.many_start,
                  'reslt_split': args.reslt_split, 'branches': args.branches, 'alpha': args.alpha,
                  'backbone': args.backbone}

    resume = args.resume or os.path.join(args.root_model, 'model_best.pth.tar' if args.evaluate else 'checkpoint.pth.tar')
    if os.path.isfile(resume):
        log("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume, map_location=args.device)
        if checkpoint.get('class_map') is not None and checkpoint['class_map'] != class_map:
            raise SystemExit('=> the checkpoint was trained with a different class order (class_map differs)')
        if checkpoint.get('backbone', args.backbone) != args.backbone:
            raise SystemExit('=> the checkpoint used backbone {}'.format(checkpoint['backbone']))
        missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
        missing = [k for k in missing if not k.startswith('encoder.')]
        if missing or unexpected:
            raise SystemExit("=> checkpoint does not match the model: missing {} unexpected {}".format(missing, unexpected))
        if not args.evaluate:
            args.start_epoch = checkpoint['epoch']
            best_score = checkpoint.get('best_score', -math.inf)
            optimizer.load_state_dict(checkpoint['optimizer'])
        log("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
    elif args.evaluate:
        raise SystemExit("=> nothing to evaluate: no checkpoint at '{}' (train first or pass --resume)".format(resume))
    elif args.resume:
        raise SystemExit("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.feature_dir:
        train_loader, val_loader = feature_cache.build_chunked_loaders(data, args)
    elif args.cache_features:
        train_loader, val_loader = feature_cache.build_cached_loaders(encoder, data, args)
    else:
        train_loader, val_loader = data.train, data.test

    masks = L.branch_masks_for(args.num_classes, args.few_end, args.reslt_split, args.branches, args.device)

    if args.evaluate:
        validate(val_loader, model, groups, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        train(train_loader, model, masks, optimizer, epoch, args)
        metrics = validate(val_loader, model, groups, args)
        score = metrics[args.select_metric]
        is_best = score > best_score
        best_score = max(score, best_score)
        save_checkpoint({'epoch': epoch + 1, 'state_dict': head_state_dict(model), 'best_score': best_score,
                         'select_metric': args.select_metric, 'optimizer': optimizer.state_dict(), **model_meta},
                        is_best)


def fuse(branch_logits, args):
    return sum(branch_logits[:args.branches])


def train(train_loader, model, masks, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    F_losses = AverageMeter('F_Loss', ':.4e')
    I_losses = AverageMeter('I_Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, F_losses, I_losses, top1],
                             prefix="Epoch: [{}]".format(epoch))
    model.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)

        branches = list(model(images))[:args.branches]
        loss, F_loss, I_loss = L.ltic_loss(branches, target, masks, args.alpha)
        logit = sum(branches)

        n = images.size(0)
        F_losses.update(F_loss.item(), n)
        I_losses.update(I_loss.item(), n)
        top1.update((logit.argmax(1) == target).float().mean().item() * 100, n)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, groups, args):
    model.eval()
    targets, preds, top5_hits = [], [], []
    with torch.no_grad():
        for images, target in val_loader:
            images = images.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            output = fuse(model(images), args)
            k = min(5, output.size(1))
            top5 = output.topk(k, 1).indices
            targets.append(target.cpu())
            preds.append(top5[:, 0].cpu())
            top5_hits.append((top5 == target.view(-1, 1)).any(1).cpu())
    y = torch.cat(targets).numpy()
    pred = torch.cat(preds).numpy()
    top5 = torch.cat(top5_hits).float().mean().item()
    m = evaluate_predictions(y, pred, args.num_classes, groups, top5)
    log((' * Acc@1 {top1:.4f} Acc@5 {top5:.4f} MacroF1 {macro_f1:.4f} MacroAcc {macro_acc:.4f} '
         'WorstClass {worst_class_acc:.4f} | HAcc {many_acc:.4f} MAcc {medium_acc:.4f} TAcc {few_acc:.4f} '
         '(pooled TAcc {few_acc_pooled:.4f}) | appendix: weighted P {weighted_precision_appendix:.4f} '
         'R {weighted_recall_appendix:.4f}').format(**m))
    return m


def save_checkpoint(state, is_best):
    filename = args.root_model + '/checkpoint.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, args.root_model + '/model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        log('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Cosine schedule over epochs."""
    lr = 0.5 * args.lr * (1 + math.cos(epoch / args.epochs * math.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
