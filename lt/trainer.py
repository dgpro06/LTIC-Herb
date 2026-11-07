"""Fast head training on cached features (plan §11.2 "Head-training loop").

Features and labels stay on the GPU, batches are sampled by index, metrics are computed every
`eval_every` epochs on V-tune only, and the epoch is chosen by macro-F1 on V-tune (plan §3.1).
V-select is never passed here.

Methods (plan §7, baselines 1-9):
    ce            CE linear probe
    la            linear probe + logit adjustment; la_mode 'posthoc' (CE, then -tau log pi) or 'loss'
    balanced_softmax
    crt           linear classifier trained with class-balanced sampling (cRT on frozen features)
    tau_norm      CE probe + test-time weight normalization w / ||w||^test_tau
    ldam_drw      cosine classifier + LDAM margins + deferred re-weighting
    reslt         ResLT-style 3-branch head without the MLP
    ltic2         IJCNN code: branches H, M; prediction H + M
    ltic3         C0: branches H, M, T; prediction H + M + T
    flat_genus    species probe + genus probe (genus trained with the logit-adjusted loss)
Every method also accepts a post-hoc 'tau' (LA at decision time) if it is in its search space.
"""
import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from lt import heads as H
from lt import losses as L
from lt.metrics import macro_f1

DEFAULTS = {
    'lr': 0.1, 'weight_decay': 1e-4, 'momentum': 0.9, 'epochs': 30, 'batch_size': 512, 'eval_every': 2,
    'tau': 0.0, 'la_mode': 'posthoc', 'loss_tau': 1.0, 'alpha': 0.85, 'gamma': 0.0, 'test_tau': 0.0,
    'ldam_max_m': 0.5, 'ldam_s': 30.0, 'drw_frac': 0.8, 'genus_weight': 1.0, 'genus_tau': 1.0,
    'width': 170, 'select_metric': 'macro_f1',
}

HEAD_OF = {'ce': 'linear', 'la': 'linear', 'balanced_softmax': 'linear', 'crt': 'linear', 'tau_norm': 'linear',
           'ldam_drw': 'linear', 'reslt': 'reslt', 'ltic2': 'ltic', 'ltic3': 'ltic', 'flat_genus': 'multitask'}


class FeatureSet(object):
    """GPU-resident features in output-index space. feats: [V, N, D] or [N, D] (fp16 is fine)."""

    def __init__(self, feats, labels, device):
        feats = torch.as_tensor(feats)
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
        self.x = feats.to(device)
        self.y = torch.as_tensor(labels, dtype=torch.long).to(device)
        self.views, self.n, self.dim = self.x.shape

    def batch(self, idx, generator=None):
        if self.views == 1:
            return self.x[0, idx].float()
        v = torch.randint(0, self.views, (len(idx),), device=self.x.device, generator=generator)
        return self.x[v, idx].float()


class _ClassBalancedSampler(object):
    def __init__(self, y, num_classes, generator):
        self.order = torch.argsort(y)
        counts = torch.bincount(y, minlength=num_classes)
        self.starts = torch.cumsum(counts, 0) - counts
        self.counts = counts
        self.present = torch.nonzero(counts > 0).squeeze(1)
        self.g = generator

    def sample(self, b):
        c = self.present[torch.randint(0, len(self.present), (b,), device=self.order.device, generator=self.g)]
        j = (torch.rand(b, device=self.order.device, generator=self.g) * self.counts[c]).long()
        return self.order[self.starts[c] + j]


def build(cfg, in_dim, num_classes, num_genera=None):
    method = cfg['method']
    kind = HEAD_OF[method]
    return H.build_head(kind, in_dim, num_classes, num_genera=num_genera, normed=method == 'ldam_drw',
                        width=cfg['width'], gamma=cfg['gamma'])


def species_logits(model, out, cfg):
    """Fused species logits for the method's prediction rule (before post-hoc tau)."""
    if cfg['method'] == 'ltic2':
        return out['branches'][0] + out['branches'][1]
    return out['species']


@torch.no_grad()
def predict(model, fs, cfg, log_pi, batch=8192):
    """Returns (species logits after post-hoc tau, genus logits or None) on the CPU."""
    model.eval()
    if cfg['method'] == 'tau_norm':
        model.test_tau = cfg['test_tau']
    sp, gn = [], []
    for s in range(0, fs.n, batch):
        out = model(fs.x[0, s:s + batch].float())
        sp.append(L.logit_adjust(species_logits(model, out, cfg), log_pi, cfg['tau']).cpu())
        if 'genus' in out:
            gn.append(out['genus'].cpu())
    return torch.cat(sp), (torch.cat(gn) if gn else None)


def train(cfg, train_fs, tune_fs, counts, few_end, many_start, seed=0, genus_of=None, log=print):
    """Train one configuration; return (model at the best V-tune epoch, info dict)."""
    cfg = dict(DEFAULTS, **cfg)
    device = train_fs.x.device
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    C = len(counts)
    num_genera = int(genus_of.max()) + 1 if genus_of is not None else None
    model = build(cfg, train_fs.dim, C, num_genera).to(device)
    log_pi = L.log_prior(counts, device)
    method = cfg['method']

    genus_t = genus_log_pi = None
    if method == 'flat_genus':
        genus_t = torch.as_tensor(genus_of, device=device)
        genus_log_pi = L.log_prior(np.bincount(genus_of, weights=counts, minlength=num_genera), device)
    masks = None
    if method in ('ltic2', 'ltic3', 'reslt'):
        masks = L.branch_masks_for(C, few_end, many_start, 2 if method == 'ltic2' else 3, device)
    ldam = None
    if method == 'ldam_drw':
        ldam = L.LDAMLoss(counts, cfg['ldam_max_m'], cfg['ldam_s'], int(cfg['drw_frac'] * cfg['epochs']),
                          device=device)

    opt = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'],
                          weight_decay=cfg['weight_decay'])
    steps_per_epoch = max(1, train_fs.n // cfg['batch_size'])
    total = steps_per_epoch * cfg['epochs']
    balanced = _ClassBalancedSampler(train_fs.y, C, g) if method == 'crt' else None

    best = {'score': -math.inf, 'epoch': -1, 'state': None}
    history = []
    t0 = time.time()
    step = 0
    for epoch in range(cfg['epochs']):
        model.train()
        perm = torch.randperm(train_fs.n, device=device, generator=g)
        for k in range(steps_per_epoch):
            lr = 0.5 * cfg['lr'] * (1 + math.cos(math.pi * step / total))
            for pg in opt.param_groups:
                pg['lr'] = lr
            idx = balanced.sample(cfg['batch_size']) if balanced else perm[k * cfg['batch_size']:(k + 1) * cfg['batch_size']]
            x, y = train_fs.batch(idx, g), train_fs.y[idx]
            out = model(x)
            if method in ('ltic2', 'ltic3', 'reslt'):
                loss = L.ltic_loss(out['branches'], y, masks, cfg['alpha'])[0]
            elif method == 'ldam_drw':
                loss = ldam(out['species'], y, epoch)
            elif method == 'balanced_softmax':
                loss = L.la_loss(out['species'], y, log_pi, 1.0)
            elif method == 'la' and cfg['la_mode'] == 'loss':
                loss = L.la_loss(out['species'], y, log_pi, cfg['loss_tau'])
            else:
                loss = F.cross_entropy(out['species'], y)
            if method == 'flat_genus':
                loss = loss + cfg['genus_weight'] * L.la_loss(out['genus'], genus_t[y], genus_log_pi, cfg['genus_tau'])
            if not torch.isfinite(loss):
                log('   non-finite loss at epoch {}; stopping this run'.format(epoch))
                return _finish(model, best, history, cfg, t0, diverged=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1

        if (epoch + 1) % cfg['eval_every'] == 0 or epoch + 1 == cfg['epochs']:
            logits, _ = predict(model, tune_fs, cfg, log_pi)
            pred = logits.argmax(1).numpy()
            y_t = tune_fs.y.cpu().numpy()
            score = macro_f1(y_t, pred, C) if cfg['select_metric'] == 'macro_f1' else float((pred == y_t).mean())
            history.append({'epoch': epoch + 1, 'loss': float(loss.detach()),'vtune_' + cfg['select_metric']: score})
            if score > best['score']:
                best = {'score': score, 'epoch': epoch + 1, 'state': copy.deepcopy(model.state_dict())}
    return _finish(model, best, history, cfg, t0)


def _finish(model, best, history, cfg, t0, diverged=False):
    if best['state'] is not None:
        model.load_state_dict(best['state'])
    info = {'best_epoch': best['epoch'], 'vtune_score': best['score'], 'history': history,
            'seconds': time.time() - t0, 'diverged': diverged, 'cfg': cfg}
    return model, info
