"""Trainable heads on cached features. Species classifiers have no bias (plan §3.1).

Every head returns a dict:
    'branches'  list of species logits per branch (one entry for single-branch heads)
    'species'   fused species logits (sum of branches)
    'genus'     genus logits (multitask head only)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def tau_norm_weight(weight, tau):
    """w / ||w||^tau (tau-norm, and LTIC's test-time gamma). tau = 0 leaves w unchanged."""
    if not tau:
        return weight
    return weight / weight.norm(dim=1, keepdim=True).clamp(min=1e-12).pow(tau)


class LinearHead(nn.Module):
    """Linear probe. normed=True gives the cosine classifier used by LDAM."""

    def __init__(self, in_dim, num_classes, normed=False):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes, bias=False)
        self.normed = normed
        self.test_tau = 0.0     # tau-norm at eval (cRT / tau-norm baseline)

    def forward(self, x):
        w = self.fc.weight
        if self.normed:
            logits = F.normalize(x, dim=1) @ F.normalize(w, dim=1).t()
        else:
            if not self.training:
                w = tau_norm_weight(w, self.test_tau)
            logits = x @ w.t()
        return {'branches': [logits], 'species': logits}


class LTICHead(nn.Module):
    """LTIC head: Linear(d, 1024) -> ReLU -> BN -> Linear(1024, width), three slices of width // 3
    sharing one bias-free classifier. resl=True drops the MLP (ResLT-style: one linear projection
    to the three branch groups). gamma is LTIC's test-time weight normalization (0 = off)."""

    def __init__(self, in_dim, num_classes, width=170, hidden=1024, gamma=0.0, reslt=False):
        super().__init__()
        if reslt:
            self.intermediate = nn.Linear(in_dim, width)
        else:
            self.intermediate = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                              nn.BatchNorm1d(hidden), nn.Linear(hidden, width))
        self.slice = width // 3
        self.fc = nn.Linear(self.slice, num_classes, bias=False)
        self.gamma = gamma

    def forward(self, x):
        z = self.intermediate(x)
        c = self.slice
        parts = [z[:, i * c:(i + 1) * c] for i in range(3)]
        w = self.fc.weight if self.training else tau_norm_weight(self.fc.weight, self.gamma)
        branches = [p @ w.t() for p in parts]
        return {'branches': branches, 'species': sum(branches)}


class MultitaskHead(nn.Module):
    """Flat species probe + genus probe on the same features (baseline 9)."""

    def __init__(self, in_dim, num_classes, num_genera):
        super().__init__()
        self.species = nn.Linear(in_dim, num_classes, bias=False)
        self.genus = nn.Linear(in_dim, num_genera, bias=False)

    def forward(self, x):
        s = self.species(x)
        return {'branches': [s], 'species': s, 'genus': self.genus(x)}


def build_head(kind, in_dim, num_classes, num_genera=None, **kw):
    if kind == 'linear':
        return LinearHead(in_dim, num_classes, normed=kw.get('normed', False))
    if kind == 'ltic':
        return LTICHead(in_dim, num_classes, width=kw.get('width', 170), gamma=kw.get('gamma', 0.0))
    if kind == 'reslt':
        return LTICHead(in_dim, num_classes, width=kw.get('width', 170), gamma=kw.get('gamma', 0.0), reslt=True)
    if kind == 'multitask':
        if not num_genera:
            raise ValueError('multitask head needs a taxonomy (num_genera)')
        return MultitaskHead(in_dim, num_classes, num_genera)
    raise ValueError('unknown head ' + kind)


def count_parameters(module, exclude_classifier=False):
    n = 0
    for name, p in module.named_parameters():
        if exclude_classifier and (name.startswith('fc.') or name.startswith('species.') or name.startswith('genus.')):
            continue
        n += p.numel()
    return n
