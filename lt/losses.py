"""Training losses and post-hoc adjustments, in output-index space (rarest class first).

All species classifiers are bias-free (plan §3.1), so frequency correction is explicit:
    post-hoc logit adjustment   logits - tau * log(pi)          (Menon et al., ICLR 2021)
    logit-adjusted loss         CE(logits + tau * log(pi))      (same paper)
    Balanced Softmax            CE(logits + log(pi))            (Ren et al., NeurIPS 2020)
    LDAM-DRW                    margin C / n^(1/4), deferred re-weighting (Cao et al., NeurIPS 2019)
    tau-norm                    w / ||w||^tau at test time       (Kang et al., ICLR 2020)
    TaxoNet dual-margin         see TaxoNetLoss                  (Low et al., arXiv 2512.18994)
"""
import numpy as np
import torch
import torch.nn.functional as F


def log_prior(counts, device=None):
    counts = torch.as_tensor(np.asarray(counts), dtype=torch.float32, device=device)
    return torch.log(counts.clamp(min=1) / counts.sum())


def logit_adjust(logits, log_pi, tau):
    """Post-hoc LA at decision time: subtract tau * log(pi)."""
    return logits - tau * log_pi if tau else logits


def la_loss(logits, target, log_pi, tau=1.0, weight=None):
    """Logit-adjusted cross-entropy; tau = 1 is Balanced Softmax."""
    return F.cross_entropy(logits + tau * log_pi, target, weight=weight)


class LDAMLoss(object):
    """LDAM with deferred re-weighting (DRW) starting at drw_epoch."""

    def __init__(self, counts, max_m=0.5, s=30.0, drw_epoch=None, drw_beta=0.9999, device=None):
        counts = np.maximum(np.asarray(counts, dtype=np.float64), 1)
        m = 1.0 / np.sqrt(np.sqrt(counts))
        m = m * (max_m / m.max())
        self.margins = torch.tensor(m, dtype=torch.float32, device=device)
        self.s = s
        self.drw_epoch = drw_epoch
        eff = 1.0 - np.power(drw_beta, counts)
        w = (1.0 - drw_beta) / eff
        w = w / w.sum() * len(counts)
        self.drw_weight = torch.tensor(w, dtype=torch.float32, device=device)

    def __call__(self, cos_logits, target, epoch):
        """cos_logits: cosine similarities from a NormedLinear head (reference recipe), in [-1, 1]."""
        adjusted = cos_logits - F.one_hot(target, cos_logits.size(1)) * self.margins[target].unsqueeze(1)
        weight = self.drw_weight if self.drw_epoch is not None and epoch >= self.drw_epoch else None
        return F.cross_entropy(self.s * adjusted, target, weight=weight)


class TaxoNetLoss(torch.nn.Module):
    """Dual-margin loss of TaxoNet (Low et al., arXiv 2512.18994, Eq. 2-5), feature-level version (baseline 11).

        L     = -log softmax_y( s * (z - m) ),  m_y = m0 + Delta_y (target),  m_k = Delta_k (k != y)
        Delta = m0 * (rho / m0) ** softplus(gamma),   gamma learnable
        rho_k ~ -log(N_k / N)  scaled so that sum_k rho_k = m0   ('sum', as stated in the paper)
                                          or max_k rho_k = m0   ('max', an option: with 1000+ classes the
                                                                  'sum' scaling makes every Delta tiny)
        total = L + lam * sum_k (Delta_k - rho_k)^2
    z are cosine logits (normalized features and prototypes, AM-Softmax style); s = 32 and m0 = 0.15 in the paper.
    Not reproduced at feature level: AugMix oversampling of tail images (images are not available) and the
    norm-guided sampling. lam and the initial gamma are not given in the paper; lam is searched, gamma starts at 0.
    """

    def __init__(self, counts, s=32.0, m0=0.15, lam=1.0, rho_norm='sum', gamma_init=0.0, device=None):
        super().__init__()
        counts = np.maximum(np.asarray(counts, dtype=np.float64), 1)
        rho = -np.log(counts / counts.sum())
        rho = rho * (m0 / (rho.sum() if rho_norm == 'sum' else rho.max()))
        self.register_buffer('rho', torch.tensor(rho, dtype=torch.float32, device=device))
        self.gamma = torch.nn.Parameter(torch.tensor(float(gamma_init), device=device))
        self.s, self.m0, self.lam = s, m0, lam

    def deltas(self):
        return self.m0 * (self.rho / self.m0).clamp(min=1e-12).pow(F.softplus(self.gamma))

    def forward(self, cos_logits, target):
        delta = self.deltas()
        m = delta.unsqueeze(0) + self.m0 * F.one_hot(target, cos_logits.size(1))
        loss = F.cross_entropy(self.s * (cos_logits - m), target)
        return loss + self.lam * ((delta - self.rho) ** 2).sum()


def branch_masks_for(num_classes, few_end, many_start, branches, device=None):
    """Class masks for the LTIC branches in output order.
    3 branches (C0): H = all, M = medium + tail, T = tail.  2 branches (IJCNN code): H, M."""
    c = torch.arange(num_classes, device=device)
    masks = [torch.ones(num_classes, dtype=torch.bool, device=device), c < many_start]
    if branches == 3:
        masks.append(c < few_end)
    return masks


def ltic_loss(branch_logits, target, masks, alpha):
    """L = (1 - alpha) * CE(sum of branches) + alpha * branch CE (plan §3.1, LTIC Eq. 2/7/8).

    Branch b is trained only on images whose class is in its mask; the branch term is the summed CE
    over (branch, image) pairs divided by the number of such pairs, as in the IJCNN code.
    """
    fused = sum(branch_logits)
    fused_loss = F.cross_entropy(fused, target)
    total, count = fused.new_zeros(()), 0
    for logits, mask in zip(branch_logits, masks):
        keep = mask[target]
        if keep.any():
            total = total + F.cross_entropy(logits[keep], target[keep], reduction='sum')
            count += int(keep.sum())
    branch_loss = total / max(count, 1)
    return (1 - alpha) * fused_loss + alpha * branch_loss, fused_loss.detach(), branch_loss.detach()
