"""C2: taxonomy-guided feature calibration on frozen features (plan §3.3, PLAN_NEXT.md §6).

For every tail species s (output index < few_end, at least one train image) we pick donors in the RAW
frozen-encoder feature space, using train statistics only:

    donors(s) = same-genus species with >= n_min train images (s excluded)
                -> else same-family species with >= n_min images
                -> else the global pool of species with >= n_min images

    mean       m_s   = (n_s mu_s + tau_mu mu_g) / (n_s + tau_mu),   mu_g = count-weighted donor mean
    covariance Sig_s = count-weighted average of the donors' covariances, kept as
                       'diag'  per-dimension variances, or
                       r (int) the top-r eigenvectors + an isotropic residual (PPCA-style)

Every epoch k synthetic features per tail species are drawn from N(m_s, Sig_s) with a seeded generator
and mixed into the training set of the base method (ce / la / crt / tau_norm / ltic3 / ...).

The expensive part (donor sets and their covariances) depends only on the features, the taxonomy, n_min and
the covariance form, so it is built once by `DonorStats` and reused across search trials; tau_mu and k are
applied by `DonorStats.sampler`.
"""
import copy

import numpy as np
import torch

LEVELS = ('genus', 'family', 'global')


def _class_moments(x, y, num_classes, chunk=65536):
    """Per-class count, mean and unbiased per-dimension variance (float64 accumulation)."""
    d = x.shape[1]
    dev = x.device
    n = torch.zeros(num_classes, dtype=torch.float64, device=dev)
    s1 = torch.zeros(num_classes, d, dtype=torch.float64, device=dev)
    s2 = torch.zeros(num_classes, d, dtype=torch.float64, device=dev)
    for i in range(0, x.shape[0], chunk):
        xb = x[i:i + chunk].double()
        yb = y[i:i + chunk]
        n.index_add_(0, yb, torch.ones_like(yb, dtype=torch.float64))
        s1.index_add_(0, yb, xb)
        s2.index_add_(0, yb, xb * xb)
    safe = n.clamp(min=1).unsqueeze(1)
    mu = s1 / safe
    var = (s2 / safe - mu * mu).clamp(min=0) * (safe / (safe - 1).clamp(min=1))
    return n, mu, var


def select_donors(counts, genus_of, family_of, few_end, n_min, global_fallback=True):
    """For each tail class: (donor class indices, level). Classes with no train image or no donor get
    (None, 'none'). Pure numpy so it can be unit-tested and reported without features."""
    counts = np.asarray(counts)
    genus_of = np.asarray(genus_of)
    family_of = np.asarray(family_of)
    eligible = counts >= n_min
    out = []
    for s in range(few_end):
        if counts[s] == 0:
            out.append((None, 'none'))
            continue
        other = eligible.copy()
        other[s] = False
        cand = np.nonzero(other & (genus_of == genus_of[s]))[0]
        if len(cand):
            out.append((cand, 'genus'))
            continue
        if family_of[s] >= 0:
            cand = np.nonzero(other & (family_of == family_of[s]))[0]
            if len(cand):
                out.append((cand, 'family'))
                continue
        cand = np.nonzero(other)[0]
        if global_fallback and len(cand):
            out.append((cand, 'global'))
        else:
            out.append((None, 'none'))
    return out


def donor_summary(counts, genus_of, family_of, few_end, n_min, global_fallback=True):
    """Share of tail species at each donor level (reported in the paper)."""
    sel = select_donors(counts, genus_of, family_of, few_end, n_min, global_fallback)
    levels = [lv for _, lv in sel]
    n = len(levels)
    out = {'tail_species': n, 'n_min': n_min}
    for lv in LEVELS + ('none',):
        c = sum(l == lv for l in levels)
        out[lv] = c
        out[lv + '_share'] = c / n if n else 0.0
    sizes = [len(d) for d, lv in sel if d is not None]
    out['donors_per_species_median'] = float(np.median(sizes)) if sizes else 0.0
    return out


class DonorStats(object):
    """Donor sets, donor means and donor covariances for every tail species.

    x: [N, D] train features (any float dtype, on the training device); y: [N] output indices.
    rank: 'diag' or an int r (clamped to D).
    """

    MAX_RANK = 64

    def __init__(self, x, y, counts, genus_of, family_of, few_end, n_min=20, rank='diag',
                 global_fallback=True, chunk=65536):
        dev = x.device
        C = len(counts)
        D = x.shape[1]
        self.rank = rank if rank == 'diag' else min(int(rank), D)
        self.dim = D
        self.device = dev
        n_cls, mu, var = _class_moments(x, y, C, chunk)
        self.class_n = n_cls
        self.class_mu = mu
        sel = select_donors(np.asarray(counts), genus_of, family_of, few_end, n_min, global_fallback)
        self.levels = [lv for _, lv in sel]
        self.summary = donor_summary(np.asarray(counts), genus_of, family_of, few_end, n_min, global_fallback)

        # group tail classes that share a donor set (usually a whole genus)
        set_id = {}
        self.tail = []           # tail classes that get synthetic features
        self.tail_set = []       # their donor-set index
        donor_sets = []
        for s, (donors, lv) in enumerate(sel):
            if donors is None:
                continue
            key = tuple(int(d) for d in donors)
            if key not in set_id:
                set_id[key] = len(donor_sets)
                donor_sets.append(np.asarray(key))
            self.tail.append(s)
            self.tail_set.append(set_id[key])
        self.tail = np.asarray(self.tail, dtype=np.int64)
        self.tail_set = np.asarray(self.tail_set, dtype=np.int64)
        self.donor_sets = donor_sets

        # per donor set: count-weighted mean, and the covariance factor
        self.set_mu = torch.zeros(len(donor_sets), D, dtype=torch.float64, device=dev)
        self.set_std = None          # diag: [S, D]
        self.set_factor = []         # rank r: list of [D, r]
        self.set_resid = None        # rank r: [S] isotropic residual std
        self._evals, self._evecs = [], []   # rank r: all eigenvalues, top-MAX_RANK eigenvectors per set
        if rank == 'diag':
            self.set_std = torch.zeros(len(donor_sets), D, dtype=torch.float64, device=dev)
        else:
            self.set_resid = torch.zeros(len(donor_sets), dtype=torch.float64, device=dev)
        for j, donors in enumerate(donor_sets):
            d_t = torch.as_tensor(donors, device=dev)
            w = n_cls[d_t] / n_cls[d_t].sum()
            self.set_mu[j] = (w.unsqueeze(1) * mu[d_t]).sum(0)
            if rank == 'diag':
                self.set_std[j] = (w.unsqueeze(1) * var[d_t]).sum(0).sqrt()
            else:
                cov = self._pooled_cov(x, y, d_t, w, mu, n_cls, C, chunk)
                evals, evecs = torch.linalg.eigh(cov)                     # ascending
                evals = evals.clamp(min=0).flip(0)                        # descending
                R = max(self.rank, min(self.MAX_RANK, D))
                self._evals.append(evals)
                self._evecs.append(evecs.flip(1)[:, :R].float().contiguous())    # float32: ~2,500 sets on H22
        if rank != 'diag':
            self._set_rank(self.rank)

    def _set_rank(self, r):
        """Top-r factor + isotropic residual from the stored eigendecompositions (no recomputation)."""
        self.rank = r
        self.set_factor = []
        for j, (ev, V) in enumerate(zip(self._evals, self._evecs)):
            self.set_factor.append(V[:, :r] * ev[:r].sqrt().float().unsqueeze(0))
            rest = ev[r:] if r < len(ev) else ev.new_zeros(1)
            self.set_resid[j] = rest.mean().sqrt()

    def at_rank(self, rank):
        """The same donor statistics with another low-rank size (<= the size they were built with)."""
        if self.rank == 'diag' or rank == 'diag':
            if rank != self.rank:
                raise ValueError('diag and low-rank statistics are built separately')
            return self
        r = min(int(rank), self.dim)
        if self._evecs and r > self._evecs[0].shape[1]:
            raise ValueError('rank {} > the {} eigenvectors kept'.format(r, self._evecs[0].shape[1]))
        if r == self.rank:
            return self
        out = copy.copy(self)
        out.set_resid = self.set_resid.clone()
        out._set_rank(r)
        return out

    @staticmethod
    def _pooled_cov(x, y, donors, w, mu, n_cls, C, chunk):
        """sum_d w_d Cov_d with Cov_d unbiased, computed from the donor rows in chunks."""
        D = x.shape[1]
        is_donor = torch.zeros(C, dtype=torch.bool, device=x.device)
        is_donor[donors] = True
        row_w = torch.zeros(C, dtype=torch.float64, device=x.device)
        row_w[donors] = w / (n_cls[donors] - 1).clamp(min=1)
        cov = torch.zeros(D, D, dtype=torch.float64, device=x.device)
        for i in range(0, x.shape[0], chunk):
            yb = y[i:i + chunk]
            keep = is_donor[yb]
            if not keep.any():
                continue
            yk = yb[keep]
            xc = x[i:i + chunk][keep].double() - mu[yk]
            xc = xc * row_w[yk].sqrt().unsqueeze(1)
            cov += xc.t() @ xc
        return cov

    def sampler(self, tau_mu, k):
        return C2Sampler(self, tau_mu, k)


class C2Sampler(object):
    """Shrunk means for the tail species and per-epoch sampling of k synthetic features each."""

    def __init__(self, stats, tau_mu, k):
        self.stats = stats
        self.k = int(k)
        dev = stats.device
        tail = torch.as_tensor(stats.tail, device=dev)
        n_s = stats.class_n[tail].unsqueeze(1)
        mu_g = stats.set_mu[torch.as_tensor(stats.tail_set, device=dev)]
        self.mean = ((n_s * stats.class_mu[tail] + tau_mu * mu_g) / (n_s + tau_mu)).float()   # [T, D]
        self.labels = tail.repeat_interleave(self.k)                                           # [T*k]
        self.n_samples = len(stats.tail) * self.k
        # rows of the output that belong to each donor set (for the low-rank path)
        self._rows = []
        if stats.rank != 'diag':
            rows = np.arange(self.n_samples).reshape(len(stats.tail), self.k) if len(stats.tail) else np.zeros((0, self.k))
            for j in range(len(stats.donor_sets)):
                r = rows[stats.tail_set == j].reshape(-1)
                self._rows.append(torch.as_tensor(r, device=dev, dtype=torch.long))
            self._factor = [f.float() for f in stats.set_factor]
            self._resid = stats.set_resid.float()
        else:
            self._std = stats.set_std[torch.as_tensor(stats.tail_set, device=dev)].float()   # [T, D]

    def sample(self, generator):
        """-> (features [T*k, D] float32, labels [T*k]) drawn with `generator` (deterministic per seed)."""
        st = self.stats
        D = st.dim
        dev = st.device
        if self.n_samples == 0:
            return torch.zeros(0, D, device=dev), self.labels
        base = self.mean.repeat_interleave(self.k, dim=0)
        eps = torch.randn(self.n_samples, D, device=dev, generator=generator)
        if st.rank == 'diag':
            return base + eps * self._std.repeat_interleave(self.k, dim=0), self.labels
        out = base.clone()
        for j, rows in enumerate(self._rows):
            if len(rows) == 0:
                continue
            z = torch.randn(len(rows), self._factor[j].shape[1], device=dev, generator=generator)
            out[rows] += z @ self._factor[j].t() + self._resid[j] * eps[rows]
        return out, self.labels
