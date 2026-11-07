"""Week-1 gate diagnostics (plan §8.1). All inputs are in output-index space.

1   donor_pooling       tail species: (images in the genus excluding the species) / (own images); median >= 3
(a) cross-genus share   lt.cf.shares_with_ci on the LA probe; >= 30%
(b) genus_rescue        on cross-genus tail FPs, how often a genus classifier predicts the true genus; >= 50%
C2  covariance_check    same-genus top-k covariance subspaces more similar than chance (genus-label
                        permutation test, p < 0.05), species with >= min_images train images
"""
import numpy as np
import torch


def donor_pooling(counts, genus_of, few_end):
    counts = np.asarray(counts, dtype=np.float64)
    genus_total = np.bincount(genus_of, weights=counts)
    tail = np.arange(few_end)
    tail = tail[counts[tail] > 0]
    others = genus_total[genus_of[tail]] - counts[tail]
    ratio = others / counts[tail]
    q = np.quantile(ratio, [0.1, 0.25, 0.5, 0.75, 0.9]) if len(ratio) else [np.nan] * 5
    return {'tail_species': int(len(tail)), 'median_ratio': float(np.median(ratio)) if len(ratio) else float('nan'),
            'quantiles_10_25_50_75_90': [float(v) for v in q],
            'share_no_congener_images': float((others == 0).mean()) if len(ratio) else float('nan'),
            'share_ratio_ge_3': float((ratio >= 3).mean()) if len(ratio) else float('nan'),
            'ratios': ratio.tolist()}


def genus_rescue(y, pred_species, genus_pred, few_end, genus_of):
    """Share of cross-genus tail false positives where the genus classifier gets the true genus."""
    y, pred_species, genus_pred = map(np.asarray, (y, pred_species, genus_pred))
    fp = (pred_species < few_end) & (pred_species != y) & (genus_of[pred_species] != genus_of[y])
    n = int(fp.sum())
    hit = genus_pred[fp] == genus_of[y[fp]]
    return {'n_cross_genus_tail_fp': n, 'genus_correct_share': float(hit.mean()) if n else float('nan')}


def _subspaces(feats, labels, species, k, device):
    U = []
    for c in species:
        x = feats[labels == c].float().to(device)
        x = x - x.mean(0, keepdim=True)
        _, _, vh = torch.linalg.svd(x, full_matrices=False)
        U.append(vh[:k].t())                   # [D, k]
    return torch.stack(U)                      # [S, D, k]


def _pair_sims(U, a, b, k, chunk=4096):
    out = []
    for s in range(0, len(a), chunk):
        ua, ub = U[a[s:s + chunk]], U[b[s:s + chunk]]
        m = torch.einsum('pdk,pdl->pkl', ua, ub)
        out.append((m ** 2).sum((1, 2)) / k)   # mean cos^2 of the principal angles, in [0, 1]
    return torch.cat(out).cpu().numpy()


def _same_genus_pairs(genus, rng, max_pairs):
    pairs = []
    for g in np.unique(genus):
        idx = np.flatnonzero(genus == g)
        if len(idx) > 1:
            i, j = np.triu_indices(len(idx), 1)
            pairs.append(np.stack([idx[i], idx[j]], 1))
    if not pairs:
        return np.zeros((0, 2), np.int64)
    pairs = np.concatenate(pairs)
    if len(pairs) > max_pairs:
        pairs = pairs[rng.choice(len(pairs), max_pairs, replace=False)]
    return pairs


def covariance_check(feats, labels, counts, genus_of, k=10, min_images=20, n_perm=1000, max_pairs=20000,
                     seed=0, device=None):
    """feats: [N, D] train features; labels: output indices. Returns statistic, null and p-value."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(seed)
    feats, labels = torch.as_tensor(feats), torch.as_tensor(labels)
    species = np.flatnonzero(np.asarray(counts) >= min_images)
    if len(species) < 3:
        return {'eligible_species': int(len(species)), 'p_value': float('nan')}
    U = _subspaces(feats, labels, species, k, device)
    genus = np.asarray(genus_of)[species]

    def stat(g):
        pairs = _same_genus_pairs(g, rng, max_pairs)
        if not len(pairs):
            return float('nan'), 0
        return float(_pair_sims(U, pairs[:, 0], pairs[:, 1], k).mean()), len(pairs)

    observed, n_pairs = stat(genus)
    null = np.array([stat(rng.permutation(genus))[0] for _ in range(n_perm)])
    null = null[~np.isnan(null)]
    a = rng.integers(0, len(species), max_pairs)
    b = rng.integers(0, len(species), max_pairs)
    keep = a != b
    random_mean = float(_pair_sims(U, a[keep], b[keep], k).mean())
    p = float((1 + (null >= observed).sum()) / (1 + len(null)))
    return {'eligible_species': int(len(species)), 'k': k, 'same_genus_pairs': int(n_pairs),
            'same_genus_mean_sim': observed, 'random_pair_mean_sim': random_mean,
            'null_mean': float(null.mean()) if len(null) else float('nan'), 'n_perm': int(len(null)),
            'p_value': p}
