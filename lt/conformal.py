"""Calibrated prediction sets with genus fallback (plan §3.5, C4). Calibrate on V-cal only.

Score: PAS (prevalence-adjusted softmax, Ding, Fermanian, Salmon, ICLR 2026),
    s(x, y) = -p(y | x) / pi(y),
with pi the train label prior; 'softmax' (s = 1 - p) is kept for comparison.
TODO before reporting: check the PAS definition against github.com/tiffanyding/long-tail-conformal.

    CP-1  species split CP at alpha                       marginal >= 1 - alpha (species)
    CP-2  genus split CP at alpha on summed probabilities  marginal >= 1 - alpha (genus)
    CP-3  species at alpha/2 and genus at alpha/2; species set if |S| <= K else genus set.
          A miss needs one of the two sets to miss, so P(miss) <= alpha for any K (union bound).
    CP-4  species at alpha + an uncalibrated genus switch (top genera up to 1 - alpha probability mass).
          No guarantee; shown to motivate CP-3.
Coverage of a genus set means the true GENUS is in it (plan §3.5).
"""
import numpy as np


def softmax(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def genus_probs(p, genus_of, num_genera):
    out = np.zeros((p.shape[0], num_genera))
    np.add.at(out.T, genus_of, p.T)          # sum species probabilities within each genus
    return out


def scores(p, prior, kind='pas'):
    if kind == 'pas':
        return -p / np.maximum(prior, 1e-12)[None, :]
    if kind == 'softmax':
        return 1.0 - p
    raise ValueError(kind)


def calibrate(cal_scores_true, alpha):
    """Split-conformal threshold: the ceil((n+1)(1-alpha))-th smallest calibration score."""
    n = len(cal_scores_true)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return float(np.sort(cal_scores_true)[k - 1])


def sets_from_threshold(s, qhat):
    return s <= qhat                          # boolean [N, C]


class GenusFallbackCP(object):
    def __init__(self, counts, genus_of, score='pas'):
        self.counts = np.asarray(counts, dtype=np.float64)
        self.genus_of = np.asarray(genus_of)
        self.G = int(self.genus_of.max()) + 1
        self.prior_s = self.counts / self.counts.sum()
        self.prior_g = np.bincount(self.genus_of, weights=self.counts, minlength=self.G) / self.counts.sum()
        self.species_per_genus = np.bincount(self.genus_of, minlength=self.G)
        self.score = score

    def _scores(self, logits):
        p = softmax(logits)
        pg = genus_probs(p, self.genus_of, self.G)
        return p, pg, scores(p, self.prior_s, self.score), scores(pg, self.prior_g, self.score)

    def fit(self, cal_logits, cal_y, alpha):
        _, _, ss, sg = self._scores(cal_logits)
        y = np.asarray(cal_y)
        idx = np.arange(len(y))
        true_s, true_g = ss[idx, y], sg[idx, self.genus_of[y]]
        self.alpha = alpha
        self.q = {'species': calibrate(true_s, alpha), 'genus': calibrate(true_g, alpha),
                  'species_half': calibrate(true_s, alpha / 2), 'genus_half': calibrate(true_g, alpha / 2)}
        return self

    def predict(self, logits, procedure, K=5):
        """Returns dict: species_set [N, C] bool, genus_set [N, G] bool, fallback [N] bool."""
        p, pg, ss, sg = self._scores(logits)
        N = p.shape[0]
        if procedure == 'CP-1':
            S = sets_from_threshold(ss, self.q['species'])
            return {'species_set': S, 'genus_set': np.zeros((N, self.G), bool), 'fallback': np.zeros(N, bool)}
        if procedure == 'CP-2':
            G = sets_from_threshold(sg, self.q['genus'])
            return {'species_set': np.zeros_like(ss, bool), 'genus_set': G, 'fallback': np.ones(N, bool)}
        if procedure == 'CP-3':
            S = sets_from_threshold(ss, self.q['species_half'])
            G = sets_from_threshold(sg, self.q['genus_half'])
            fb = S.sum(1) > K
            return {'species_set': S & ~fb[:, None], 'genus_set': G & fb[:, None], 'fallback': fb}
        if procedure == 'CP-4':
            S = sets_from_threshold(ss, self.q['species'])
            fb = S.sum(1) > K
            order = np.argsort(-pg, axis=1)
            cum = np.cumsum(np.take_along_axis(pg, order, 1), 1)
            keep_sorted = (cum - np.take_along_axis(pg, order, 1)) < (1 - self.alpha)   # smallest prefix reaching 1-alpha
            G = np.zeros_like(pg, bool)
            np.put_along_axis(G, order, keep_sorted, 1)
            return {'species_set': S & ~fb[:, None], 'genus_set': G & fb[:, None], 'fallback': fb}
        raise ValueError(procedure)

    def metrics(self, out, y):
        y = np.asarray(y)
        idx = np.arange(len(y))
        fb = out['fallback']
        covered = np.where(fb, out['genus_set'][idx, self.genus_of[y]], out['species_set'][idx, y])
        # set size in leaves: a genus set counts every species of its genera
        leaves = np.where(fb, out['genus_set'].astype(np.int64) @ self.species_per_genus, out['species_set'].sum(1))
        per_class = np.array([covered[y == c].mean() for c in np.unique(y)])
        return {'coverage': float(covered.mean()), 'macro_coverage': float(per_class.mean()),
                'classes_below_50pct': int((per_class < 0.5).sum()), 'classes_evaluated': int(len(per_class)),
                'avg_set_size_leaves': float(leaves.mean()), 'median_set_size_leaves': float(np.median(leaves)),
                'fallback_rate': float(fb.mean()), 'empty_sets': float((leaves == 0).mean())}
