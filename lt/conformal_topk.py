"""Prediction sets (CP-1..CP-4 of lt.conformal) from compact 'topk' predictions (H22: top-K logits + logsumexp).

Only the K highest-scoring classes of each image are stored. Every class outside them gets the WORST score of the
chosen kind (0 for PAS, whose scores are -p/pi <= 0; 1 for softmax, whose scores are 1 - p <= 1), and a genus is
scored on the probability of its species inside the top K (a lower bound on its true probability). This is a
fixed scoring rule, applied the same way to V-cal and test, so split-conformal coverage holds exactly; the sets
can only be larger than with full logits. If a threshold reaches the worst score, the set is the whole label
space (`full` below; counted with all leaves). With K = C this module gives the same sets as lt.conformal.
"""
import numpy as np

from lt.conformal import calibrate


def topk_probs(top_val, logsumexp):
    """Exact softmax probabilities of the stored classes."""
    return np.exp(top_val.astype(np.float64) - logsumexp.astype(np.float64)[:, None])


class TopKGenusFallbackCP(object):
    def __init__(self, counts, genus_of, score='pas'):
        if score not in ('pas', 'softmax'):
            raise ValueError(score)
        self.counts = np.asarray(counts, dtype=np.float64)
        self.genus_of = np.asarray(genus_of)
        self.C = len(self.counts)
        self.G = int(self.genus_of.max()) + 1
        self.prior_s = self.counts / self.counts.sum()
        self.prior_g = np.bincount(self.genus_of, weights=self.counts, minlength=self.G) / self.counts.sum()
        self.species_per_genus = np.bincount(self.genus_of, minlength=self.G)
        self.score = score
        self.worst = 0.0 if score == 'pas' else 1.0

    def _s(self, p, prior):
        return -p / np.maximum(prior, 1e-12) if self.score == 'pas' else 1.0 - p

    def _scores(self, idx, p):
        """Species scores [N, K]; genus ids, probability lower bounds and scores [N, K] (one slot per distinct
        genus in the row, the remaining slots are -1 / worst)."""
        ss = self._s(p, self.prior_s[idx])
        g = self.genus_of[idx]
        N, K = idx.shape
        order = np.argsort(g, axis=1, kind='stable')
        gs = np.take_along_axis(g, order, 1)
        ps = np.take_along_axis(p, order, 1)
        start = np.ones((N, K), bool)
        start[:, 1:] = gs[:, 1:] != gs[:, :-1]
        gid = np.full((N, K), -1, np.int64)
        pg = np.zeros((N, K))
        slot = np.cumsum(start, 1) - 1                       # slot of each entry's genus within the row
        rows = np.repeat(np.arange(N), K)
        np.add.at(pg, (rows, slot.ravel()), ps.ravel())
        gid[rows[start.ravel()], slot[start]] = gs[start]
        sg = np.where(gid >= 0, self._s(pg, self.prior_g[np.maximum(gid, 0)]), self.worst)
        return ss, gid, pg, sg

    def _true(self, idx, ss, gid, sg, y):
        hit_s = idx == y[:, None]
        true_s = np.where(hit_s.any(1), (np.where(hit_s, ss, 0)).sum(1), self.worst)
        hit_g = gid == self.genus_of[y][:, None]
        true_g = np.where(hit_g.any(1), (np.where(hit_g, sg, 0)).sum(1), self.worst)
        return true_s, true_g

    def fit(self, idx, p, y, alpha):
        ss, gid, _, sg = self._scores(idx, p)
        true_s, true_g = self._true(idx, ss, gid, sg, np.asarray(y))
        self.alpha = alpha
        self.q = {'species': calibrate(true_s, alpha), 'genus': calibrate(true_g, alpha),
                  'species_half': calibrate(true_s, alpha / 2), 'genus_half': calibrate(true_g, alpha / 2)}
        return self

    def _sets(self, s, q):
        full = np.full(s.shape[0], q >= self.worst)
        return (s <= q) & ~full[:, None], full

    def predict(self, idx, p, procedure, K=5):
        """Returns species/genus membership over the stored slots, `full` flags and the fallback flag."""
        ss, gid, pg, sg = self._scores(idx, p)
        N = idx.shape[0]
        none = np.zeros(idx.shape, bool)
        no = np.zeros(N, bool)
        if procedure == 'CP-1':
            S, fs = self._sets(ss, self.q['species'])
            return dict(idx=idx, gid=gid, S=S, S_full=fs, G=none, G_full=no, fallback=no)
        if procedure == 'CP-2':
            G, fg = self._sets(sg, self.q['genus'])
            return dict(idx=idx, gid=gid, S=none, S_full=no, G=G & (gid >= 0), G_full=fg, fallback=~no)
        if procedure == 'CP-3':
            S, fs = self._sets(ss, self.q['species_half'])
            G, fg = self._sets(sg, self.q['genus_half'])
            fb = fs | (S.sum(1) > K)
            return dict(idx=idx, gid=gid, S=S & ~fb[:, None], S_full=no, G=G & (gid >= 0) & fb[:, None],
                        G_full=fg & fb, fallback=fb)
        if procedure == 'CP-4':
            S, fs = self._sets(ss, self.q['species'])
            fb = fs | (S.sum(1) > K)
            pgs = np.where(gid >= 0, pg, -1.0)
            order = np.argsort(-pgs, axis=1)
            sorted_p = np.take_along_axis(np.maximum(pgs, 0), order, 1)
            keep_sorted = (np.cumsum(sorted_p, 1) - sorted_p) < (1 - self.alpha)
            G = np.zeros(idx.shape, bool)
            np.put_along_axis(G, order, keep_sorted, 1)
            G &= gid >= 0
            g_full = sorted_p.sum(1) < (1 - self.alpha)       # the stored genera never reach 1 - alpha: all genera
            return dict(idx=idx, gid=gid, S=S & ~fb[:, None], S_full=no, G=G & fb[:, None], G_full=g_full & fb,
                        fallback=fb)
        raise ValueError(procedure)

    def metrics(self, out, y):
        y = np.asarray(y)
        fb = out['fallback']
        in_s = out['S_full'] | (out['S'] & (out['idx'] == y[:, None])).any(1)
        in_g = out['G_full'] | (out['G'] & (out['gid'] == self.genus_of[y][:, None])).any(1)
        covered = np.where(fb, in_g, in_s)
        g_leaves = np.where(out['G'], self.species_per_genus[np.maximum(out['gid'], 0)], 0).sum(1)
        leaves = np.where(fb, np.where(out['G_full'], self.C, g_leaves),
                          np.where(out['S_full'], self.C, out['S'].sum(1)))
        cls, inv = np.unique(y, return_inverse=True)
        per_class = np.bincount(inv, weights=covered) / np.bincount(inv)
        return {'coverage': float(covered.mean()), 'macro_coverage': float(per_class.mean()),
                'classes_below_50pct': int((per_class < 0.5).sum()), 'classes_evaluated': int(len(cls)),
                'avg_set_size_leaves': float(leaves.mean()), 'median_set_size_leaves': float(np.median(leaves)),
                'fallback_rate': float(fb.mean()), 'empty_sets': float((leaves == 0).mean()),
                'full_sets': float(np.where(fb, out['G_full'], out['S_full']).mean())}
