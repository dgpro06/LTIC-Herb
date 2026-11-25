import numpy as np
import pytest

from lt.conformal import GenusFallbackCP, softmax
from lt.conformal_topk import TopKGenusFallbackCP, topk_probs


def _data(n, C, G, seed):
    rng = np.random.default_rng(seed)
    genus_of = np.sort(rng.integers(0, G, C))
    genus_of = np.unique(genus_of, return_inverse=True)[1]             # genera 0..G'-1, all used
    counts = np.sort(rng.integers(1, 200, C))
    y = rng.integers(0, C, n)
    logits = rng.normal(size=(n, C)) * 2
    logits[np.arange(n), y] += rng.normal(2.0, 1.0, n)                 # informative but imperfect
    return genus_of, counts, y, logits


def _topk(logits, k):
    idx = np.argsort(-logits, axis=1)[:, :k]
    val = np.take_along_axis(logits, idx, 1)
    m = logits.max(1, keepdims=True)
    lse = (m + np.log(np.exp(logits - m).sum(1, keepdims=True)))[:, 0]
    return idx, topk_probs(val, lse)


@pytest.mark.parametrize('score', ['pas', 'softmax'])
def test_topk_with_all_classes_equals_dense(score):
    genus_of, counts, y, logits = _data(600, 40, 12, seed=1)
    cal, test = slice(0, 300), slice(300, 600)
    dense = GenusFallbackCP(counts, genus_of, score=score)
    tk = TopKGenusFallbackCP(counts, genus_of, score=score)
    idx_c, p_c = _topk(logits[cal], 40)
    idx_t, p_t = _topk(logits[test], 40)
    assert np.allclose(np.take_along_axis(softmax(logits[test]), idx_t, 1), p_t)
    for alpha in (0.05, 0.1, 0.3):
        dense.fit(logits[cal], y[cal], alpha)
        tk.fit(idx_c, p_c, y[cal], alpha)
        for key in dense.q:
            assert np.isclose(dense.q[key], tk.q[key]), (key, dense.q[key], tk.q[key])
        for proc in ('CP-1', 'CP-2', 'CP-3', 'CP-4'):
            for K in (1, 3, 8):
                a = dense.metrics(dense.predict(logits[test], proc, K), y[test])
                b = tk.metrics(tk.predict(idx_t, p_t, proc, K), y[test])
                for m in a:
                    assert np.isclose(a[m], b[m]), (score, alpha, proc, K, m, a[m], b[m])


@pytest.mark.parametrize('score', ['pas', 'softmax'])
def test_topk_truncated_keeps_coverage(score):
    genus_of, counts, y, logits = _data(8000, 300, 60, seed=2)
    idx, p = _topk(logits, 10)
    tk = TopKGenusFallbackCP(counts, genus_of, score=score)
    for alpha in (0.1, 0.2):
        tk.fit(idx[:4000], p[:4000], y[:4000], alpha)
        for proc in ('CP-1', 'CP-2', 'CP-3'):
            m = tk.metrics(tk.predict(idx[4000:], p[4000:], proc, 5), y[4000:])
            assert m['coverage'] >= 1 - alpha - 0.02, (score, alpha, proc, m)
