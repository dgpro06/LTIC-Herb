"""C2 (taxonomy-guided feature calibration) tests: donor selection, shrinkage, covariance, sampling
determinism, and C2 on top of every base method (PLAN_NEXT.md §6 "Done when")."""
import csv
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lt import calibration as C  # noqa: E402
from lt import harness, metrics  # noqa: E402
from lt import trainer as T  # noqa: E402
from tests.synthetic import make_dataset  # noqa: E402


# classes in output order (rarest first): 0-3 tail (few_end = 4), 4-7 donors
COUNTS = np.array([3, 5, 0, 8, 30, 40, 50, 60])
GENUS = np.array([0, 1, 2, 3, 0, 0, 4, 5])
FAMILY = np.array([0, 1, 2, -1, 0, 0, 1, 3])


def test_donor_levels_and_self_exclusion():
    sel = C.select_donors(COUNTS, GENUS, FAMILY, few_end=4, n_min=20)
    assert sel[0][1] == 'genus' and list(sel[0][0]) == [4, 5]          # congeners with >= n_min images
    assert sel[1][1] == 'family' and list(sel[1][0]) == [6]            # no congener -> same family
    assert sel[2] == (None, 'none')                                    # no train image -> no synthesis
    assert sel[3][1] == 'global' and list(sel[3][0]) == [4, 5, 6, 7]   # unknown family -> global pool
    # a class never donates to itself, even when it is eligible
    sel = C.select_donors(COUNTS, GENUS, FAMILY, few_end=4, n_min=3)
    assert 0 not in sel[0][0]
    # without the global fallback the last level is 'none'
    assert C.select_donors(COUNTS, GENUS, FAMILY, 4, 20, global_fallback=False)[3] == (None, 'none')
    s = C.donor_summary(COUNTS, GENUS, FAMILY, 4, 20)
    assert (s['genus'], s['family'], s['global'], s['none']) == (1, 1, 1, 1)


def _features(seed=0, dim=6):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c, n in enumerate(COUNTS):
        xs.append(rng.normal(c, 1.0 + 0.2 * c, (n, dim)))
        ys.append(np.full(n, c))
    return torch.tensor(np.concatenate(xs), dtype=torch.float16), torch.tensor(np.concatenate(ys))


def test_shrunk_mean_and_diag_covariance():
    x, y = _features()
    st = C.DonorStats(x, y, COUNTS, GENUS, FAMILY, few_end=4, n_min=20, rank='diag')
    assert list(st.tail) == [0, 1, 3]
    xf = x.double()
    mu = torch.stack([xf[y == c].mean(0) if COUNTS[c] else torch.zeros(6, dtype=torch.float64)
                      for c in range(len(COUNTS))])
    w = torch.tensor([30., 40.], dtype=torch.float64) / 70
    mu_g = w[0] * mu[4] + w[1] * mu[5]
    var_g = w[0] * xf[y == 4].var(0) + w[1] * xf[y == 5].var(0)
    assert torch.allclose(st.set_mu[st.tail_set[0]], mu_g, atol=1e-6)
    assert torch.allclose(st.set_std[st.tail_set[0]] ** 2, var_g, atol=1e-6)
    # tau_mu = 0 keeps the species mean; a huge tau_mu moves it to the donor mean; in between: the formula
    assert torch.allclose(st.sampler(0.0, 2).mean[0].double(), mu[0], atol=1e-4)
    assert torch.allclose(st.sampler(1e9, 2).mean[0].double(), mu_g, atol=1e-4)
    m = st.sampler(6.0, 2).mean[0].double()
    assert torch.allclose(m, (3 * mu[0] + 6.0 * mu_g) / 9.0, atol=1e-4)


def test_low_rank_covariance_matches_pooled_covariance():
    x, y = _features(dim=5)
    st = C.DonorStats(x, y, COUNTS, GENUS, FAMILY, few_end=4, n_min=20, rank=5)    # full rank: exact
    xf = x.double()
    w = torch.tensor([30., 40.], dtype=torch.float64) / 70
    cov = w[0] * torch.cov(xf[y == 4].t()) + w[1] * torch.cov(xf[y == 5].t())
    f = st.set_factor[st.tail_set[0]]
    assert torch.allclose(f @ f.t(), cov, atol=1e-6)
    st2 = C.DonorStats(x, y, COUNTS, GENUS, FAMILY, few_end=4, n_min=20, rank=2)   # truncated + residual
    f2 = st2.set_factor[st2.tail_set[0]]
    ev = torch.linalg.eigvalsh(cov)
    assert f2.shape == (5, 2)
    assert abs(float(st2.set_resid[st2.tail_set[0]] ** 2) - float(ev[:3].mean())) < 1e-6


def test_rank_slicing_matches_a_direct_build():
    x, y = _features()
    big = C.DonorStats(x, y, COUNTS, GENUS, FAMILY, few_end=4, n_min=20, rank=5)
    direct = C.DonorStats(x, y, COUNTS, GENUS, FAMILY, few_end=4, n_min=20, rank=2)
    sliced = big.at_rank(2)
    assert big.rank == 5 and sliced.rank == 2                    # the cached object is not modified
    for a, b in zip(sliced.set_factor, direct.set_factor):
        assert torch.allclose(a @ a.t(), b @ b.t(), atol=1e-8)
    assert torch.allclose(sliced.set_resid, direct.set_resid, atol=1e-8)
    with pytest.raises(ValueError):
        big.at_rank('diag')


@pytest.mark.parametrize('rank', ['diag', 3])
def test_sampling_is_seeded_and_labels_are_tail(rank):
    x, y = _features()
    sp = C.DonorStats(x, y, COUNTS, GENUS, FAMILY, few_end=4, n_min=20, rank=rank).sampler(5.0, 7)
    g1, g2, g3 = (torch.Generator().manual_seed(s) for s in (0, 0, 1))
    a, la = sp.sample(g1)
    b, lb = sp.sample(g2)
    c, _ = sp.sample(g3)
    assert a.shape == (21, 6) and torch.equal(a, b) and not torch.equal(a, c)
    assert sorted(set(la.tolist())) == [0, 1, 3] and torch.equal(la, lb)
    # draws are centred on the shrunk mean (large sample)
    big = C.DonorStats(x, y, COUNTS, GENUS, FAMILY, few_end=4, n_min=20, rank=rank).sampler(5.0, 4000)
    s, l = big.sample(torch.Generator().manual_seed(0))
    assert torch.allclose(s[l == 0].mean(0), big.mean[0], atol=0.15)


def test_parse_method_suffixes():
    assert harness.parse_method('crt+c2') == ('crt', {'c2': True})
    assert harness.parse_method('ltic3+c2shuf@gamma=0') == ('ltic3', {'c2': True, 'c2_shuffle': True, 'gamma': 0})
    with pytest.raises(ValueError):
        harness.parse_method('crt+c3')
    cfg = harness.sample_config('crt', np.random.default_rng(0), {'c2': True})
    assert cfg['c2'] and set(harness.C2_SPACE) <= set(cfg)
    assert 'c2_k' not in harness.sample_config('crt', np.random.default_rng(0))


@pytest.fixture(scope='module')
def synth(tmp_path_factory):
    return make_dataset(str(tmp_path_factory.mktemp('synth_c2')))


def test_c2_on_every_base_method(synth):
    conf, _, _, _ = synth
    data = harness.Data(conf, device=torch.device('cpu'))
    for name in ('ce+c2', 'la+c2', 'crt+c2', 'tau_norm+c2', 'ltic3+c2', 'flat_genus+c2', 'crt+c2shuf'):
        method, fixed = harness.parse_method(name)
        cfg = harness.sample_config(method, np.random.default_rng(0), fixed, base={'epochs': 4, 'eval_every': 2})
        cfg.update(lr=0.05, weight_decay=1e-4, batch_size=256, c2_rank=8, c2_nmin=20)
        model, info = harness.train_one(cfg, data, seed=0)
        assert info['best_epoch'] > 0 and 'c2_summary' in info, name
        logits, _ = T.predict(model, data.fs['test'], info['cfg'], data.log_pi)
        m = metrics.evaluate(logits, data.fs['test'].y.cpu().numpy(), data.groups)
        assert m['top1'] > 3.0 / conf['num_classes'], (name, m['top1'])
    # same seed -> same model; the donor statistics are cached across trials
    cfg = harness.sample_config('crt', np.random.default_rng(1), {'c2': True}, base={'epochs': 2, 'eval_every': 2})
    m1, _ = harness.train_one(cfg, data, seed=3)
    m2, _ = harness.train_one(cfg, data, seed=3)
    assert all(torch.equal(a, b) for a, b in zip(m1.state_dict().values(), m2.state_dict().values()))
    n_cache = len(data.c2_cache)
    harness.train_one(dict(cfg, c2_tau_mu=1.0, c2_k=5), data, seed=0)
    assert len(data.c2_cache) == n_cache
    # A9 uses a different (shuffled) donor structure
    real = data.c2_stats(dict(cfg, c2_shuffle=False))
    shuf = data.c2_stats(dict(cfg, c2_shuffle=True))
    assert real is not shuf
    assert [tuple(d) for d in real.donor_sets] != [tuple(d) for d in shuf.donor_sets]


def test_runner_end_to_end_with_c2(synth, tmp_path):
    _, conf_path, _, _ = synth
    results, preds = str(tmp_path / 'all_runs.csv'), str(tmp_path / 'preds')
    out = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'run_baselines.py'), '--config', conf_path,
                          '--methods', 'crt+c2,crt+c2shuf', '--trials', '2', '--seeds', '0', '--epochs', '2',
                          '--results', results, '--preds', preds], cwd=str(tmp_path), capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-3000:]
    rows = list(csv.DictReader(open(results, encoding='utf-8')))
    assert {r['method'] for r in rows} == {'crt+c2', 'crt+c2shuf'}
    assert all(json.loads(r['hparams'])['c2'] for r in rows)
    assert all(json.loads(r['hparams'])['c2_shuffle'] for r in rows if r['method'] == 'crt+c2shuf')
    assert os.path.isfile(os.path.join(preds, 'synth_synth_crt+c2_test_seed0.npz'))
    donors = json.load(open(tmp_path / 'c2' / 'synth_synth_crt+c2_donors.json', encoding='utf-8'))
    assert donors['tail_species'] > 0
