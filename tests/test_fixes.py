"""Tests for the 2026-11-18 fixes (after the week-2 gate review):
  1. class-stratified paired bootstrap for macro-F1 (lt.metrics)
  2. McNemar counts for C2 only when C2 is ahead (tools/week2_gate.compare)
  3. A9 control searched on the same candidate configurations as its C2 method (lt.harness.search_seed_key)
  4. C2 on a frozen base: +c2f / +c2fshuf (lt.harness, tools/run_baselines.py)
  5. post-hoc knob grid on V-tune (dataset config "posthoc_grid")
  6. prior that counts C2's synthetic tail samples (c2_prior='augmented')
"""
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

from lt import harness, metrics  # noqa: E402
from lt import trainer as T  # noqa: E402
from tests.synthetic import make_dataset  # noqa: E402
from tools import week2_gate  # noqa: E402


@pytest.fixture(scope='module')
def synth(tmp_path_factory):
    return make_dataset(str(tmp_path_factory.mktemp('synth_fixes')))


# ----------------------------------------------------------------------------- 1. stratified bootstrap
def test_stratified_resample_keeps_every_class_support():
    y = np.array([0, 1, 2, 2, 2, 3, 3, 3, 3, 3, 7])
    rng = np.random.default_rng(0)
    for _ in range(50):
        idx = metrics.stratified_resample(y, rng)
        assert np.array_equal(np.bincount(y[idx], minlength=8), np.bincount(y, minlength=8))
        assert np.array_equal(np.sort(y[idx]), np.sort(y))


def _singleton_case():
    """300 singleton classes both methods get right; a big class (300) of 700 images. Method A sends 100 big-class
    images to singleton classes 0..99, method B sends 100 to an absent class 301. Image resampling drops singleton
    classes and then scores A's still-predicted ones as 0, biasing A - B downwards."""
    y = np.concatenate([np.arange(300), np.full(700, 300)])
    a, b = y.copy(), y.copy()
    a[300:400] = np.arange(100)
    b[300:400] = 301
    return y, a, b


def test_stratified_bootstrap_is_less_off_centre_than_image_bootstrap():
    y, a, b = _singleton_case()
    f1 = lambda t, p: metrics.macro_f1(t, p, 302)                                   # noqa: E731
    d, ci_img, _ = metrics.paired_bootstrap(f1, y, a, b, n_boot=300, seed=0)
    d2, ci_str, _ = metrics.paired_bootstrap(f1, y, a, b, n_boot=300, seed=0, stratified=True)
    assert d == d2
    assert not (ci_img[0] <= d <= ci_img[1])            # the bias the week-2 gate showed: delta outside its own CI
    off_img = abs(0.5 * (ci_img[0] + ci_img[1]) - d)
    off_str = abs(0.5 * (ci_str[0] + ci_str[1]) - d)
    assert off_str < 0.5 * off_img                      # much smaller, but not zero in this extreme case:
    # that is why the significance decision uses the randomization test below, not this interval


def test_randomization_test_direction_and_ties():
    y, a, b = _singleton_case()
    f1 = lambda t, p: metrics.macro_f1(t, p, 302)                                   # noqa: E731
    d, p_gt, p_two = metrics.paired_randomization(f1, y, a, b, n_perm=300, seed=0)
    assert d < 0 and p_gt > 0.9 and p_two < 0.01        # A is clearly worse
    d, p_gt, _ = metrics.paired_randomization(f1, y, b, a, n_perm=300, seed=0)
    assert d > 0 and p_gt < 0.01                        # B is clearly better
    d, p_gt, p_two = metrics.paired_randomization(f1, y, a, a, n_perm=50, seed=0)
    assert d == 0 and p_gt == 1.0 and p_two == 1.0      # identical predictions are never significant


def test_randomization_test_false_positive_rate_is_controlled():
    """Two methods drawn from the SAME process: one-sided p < 0.05 should happen ~5% of the time."""
    rng = np.random.default_rng(1)
    y = np.concatenate([np.arange(150), rng.integers(150, 160, 450)])    # 150 singletons + 10 bigger classes
    f1 = lambda t, p: metrics.macro_f1(t, p, 160)                                   # noqa: E731
    hits = 0
    reps = 60
    for r in range(reps):
        noisy = lambda: np.where(rng.random(len(y)) < 0.3, rng.integers(0, 160, len(y)), y)   # noqa: E731
        _, p_gt, _ = metrics.paired_randomization(f1, y, noisy(), noisy(), n_perm=199, seed=r)
        hits += p_gt < 0.05
    assert hits / reps <= 0.15


# ----------------------------------------------------------------------------- 2. McNemar direction
def _gate_inputs(a_ahead):
    """Few-shot classes 0..9 (few_end = 10), 2 images each; a big class 10 with 200 images."""
    y = np.concatenate([np.repeat(np.arange(10), 2), np.full(200, 10)])
    a, b = y.copy(), y.copy()
    if a_ahead:
        b[20:80] = 0                                    # baseline wrong on 60 images, C2 right
    else:
        a[20:80] = 0                                    # C2 wrong on 60 images, baseline right
    mk = lambda p: {'y': y, 'pred': p, 'few_end': 10, 'C': 11}                       # noqa: E731
    return mk(a), mk(b)


@pytest.mark.parametrize('a_ahead', [True, False])
def test_mcnemar_counts_only_when_c2_is_ahead(monkeypatch, a_ahead):
    # a bootstrap CI that includes 0 with a positive macro-F1 delta: only McNemar can decide criterion 1
    monkeypatch.setattr(week2_gate.M, 'paired_bootstrap', lambda *a, **k: (0.01, (-0.01, 0.02), False))
    a, b = _gate_inputs(a_ahead)
    out = week2_gate.compare(a, b, 'c2', 'base', n_boot=10)
    mf = out['macro_f1']
    assert mf['mcnemar_p_all_images'] < 0.05
    assert (mf['mcnemar_a_only_right'] > mf['mcnemar_b_only_right']) == a_ahead
    assert mf['pass'] is a_ahead                        # before the fix this was True in both cases
    # few-shot criterion: same direction rule
    fs = out['few_shot_pooled_acc']
    assert fs['pass'] is False                          # nothing differs on the few-shot images here
    # the randomization rule ignores McNemar: C2 behind on every differing image -> never a pass
    if not a_ahead:
        r = week2_gate.compare(a, b, 'c2', 'base', n_boot=99, f1_test='randomization')['macro_f1']
        assert r['decided_by'] == 'randomization' and r['pass'] is False and r['randomization_p_greater'] > 0.5


# ----------------------------------------------------------------------------- 3. paired A9 candidates
def test_search_seed_key_pairs_the_a9_control():
    k = harness.search_seed_key
    assert k('crt+c2shuf') == k('crt+c2') == 'crt+c2'
    assert k('crt+c2shuf@c2_shuffle_seed=2') == 'crt+c2'
    assert k('tau_norm+c2fshuf@c2_shuffle_seed=1') == k('tau_norm+c2f') == 'tau_norm+c2f'
    assert k('ltic3+c2shuf@gamma=0,c2_shuffle_seed=1') == 'ltic3+c2@gamma=0'
    assert k('ltic3@gamma=0') == 'ltic3@gamma=0'
    assert k('crt+c2') != k('crt+c2f')                  # frozen and non-frozen searches stay separate


def test_real_and_shuffled_c2_draw_identical_candidates():
    def draws(name):
        method, fixed = harness.parse_method(name)
        rng = np.random.default_rng(int(__import__('hashlib').sha1(harness.search_seed_key(name).encode())
                                        .hexdigest(), 16) % (2 ** 32))
        return [harness.sample_config(method, rng, fixed) for _ in range(5)]
    real, shuf, shuf2 = draws('crt+c2'), draws('crt+c2shuf'), draws('crt+c2shuf@c2_shuffle_seed=3')
    for r, s, s2 in zip(real, shuf, shuf2):
        drop = ('c2_shuffle', 'c2_shuffle_seed')
        assert {k: v for k, v in r.items() if k not in drop} == {k: v for k, v in s.items() if k not in drop}
        assert {k: v for k, v in r.items() if k not in drop} == {k: v for k, v in s2.items() if k not in drop}
        assert s['c2_shuffle'] and s2['c2_shuffle_seed'] == 3


# ----------------------------------------------------------------------------- 4. frozen base
def test_base_method_name_and_frozen_settings():
    assert harness.base_method_name('tau_norm+c2f') == 'tau_norm'
    assert harness.base_method_name('tau_norm+c2fshuf@c2_shuffle_seed=2') == 'tau_norm'
    assert harness.base_method_name('ltic3+c2f@gamma=0,c2_rank=8') == 'ltic3@gamma=0'
    base = {'lr': 0.05, 'weight_decay': 1e-4, 'batch_size': 256, 'epochs': 2, 'test_tau': 0.7, 'tau': 0.0,
            'method': 'tau_norm', 'c2_k': 5}
    fs = harness.frozen_settings('tau_norm', base)
    assert fs == {'lr': 0.05, 'weight_decay': 1e-4, 'batch_size': 256, 'epochs': 2}   # no post-hoc knobs, no C2


def test_frozen_base_search_only_varies_c2_settings(synth, tmp_path):
    conf, _, _, _ = synth
    data = harness.Data(conf, device=torch.device('cpu'))
    frozen = dict(T.DEFAULTS, lr=0.05, weight_decay=1e-4, batch_size=256, epochs=2, test_tau=0.7, method='tau_norm')
    results = str(tmp_path / 'runs.csv')
    with pytest.raises(ValueError):
        harness.search('tau_norm+c2f', data, 1, results, base={'epochs': 2, 'eval_every': 2})
    harness.search('tau_norm+c2f', data, 3, results, base={'epochs': 2, 'eval_every': 2}, frozen_base=frozen,
                   log=lambda *_: None)
    rows = [json.loads(r['hparams']) for r in csv.DictReader(open(results, encoding='utf-8'))]
    assert len(rows) == 3
    for h in rows:
        assert (h['lr'], h['weight_decay'], h['batch_size'], h['epochs']) == (0.05, 1e-4, 256, 2)
        assert h['c2'] and h['c2_frozen_base']
    assert len({(h['c2_k'], h['c2_rank'], h['c2_nmin'], round(h['c2_tau_mu'], 6)) for h in rows}) > 1
    assert len({round(h['test_tau'], 6) for h in rows}) > 1         # the post-hoc knob is re-tuned, not frozen


def test_runner_frozen_base_end_to_end(synth, tmp_path):
    _, conf_path, _, _ = synth
    results, preds = str(tmp_path / 'all_runs.csv'), str(tmp_path / 'preds')
    run = lambda methods: subprocess.run(                                            # noqa: E731
        [sys.executable, os.path.join(ROOT, 'tools', 'run_baselines.py'), '--config', conf_path, '--methods', methods,
         '--trials', '2', '--seeds', '0', '--epochs', '2', '--results', results, '--preds', preds],
        cwd=str(tmp_path), capture_output=True, text=True)
    bad = run('crt+c2f')                                # base not run yet -> clear error, nothing written
    assert bad.returncode != 0 and 'run crt first' in (bad.stdout + bad.stderr)
    ok = run('tau_norm,tau_norm+c2f,tau_norm+c2fshuf')
    assert ok.returncode == 0, ok.stderr[-3000:]
    base = json.load(open(tmp_path / 'configs' / 'synth_synth_tau_norm.json', encoding='utf-8'))
    for m in ('tau_norm+c2f', 'tau_norm+c2fshuf'):
        cfg = json.load(open(tmp_path / 'configs' / 'synth_synth_{}.json'.format(m), encoding='utf-8'))
        assert all(cfg[k] == base[k] for k in ('lr', 'weight_decay', 'batch_size', 'epochs'))
        assert cfg['c2_frozen_base'] and cfg['c2_shuffle'] == (m.endswith('shuf'))


# ----------------------------------------------------------------------------- confirmatory gate (H22 rules)
def test_confirmatory_gate_end_to_end(synth, tmp_path):
    _, conf_path, _, _ = synth
    results, preds = str(tmp_path / 'all_runs.csv'), str(tmp_path / 'preds')
    methods = 'ce,tau_norm,tau_norm+c2f,tau_norm+c2fshuf,tau_norm+c2fshuf@c2_shuffle_seed=1'
    run = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'run_baselines.py'), '--config', conf_path,
                          '--methods', methods, '--trials', '1', '--seeds', '0', '--epochs', '2',
                          '--results', results, '--preds', preds], cwd=str(tmp_path), capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-3000:]
    with open(os.path.join(ROOT, 'gates', 'h22_confirmatory.json'), encoding='utf-8') as f:
        rules = json.load(f)                            # the real draft, pointed at the synthetic dataset
    rules.update(dataset='synth', encoders={'primary': ['synth']}, baselines=['ce', 'tau_norm'],
                 c2_methods=['tau_norm+c2f'], a9=dict(rules['a9'], shuffle_seeds=[0, 1]),
                 criteria=dict(rules['criteria'], n_perm=99))
    rules_path = tmp_path / 'rules.json'
    rules_path.write_text(json.dumps(rules), encoding='utf-8')
    cmd = [sys.executable, os.path.join(ROOT, 'tools', 'confirmatory_gate.py'), '--rules', str(rules_path),
           '--results', results, '--preds', preds]
    g = subprocess.run(cmd + ['--encoder', 'synth'], cwd=str(tmp_path), capture_output=True, text=True)
    assert g.returncode == 0, g.stderr[-3000:]
    res = json.load(open(tmp_path / 'confirmatory_synth_synth.json', encoding='utf-8'))
    assert res['selection']['c2_method'] == 'tau_norm+c2f' and res['selection']['best_baseline'] in ('ce', 'tau_norm')
    assert res['criterion1']['macro_f1']['decided_by'] == 'randomization'
    assert set(res['criterion2_a9']['per_shuffle_seed']) == {'tau_norm+c2fshuf', 'tau_norm+c2fshuf@c2_shuffle_seed=1'}
    assert res['criterion2_a9']['base'] == 'tau_norm' and isinstance(res['pass'], bool)
    s = subprocess.run(cmd + ['--summary'], cwd=str(tmp_path), capture_output=True, text=True)
    assert s.returncode == 0 and json.loads(s.stdout)['verdict'] in rules['verdict'].values()
    assert subprocess.run(cmd + ['--encoder', 'synth'], cwd=str(tmp_path), capture_output=True,
                          text=True).returncode != 0                                   # recorded once


# ----------------------------------------------------------------------------- 5. post-hoc grid
def test_posthoc_grid_picks_the_best_knob_on_vtune(synth):
    conf, _, _, _ = synth
    data = harness.Data(dict(conf, posthoc_grid=True), device=torch.device('cpu'))
    cfg = harness.sample_config('tau_norm', np.random.default_rng(0), base={'epochs': 2, 'eval_every': 2})
    cfg['test_tau'] = 1.37                                                  # off the grid on purpose
    model, info = harness.train_one(cfg, data, seed=0)
    best_cfg, best = harness.tune_posthoc(model, info['cfg'], data)
    assert best_cfg['test_tau'] in harness.POSTHOC_GRID
    y = data.fs['vtune'].y.cpu().numpy()
    for v in (0.0, 1.0, 2.0):
        logits, _ = T.predict(model, data.fs['vtune'], dict(info['cfg'], test_tau=v), data.log_pi)
        assert best >= metrics.macro_f1(y, logits.argmax(1).numpy(), conf['num_classes']) - 1e-12
    # a knob fixed in the method name is left alone; methods without a post-hoc knob are untouched
    assert harness.posthoc_knob(dict(cfg), {'test_tau': 0.0}) is None
    assert harness.posthoc_knob({'method': 'crt'}) is None
    assert harness.posthoc_knob({'method': 'la', 'la_mode': 'loss'}) is None
    assert harness.posthoc_knob({'method': 'la', 'la_mode': 'posthoc'}) == 'tau'


# ----------------------------------------------------------------------------- 6. augmented prior
def test_prior_counts():
    class Stats:
        tail = np.array([0, 2])
    counts = np.array([3, 30, 5, 100])
    assert np.array_equal(T.prior_counts({'c2': False}, counts), counts)
    assert np.array_equal(T.prior_counts({'c2': True}, counts, Stats()), counts)            # default 'real'
    assert np.array_equal(T.prior_counts({'c2': True, 'c2_prior': 'augmented', 'c2_k': 5}, counts, Stats()),
                          [8, 30, 10, 100])
    with pytest.raises(ValueError):
        T.prior_counts({'c2': True, 'c2_prior': 'bogus'}, counts, Stats())
    with pytest.raises(ValueError):
        T.prior_counts({'c2': True, 'c2_prior': 'augmented'}, counts)


def test_augmented_prior_is_used_and_saved(synth, tmp_path):
    conf, _, _, _ = synth
    data = harness.Data(conf, device=torch.device('cpu'))
    cfg = harness.sample_config('la', np.random.default_rng(0), {'c2': True},
                                base={'epochs': 2, 'eval_every': 2, 'c2_prior': 'augmented'})
    cfg.update(la_mode='posthoc', tau=1.0, c2_k=7)
    stats = data.c2_stats(cfg)
    expect = data.index.counts.astype(np.float64).copy()
    expect[stats.tail] += 7
    assert np.array_equal(data.prior_counts(cfg), expect)
    assert not torch.allclose(data.log_pi_for(cfg), data.log_pi)
    assert data.log_pi_for(dict(cfg, c2_prior='real')) is data.log_pi
    harness.final_runs('la+c2', cfg, data, [0], str(tmp_path / 'runs.csv'), save_dir=str(tmp_path / 'preds'))
    d = np.load(tmp_path / 'preds' / 'synth_synth_la+c2_test_seed0.npz')
    assert np.array_equal(d['prior_counts'], expect) and np.array_equal(d['counts'], data.index.counts)
    # the gate applies post-hoc tau with the saved training prior
    p = week2_gate.load_pred(str(tmp_path / 'preds'), 'synth', 'synth', 'la+c2', split='test')
    logits = d['logits'].astype(np.float64) - 1.0 * np.log(expect / expect.sum())[None, :]
    assert np.array_equal(p['pred'], logits.argmax(1))


# ----------------------------------------------------------------------------- 7. compact predictions (H22)
@pytest.mark.parametrize('method,extra', [('tau_norm', {'test_tau': 0.8}), ('la', {'la_mode': 'posthoc', 'tau': 1.0}),
                                          ('ltic3', {'gamma': 0.3})])
def test_compact_predictions_match_full(synth, tmp_path, method, extra):
    """pred_format='topk' must give the same metrics, rows and gate predictions as the full logits."""
    conf, _, _, _ = synth
    cfg = harness.sample_config(method, np.random.default_rng(1), base={'epochs': 2, 'eval_every': 2})
    cfg.update(extra)
    out = {}
    for fmt in ('full', 'topk'):
        data = harness.Data(dict(conf, pred_format=fmt, pred_topk=7), device=torch.device('cpu'))
        res = str(tmp_path / '{}.csv'.format(fmt))
        preds = str(tmp_path / fmt)
        vs = harness.score_vselect(method, cfg, data, res, save_dir=preds)
        fin = harness.final_runs(method, cfg, data, [0], res, save_dir=preds, splits=('test', 'vcal'))
        out[fmt] = (vs, fin, preds, data)
    assert out['full'][0] == out['topk'][0]
    assert [m for _, _, m in out['full'][1]] == [m for _, _, m in out['topk'][1]]
    full_dir, top_dir, data = out['full'][2], out['topk'][2], out['topk'][3]
    reported = dict([('vselect', out['topk'][0])] + [(s, m) for _, s, m in out['topk'][1]])
    for split in ('vselect', 'test', 'vcal'):
        a = week2_gate.load_pred(full_dir, 'synth', 'synth', method, split=split)
        b = week2_gate.load_pred(top_dir, 'synth', 'synth', method, split=split)
        assert a['C'] == b['C'] and a['few_end'] == b['few_end'] and np.array_equal(a['y'], b['y'])
        # compact stores the exact decision: it reproduces the reported metric exactly
        assert metrics.macro_f1(b['y'], b['pred'], b['C']) == reported[split]['macro_f1']
        # the full format re-derives it from fp16 logits, which can flip a rare near-tie
        assert (a['pred'] == b['pred']).mean() > 0.97
    d = np.load(os.path.join(top_dir, 'synth_synth_{}_test_seed0.npz'.format(method)))
    assert 'logits' not in d.files and d['top_idx'].shape == (len(d['y']), 7)
    assert np.array_equal(d['top_idx'][:, 0], d['pred'])
    # the stored top-k logits + logsumexp give the exact softmax of the final logits
    model, info = harness.train_one(cfg, data, 0)
    logits, _ = T.predict(model, data.fs['test'], info['cfg'], data.log_pi)
    p = torch.softmax(logits.float(), 1).gather(1, torch.as_tensor(d['top_idx']).long()).numpy()
    q = np.exp(d['top_val'].astype(np.float64) - d['logsumexp'][:, None])
    assert d['top_val'].dtype == np.float32
    assert np.allclose(p, q, atol=1e-5)


def test_predict_labels_equals_full_argmax(synth):
    conf, _, _, _ = synth
    data = harness.Data(conf, device=torch.device('cpu'))
    cfg = harness.sample_config('tau_norm', np.random.default_rng(2), base={'epochs': 2, 'eval_every': 2})
    model, info = harness.train_one(cfg, data, 0)
    for v in (0.0, 0.5, 1.5):
        c = dict(info['cfg'], test_tau=v, tau=0.7)
        logits, _ = T.predict(model, data.fs['vtune'], c, data.log_pi)
        assert np.array_equal(T.predict_labels(model, data.fs['vtune'], c, data.log_pi), logits.argmax(1).numpy())
    with pytest.raises(ValueError):
        harness.compact_preds({'pred_format': 'bogus'})
