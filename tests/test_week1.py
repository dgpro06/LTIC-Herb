"""Week-1 unit and integration tests (plan §15: "unit test passes on the synthetic set").

    python -m pytest tests -q
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lt import cf, conformal, diagnostics, harness, heads, losses, metrics  # noqa: E402
from lt import trainer as T  # noqa: E402
from lt.classes import ClassIndex, ShotGroups  # noqa: E402
from lt.features import load_split, save_chunk, write_manifest  # noqa: E402
from lt.taxonomy import Taxonomy  # noqa: E402
from tests.synthetic import make_dataset  # noqa: E402


# ----------------------------------------------------------------------------- class order, R1
def test_class_index_is_stable_and_ascending():
    labels = np.array([0, 0, 1, 2, 2, 3, 3, 4])            # counts 2,1,2,2,1 -> ties
    a, b = ClassIndex(labels, 6), ClassIndex(labels[::-1], 6)
    assert np.array_equal(a.class_map, b.class_map)          # order does not depend on file order
    assert list(a.counts) == [0, 1, 1, 2, 2, 2]
    assert list(a.inverse[:3]) == [5, 1, 4]                  # ties keep original-label order
    assert a.empty_classes == 1
    with pytest.raises(RuntimeError):
        a.check(np.roll(a.class_map, 1))


def test_shot_groups_masks_and_empty_many_group():
    counts = np.sort(np.array([1, 5, 19, 20, 50, 51, 80]))
    g = ShotGroups(counts, few_thr=20, many_thr=50)
    assert (g.few_end, g.many_start) == (3, 5)
    H, M, Tm = g.masks()
    assert H.all() and M.sum() == 5 and Tm.sum() == 3
    with pytest.raises(ValueError):                         # H22 with the default many_thr = 100
        ShotGroups(counts, few_thr=20, many_thr=100)
    with pytest.raises(ValueError):
        ShotGroups(counts, few_thr=None, many_thr=50)


# ----------------------------------------------------------------------------- metrics
def test_macro_f1_matches_sklearn():
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 30, 500), rng.integers(0, 35, 500)
    assert abs(metrics.macro_f1(y, p, 40) - f1_score(y, p, average='macro')) < 1e-9


def test_mcnemar_and_paired_bootstrap():
    y = np.zeros(200, int)
    a = np.zeros(200, int)
    b = np.zeros(200, int)
    b[:40] = 1                                               # b wrong on 40 images where a is right
    _, _, p = metrics.mcnemar(y, a, b)
    assert p < 1e-6
    d, (lo, hi), sig = metrics.paired_bootstrap(lambda t, q: float((t == q).mean()), y, a, b, n_boot=500)
    assert abs(d - 0.2) < 1e-9 and sig and lo > 0
    assert metrics.mcnemar(y, a, a)[2] == 1.0


# ----------------------------------------------------------------------------- heads and losses
def test_ltic_head_has_no_bias_and_gamma0_matches_train():
    h = heads.LTICHead(16, 10, gamma=0.0)
    assert h.fc.bias is None
    x = torch.randn(8, 16)
    h.train()
    h.intermediate[2].eval()                                 # freeze BN statistics for the comparison
    out_train = h(x)['species']
    h.eval()
    assert torch.allclose(out_train, h(x)['species'], atol=1e-6)


def test_ltic_loss_branch_masks():
    masks = losses.branch_masks_for(10, few_end=3, many_start=6, branches=3)
    assert [int(m.sum()) for m in masks] == [10, 6, 3]
    assert len(losses.branch_masks_for(10, 3, 6, branches=2)) == 2
    logits = [torch.zeros(4, 10, requires_grad=True) for _ in range(3)]
    target = torch.tensor([0, 4, 8, 9])                     # only class 0 reaches T, 0 and 4 reach M
    loss, fused, branch = losses.ltic_loss(logits, target, masks, alpha=0.5)
    assert abs(float(branch) - np.log(10)) < 1e-5 and abs(float(fused) - np.log(10)) < 1e-5
    # the fused term reaches every branch; with alpha = 1 only the masked branch term remains
    logits = [torch.zeros(4, 10, requires_grad=True) for _ in range(3)]
    losses.ltic_loss(logits, target, masks, alpha=1.0)[0].backward()
    assert logits[2].grad[1:].abs().sum() == 0               # T: no gradient from non-tail images
    assert logits[1].grad[2:].abs().sum() == 0 and logits[1].grad[:2].abs().sum() > 0   # M: images 0, 1 only


def test_logit_adjustment_and_balanced_softmax():
    log_pi = losses.log_prior([90, 10])
    z = torch.zeros(1, 2)
    assert losses.logit_adjust(z, log_pi, 1.0).argmax() == 1   # LA favours the rare class
    t = torch.tensor([0])
    assert torch.isclose(losses.la_loss(z, t, log_pi, 1.0), torch.tensor(-np.log(0.9), dtype=torch.float32), atol=1e-5)


# ----------------------------------------------------------------------------- features
def test_feature_store_roundtrip_and_missing_chunk(tmp_path):
    root = str(tmp_path)
    X = torch.randn(25, 4).half()
    y = torch.arange(25) % 3
    write_manifest(root, 'train', 'enc', {'num_images': 25, 'num_chunks': 3, 'chunk_size': 10, 'views': 1, 'txt_sha1': 'x'})
    for i in range(3):
        s, e = i * 10, min(25, (i + 1) * 10)
        save_chunk(root, 'train', 'enc', i, {'feats': X[s:e].unsqueeze(0), 'labels': y[s:e], 'start': s, 'end': e})
    f, lab = load_split(root, 'train', 'enc')
    assert f.shape == (1, 25, 4) and torch.equal(lab, y) and torch.equal(f[0], X)
    os.remove(os.path.join(root, 'train', 'enc', 'chunk_00001.pt'))
    with pytest.raises(RuntimeError):
        load_split(root, 'train', 'enc')


def test_extractor_resumes(tmp_path):
    """Kill-and-resume: deleting a chunk and re-running recomputes only that chunk."""
    from PIL import Image
    from tools import extract_features as X
    from networks.backbones import build_backbone
    img_root = tmp_path / 'imgs'
    img_root.mkdir()
    lines = []
    rng = np.random.default_rng(0)
    for i in range(9):
        Image.fromarray(rng.integers(0, 255, (40, 30, 3), dtype=np.uint8)).save(img_root / '{}.png'.format(i))
        lines.append('{}.png {}'.format(i, i % 3))
    txt = tmp_path / 'split.txt'
    txt.write_text('\n'.join(lines) + '\n')

    class A:
        root, out, chunk, batch, workers, views, size, resize, seed, limit = (
            str(img_root), str(tmp_path / 'feats'), 4, 3, 0, 1, 32, 36, 0, 0)
    enc = build_backbone('debug_tiny')
    encoders = {'debug_tiny': (enc, torch.tensor(enc.mean).view(1, 3, 1, 1), torch.tensor(enc.std).view(1, 3, 1, 1))}
    X.extract_split(A, 'train', str(txt), encoders, torch.device('cpu'))
    f1, l1 = load_split(A.out, 'train', 'debug_tiny')
    os.remove(os.path.join(A.out, 'train', 'debug_tiny', 'chunk_00001.pt'))
    X.extract_split(A, 'train', str(txt), encoders, torch.device('cpu'))
    f2, l2 = load_split(A.out, 'train', 'debug_tiny')
    assert f1.shape == (1, 9, 32) and torch.equal(f1, f2) and torch.equal(l1, l2)
    assert l1.tolist() == [i % 3 for i in range(9)]           # original labels are stored


# ----------------------------------------------------------------------------- splits
def test_h22_per_class_split_examples():
    from tools.make_h22_clean import per_class_split
    assert per_class_split(5) == (3, 1, 1)
    assert per_class_split(3) == (1, 1, 1)
    assert per_class_split(79) == (63, 8, 8)
    assert per_class_split(80) == (64, 8, 8)
    assert per_class_split(2) == (2, 0, 0)
    assert per_class_split(15) == (11, 2, 2)                  # floor(1.5 + 0.5) = 2, not round-half-even


def test_val_parts_are_stratified():
    from tools.make_val_parts import assign_parts
    labels = np.repeat(np.arange(200), 10)
    part = assign_parts(labels, [40, 30, 30], seed=0)
    shares = np.bincount(part, minlength=3) / len(part)
    assert np.allclose(shares, [0.4, 0.3, 0.3], atol=0.02)
    per_class = np.array([np.bincount(part[labels == c], minlength=3) for c in range(200)])
    assert per_class.min() >= 2                                # 10 images spread over all three parts


# ----------------------------------------------------------------------------- taxonomy, C-F
def test_taxonomy_shuffle_keeps_genus_sizes():
    tax = Taxonomy(np.array([0, 0, 1, 1, 1, 2]), np.array([0, 0, 1, 1, 1, 1]), ['a', 'b', 'c'], ['F', 'G'])
    sh = tax.shuffled(seed=1)
    assert sorted(np.bincount(sh.genus_of)) == sorted(np.bincount(tax.genus_of))


def test_cf_buckets():
    genus_of = np.array([0, 0, 1, 2, 3])       # classes 0..4; tail = classes 0, 1 (few_end = 2)
    family_of = np.array([0, 0, 0, 1, -1])
    y = np.array([1, 2, 3, 4, 0, 1])
    pred = np.array([0, 0, 0, 0, 0, 1])          # four tail FPs and one correct
    fp, b = cf.tail_fp_buckets(y, pred, 2, genus_of, family_of)
    assert list(fp) == [0, 1, 2, 3]
    assert list(b) == [0, 1, 2, 3]               # same genus, same family, different family, unknown
    s = cf.shares(b)
    assert s['cross_genus'] == 0.75


# ----------------------------------------------------------------------------- conformal
def test_cp3_union_bound_coverage():
    rng = np.random.default_rng(0)
    C, G, n = 40, 8, 6000
    genus_of = np.arange(C) % G
    counts = rng.integers(1, 100, C)
    y = rng.choice(C, n, p=counts / counts.sum())
    logits = rng.normal(0, 1, (n, C))
    logits[np.arange(n), y] += 2.0
    logits[:, :] += 1.0 * (genus_of[None, :] == genus_of[y][:, None])
    cp = conformal.GenusFallbackCP(counts, genus_of)
    cp.fit(logits[:3000], y[:3000], alpha=0.1)
    for proc in ('CP-1', 'CP-2', 'CP-3'):
        m = cp.metrics(cp.predict(logits[3000:], proc, K=3), y[3000:])
        assert m['coverage'] >= 0.87, (proc, m)               # 1 - alpha minus sampling slack
    m3 = cp.metrics(cp.predict(logits[3000:], 'CP-3', K=3), y[3000:])
    assert 0 < m3['fallback_rate'] < 1


# ----------------------------------------------------------------------------- integration
@pytest.fixture(scope='module')
def synth(tmp_path_factory):
    root = str(tmp_path_factory.mktemp('synth'))
    return make_dataset(root)


def test_every_baseline_trains(synth):
    conf, _, _, _ = synth
    data = harness.Data(conf, device=torch.device('cpu'))
    assert np.all(np.diff(data.index.counts) >= 0)
    for method in harness.SPACES:
        cfg = harness.sample_config(method, np.random.default_rng(0), base={'epochs': 4, 'eval_every': 2})
        cfg.update(lr=0.05, weight_decay=1e-4, batch_size=256)
        model, info = harness.train_one(cfg, data, seed=0)
        assert info['best_epoch'] > 0, method
        logits, _ = T.predict(model, data.fs['test'], info['cfg'], data.log_pi)
        m = metrics.evaluate(logits, data.fs['test'].y.cpu().numpy(), data.groups)
        assert m['top1'] > 1.0 / conf['num_classes'] * 3, (method, m['top1'])   # well above chance


def test_harness_end_to_end(synth, tmp_path):
    conf, conf_path, _, _ = synth
    results, preds = str(tmp_path / 'all_runs.csv'), str(tmp_path / 'preds')
    out = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'run_baselines.py'), '--config', conf_path,
                          '--methods', 'la,ltic3@gamma=0', '--trials', '2', '--seeds', '0,1',
                          '--results', results, '--preds', preds], cwd=str(tmp_path), capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-3000:]
    import csv
    rows = list(csv.DictReader(open(results)))
    assert {r['stage'] for r in rows} == {'trial', 'vselect', 'final'}
    assert sum(r['stage'] == 'trial' for r in rows) == 4
    assert all(json.loads(r['hparams'])['gamma'] == 0 for r in rows if r['method'] == 'ltic3@gamma=0')
    la_vselect = os.path.join(preds, 'synth_synth_la_vselect_seed0.npz')
    la_test = os.path.join(preds, 'synth_synth_la_test_seed0.npz')
    la_vcal = os.path.join(preds, 'synth_synth_la_vcal_seed0.npz')
    for p in (la_vselect, la_test, la_vcal):
        assert os.path.isfile(p), os.listdir(preds)

    gate = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'week1_diagnostics.py'), '--config', conf_path,
                           '--la_preds', la_vselect, '--thresholds', os.path.join(ROOT, 'gates', 'week1_thresholds.json'),
                           '--out', str(tmp_path / 'gate.json'), '--genus_trials', '1', '--n_perm', '50',
                           '--n_boot', '100', '--cov_k', '3'],
                          cwd=str(tmp_path), capture_output=True, text=True)
    assert gate.returncode == 0, gate.stderr[-3000:]
    g = json.load(open(tmp_path / 'gate.json'))
    assert set(g['checks']) == {'C1_donor_median_ge', 'C1a_cross_genus_share_ge', 'C1b_genus_correct_ge', 'C2_p_value_lt'}

    rep = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'cf_report.py'), '--preds', la_test,
                          '--taxonomy', conf['taxonomy'], '--out', str(tmp_path / 'cf'), '--n_boot', '100'],
                         cwd=str(tmp_path), capture_output=True, text=True)
    assert rep.returncode == 0, rep.stderr[-3000:]
    assert os.path.isfile(tmp_path / 'cf' / 'synth_synth_la_test_seed0_cf_tau.csv')

    cpo = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'conformal_eval.py'), '--cal', la_vcal,
                          '--test', la_test, '--taxonomy', conf['taxonomy'], '--out', str(tmp_path / 'cp.csv')],
                         cwd=ROOT, capture_output=True, text=True)
    assert cpo.returncode == 0, cpo.stderr[-3000:]


def test_diagnostics_find_genus_structure(synth):
    conf, _, counts, _ = synth
    data = harness.Data(conf, device=torch.device('cpu'), splits=('train',))
    tr = data.fs['train']
    r = diagnostics.covariance_check(tr.x[0], tr.y, data.index.counts, data.taxonomy.genus_of, k=2,
                                     min_images=20, n_perm=200, device='cpu')
    assert r['same_genus_mean_sim'] > r['random_pair_mean_sim'] and r['p_value'] < 0.05, r
    d = diagnostics.donor_pooling(data.index.counts, data.taxonomy.genus_of, data.groups.few_end)
    assert d['tail_species'] == data.groups.few_end and d['median_ratio'] > 0


def test_legacy_training_script_smoke(tmp_path):
    """CLIP_VIT_LONGTAIL.py end to end on tiny images with the debug encoder: C0 run + evaluate."""
    from PIL import Image
    rng = np.random.default_rng(0)
    counts = [2, 3, 25, 30, 120, 130]                         # few: 2 classes, medium: 2, many: 2
    lines_tr, lines_va = [], []
    (tmp_path / 'img').mkdir()
    k = 0
    for c, n in enumerate(counts):
        for j in range(n + 2):
            arr = np.clip(rng.normal(40 * c, 20, (24, 24, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(tmp_path / 'img' / '{}.png'.format(k))
            (lines_tr if j < n else lines_va).append('img/{}.png {}'.format(k, c))
            k += 1
    (tmp_path / 'tr.txt').write_text('\n'.join(lines_tr) + '\n')
    (tmp_path / 'va.txt').write_text('\n'.join(lines_va) + '\n')
    common = [sys.executable, os.path.join(ROOT, 'CLIP_VIT_LONGTAIL.py'), '-dataset', 'plantnetDataset',
              '--data_path', str(tmp_path), '--train_txt', str(tmp_path / 'tr.txt'), '--val_txt', str(tmp_path / 'va.txt'),
              '--root_path', str(tmp_path / 'runs'), '--mark', 'smoke', '--backbone', 'debug_tiny', '--num_classes', '6',
              '--few_thr', '20', '--many_thr', '100', '-b', '16', '--num_works', '0', '--cache_features',
              '--cache_dir', str(tmp_path / 'cache')]
    tr = subprocess.run(common + ['--epochs', '2'], cwd=ROOT, capture_output=True, text=True)
    assert tr.returncode == 0, tr.stderr[-3000:]
    assert os.path.isfile(tmp_path / 'runs' / 'plantnetDataset' / 'smoke' / 'model_best.pth.tar')
    ev = subprocess.run(common + ['--evaluate'], cwd=ROOT, capture_output=True, text=True)
    assert ev.returncode == 0 and 'MacroF1' in ev.stdout, ev.stderr[-3000:]
    bad = subprocess.run(common + ['--evaluate', '--mark', 'never_trained'], cwd=ROOT,
                         capture_output=True, text=True)
    assert bad.returncode != 0 and 'nothing to evaluate' in (bad.stdout + bad.stderr)
