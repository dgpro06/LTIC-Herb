"""Finding C-F (plan §4): where logit adjustment's tail false positives come from.

Tail false positive: a test image whose predicted class is a tail (few-shot) species different from
the true class. Each is put in one bucket:
    same_genus        true genus == predicted species' genus
    same_family       same family, different genus
    diff_family       different family
    unknown_family    different genus and a family is missing for one side
cross_genus = everything except same_genus (week-1 diagnostic (a), plan §8.1).
"""
import numpy as np

BUCKETS = ('same_genus', 'same_family', 'diff_family', 'unknown_family')


def predictions(logits, log_pi, tau):
    logits = np.asarray(logits, dtype=np.float32)
    return (logits - tau * np.asarray(log_pi, dtype=np.float32)[None, :]).argmax(1)


def tail_fp_buckets(y, pred, few_end, genus_of, family_of):
    """Returns (indices of tail FPs, bucket id per tail FP)."""
    y, pred = np.asarray(y), np.asarray(pred)
    fp = np.flatnonzero((pred < few_end) & (pred != y))
    gy, gp = genus_of[y[fp]], genus_of[pred[fp]]
    fy, fpm = family_of[y[fp]], family_of[pred[fp]]
    bucket = np.full(len(fp), 3)
    bucket[gy == gp] = 0
    known = (fy >= 0) & (fpm >= 0)
    bucket[(gy != gp) & known & (fy == fpm)] = 1
    bucket[(gy != gp) & known & (fy != fpm)] = 2
    return fp, bucket


def shares(bucket):
    n = len(bucket)
    out = {b: (float((bucket == i).sum()) / n if n else float('nan')) for i, b in enumerate(BUCKETS)}
    out['cross_genus'] = 1.0 - out['same_genus'] if n else float('nan')
    return out


def shares_with_ci(y, pred, few_end, genus_of, family_of, n_boot=2000, seed=0, alpha=0.05):
    """Point shares and percentile CIs from resampling test images (plan §4)."""
    y, pred = np.asarray(y), np.asarray(pred)
    fp, bucket = tail_fp_buckets(y, pred, few_end, genus_of, family_of)
    point = shares(bucket)
    # per-image bucket (-1 = not a tail FP) so resampling images resamples tail FPs correctly
    img_bucket = np.full(len(y), -1)
    img_bucket[fp] = bucket
    rng = np.random.default_rng(seed)
    keys = list(BUCKETS) + ['cross_genus']
    boot = {k: [] for k in keys}
    for _ in range(n_boot):
        b = img_bucket[rng.integers(0, len(y), len(y))]
        b = b[b >= 0]
        if not len(b):
            continue
        s = shares(b)
        for k in keys:
            boot[k].append(s[k])
    ci = {k: (float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))) if v else (np.nan, np.nan)
          for k, v in boot.items()}
    return {'n_images': int(len(y)), 'n_tail_fp': int(len(fp)), 'share': point, 'ci95': ci}


def tau_curve(logits, y, log_pi, few_end, genus_of, family_of, taus):
    rows = []
    for t in taus:
        pred = predictions(logits, log_pi, t)
        _, bucket = tail_fp_buckets(y, pred, few_end, genus_of, family_of)
        s = shares(bucket)
        rows.append(dict(tau=float(t), n_tail_fp=int(len(bucket)), top1=float((pred == y).mean()), **s))
    return rows


def count_curve(y, pred, few_end, genus_of, family_of, counts, edges=(1, 3, 6, 11, 20)):
    """Cross-genus share vs the predicted tail class's train count, binned by `edges` ([lo, hi))."""
    fp, bucket = tail_fp_buckets(y, pred, few_end, genus_of, family_of)
    c = np.asarray(counts)[np.asarray(pred)[fp]]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (c >= lo) & (c < hi)
        s = shares(bucket[sel])
        rows.append(dict(count_lo=int(lo), count_hi=int(hi), n_tail_fp=int(sel.sum()), **s))
    return rows
