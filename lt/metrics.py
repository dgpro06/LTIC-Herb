"""Metrics on output-index predictions (labels sorted rarest first).

Headline metrics (plan §12): macro-F1 and mean per-class accuracy. Weighted precision / recall
are kept only for the appendix and use zero_division=0 (the IJCNN code used 1, which counts a
never-predicted class as precision 1).
"""
import math

import numpy as np
import torch


def _np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def confusion_counts(y_true, y_pred, num_classes):
    """Per-class true positives, support (true count) and predicted count."""
    y_true, y_pred = _np(y_true).astype(np.int64), _np(y_pred).astype(np.int64)
    support = np.bincount(y_true, minlength=num_classes)
    predicted = np.bincount(y_pred, minlength=num_classes)
    tp = np.bincount(y_true[y_true == y_pred], minlength=num_classes)
    return tp, support, predicted


def macro_f1(y_true, y_pred, num_classes, labels='union'):
    """Macro-F1. labels='union' matches sklearn's f1_score(average='macro') (classes seen in
    y_true or y_pred, the Kaggle H22 metric); labels='true' averages over classes with support."""
    tp, support, predicted = confusion_counts(y_true, y_pred, num_classes)
    denom = support + predicted
    f1 = np.where(denom > 0, 2.0 * tp / np.maximum(denom, 1), 0.0)
    keep = (support > 0) | (predicted > 0) if labels == 'union' else support > 0
    return float(f1[keep].mean()) if keep.any() else float('nan')


def per_class_accuracy(y_true, y_pred, num_classes):
    """Recall per class; NaN for classes without support."""
    tp, support, _ = confusion_counts(y_true, y_pred, num_classes)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(support > 0, tp / np.maximum(support, 1), np.nan)


def topk_accuracy(logits, y_true, k=5):
    logits = torch.as_tensor(logits)
    y_true = torch.as_tensor(y_true)
    k = min(k, logits.size(1))
    return float((logits.topk(k, dim=1).indices == y_true.view(-1, 1)).any(1).float().mean())


def weighted_precision_recall(y_true, y_pred, num_classes):
    """Appendix only. zero_division=0."""
    tp, support, predicted = confusion_counts(y_true, y_pred, num_classes)
    w = support / max(support.sum(), 1)
    prec = np.where(predicted > 0, tp / np.maximum(predicted, 1), 0.0)
    rec = np.where(support > 0, tp / np.maximum(support, 1), 0.0)
    return float((w * prec).sum()), float((w * rec).sum())


def evaluate(logits, y_true, groups, prefix=''):
    """All per-run metrics for one split. logits: [N, C] (any adjustment already applied)."""
    logits = torch.as_tensor(logits).float()
    y = _np(y_true).astype(np.int64)
    pred = logits.argmax(1).cpu().numpy()
    top5 = topk_accuracy(logits, y, 5)
    return evaluate_predictions(y, pred, logits.size(1), groups, top5, prefix)


def evaluate_predictions(y_true, pred, num_classes, groups, top5=float('nan'), prefix=''):
    """Same metrics from top-1 predictions (for splits too large to keep all logits)."""
    y, pred, C = _np(y_true).astype(np.int64), _np(pred).astype(np.int64), num_classes
    acc_c = per_class_accuracy(y, pred, C)
    g_of_class = groups.group_of(np.arange(C))
    g_of_img = groups.group_of(y)
    out = {
        'top1': float((pred == y).mean()),
        'top5': float(top5),
        'macro_f1': macro_f1(y, pred, C),
        'macro_acc': float(np.nanmean(acc_c)),
        'worst_class_acc': float(np.nanmin(acc_c)),
        'n': int(len(y)),
    }
    for gid, name in enumerate(('few', 'medium', 'many')):
        cls = (g_of_class == gid) & ~np.isnan(acc_c)
        img = g_of_img == gid
        out[name + '_acc'] = float(acc_c[cls].mean()) if cls.any() else float('nan')       # per-class mean
        out[name + '_acc_pooled'] = float((pred[img] == y[img]).mean()) if img.any() else float('nan')
        out[name + '_n_img'] = int(img.sum())
        out[name + '_n_cls'] = int(cls.sum())
    wp, wr = weighted_precision_recall(y, pred, C)
    out['weighted_precision_appendix'] = wp
    out['weighted_recall_appendix'] = wr
    return {prefix + k: v for k, v in out.items()}


# ---------------------------------------------------------------------------------------------
# Uncertainty (plan §5.2 item 8, §8.2, §10 item 9)
# ---------------------------------------------------------------------------------------------

def bootstrap_ci(stat_fn, n, n_boot=10000, seed=0, alpha=0.05):
    """Percentile CI for stat_fn(index_array) over image resamples."""
    rng = np.random.default_rng(seed)
    vals = np.array([stat_fn(rng.integers(0, n, n)) for _ in range(n_boot)])
    vals = vals[~np.isnan(vals)]
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def group_accuracy_ci(y_true, y_pred, class_ids, n_boot=2000, seed=0, alpha=0.05):
    """Per-class mean accuracy over class_ids with a two-stage bootstrap: resample test images
    within each class, then resample classes (plan §5.2 item 8)."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = _np(y_true), _np(y_pred)
    correct_by_class = [(y_pred[y_true == c] == c).astype(np.float64) for c in class_ids]
    correct_by_class = [c for c in correct_by_class if len(c)]
    if not correct_by_class:
        return float('nan'), (float('nan'), float('nan'))
    point = float(np.mean([c.mean() for c in correct_by_class]))
    k = len(correct_by_class)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.integers(0, k, k)
        stats[b] = np.mean([correct_by_class[i][rng.integers(0, len(correct_by_class[i]),
                                                              len(correct_by_class[i]))].mean()
                            for i in picked])
    return point, (float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2)))


def paired_bootstrap(metric_fn, y_true, pred_a, pred_b, n_boot=10000, seed=0, alpha=0.05):
    """Difference metric(a) - metric(b) on the same images, resampled jointly (plan §8.2).
    metric_fn(y_true, y_pred) -> float. Returns (delta, (lo, hi), excludes_zero)."""
    rng = np.random.default_rng(seed)
    y_true, pred_a, pred_b = _np(y_true), _np(pred_a), _np(pred_b)
    n = len(y_true)
    delta = metric_fn(y_true, pred_a) - metric_fn(y_true, pred_b)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = metric_fn(y_true[idx], pred_a[idx]) - metric_fn(y_true[idx], pred_b[idx])
    lo, hi = float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))
    return float(delta), (lo, hi), bool(lo > 0 or hi < 0)


def mcnemar(y_true, pred_a, pred_b):
    """Exact two-sided McNemar test on top-1 correctness. Returns (b, c, p): b = a right & b wrong."""
    y_true, pred_a, pred_b = _np(y_true), _np(pred_a), _np(pred_b)
    ra, rb = pred_a == y_true, pred_b == y_true
    b, c = int((ra & ~rb).sum()), int((~ra & rb).sum())
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    # exact binomial tail, two-sided; log-space for large n
    log_terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2)
                 for i in range(k + 1)]
    m = max(log_terms)
    p = 2 * math.exp(m) * sum(math.exp(t - m) for t in log_terms)
    return b, c, min(1.0, p)
