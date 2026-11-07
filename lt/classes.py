"""Class order and shot groups.

LTIC's branch masks are label-index thresholds, so labels must be sorted by train count,
rarest first (plan §3.1, R1). Split files and feature chunks keep the ORIGINAL split-file
labels; this module builds the remap at load time with a stable sort, so every machine
agrees on the order of tied classes.
"""
import json

import numpy as np


class ClassIndex(object):
    """Maps original split-file labels to output indices sorted by train count (ascending).

    class_map[original_label] = output index; inverse[output_index] = original label.
    counts[i] = train images of output class i (non-decreasing).
    """

    def __init__(self, train_labels, num_classes):
        train_labels = np.asarray(train_labels, dtype=np.int64)
        if train_labels.size and (train_labels.min() < 0 or train_labels.max() >= num_classes):
            raise ValueError('train labels outside [0, {})'.format(num_classes))
        orig_counts = np.bincount(train_labels, minlength=num_classes)
        # kind='stable': ties keep their original-label order on every machine (plan §3.1)
        self.inverse = np.argsort(orig_counts, kind='stable')
        self.class_map = np.empty(num_classes, dtype=np.int64)
        self.class_map[self.inverse] = np.arange(num_classes)
        self.counts = orig_counts[self.inverse]
        self.num_classes = num_classes
        assert np.all(np.diff(self.counts) >= 0), 'remapped counts must be non-decreasing'

    def remap(self, labels):
        return self.class_map[np.asarray(labels, dtype=np.int64)]

    @property
    def empty_classes(self):
        return int((self.counts == 0).sum())

    def prior(self):
        """Train label prior pi in output order (for logit adjustment)."""
        return self.counts / max(self.counts.sum(), 1)

    def check(self, class_map):
        """Raise if a checkpoint's class_map differs from this one (resume across machines)."""
        if class_map is not None and not np.array_equal(np.asarray(class_map), self.class_map):
            raise RuntimeError('class_map in the checkpoint differs from the one built from the train file; '
                               'the class order changed (different train file or unstable sort)')

    def to_json(self):
        return {'class_map': self.class_map.tolist(), 'counts': self.counts.tolist()}


class ShotGroups(object):
    """few = [0, few_end), medium = [few_end, many_start), many = [many_start, C) in output order.

    few: fewer than few_thr train images; many: more than many_thr (plan §5.2).
    The same groups define LTIC's masks: H = all classes, M = label < many_start, T = label < few_end.
    """

    def __init__(self, counts, few_thr, many_thr):
        if few_thr is None or many_thr is None:
            raise ValueError('few_thr and many_thr must be passed explicitly (plan §3.1)')
        counts = np.asarray(counts)
        assert np.all(np.diff(counts) >= 0), 'counts must be sorted ascending (rarest first)'
        self.few_thr, self.many_thr = few_thr, many_thr
        self.few_end = int((counts < few_thr).sum())
        self.many_start = int((counts <= many_thr).sum())
        self.num_classes = len(counts)
        # R1 asserts: the label-index masks equal the count-defined groups
        few = counts < few_thr
        many = counts > many_thr
        assert few[:self.few_end].all() and not few[self.few_end:].any(), 'few-shot mask mismatch'
        assert many[self.many_start:].all() and not many[:self.many_start].any(), 'many-shot mask mismatch'
        assert self.few_end <= self.many_start
        if self.many_start == self.num_classes:
            raise ValueError('many-shot group is empty (largest class has {} images, many_thr={}): '
                             'the M branch would equal H. Lower many_thr (plan §3.1: H22 uses 50)'
                             .format(int(counts.max()), many_thr))

    @classmethod
    def from_bounds(cls, num_classes, few_end, many_start):
        """Explicit label-index bounds (only to reproduce the IJCNN configuration; no R1 asserts)."""
        g = cls.__new__(cls)
        g.few_thr = g.many_thr = None
        g.few_end, g.many_start, g.num_classes = int(few_end), int(many_start), int(num_classes)
        return g

    def group_of(self, idx):
        """0 = few, 1 = medium, 2 = many, for output indices."""
        idx = np.asarray(idx)
        return (idx >= self.few_end).astype(np.int64) + (idx >= self.many_start).astype(np.int64)

    def masks(self):
        """Boolean class masks (H, M, T) in output order."""
        c = np.arange(self.num_classes)
        return np.ones(self.num_classes, bool), c < self.many_start, c < self.few_end

    def sizes(self):
        return {'few': self.few_end, 'medium': self.many_start - self.few_end,
                'many': self.num_classes - self.many_start}

    def describe(self):
        s = self.sizes()
        return ('few [0, {}) {} classes | medium [{}, {}) {} | many [{}, {}) {} | thresholds <{} / >{}'
                .format(self.few_end, s['few'], self.few_end, self.many_start, s['medium'],
                        self.many_start, self.num_classes, s['many'], self.few_thr, self.many_thr))


def read_split_file(txt):
    """Return (relative paths, original labels) from a 'path label' split file."""
    paths, labels = [], []
    with open(txt) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            paths.append(parts[0])
            labels.append(int(parts[1]))
    return paths, np.asarray(labels, dtype=np.int64)


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1)
