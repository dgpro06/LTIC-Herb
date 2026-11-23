"""tools/feature_geometry.py: the donor subspace equals the top eigenvectors of C2's pooled donor covariance."""
import os
import tempfile
import unittest

import numpy as np
import torch

from lt.calibration import select_donors
from lt.classes import ClassIndex, ShotGroups
from lt.features import load_split
from lt.taxonomy import Taxonomy
from tests.synthetic import make_dataset
from tools import feature_geometry as G


class TestGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.conf, _, _, _ = make_dataset(cls.tmp.name, num_classes=120, num_genera=20, dim=12)
        x, y0 = load_split(cls.tmp.name, 'train', 'synth', views=1)
        cls.x = x[0].float()
        idx = ClassIndex(y0.numpy(), cls.conf['num_classes'])
        cls.y = idx.remap(y0.numpy())
        cls.counts = idx.counts
        cls.few_end = ShotGroups(idx.counts, 20, 100).few_end
        cls.tax = Taxonomy.from_csv(cls.conf['taxonomy'], idx)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def direct_share(self, c, donors, r):
        """C2's definition: sum_d w_d Cov_d (w = counts, Cov unbiased); share of S_c in its top-r subspace."""
        x, y = self.x.double(), torch.as_tensor(self.y)
        M = 0
        for d in donors:
            xd = x[y == d]
            M = M + len(xd) * torch.cov(xd.T)
        v = torch.linalg.eigh(M)[1].flip(1)[:, :r]
        xc = x[y == c] - x[y == c].mean(0)
        S = xc.T @ xc
        return float((v * (S @ v)).sum() / torch.trace(S))

    def test_tail_transfer_matches_c2_donors(self):
        nmin, ranks = 10, [3, 6]
        sc = G.Scatter(self.x, self.y, len(self.counts))
        g = torch.zeros(12, 12, dtype=torch.float64)
        for c in np.nonzero(self.counts >= nmin)[0]:
            g += sc.T(c)
        tail = [c for c in range(self.few_end) if self.counts[c] >= 3]
        out, _ = G.transfer(sc, self.counts, self.tax, self.few_end, nmin, [], tail, ranks, G.top_evecs(g), {})
        sel = select_donors(self.counts, self.tax.genus_of, self.tax.family_of, self.few_end, nmin)
        checked = 0
        for c in tail[:25]:
            donors, level = sel[c]
            if donors is None:
                continue
            self.assertEqual(out[c]['donor_level_eval'], level)
            for r in ranks:
                self.assertAlmostEqual(out[c]['transfer_r{}'.format(r)], self.direct_share(c, donors, r), places=6)
            checked += 1
        self.assertGreater(checked, 5)

    def test_shuffled_taxonomy_has_less_structure(self):
        cent = torch.zeros(len(self.counts), 12)
        xn = torch.nn.functional.normalize(self.x, dim=1)
        cent.index_add_(0, torch.as_tensor(self.y), xn)
        cent = torch.nn.functional.normalize(cent, dim=1)
        has = self.counts > 0
        real = G.centroid_margins(cent, has, self.tax.genus_of, self.tax.family_of)[0]
        sh = self.tax.shuffled(0)
        shuf = G.centroid_margins(cent, has, sh.genus_of, sh.family_of)[0]
        self.assertGreater(np.nanmedian(real), np.nanmedian(shuf))
        kg, _ = G.knn_purity(xn, self.y, np.arange(self.few_end), self.tax.genus_of, self.tax.family_of, 5)
        kgs, _ = G.knn_purity(xn, self.y, np.arange(self.few_end), sh.genus_of, sh.family_of, 5)
        self.assertGreater(np.nanmean(kg), np.nanmean(kgs))
        self.assertTrue(np.all((kg[~np.isnan(kg)] >= 0) & (kg[~np.isnan(kg)] <= 1)))


if __name__ == '__main__':
    unittest.main()
