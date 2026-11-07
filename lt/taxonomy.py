"""Species -> genus -> family tables (taxonomy/<dataset>.csv).

CSV columns: label (original split-file label), source_id, species, genus, family.
Arrays returned here are indexed by OUTPUT index (rarest class first).
"""
import csv

import numpy as np


class Taxonomy(object):
    def __init__(self, genus_of, family_of, genus_names, family_names, species_names=None):
        self.genus_of = np.asarray(genus_of, dtype=np.int64)       # [C] genus id per output class
        self.family_of = np.asarray(family_of, dtype=np.int64)     # [C] family id (-1 = unknown)
        self.genus_names = list(genus_names)
        self.family_names = list(family_names)
        self.species_names = species_names
        self.num_genera = len(self.genus_names)
        self.num_families = len(self.family_names)

    @classmethod
    def from_csv(cls, path, class_index):
        """class_index: lt.classes.ClassIndex (maps original labels to output indices)."""
        rows = {}
        with open(path, newline='') as f:
            for r in csv.DictReader(f):
                rows[int(r['label'])] = r
        C = class_index.num_classes
        missing = [i for i in range(C) if i not in rows]
        if missing:
            raise ValueError('{}: {} labels without a taxonomy row (first {})'.format(path, len(missing), missing[:5]))
        genus = [None] * C
        family = [None] * C
        species = [None] * C
        for orig, r in rows.items():
            if orig >= C:
                continue
            o = int(class_index.class_map[orig])
            genus[o] = (r.get('genus') or '').strip() or 'unknown_genus_{}'.format(orig)
            family[o] = (r.get('family') or '').strip()
            species[o] = r.get('species')
        genus_names = sorted(set(genus))
        fam_names = sorted({f for f in family if f})
        g_id = {g: i for i, g in enumerate(genus_names)}
        f_id = {f: i for i, f in enumerate(fam_names)}
        return cls([g_id[g] for g in genus], [f_id[f] if f else -1 for f in family],
                   genus_names, fam_names, species)

    def shuffled(self, seed):
        """A9 control: random genus assignment with the same genus sizes; families follow the
        shuffled genera (each shuffled genus keeps its original family)."""
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(self.genus_of))
        genus_of = self.genus_of[perm]
        fam_of_genus = np.full(self.num_genera, -1)
        for c, g in enumerate(self.genus_of):
            fam_of_genus[g] = self.family_of[c]
        return Taxonomy(genus_of, fam_of_genus[genus_of], self.genus_names, self.family_names, self.species_names)

    def genus_counts(self, counts):
        return np.bincount(self.genus_of, weights=counts, minlength=self.num_genera)

    def species_per_genus(self):
        return np.bincount(self.genus_of, minlength=self.num_genera)
