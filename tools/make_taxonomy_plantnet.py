"""taxonomy/plantnet.csv for Pl@ntNet-300K: genus from the species name, family from GBIF.

    python tools/make_taxonomy_plantnet.py --label_map splits/plantnet_label_map.json

plantnet_label_map.json is written by tools/make_plantnet_splits.py. Families are looked up with the
public GBIF species-match API (https://api.gbif.org/v1/species/match) once per genus, sends only
genus names, and caches answers in taxonomy/gbif_cache.json. --offline skips the lookup (family empty).
Expected (Garcin et al. 2021): 1,081 species in 303 genera.
"""
import argparse
import csv
import json
import os
import time
import urllib.parse
import urllib.request

GBIF = 'https://api.gbif.org/v1/species/match?rank=GENUS&kingdom=Plantae&name='


def genus_from_name(name):
    tokens = [t for t in name.replace('×', ' ').split() if t.lower() != 'x']
    return tokens[0] if tokens else ''


def gbif_family(genus, cache, sleep=0.1):
    if genus in cache:
        return cache[genus]
    req = urllib.request.Request(GBIF + urllib.parse.quote(genus), headers={'User-Agent': 'ltic-herb-taxonomy'})
    with urllib.request.urlopen(req, timeout=30) as r:
        ans = json.load(r)
    cache[genus] = {'family': ans.get('family'), 'matchType': ans.get('matchType'),
                    'status': ans.get('status'), 'rank': ans.get('rank')}
    time.sleep(sleep)
    return cache[genus]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--label_map', required=True)
    p.add_argument('--out', default='taxonomy/plantnet.csv')
    p.add_argument('--cache', default='taxonomy/gbif_cache.json')
    p.add_argument('--offline', action='store_true')
    args = p.parse_args()

    with open(args.label_map) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    missing_names = [k for k, v in label_map.items() if not v.get('name')]
    if missing_names:
        raise SystemExit('{} labels have no species name (plantnet300K_species_id_2_name.json not found when '
                         'the splits were made?)'.format(len(missing_names)))

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    cache = {}
    if os.path.isfile(args.cache):
        with open(args.cache) as f:
            cache = json.load(f)
    genera = sorted({genus_from_name(v['name']) for v in label_map.values()})
    if not args.offline:
        for i, g in enumerate(genera):
            gbif_family(g, cache)
            if i % 50 == 0:
                with open(args.cache, 'w') as f:
                    json.dump(cache, f, indent=1)
        with open(args.cache, 'w') as f:
            json.dump(cache, f, indent=1)

    unresolved = []
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['label', 'source_id', 'species', 'genus', 'family'])
        for label in sorted(label_map):
            v = label_map[label]
            g = genus_from_name(v['name'])
            fam = (cache.get(g) or {}).get('family') or ''
            if not fam:
                unresolved.append(g)
            w.writerow([label, v['species_id'], v['name'], g, fam])

    species_per_genus = {}
    for v in label_map.values():
        g = genus_from_name(v['name'])
        species_per_genus[g] = species_per_genus.get(g, 0) + 1
    fuzzy = sorted(g for g in genera if (cache.get(g) or {}).get('matchType') not in (None, 'EXACT'))
    print('{} species, {} genera ({} with one species), {} families'.format(
        len(label_map), len(genera), sum(1 for n in species_per_genus.values() if n == 1),
        len({(cache.get(g) or {}).get('family') for g in genera} - {None})))
    if unresolved:
        print('genera without a family: {}'.format(sorted(set(unresolved))))
    if fuzzy:
        print('check these non-exact GBIF matches by hand: {}'.format(fuzzy))
    print('-> {}'.format(args.out))


if __name__ == '__main__':
    main()
