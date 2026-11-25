"""taxonomy/plantnet.csv for Pl@ntNet-300K: genus from the species name, family from GBIF.

    python tools/make_taxonomy_plantnet.py --label_map splits/plantnet_label_map.json

plantnet_label_map.json is written by tools/make_plantnet_splits.py. Families are looked up with the
public GBIF species-match API (https://api.gbif.org/v1/species/match) once per genus, sends only
genus names, and caches answers in taxonomy/gbif_cache.json. Failed requests are retried; if GBIF stays
unreachable the family is left empty and the run continues. Genera GBIF cannot match are filled from
MANUAL_FAMILIES (APG IV); the CSV's family_source column says which (gbif / manual / none).
--offline skips the lookup.
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

# Genera that GBIF's match API returned without a family on Pl@ntNet-300K (2026-11-15 Kaggle run).
# Families follow APG IV; they fill gaps only and never override a GBIF answer.
MANUAL_FAMILIES = {
    'Arthraxon': 'Poaceae', 'Cordyla': 'Fabaceae', 'Cyrtanthus': 'Amaryllidaceae',
    'Elodea': 'Hydrocharitaceae', 'Fittonia': 'Acanthaceae', 'Hackelia': 'Boraginaceae',
    'Hepatica': 'Ranunculaceae', 'Kniphofia': 'Asphodelaceae', 'Leersia': 'Poaceae',
    'Mertensia': 'Boraginaceae', 'Pogonophora': 'Peraceae',
}


def genus_from_name(name):
    tokens = [t for t in name.replace('×', ' ').split() if t.lower() != 'x']
    return tokens[0] if tokens else ''


def gbif_family(genus, cache, sleep=0.1, retries=4, backoff=2.0):
    """GBIF answer for a genus, or None if GBIF stays unreachable (not cached, so a re-run retries)."""
    if genus in cache:
        return cache[genus]
    req = urllib.request.Request(GBIF + urllib.parse.quote(genus), headers={'User-Agent': 'ltic-herb-taxonomy'})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                ans = json.load(r)
            break
        except Exception as e:      # network errors, timeouts, HTTP 5xx, bad JSON
            if attempt == retries - 1:
                print('warning: GBIF lookup failed for {} ({}); family left empty'.format(genus, e))
                return None
            time.sleep(backoff * (attempt + 1))
    cache[genus] = {'family': ans.get('family'), 'matchType': ans.get('matchType'),
                    'status': ans.get('status'), 'rank': ans.get('rank')}
    time.sleep(sleep)
    return cache[genus]


def family_of(genus, cache):
    """(family, source): GBIF first, then MANUAL_FAMILIES, else ('', 'none')."""
    fam = (cache.get(genus) or {}).get('family')
    if fam:
        return fam, 'gbif'
    if genus in MANUAL_FAMILIES:
        return MANUAL_FAMILIES[genus], 'manual'
    return '', 'none'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--label_map', required=True)
    p.add_argument('--out', default='taxonomy/plantnet.csv')
    p.add_argument('--cache', default='taxonomy/gbif_cache.json')
    p.add_argument('--offline', action='store_true')
    args = p.parse_args()

    with open(args.label_map, encoding='utf-8') as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    missing_names = [k for k, v in label_map.items() if not v.get('name')]
    if missing_names:
        raise SystemExit('{} labels have no species name (plantnet300K_species_id_2_name.json not found when '
                         'the splits were made?)'.format(len(missing_names)))

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    cache = {}
    if os.path.isfile(args.cache):
        with open(args.cache, encoding='utf-8') as f:
            cache = json.load(f)
    genera = sorted({genus_from_name(v['name']) for v in label_map.values()})
    if not args.offline:
        for i, g in enumerate(genera):
            gbif_family(g, cache)
            if i % 50 == 0:
                with open(args.cache, 'w', encoding='utf-8') as f:
                    json.dump(cache, f, indent=1)
        with open(args.cache, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=1)

    unresolved = []
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['label', 'source_id', 'species', 'genus', 'family', 'family_source'])
        for label in sorted(label_map):
            v = label_map[label]
            g = genus_from_name(v['name'])
            fam, src = family_of(g, cache)
            if not fam:
                unresolved.append(g)
            w.writerow([label, v['species_id'], v['name'], g, fam, src])

    species_per_genus = {}
    for v in label_map.values():
        g = genus_from_name(v['name'])
        species_per_genus[g] = species_per_genus.get(g, 0) + 1
    manual = sorted(g for g in genera if family_of(g, cache)[1] == 'manual')
    print('{} species, {} genera ({} with one species), {} families ({} genera filled from MANUAL_FAMILIES)'.format(
        len(label_map), len(genera), sum(1 for n in species_per_genus.values() if n == 1),
        len({family_of(g, cache)[0] for g in genera} - {''}), len(manual)))
    if unresolved:
        print('genera without a family (add them to MANUAL_FAMILIES): {}'.format(sorted(set(unresolved))))
    print('-> {}'.format(args.out))


if __name__ == '__main__':
    main()
