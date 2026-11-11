"""Write Pl@ntNet-300K split files in the "relpath label" format read by datasets/herbarium.py.

Download plantnet_300K.zip from https://zenodo.org/records/5645731, unzip it, then:

    python tools/make_plantnet_splits.py --root /path/to/plantnet_300K

--root may also be a parent folder (e.g. /kaggle/input): the dataset folder is searched for below it.
Both layouts are accepted: images/{train,val,test}/<species_id>/ and images_{train,val,test}/<species_id>/.
Paths in the split files are relative to --root, so train with --data_path set to the same folder.
Writes plantnet_{train,val,test}.txt and plantnet_label_map.json into --out (default: --root).
"""
import argparse
import json
import os

import numpy as np

IMG_EXTS = ('.jpg', '.jpeg', '.png')
SPLITS = ('train', 'val', 'test')


def split_dir(dataset_dir, split):
    for candidate in (os.path.join(dataset_dir, 'images', split), os.path.join(dataset_dir, 'images_' + split)):
        if os.path.isdir(candidate):
            return candidate
    return None


def find_dataset(root, max_depth=5):
    """Breadth-first search below root for the folder holding the train images."""
    level = [root]
    for _ in range(max_depth + 1):
        for d in level:
            if split_dir(d, 'train'):
                return d
        level = [os.path.join(d, c) for d in level for c in sorted(os.listdir(d))
                 if not c.startswith(('images', '.')) and os.path.isdir(os.path.join(d, c))]
    return None


NAMES_FILE = 'plantnet300K_species_id_2_name.json'


def find_names_file(root, dataset_dir, max_depth=4):
    """The species-name file next to the images, or anywhere below root (image folders are skipped)."""
    for d in (dataset_dir, os.path.dirname(dataset_dir)):
        if os.path.isfile(os.path.join(d, NAMES_FILE)):
            return os.path.join(d, NAMES_FILE)
    level = [root]
    for _ in range(max_depth + 1):
        for d in level:
            if os.path.isfile(os.path.join(d, NAMES_FILE)):
                return os.path.join(d, NAMES_FILE)
        level = [os.path.join(d, c) for d in level for c in sorted(os.listdir(d))
                 if not c.startswith(('images', '.')) and os.path.isdir(os.path.join(d, c))]
    return None


def list_split(path):
    """{species_id: [image file names]} for <path>/<species_id>/*."""
    files = {}
    for species_id in sorted(os.listdir(path)):
        class_dir = os.path.join(path, species_id)
        if os.path.isdir(class_dir):
            files[species_id] = sorted(f for f in os.listdir(class_dir) if f.lower().endswith(IMG_EXTS))
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', required=True, help='plantnet_300K folder, or a folder above it')
    parser.add_argument('--out', default=None, help='where to write the split files (default: --root)')
    parser.add_argument('--names', default=None, help='plantnet300K_species_id_2_name.json (default: searched below --root)')
    parser.add_argument('--many_thr', default=100, type=int, help='many-shot: more train images than this')
    parser.add_argument('--few_thr', default=20, type=int, help='few-shot: fewer train images than this')
    args = parser.parse_args()
    out_dir = args.out or args.root
    os.makedirs(out_dir, exist_ok=True)

    dataset_dir = find_dataset(args.root)
    if dataset_dir is None:
        raise SystemExit("no images/train or images_train folder found below '{}'".format(args.root))
    print('dataset folder: {}'.format(dataset_dir))

    split_files = {split: list_split(split_dir(dataset_dir, split)) for split in SPLITS if split_dir(dataset_dir, split)}
    train = split_files['train']
    species = sorted(train)
    label_of = {species_id: label for label, species_id in enumerate(species)}

    for split in SPLITS:
        if split not in split_files:
            print('skipping {}: no images/{} or images_{} folder'.format(split, split, split))
            continue
        files = split_files[split]
        unknown = sorted(set(files) - set(label_of))
        if unknown:
            print('warning: {} {} classes have no train images and are skipped'.format(len(unknown), split))
        prefix = os.path.relpath(split_dir(dataset_dir, split), args.root).replace(os.sep, '/')
        out = os.path.join(out_dir, 'plantnet_{}.txt'.format(split))
        n = 0
        with open(out, 'w') as f:
            for species_id in sorted(set(files) & set(label_of)):
                for name in files[species_id]:
                    f.write('{}/{}/{} {}\n'.format(prefix, species_id, name, label_of[species_id]))
                    n += 1
        print('{}: {} images -> {}'.format(split, n, out))

    names_file = args.names or find_names_file(args.root, dataset_dir)
    names = json.load(open(names_file)) if names_file and os.path.isfile(names_file) else {}
    print('species names: {}'.format(names_file if names else 'NOT FOUND (taxonomy/plantnet.csv needs them; '
                                                            'pass --names path/to/' + NAMES_FILE + ')'))
    label_map = {label: {'species_id': s, 'name': names.get(s)} for label, s in enumerate(species)}
    with open(os.path.join(out_dir, 'plantnet_label_map.json'), 'w') as f:
        json.dump(label_map, f, indent=1)

    counts = np.array([len(train[s]) for s in species])
    print('classes: {}  train images per class: min {}  median {}  max {}'.format(
        len(counts), counts.min(), int(np.median(counts)), counts.max()))
    print('many-shot (>{}): {}  medium-shot: {}  few-shot (<{}): {}'.format(
        args.many_thr, (counts > args.many_thr).sum(),
        ((counts >= args.few_thr) & (counts <= args.many_thr)).sum(),
        args.few_thr, (counts < args.few_thr).sum()))


if __name__ == '__main__':
    main()
