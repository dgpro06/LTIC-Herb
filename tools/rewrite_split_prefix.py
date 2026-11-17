"""Rewrite the image-path prefix of split files, e.g. to use Kaggle-made splits on another machine.

The committed Pl@ntNet split files carry Kaggle's folder layout:
    datasets/gabrielneto583/plantnet300k/plantnet_300K/images/train/1355868/xxx.jpg 0
On the 4070, with the dataset in D:/data/plantnet_300K:
    python tools/rewrite_split_prefix.py --old datasets/gabrielneto583/plantnet300k/plantnet_300K/ --new "" \
        --out splits_local splits/plantnet_*.txt
then pass --root D:/data/plantnet_300K and the files in splits_local/ to tools/extract_features.py.

Only the prefix changes: image order and labels stay the same. Every line must start with --old (checked).
Features already extracted from the original files can still be loaded (loading does not re-read split files).
"""
import argparse
import glob
import os


def rewrite(lines, old, new):
    out = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        path, _, rest = line.rstrip('\n').partition(' ')
        if not path.startswith(old):
            raise ValueError('line {}: "{}" does not start with "{}"'.format(i + 1, path, old))
        out.append('{}{} {}\n'.format(new, path[len(old):], rest))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('files', nargs='+', help='split files (globs are expanded)')
    p.add_argument('--old', required=True, help='prefix to remove (use "" for none)')
    p.add_argument('--new', required=True, help='prefix to put instead (use "" for none)')
    p.add_argument('--out', required=True, help='output folder (never overwrites the input files)')
    args = p.parse_args()
    files = sorted({f for pat in args.files for f in (glob.glob(pat) or [pat])})
    os.makedirs(args.out, exist_ok=True)
    for f in files:
        dst = os.path.join(args.out, os.path.basename(f))
        if os.path.abspath(dst) == os.path.abspath(f):
            raise SystemExit('--out must be a different folder from the input files')
        with open(f, encoding='utf-8') as fh:
            lines = rewrite(fh.readlines(), args.old, args.new)
        with open(dst, 'w', encoding='utf-8', newline='\n') as fh:
            fh.writelines(lines)
        print('{}: {} lines -> {}'.format(f, len(lines), dst))


if __name__ == '__main__':
    main()
