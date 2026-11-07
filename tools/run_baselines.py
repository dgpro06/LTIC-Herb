"""Tune and run baselines on cached features (plan §7, §15 "A baseline").

    python tools/run_baselines.py --config configs/plantnet_clip_b32.json --methods ce,la,balanced_softmax,crt
    python tools/run_baselines.py --config configs/plantnet_clip_b32.json --methods la --trials 20 \
        --stages search,vselect,final --seeds 0,1,2

Stages:
    search   `trials` random-search trials on V-tune; the winner is saved to results/configs/
    vselect  ONE score of the saved winner on V-select (+ its V-select logits)
    final    the saved winner with each seed on test and V-cal (+ logits for C-F and C4)
Method names may fix settings: 'ltic3@gamma=0' (the gamma = 0 row).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lt import harness  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--methods', required=True)
    p.add_argument('--trials', type=int, default=None, help='default: config "trials" (20)')
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--stages', default='search,vselect,final')
    p.add_argument('--results', default='results/all_runs.csv')
    p.add_argument('--preds', default='results/preds')
    p.add_argument('--epochs', type=int, default=None, help='override config epochs')
    args = p.parse_args()

    with open(args.config) as f:
        conf = json.load(f)
    trials = args.trials or conf.get('trials', 20)
    base = {'epochs': args.epochs or conf.get('epochs', 30), 'eval_every': conf.get('eval_every', 2)}
    seeds = [int(s) for s in args.seeds.split(',')]
    stages = set(args.stages.split(','))

    t0 = time.time()
    data = harness.Data(conf)
    print('=> ' + data.describe() + ' ({:.0f}s to load)'.format(time.time() - t0), flush=True)
    cfg_dir = os.path.join(os.path.dirname(args.results) or '.', 'configs')
    os.makedirs(cfg_dir, exist_ok=True)

    for name in args.methods.split(','):
        cfg_path = os.path.join(cfg_dir, '{}_{}_{}.json'.format(conf['dataset'], conf['encoder'],
                                                                 name.replace('@', '-').replace('=', '').replace(',', '-')))
        if 'search' in stages:
            print('=> search {} ({} trials)'.format(name, trials), flush=True)
            cfg = harness.search(name, data, trials, args.results, seed=0, base=base)
            with open(cfg_path, 'w') as f:
                json.dump(cfg, f, indent=1)
            print('   chosen: {}'.format(json.dumps(cfg)))
        with open(cfg_path) as f:
            cfg = json.load(f)
        if 'vselect' in stages:
            m = harness.score_vselect(name, cfg, data, args.results, seed=0, save_dir=args.preds)
            print('=> {} V-select: macro-F1 {:.4f} top-1 {:.4f} few {:.4f} (pooled {:.4f}) med {:.4f} many {:.4f}'.format(
                name, m['macro_f1'], m['top1'], m['few_acc'], m['few_acc_pooled'], m['medium_acc'], m['many_acc']))
        if 'final' in stages:
            out = harness.final_runs(name, cfg, data, seeds, args.results, save_dir=args.preds, splits=('test', 'vcal'))
            f1 = [m['macro_f1'] for _, split, m in out if split == 'test']
            if f1:
                print('=> {} test macro-F1 over seeds {}: {}'.format(name, seeds, ', '.join('{:.4f}'.format(v) for v in f1)))
    print('=> done in {:.1f} min; rows in {}'.format((time.time() - t0) / 60, args.results))


if __name__ == '__main__':
    main()
