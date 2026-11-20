"""Tune and run baselines on cached features (plan §7, §15 "A baseline").

    python tools/run_baselines.py --config configs/plantnet_clip_b32.json --methods ce,la,balanced_softmax,crt
    python tools/run_baselines.py --config configs/plantnet_clip_b32.json --methods la --trials 20 \
        --stages search,vselect,final --seeds 0,1,2

Stages:
    search   `trials` random-search trials on V-tune; the winner is saved to results/configs/
    vselect  ONE score of the saved winner on V-select (+ its V-select logits)
    final    the saved winner with each seed on test and V-cal (+ logits for C-F and C4)
Method names may fix settings: 'ltic3@gamma=0' (the gamma = 0 row).
C2 on top of a base method: 'crt+c2', 'la+c2', 'ltic3+c2', ...; A9 control: 'crt+c2shuf'. The donor-level shares
of the chosen C2 configuration are written to results/c2/.
C2 on a FROZEN base: 'tau_norm+c2f' (A9: 'tau_norm+c2fshuf', more shuffle seeds: 'tau_norm+c2fshuf@c2_shuffle_seed=1').
The base must have been run first: its results/configs/ file supplies every training setting, and only C2's
settings and the post-hoc knobs are searched. Real and shuffled C2 share the same candidate configurations.
Dataset config keys: "posthoc_grid": true grid-searches tau-norm's test_tau / LA's tau (0..2, step 0.1) on
V-tune for every trial; "c2_prior": "augmented" makes C2 runs use the prior that counts synthetic samples.
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
    p.add_argument('--skip_done', action='store_true',
                   help='skip a method whose chosen config and test predictions for every seed already exist '
                        '(lets a session that hit the time limit be resumed)')
    p.add_argument('--deadline_min', type=float, default=0,
                   help='do not START a method if (minutes since --t0) + its estimated time would pass this '
                        '(Kaggle kills a session at 12 h and keeps no output); 0 = off')
    p.add_argument('--t0', type=float, default=None, help='session start (unix time); default: now')
    p.add_argument('--budget_min', type=float, default=150,
                   help='time estimate for the first method of this call; later methods use 1.25 x the '
                        'longest method seen so far in this call')
    args = p.parse_args()
    t_session = args.t0 or time.time()
    longest = 0.0

    with open(args.config, encoding='utf-8') as f:
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
        tag = name.replace('@', '-').replace('=', '').replace(',', '-')
        cfg_path = os.path.join(cfg_dir, '{}_{}_{}.json'.format(conf['dataset'], conf['encoder'], tag))
        done = [os.path.join(args.preds, '{}_{}_{}_test_seed{}.npz'.format(conf['dataset'], conf['encoder'], tag, s))
                for s in seeds]
        if args.skip_done and os.path.isfile(cfg_path) and all(os.path.isfile(d) for d in done):
            print('=> {} already done (config and test predictions for seeds {}); skipped'.format(name, seeds))
            continue
        if args.deadline_min:
            used = (time.time() - t_session) / 60
            need = 1.25 * longest if longest else args.budget_min
            if used + need > args.deadline_min:
                print('=> DEADLINE: {:.0f} min used, {} needs ~{:.0f} min, limit {:.0f}; stopping before it '
                      '(the next session continues with PREV=auto)'.format(used, name, need, args.deadline_min),
                      flush=True)
                break
        t_method = time.time()
        if 'search' in stages:
            print('=> search {} ({} trials)'.format(name, trials), flush=True)
            _, fixed = harness.parse_method(name)
            m_base = dict(base)
            if fixed.get('c2') and 'c2_prior' in conf:
                m_base['c2_prior'] = conf['c2_prior']           # dataset-level choice (e.g. H22 pre-registration)
            frozen = None
            if fixed.get('c2_frozen_base'):
                b_name = harness.base_method_name(name)
                b_tag = b_name.replace('@', '-').replace('=', '').replace(',', '-')
                b_path = os.path.join(cfg_dir, '{}_{}_{}.json'.format(conf['dataset'], conf['encoder'], b_tag))
                if not os.path.isfile(b_path):
                    raise SystemExit('{} freezes the tuned config of {}, but {} does not exist: run {} first'.format(
                        name, b_name, b_path, b_name))
                with open(b_path, encoding='utf-8') as f:
                    frozen = json.load(f)
                print('   frozen base: {}'.format(b_path))
            cfg = harness.search(name, data, trials, args.results, seed=0, base=m_base, frozen_base=frozen)
            with open(cfg_path, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=1)
            print('   chosen: {}'.format(json.dumps(cfg)))
        with open(cfg_path, encoding='utf-8') as f:
            cfg = json.load(f)
        if cfg.get('c2'):
            summary = data.c2_stats(cfg).summary
            c2_dir = os.path.join(os.path.dirname(args.results) or '.', 'c2')
            os.makedirs(c2_dir, exist_ok=True)
            with open(os.path.join(c2_dir, '{}_{}_{}_donors.json'.format(conf['dataset'], conf['encoder'], tag)),
                      'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=1)
            print('=> {} donors (n_min {}): genus {:.1%}, family {:.1%}, global {:.1%}, none {}'.format(
                name, summary['n_min'], summary['genus_share'], summary['family_share'], summary['global_share'],
                summary['none']))
        if 'vselect' in stages:
            m = harness.score_vselect(name, cfg, data, args.results, seed=0, save_dir=args.preds)
            print('=> {} V-select: macro-F1 {:.4f} top-1 {:.4f} few {:.4f} (pooled {:.4f}) med {:.4f} many {:.4f}'.format(
                name, m['macro_f1'], m['top1'], m['few_acc'], m['few_acc_pooled'], m['medium_acc'], m['many_acc']))
        if 'final' in stages:
            out = harness.final_runs(name, cfg, data, seeds, args.results, save_dir=args.preds, splits=('test', 'vcal'))
            f1 = [m['macro_f1'] for _, split, m in out if split == 'test']
            if f1:
                print('=> {} test macro-F1 over seeds {}: {}'.format(name, seeds, ', '.join('{:.4f}'.format(v) for v in f1)))
        dt = (time.time() - t_method) / 60
        longest = max(longest, dt)
        print('=> {} took {:.1f} min ({:.0f} s)'.format(name, dt, dt * 60), flush=True)
    print('=> done in {:.1f} min; rows in {}'.format((time.time() - t0) / 60, args.results))


if __name__ == '__main__':
    main()
