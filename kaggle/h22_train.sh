#!/bin/bash
# H22-clean head training on Kaggle (Phase 3): baselines, C2 and the A9 controls of gates/h22_confirmatory.json,
# on the features made by kaggle/h22_kaggle.sh. Guide: KAGGLE_H22.md ("Training").
#
#   MODE=timing  1 trial x 2 epochs of tau_norm, tau_norm+c2f and ltic3 per encoder, V-tune only, nothing kept:
#                checks the pipeline end to end and measures speed. Touches no V-select or test image.
#   MODE=full    the pre-registered runs. Refuses to start unless gates/h22_confirmatory.json is AGREED.
#
# One encoder per GPU (T4 x2 -> two encoders at once). With fewer encoders than GPUs (e.g. clip_b32 alone), the
# methods of each encoder are split: GPU 0 runs tau_norm, C2 (and IJCNN=1) then A9, GPU 1 the other baselines.
# Every call stops cleanly before DEADLINE_MIN; the next session attaches this notebook's earlier output
# (PREV=auto) and skips finished methods.
#
#   IJCNN=1      also run the IJCNN 2025 configuration (LTIC-2, the paper's settings from sh/R50.sh, untuned,
#                200 epochs, epoch chosen by top-1, training seed 0) on clip_b32: the corrected H22-clean number.
set -eo pipefail

MODE=${MODE:-timing}
ENCODERS=${ENCODERS:-dinov2_b14,bioclip2}
TRIALS=${TRIALS:-}                          # default: the config's (10, pre-registered)
SEEDS=${SEEDS:-0,1,2}
A9_SEEDS=${A9_SEEDS:-0}                     # the A9 controls are compared on test seed 0 only
DEADLINE_MIN=${DEADLINE_MIN:-600}           # Kaggle kills at 12 h and then keeps NO output
BUDGET_MIN=${BUDGET_MIN:-150}               # time allowed for the first method of each call (later: 1.25 x longest)
PREV=${PREV:-auto}
IJCNN=${IJCNN:-0}
INPUT=${INPUT:-/kaggle/input}
WORK=${WORK:-/kaggle/working}
SCRATCH=${SCRATCH:-/kaggle/tmp}             # not saved as Output

# the pre-registered order: tau_norm before tau_norm+c2f (it freezes tau_norm's tuned config)
C2_METHODS="tau_norm,tau_norm+c2f"
A9_METHODS="tau_norm+c2fshuf,tau_norm+c2fshuf@c2_shuffle_seed=1,tau_norm+c2fshuf@c2_shuffle_seed=2"
BASE_METHODS="la,crt,ce,ltic3,ltic2"
# sh/R50.sh: -b 256 --epochs 200 --lr 0.1 --weight-decay 1e-4 --alpha 0.85 --gamma 0.3 --select_metric top1
IJCNN_METHOD="ltic2@lr=0.1,weight_decay=0.0001,batch_size=256,epochs=200,alpha=0.85,gamma=0.3,select_metric=top1"

CODE="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONUNBUFFERED=1
T0=$(date +%s)
mkdir -p "$WORK" "$SCRATCH"
cd "$WORK"
stamp() { echo; echo "=== [$(date +%H:%M:%S)] $*"; }

stamp "STEP 1  GPU, code and rules (MODE=$MODE)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || { echo "No GPU: Session options > Accelerator > GPU T4 x2"; exit 1; }
NGPU=$(nvidia-smi -L | wc -l)
git -C "$CODE" log --oneline -1 || true
STATUS=$(python -c "import json;print(json.load(open('$CODE/gates/h22_confirmatory.json',encoding='utf-8'))['_status'])")
echo "rules: $STATUS"
if [ "$MODE" = full ] && [[ "$STATUS" != AGREED* ]]; then
  echo "MODE=full needs the AGREED rules file (gates/h22_confirmatory.json \"_status\": \"AGREED ...\")."
  echo "Pre-registration: no H22 model run before the rules are agreed and committed. Use MODE=timing meanwhile."
  exit 1
fi
if [ "$MODE" != full ] && [ "$MODE" != timing ]; then echo "MODE must be timing or full"; exit 1; fi

stamp "STEP 2  Features (attached Output or Dataset of h22_extract) and split check"
FEAT=""
for m in $(find "$INPUT" -maxdepth 8 -path "*/h22clean/train/*/manifest.json" -print 2>/dev/null); do
  FEAT="${m%/train/*/manifest.json}"; break
done
if [ -z "$FEAT" ]; then
  echo "No H22 features under $INPUT. Add Input > Your Work > h22_extract (or the h22clean-feats Dataset)."; exit 1
fi
echo "features: $FEAT"
mkdir -p taxonomy && cp "$CODE/taxonomy/h22.csv" taxonomy/h22.csv
python - "$FEAT" "$CODE" "$ENCODERS" <<'EOF'
import hashlib, json, os, sys
feat, code, encs = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
bad = []
for split in ('train', 'vtune', 'vselect', 'vcal', 'test'):
    sha = hashlib.sha1(open(os.path.join(code, 'splits', 'h22clean_{}.txt'.format(split)), 'rb').read()).hexdigest()
    for enc in encs:
        d = os.path.join(feat, split, enc)
        try:
            m = json.load(open(os.path.join(d, 'manifest.json'), encoding='utf-8'))
        except FileNotFoundError:
            bad.append('{}/{}: no features'.format(split, enc)); continue
        miss = [i for i in range(m['num_chunks']) if not os.path.isfile(os.path.join(d, 'chunk_{:05d}.pt'.format(i)))]
        if m['txt_sha1'] != sha:
            bad.append('{}/{}: made from a different split file than the committed one'.format(split, enc))
        if miss:
            bad.append('{}/{}: {} chunks missing'.format(split, enc, len(miss)))
    print('  {:8s} split sha1 {}  ok for {}'.format(split, sha[:12], ','.join(encs)))
if bad:
    print('\n'.join(bad)); sys.exit(1)
EOF

if [ "$MODE" = full ]; then
  RES="$WORK/results"
  if [ "$PREV" = auto ]; then
    for hit in $(find "$INPUT" -maxdepth 5 -type f -name "all_runs_h22clean_*.csv" -print 2>/dev/null | xargs -r -n1 dirname | sort -u); do
      stamp "Copying earlier results $hit"
      mkdir -p "$RES"; cp -rn "$hit/." "$RES/"
    done
  fi
else
  RES="$SCRATCH/timing_results"; rm -rf "$RES"
fi
mkdir -p "$RES" configs logs

for enc in $(echo "$ENCODERS" | tr ',' ' '); do
  python - "$CODE/configs/h22clean_${enc}.json" "configs/h22clean_${enc}.json" "$FEAT" "$WORK/taxonomy/h22.csv" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1], encoding='utf-8'))
c.update(feature_root=sys.argv[3], taxonomy=sys.argv[4])
json.dump(c, open(sys.argv[2], 'w', encoding='utf-8'), indent=1)
EOF
done

run() {   # run <encoder> <methods> <seeds> [extra args]; CSV_PART keeps parallel workers' CSVs apart
  local enc=$1 methods=$2 seeds=$3; shift 3
  python "$CODE/tools/run_baselines.py" --config "configs/h22clean_${enc}.json" --methods "$methods" \
    --seeds "$seeds" --results "$RES/all_runs_h22clean_${enc}${CSV_PART}.csv" --preds "$RES/preds" \
    --t0 "$T0" --deadline_min "$DEADLINE_MIN" --budget_min "$BUDGET_MIN" "$@"
}

worker() {   # worker <gpu> <part: all|c2|base> <encoders...>; keeps going after a failed call, exit 1 if any failed
  local gpu=$1 part=$2 rc=0; shift 2
  export CUDA_VISIBLE_DEVICES=$gpu
  [ "$part" = all ] && CSV_PART="" || CSV_PART="_$part"
  for enc in "$@"; do
    stamp "GPU $gpu: $enc ($MODE, $part)"
    if [ "$MODE" = timing ]; then
      run "$enc" "tau_norm,tau_norm+c2f,ltic3" 0 --trials 1 --epochs 2 --stages search || rc=1
      continue
    fi
    local tr=(); [ -n "$TRIALS" ] && tr=(--trials "$TRIALS")
    if [ "$part" != base ]; then
      run "$enc" "$C2_METHODS" "$SEEDS" --skip_done "${tr[@]}" || rc=1
      if [ "$IJCNN" = 1 ] && [ "$enc" = clip_b32 ]; then   # all settings fixed: 1 "trial" = the paper's config
        run "$enc" "$IJCNN_METHOD;" 0 --skip_done --trials 1 --stages search,final || rc=1
      fi
      run "$enc" "$A9_METHODS" "$A9_SEEDS" --skip_done "${tr[@]}" || rc=1
    fi
    if [ "$part" != c2 ]; then
      run "$enc" "$BASE_METHODS" "$SEEDS" --skip_done "${tr[@]}" || rc=1
    fi
  done
  return $rc
}

stamp "STEP 3  Training: $ENCODERS on $NGPU GPU(s), stop before ${DEADLINE_MIN} min"
ENC_ARR=($(echo "$ENCODERS" | tr ',' ' '))
if [ "$MODE" = full ] && [ "$NGPU" -ge 2 ] && [ ${#ENC_ARR[@]} -lt "$NGPU" ]; then
  # fewer encoders than GPUs: encoder i gets GPU 2i (tau_norm, C2, IJCNN, A9) and GPU 2i+1 (other baselines)
  for i in "${!ENC_ARR[@]}"; do
    for part in c2 base; do
      g=$((2 * i)); [ "$part" = base ] && g=$((g + 1))
      [ "$g" -ge "$NGPU" ] && continue
      ( set +e; worker "$g" "$part" "${ENC_ARR[$i]}" > "logs/gpu${g}.log" 2>&1; echo $? > "logs/gpu${g}.exit" ) &
      echo "GPU $g: ${ENC_ARR[$i]} ($part methods)  (log: logs/gpu${g}.log)"
    done
  done
else
  for g in $(seq 0 $((NGPU - 1))); do
    mine=()
    for i in "${!ENC_ARR[@]}"; do [ $((i % NGPU)) -eq "$g" ] && mine+=("${ENC_ARR[$i]}"); done
    [ ${#mine[@]} -eq 0 ] && continue
    ( set +e; worker "$g" all "${mine[@]}" > "logs/gpu${g}.log" 2>&1; echo $? > "logs/gpu${g}.exit" ) &
    echo "GPU $g: ${mine[*]}  (log: logs/gpu${g}.log)"
  done
fi
# progress every 5 min while any worker runs (the notebook log only shows this process's output)
n=0
while [ -n "$(jobs -rp)" ]; do
  sleep 60; n=$((n + 1))
  if [ $((n % 5)) -eq 0 ]; then
    for f in logs/gpu*.log; do
      echo "--- $(date +%H:%M) $f"; grep -E "^=>|trial|DEADLINE|Error|error" "$f" | tail -4 || true
    done
  fi
done
wait || true

stamp "Worker logs"
fail=0
for f in logs/gpu*.log; do
  g=${f#logs/gpu}; g=${g%.log}
  code=$(cat "logs/gpu${g}.exit" 2>/dev/null || echo missing)
  echo "----- $f (exit $code)"; cat "$f"
  [ "$code" = 0 ] || fail=1
done

if [ "$MODE" = full ]; then
  stamp "Results"
  python - "$RES" <<'EOF'
import csv, glob, os, sys
res = sys.argv[1]
parts = sorted(glob.glob(os.path.join(res, 'all_runs_h22clean_*.csv')))
rows, header = [], None
for p in parts:
    with open(p, newline='', encoding='utf-8') as f:
        r = csv.reader(f); h = next(r); header = header or h; rows += list(r)
if header:
    with open(os.path.join(res, 'all_runs_h22clean.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
print('{} rows from {} -> all_runs_h22clean.csv'.format(len(rows), [os.path.basename(p) for p in parts]))
EOF
  IJCNN=$IJCNN IJCNN_METHOD=$IJCNN_METHOD python - "$RES" "$ENCODERS" <<'EOF'
import json, os, sys
res, encs = sys.argv[1], sys.argv[2].split(',')
need = ['tau_norm', 'tau_norm+c2f', 'tau_norm+c2fshuf', 'tau_norm+c2fshuf@c2_shuffle_seed=1',
        'tau_norm+c2fshuf@c2_shuffle_seed=2', 'la', 'crt', 'ce', 'ltic3', 'ltic2']
for enc in encs:
    done = []
    for m in need:
        tag = m.replace('@', '-').replace('=', '').replace(',', '-')
        seeds = [0] if 'shuf' in m else [0, 1, 2]
        ok = os.path.isfile(os.path.join(res, 'configs', 'h22clean_{}_{}.json'.format(enc, tag))) and all(
            os.path.isfile(os.path.join(res, 'preds', 'h22clean_{}_{}_test_seed{}.npz'.format(enc, tag, s))) for s in seeds)
        done.append(ok)
    print('  {:11s} {}/{} methods done{}'.format(enc, sum(done), len(need),
          '' if all(done) else '  todo: ' + ','.join(m for m, d in zip(need, done) if not d)))
    if all(done):
        print('  {:11s} READY for: python tools/confirmatory_gate.py --rules gates/h22_confirmatory.json '
              '--results results/all_runs_h22clean.csv --encoder {}'.format('', enc))
    if enc == 'clip_b32' and os.environ.get('IJCNN') == '1':
        ij = os.environ['IJCNN_METHOD'].replace('@', '-').replace('=', '').replace(',', '-')
        ok = os.path.isfile(os.path.join(res, 'preds', 'h22clean_clip_b32_{}_test_seed0.npz'.format(ij)))
        print('  {:11s} IJCNN configuration: {}'.format('', 'done' if ok else 'todo'))
EOF
else
  stamp "Timing summary (2 epochs, 1 trial; a full run is 30 epochs)"
  grep -h -E "trial|took|load" logs/gpu*.log || true
  rm -rf "$RES"
fi

stamp "DONE ($(( ($(date +%s) - T0) / 60 )) min)"
du -sh "$WORK/results" 2>/dev/null || true
exit $fail
