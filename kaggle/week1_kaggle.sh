#!/bin/bash
# Week-1 pipeline (plan v9.1) on Pl@ntNet-300K in a Kaggle GPU notebook. Guide: KAGGLE_WEEK1.md.
#
#   splits + V-parts (§8.0) -> taxonomy (GBIF) -> chunked features -> LA probe -> C-F, gate diagnostics,
#   conformal -> optional baselines
#
# Everything is written to $WORK (/kaggle/working = the notebook's saved Output). Every step skips work
# that is already there, so a later session can continue: attach the earlier version's Output as an
# input and set PREV to its folder (splits, taxonomy, features and results are copied back first).
set -eo pipefail

ENCODERS=${ENCODERS:-clip_b32}              # comma-separated: clip_b32,clip_b16,bioclip2,dinov2_b14
RUN_GATE=${RUN_GATE:-1}                     # LA probe + C-F + week-1 diagnostics + conformal, per encoder
GATE_ENCODERS=${GATE_ENCODERS:-${GATE_ENCODER:-$ENCODERS}}   # default: every encoder in ENCODERS
BASELINES=${BASELINES:-}                    # e.g. ce,balanced_softmax,crt,tau_norm,ldam_drw,reslt,ltic2,ltic3,ltic3@gamma=0,flat_genus
BASELINE_ENCODERS=${BASELINE_ENCODERS:-$ENCODERS}
TRIALS=${TRIALS:-20}                        # random-search trials per method (cut to 10 for ALL methods if runs are slow)
EPOCHS=${EPOCHS:-}                          # default: the config's 30
SEEDS=${SEEDS:-0,1,2}
CF_FROM_CE=${CF_FROM_CE:-0}                 # 1 = C-F tau curve from the CE baseline's test predictions
BATCH=${BATCH:-128}                         # extraction batch size
WORKERS=${WORKERS:-4}                       # Kaggle notebooks have 4 CPU cores
INPUT=${INPUT:-/kaggle/input}
WORK=${WORK:-/kaggle/working}
PREV=${PREV:-}                              # e.g. /kaggle/input/<your-notebook>  (an earlier version's Output)
GBIF_OFFLINE=${GBIF_OFFLINE:-0}
SKIP_INSTALL=${SKIP_INSTALL:-0}
REQUIRE_GPU=${REQUIRE_GPU:-1}

CODE="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONUNBUFFERED=1
mkdir -p "$WORK"
cd "$WORK"                                  # configs use paths relative to here: feats/, taxonomy/, results/
stamp() { echo; echo "=== [$(date +%H:%M:%S)] $*"; }

stamp "STEP 1  GPU and code"
if [ "$REQUIRE_GPU" = 1 ] && ! nvidia-smi --query-gpu=name,memory.total --format=csv; then
  echo "No GPU. Notebook sidebar: Session options > Accelerator > GPU T4 x2, then run again."; exit 1
fi
git -C "$CODE" log --oneline -1 || true
if [ "$SKIP_INSTALL" != 1 ]; then pip install -q --upgrade open_clip_torch; fi
python -c "import torch; print('torch', torch.__version__, '| GPUs', torch.cuda.device_count())"

if [ "$PREV" = auto ]; then
  # an earlier version's Output attached as an input: the folder that holds feats/plantnet
  HIT=$(find "$INPUT" -maxdepth 7 -name images -prune -o -type d -path "*/feats/plantnet" -print 2>/dev/null | head -1)
  if [ -z "$HIT" ]; then
    echo "PREV=auto: no earlier output with feats/plantnet under $INPUT. Attach the earlier version's Output"
    echo "(Input > + Add Input > Your Work) or set PREV to its folder."; exit 1
  fi
  PREV="${HIT%/feats/plantnet}"
fi
if [ -n "$PREV" ]; then
  [ -d "$PREV" ] || { echo "PREV folder not found: $PREV"; exit 1; }
  stamp "Resuming from $PREV"
  for d in splits taxonomy feats results; do
    if [ -d "$PREV/$d" ]; then mkdir -p "$WORK/$d"; cp -rn "$PREV/$d/." "$WORK/$d/"; echo "copied $d"; fi
  done
fi

stamp "STEP 2  Split files and V-tune / V-select / V-cal (plan 8.0)"
if [ ! -f splits/plantnet_vcal.txt ]; then
  python "$CODE/tools/make_plantnet_splits.py" --root "$INPUT" --out splits
  python "$CODE/tools/make_val_parts.py" --train splits/plantnet_train.txt --val splits/plantnet_val.txt \
    --out splits --prefix plantnet --few_thr 20 --auto_ratios
else
  echo "splits already there"
fi
cat splits/plantnet_vparts.json | python -c "import json,sys; r=json.load(sys.stdin); print('ratios', r['ratios'], '|', r['decision_8_0']); print('V-select few-shot images:', r['parts']['vselect']['few_images'])"
# split-file paths are relative to --root of make_plantnet_splits.py, so images are read from $INPUT
DATA_ROOT="$INPUT"

stamp "STEP 3  Taxonomy (genus from the species name, family from GBIF)"
TAX_REBUILT=0
# rebuild if missing, or if made by older code (no family_source column: unmatched genera had no family);
# the GBIF answers are cached in taxonomy/gbif_cache.json, so a rebuild makes no new requests
if [ ! -f taxonomy/plantnet.csv ] || ! head -1 taxonomy/plantnet.csv | grep -q family_source; then
  OFF=""; [ "$GBIF_OFFLINE" = 1 ] && OFF="--offline"
  python "$CODE/tools/make_taxonomy_plantnet.py" --label_map splits/plantnet_label_map.json $OFF
  TAX_REBUILT=1
else
  echo "taxonomy already there"
fi

stamp "STEP 4  Features: $ENCODERS (chunked, resumable)"
python "$CODE/tools/extract_features.py" --root "$DATA_ROOT" --out feats/plantnet --encoders "$ENCODERS" \
  --batch "$BATCH" --workers "$WORKERS" \
  --split train=splits/plantnet_train.txt --split vtune=splits/plantnet_vtune.txt \
  --split vselect=splits/plantnet_vselect.txt --split vcal=splits/plantnet_vcal.txt \
  --split test=splits/plantnet_test.txt

# one config per encoder, from the committed template (paths are relative to $WORK)
mkdir -p configs
for enc in $(echo "$ENCODERS,$BASELINE_ENCODERS" | tr ',' ' ' | xargs -n1 | sort -u); do
  python - "$CODE/configs/plantnet_clip_b32.json" "configs/plantnet_${enc}.json" "$enc" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1], encoding='utf-8')); c['encoder'] = sys.argv[3]
c['num_classes'] = len(json.load(open('splits/plantnet_label_map.json', encoding='utf-8')))   # 1081 on Pl@ntNet-300K
json.dump(c, open(sys.argv[2], 'w', encoding='utf-8'), indent=1)
EOF
done
EP=""; [ -n "$EPOCHS" ] && EP="--epochs $EPOCHS"

if [ "$TAX_REBUILT" = 1 ] && [ "$RUN_GATE" != 1 ]; then
  # families changed: redo the C-F reports of earlier LA probes (the gate checks do not use families)
  for f in results/preds/plantnet_*_la_test_seed0.npz; do
    [ -e "$f" ] || continue
    stamp "C-F report redone with the updated taxonomy: $f"
    python "$CODE/tools/cf_report.py" --preds "$f" --taxonomy taxonomy/plantnet.csv --out results/cf
  done
fi

if [ "$RUN_GATE" = 1 ]; then
  for genc in $(echo "$GATE_ENCODERS" | tr ',' ' '); do
    stamp "STEP 5  LA probe on $genc ($TRIALS trials, seeds $SEEDS; skipped if already done)"
    python "$CODE/tools/run_baselines.py" --config "configs/plantnet_${genc}.json" --methods la \
      --trials "$TRIALS" --seeds "$SEEDS" $EP --skip_done
    P="results/preds/plantnet_${genc}_la"

    stamp "STEP 6  Finding C-F on $genc (test split)"
    python "$CODE/tools/cf_report.py" --preds "${P}_test_seed0.npz" --taxonomy taxonomy/plantnet.csv --out results/cf

    stamp "STEP 7  Week-1 gate diagnostics on $genc (V-select)"
    # never overwrite a recorded gate result: a re-run (e.g. for the secondary analyses) gets its own file
    GATE_OUT="results/week1_gate_plantnet_${genc}.json"
    [ -e "$GATE_OUT" ] && GATE_OUT="results/week1_gate_plantnet_${genc}_rerun.json"
    python "$CODE/tools/week1_diagnostics.py" --config "configs/plantnet_${genc}.json" \
      --la_preds "${P}_vselect_seed0.npz" --thresholds "$CODE/gates/week1_thresholds.json" --out "$GATE_OUT"

    stamp "STEP 8  Conformal CP-1..CP-4 on $genc (calibrate on V-cal, report on test)"
    python "$CODE/tools/conformal_eval.py" --cal "${P}_vcal_seed0.npz" --test "${P}_test_seed0.npz" \
      --taxonomy taxonomy/plantnet.csv --out "results/conformal/plantnet_${genc}_la_cp.csv"
  done
fi

if [ -n "$BASELINES" ]; then
  for enc in $(echo "$BASELINE_ENCODERS" | tr ',' ' '); do
    stamp "STEP 9  Baselines on $enc: $BASELINES (finished methods are skipped)"
    python "$CODE/tools/run_baselines.py" --config "configs/plantnet_${enc}.json" --methods "$BASELINES" \
      --trials "$TRIALS" --seeds "$SEEDS" $EP --skip_done
  done
fi

if [ "$CF_FROM_CE" = 1 ]; then
  # the paper's "tail-FP share vs tau" curve, drawn from the plain CE probe (tau = 0 is the untouched model)
  for enc in $(echo "$BASELINE_ENCODERS" | tr ',' ' '); do
    f="results/preds/plantnet_${enc}_ce_test_seed0.npz"
    [ -e "$f" ] || { echo "no CE predictions for $enc ($f); C-F tau curve skipped"; continue; }
    stamp "C-F tau curve from the CE baseline: $enc"
    python "$CODE/tools/cf_report.py" --preds "$f" --taxonomy taxonomy/plantnet.csv --out results/cf --tau 0 --tau_max 2
  done
fi

stamp "DONE. Saved in $WORK:"
for f in splits/plantnet_vparts.json taxonomy/plantnet.csv results/all_runs.csv results/week1_gate_plantnet_*.json; do
  [ -e "$f" ] && echo "  $f"
done
if ls results/week1_gate_plantnet_*.json >/dev/null 2>&1; then
  stamp "Gate summary"
  python - <<'EOF'
import glob, json
for path in sorted(glob.glob('results/week1_gate_plantnet_*.json')):
    g = json.load(open(path, encoding='utf-8'))
    checks = ' | '.join('{} {:.3f} {}'.format(k.split('_')[0], v['value'], 'PASS' if v['pass'] else 'FAIL')
                        for k, v in g['checks'].items())
    print('{:45s} {}  -> {}'.format(path.split('/')[-1], checks, g['verdict']))
EOF
fi
du -sh feats results 2>/dev/null || true
