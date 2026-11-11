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
GATE_ENCODER=${GATE_ENCODER:-${ENCODERS%%,*}}
RUN_GATE=${RUN_GATE:-1}                     # LA probe + C-F + week-1 diagnostics + conformal
BASELINES=${BASELINES:-}                    # e.g. ce,balanced_softmax,crt,tau_norm,ldam_drw,reslt,ltic2,ltic3,ltic3@gamma=0,flat_genus
BASELINE_ENCODERS=${BASELINE_ENCODERS:-$ENCODERS}
TRIALS=${TRIALS:-20}                        # random-search trials per method (cut to 10 for ALL methods if runs are slow)
EPOCHS=${EPOCHS:-}                          # default: the config's 30
SEEDS=${SEEDS:-0,1,2}
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

if [ -n "$PREV" ]; then
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
if [ ! -f taxonomy/plantnet.csv ]; then
  OFF=""; [ "$GBIF_OFFLINE" = 1 ] && OFF="--offline"
  python "$CODE/tools/make_taxonomy_plantnet.py" --label_map splits/plantnet_label_map.json $OFF
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
c = json.load(open(sys.argv[1])); c['encoder'] = sys.argv[3]
c['num_classes'] = len(json.load(open('splits/plantnet_label_map.json')))   # 1081 on Pl@ntNet-300K
json.dump(c, open(sys.argv[2], 'w'), indent=1)
EOF
done
EP=""; [ -n "$EPOCHS" ] && EP="--epochs $EPOCHS"

if [ "$RUN_GATE" = 1 ]; then
  stamp "STEP 5  LA probe on $GATE_ENCODER ($TRIALS trials, seeds $SEEDS)"
  python "$CODE/tools/run_baselines.py" --config "configs/plantnet_${GATE_ENCODER}.json" --methods la \
    --trials "$TRIALS" --seeds "$SEEDS" $EP
  P="results/preds/plantnet_${GATE_ENCODER}_la"

  stamp "STEP 6  Finding C-F (test split)"
  python "$CODE/tools/cf_report.py" --preds "${P}_test_seed0.npz" --taxonomy taxonomy/plantnet.csv --out results/cf

  stamp "STEP 7  Week-1 gate diagnostics (V-select)"
  python "$CODE/tools/week1_diagnostics.py" --config "configs/plantnet_${GATE_ENCODER}.json" \
    --la_preds "${P}_vselect_seed0.npz" --thresholds "$CODE/gates/week1_thresholds.json"

  stamp "STEP 8  Conformal CP-1..CP-4 (calibrate on V-cal, report on test)"
  python "$CODE/tools/conformal_eval.py" --cal "${P}_vcal_seed0.npz" --test "${P}_test_seed0.npz" \
    --taxonomy taxonomy/plantnet.csv --out "results/conformal/plantnet_${GATE_ENCODER}_la_cp.csv"
fi

if [ -n "$BASELINES" ]; then
  for enc in $(echo "$BASELINE_ENCODERS" | tr ',' ' '); do
    stamp "STEP 9  Baselines on $enc: $BASELINES"
    python "$CODE/tools/run_baselines.py" --config "configs/plantnet_${enc}.json" --methods "$BASELINES" \
      --trials "$TRIALS" --seeds "$SEEDS" $EP
  done
fi

stamp "DONE. Saved in $WORK:"
for f in splits/plantnet_vparts.json taxonomy/plantnet.csv results/all_runs.csv results/week1_gate_plantnet_${GATE_ENCODER}.json; do
  [ -e "$f" ] && echo "  $f"
done
du -sh feats results 2>/dev/null || true
