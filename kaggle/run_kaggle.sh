#!/bin/bash
# Full Pl@ntNet-300K run on a Kaggle GPU notebook: setup -> split files -> training -> test evaluation.
# Step-by-step guide: KAGGLE.md.   Usage (in a notebook cell):  !bash kaggle/run_kaggle.sh
set -eo pipefail

BACKBONE=${BACKBONE:-bioclip2}          # bioclip2 | dinov2_b14 | clip_b32 | clip_b16
EPOCHS=${EPOCHS:-50}
CACHE_VIEWS=${CACHE_VIEWS:-1}           # 1 except A12 (plan §11.2)
WORKERS=${WORKERS:-4}                   # Kaggle GPU notebooks have 4 CPU cores
INPUT=${INPUT:-/kaggle/input}           # attached datasets are mounted here (read-only)
WORK=${WORK:-/kaggle/working}           # writable, saved as the notebook output

export PYTHONUNBUFFERED=1               # show Python logs live in the notebook
cd "$(dirname "$0")/.."

echo "=== STEP 1/5  Checking the GPU"
if ! nvidia-smi --query-gpu=name,memory.total --format=csv; then
  echo "No GPU found. In the notebook sidebar: Settings > Accelerator > GPU T4 x2, then run again."
  exit 1
fi

echo "=== STEP 2/5  Installing open_clip (BioCLIP 2 / CLIP loader)"
pip install -q --upgrade open_clip_torch
python -c "import torch, open_clip; print('torch', torch.__version__, '| open_clip', open_clip.__version__, '| GPUs', torch.cuda.device_count())"

echo "=== STEP 3/5  Writing the Pl@ntNet-300K split files"
python tools/make_plantnet_splits.py --root "$INPUT" --out "$WORK/splits"
python tools/make_val_parts.py --train "$WORK/splits/plantnet_train.txt" --val "$WORK/splits/plantnet_val.txt" \
  --out "$WORK/splits" --prefix plantnet --few_thr 20 --auto_ratios

export DATA="$INPUT" SPLITS="$WORK/splits" RUNS="$WORK/runs" CACHE_DIR="$WORK/feat_cache"
export BACKBONE EPOCHS CACHE_VIEWS WORKERS

echo "=== STEP 4/5  Training LTIC with $BACKBONE for $EPOCHS epochs (features are extracted first)"
bash sh/plantnet.sh

echo "=== STEP 5/5  Evaluating the best checkpoint on the test split"
bash sh/plantnet_eval.sh

echo "=== DONE"
echo "Best validation epoch (from training):"
grep '\* Acc' "$RUNS/plantnetDataset/plantnet_${BACKBONE}_bt${BATCH:-256}/train.log" | sort -t' ' -k4 -g | tail -n 1
echo "Test split:"
grep '\* Acc' "$RUNS/plantnetDataset/plantnet_${BACKBONE}_bt${BATCH:-256}_test/train.log" | tail -n 1
echo "Logs and checkpoints: $RUNS/plantnetDataset/"
