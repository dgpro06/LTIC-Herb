#!/bin/bash
# One LTIC run on Pl@ntNet-300K with a frozen encoder (C0: 3 branches, no classifier bias).
# For tuning use tools/run_baselines.py on chunked features (much faster); this script is the
# image-path entry point.
# Once, before training:
#   python tools/make_plantnet_splits.py --root $DATA --out $SPLITS
#   python tools/make_val_parts.py --train $SPLITS/plantnet_train.txt --val $SPLITS/plantnet_val.txt \
#       --out $SPLITS --prefix plantnet --few_thr 20 --auto_ratios
# Other encoders: BACKBONE=dinov2_b14 bash sh/plantnet.sh   (clip_b32 | clip_b16 | bioclip2 | dinov2_l14)

DATA=${DATA:-/path/to/plantnet_300K}   # the --root given to make_plantnet_splits.py
SPLITS=${SPLITS:-$DATA}                 # folder holding plantnet_{train,vtune,...}.txt
BACKBONE=${BACKBONE:-bioclip2}
EPOCHS=${EPOCHS:-50}
WORKERS=${WORKERS:-16}
BATCH=${BATCH:-256}
ALPHA=${ALPHA:-0.85}
GAMMA=${GAMMA:-0.0}
SEED=${SEED:-0}
RUNS=${RUNS:-./data}                    # logs and checkpoints
CACHE_DIR=${CACHE_DIR:-./feat_cache}
CACHE_VIEWS=${CACHE_VIEWS:-1}           # 1 everywhere except A12 (plan §11.2)

python CLIP_VIT_LONGTAIL.py \
  --backbone $BACKBONE \
  --mark plantnet_${BACKBONE}_bt${BATCH} \
  -dataset plantnetDataset \
  --data_path "$DATA" \
  --train_txt "$SPLITS/plantnet_train.txt" \
  --val_txt "$SPLITS/plantnet_vtune.txt" \
  --root_path "$RUNS" \
  -b $BATCH \
  --epochs $EPOCHS \
  --num_works $WORKERS \
  -p 100 \
  --lr 0.1 \
  --weight-decay 1e-4 \
  --alpha $ALPHA \
  --gamma $GAMMA \
  --seed $SEED \
  --few_thr 20 \
  --many_thr 100 \
  --num_classes 1081 \
  --cache_features \
  --cache_dir "$CACHE_DIR" \
  --cache_views $CACHE_VIEWS
