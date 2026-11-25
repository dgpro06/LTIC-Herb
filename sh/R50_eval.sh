#!/bin/bash
# Evaluate the TRAINED head from sh/R50.sh. The old script passed no --resume and evaluated
# random head weights (plan §12); evaluation now refuses to run without a checkpoint.

RUNS=${RUNS:-./data}

python CLIP_VIT_LONGTAIL.py \
  --mark clip_VIT_bt256 \
  -dataset herbariumDataset \
  --data_path /media/intisar/dataset/visual_categorization/herbarium-2022-fgvc9/ \
  --root_path "$RUNS" \
  --resume "$RUNS/herbariumDataset/clip_VIT_bt256/model_best.pth.tar" \
  -b 112 \
  --num_works 20 \
  --alpha 0.85 \
  --branches 2 \
  --gamma 0.3 \
  --num_classes 15505 \
  --backbone clip_b32 \
  --dataset_norm \
  --few_end 6799 \
  --many_start 13900 \
  --reslt_split 12568 \
  --evaluate
