#!/bin/bash
# IJCNN 2025 configuration on the ORIGINAL H22 files (train_hbm.txt / val_hbm.txt).
# Caution: 90% of val_hbm.txt is also in train_hbm.txt (plan §12), so these numbers are not clean.
# This runs the IJCNN settings on the C0 code (no classifier bias, macro-F1 logging). For the
# "old numbers" column of the corrected-evaluation table, run the original commit instead.

python CLIP_VIT_LONGTAIL.py \
  --mark clip_VIT_bt256 \
  -dataset herbariumDataset \
  --data_path /media/intisar/dataset1/visual_categorization/herbarium-2022-fgvc9/ \
  -b 256 \
  --epochs 200 \
  --num_works 20 \
  --lr 0.1 \
  --weight-decay 1e-4 \
  --alpha 0.85 \
  --branches 2 \
  --gamma 0.3 \
  --select_metric top1 \
  --num_classes 15505 \
  --backbone clip_b32 \
  --dataset_norm \
  --few_end 6799 \
  --many_start 13900 \
  --reslt_split 12568
