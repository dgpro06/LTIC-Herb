# Week 1 runbook (plan v9.1)

Commands to reach the week-1 gate (§8.1) on Pl@ntNet-300K, then repeat it on H22. Run them from the
repo root. `python -m pytest tests -q` must pass first; it covers every step below on a synthetic set.

## 0. Before computing anything
- Agree on the thresholds in [gates/week1_thresholds.json](gates/week1_thresholds.json) and commit them.
  The commit is the record that they were fixed before any diagnostic was run (§8.1).

## 1. Splits and taxonomy (Track C)
```bash
# Pl@ntNet official splits -> splits/plantnet_{train,val,test}.txt + plantnet_label_map.json
python tools/make_plantnet_splits.py --root $PLANTNET --out splits
# V-tune / V-select / V-cal + the §8.0 tail count (writes splits/plantnet_vparts.json)
python tools/make_val_parts.py --train splits/plantnet_train.txt --val splits/plantnet_val.txt \
    --out splits --prefix plantnet --few_thr 20 --auto_ratios
# taxonomy/plantnet.csv (genus from the name, family from GBIF; cached in taxonomy/gbif_cache.json)
python tools/make_taxonomy_plantnet.py --label_map splits/plantnet_label_map.json
```
Record the printed V-select few-shot count and the chosen ratios in the project notes.

H22-clean is built from the **official** metadata (on Kaggle). The local `*_hbm.txt` files give
780,955 images and are only good for a dry run (`--txt train_hbm.txt --txt val_hbm.txt`):
```bash
python tools/make_h22_clean.py --metadata $H22/train_metadata.json --out splits   # also writes taxonomy/h22.csv
python tools/make_val_parts.py --train splits/h22clean_train.txt --val splits/h22clean_val.txt \
    --out splits --prefix h22clean --few_thr 20 --auto_ratios
```

## 2. Features (Track A)
Throughput test first (10k images on each machine; update the hours in §11.2):
```bash
python tools/extract_features.py --root $PLANTNET --out /tmp/bench --encoders clip_b32 \
    --split train=splits/plantnet_train.txt --limit 10000
```
Full extraction. It is chunked, resumable (re-run the same command after a crash) and decodes each image once for all encoders:
```bash
python tools/extract_features.py --root $PLANTNET --out feats/plantnet \
    --encoders clip_b32,clip_b16,bioclip2,dinov2_b14 \
    --split train=splits/plantnet_train.txt --split vtune=splits/plantnet_vtune.txt \
    --split vselect=splits/plantnet_vselect.txt --split vcal=splits/plantnet_vcal.txt \
    --split test=splits/plantnet_test.txt
```
On the 12 GB 4070, if four encoders don't fit together, run them in two calls with the same `--out`.

## 3. Baselines and the LA probe (Track B)
```bash
python tools/run_baselines.py --config configs/plantnet_clip_b32.json --methods la            # LA probe first
python tools/run_baselines.py --config configs/plantnet_clip_b32.json \
    --methods ce,balanced_softmax,crt,tau_norm,ldam_drw,reslt,ltic2,ltic3,ltic3@gamma=0,flat_genus
```
Each method gets 20 random-search trials on V-tune, one score on V-select, and seeds 0/1/2 on test and V-cal.
Rows go to `results/all_runs.csv`, logits to `results/preds/`. Time the first run: if a run takes more than 2 min,
cut `--trials` to 10 for **every** method (§11.2).

## 4. C-F and the gate (Track C)
```bash
python tools/cf_report.py --preds results/preds/plantnet_clip_b32_la_test_seed0.npz --taxonomy taxonomy/plantnet.csv
python tools/week1_diagnostics.py --config configs/plantnet_clip_b32.json \
    --la_preds results/preds/plantnet_clip_b32_la_vselect_seed0.npz
python tools/conformal_eval.py --cal results/preds/plantnet_clip_b32_la_vcal_seed0.npz \
    --test results/preds/plantnet_clip_b32_la_test_seed0.npz --taxonomy taxonomy/plantnet.csv
```
`week1_diagnostics.py` prints PASS/FAIL for the donor median, the cross-genus share (a), the genus rescue (b)
and the C2 covariance test. The verdict, with every number, is saved to `results/week1_gate_<dataset>_<encoder>.json`.
Gate diagnostics use V-select; the C-F figure uses test.

## Not done yet
- SMART and the feature-level TaxoNet loss (baselines 10 and 11)
- LIFT setup (Track A, Thu Nov 5)
- iNat-2018 loader and taxonomy
- the CP-5 option, and a check of the PAS score definition against Ding et al.'s code (`lt/conformal.py`, TODO)
