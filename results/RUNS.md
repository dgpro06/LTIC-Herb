# Run log

Which Kaggle run produced which file. Notebook: https://www.kaggle.com/code/dgpro06/first-test
(GPU T4 x2, dataset `gabrielneto583/plantnet300k`). All on Pl@ntNet-300K; gate checks on V-select, C-F and
baselines on test (3 seeds) unless noted.

| Session | Kaggle version | Date | Code commit | Settings | Produced |
|---|---|---|---|---|---|
| 1 | Version 1 | 2026-11-15 | `e2dae67` | `ENCODERS` = all four, `RUN_GATE` on (gate on the first encoder only at that code version) | splits, taxonomy (pre-fix), features ×4, LA probe + gate + C-F + conformal on **clip_b32** → `week1_gate_plantnet_clip_b32.json` (the recorded gate result; never overwritten) |
| 2 | Version 2 "session-2" | 2026-11-16 | `ea25be9` | `PREV=auto`, gate on all four, baselines 1–9 on clip_b32, 20 trials, seeds 0–2 | `week1_gate_plantnet_{clip_b16,dinov2_b14,bioclip2}.json`, `week1_gate_plantnet_clip_b32_rerun.json` (adds the secondary analyses), all `cf/` files (clip_b32 redone with the completed taxonomy), `conformal/` ×4, `configs/` ×14, `all_runs.csv` (378 rows, includes session 1's 27) |

| 3 | Version 3 (named "session-2" again in Kaggle) | 2026-11-17 | `a0baa85` | `ENCODERS=dinov2_b14,bioclip2`, `RUN_GATE=False`, baselines 1–9 on both, 20 trials, seeds 0–2, input = Version 2 output | `configs/` ×20 (dinov2_b14, bioclip2), `all_runs.csv` (918 rows, sessions 1–3; 540 new runs, 5.2 h of head training) |
| 4 | Version 4 | 2026-11-17 | `ae60352`/`dc7e6ef` (no code change for baselines) | `ENCODERS=clip_b16`, `RUN_GATE=False`, baselines 1–9 on clip_b16, 20 trials, seeds 0–2, input = Version 3 output | `configs/` ×10 (clip_b16), `all_runs.csv` (1,188 rows, sessions 1–4; 270 new runs, 2.7 h of head training) |
| 5 | Version 5 | 2026-11-17 → 18 (rows 17:39 – 00:23) | `c30d91a` (C2 + TaxoNet code from `dc7e6ef`/`3dee815`; confirm in the Kaggle log) | `ENCODERS = BASELINE_ENCODERS = dinov2_b14,bioclip2,clip_b32`, `RUN_GATE=False`, `BASELINES = crt+c2,crt+c2shuf,tau_norm+c2,la+c2,ltic3+c2,ltic3+c2shuf,taxonet`, 20 trials, seeds 0–2, `CF_FROM_CE=True`, input = Version 4 output | `c2/` donor files ×18, `week2_gate_plantnet_{dinov2_b14,bioclip2,clip_b32}.txt` (JSON saved as .txt by the browser), `all_runs.csv` (1,755 rows, sessions 1–5; 567 new runs, 7.5 h of head training). Committed in `98ba06b`. The C2 configs and the CE-based C-F files are not in the repo |

Session 1's original C-F files and `all_runs.csv` for clip_b32 are in commit `413fd74`; session 2's
`all_runs.csv` is in `a0baa85`.

## Headline numbers

**Gate (V-select):** verdict "keep C2" on all four encoders.
| Encoder | Tail FPs | Cross-genus share (a) | Genus rescue (b), recorded | (b) with a plain-CE genus probe (secondary) | C2 p |
|---|---|---|---|---|---|
| clip_b32 | 167 | 0.772 [0.707, 0.835] | 0.070 (9/129) | 0.310 | 0.001 |
| clip_b16 | 88 | 0.795 [0.709, 0.877] | 0.071 (5/70) | 0.286 | 0.001 |
| dinov2_b14 | 108 | 0.417 [0.321, 0.514] | 0.267 (12/45) | 0.356 | 0.001 |
| bioclip2 | 91 | 0.253 [0.167, 0.343] | 0.217 (5/23) | 0.304 | 0.001 |

**C-F (test):** cross-genus share of LA tail false positives: clip_b32 0.733, clip_b16 0.812, dinov2_b14 0.376,
bioclip2 0.388.

**LA probe (test, mean of 3 seeds):** top-1 / macro-F1 / few-shot (per-class): clip_b32 0.699 / 0.306 / 0.129 ·
clip_b16 0.730 / 0.296 / 0.073 · dinov2_b14 0.819 / 0.500 / 0.357 · bioclip2 0.838 / 0.536 / 0.405.

**Baselines on clip_b32 (test macro-F1, mean of 3 seeds):** ltic3@γ=0 0.343 · ltic3 0.338 · crt 0.329 · reslt 0.326 ·
la 0.306 · ce 0.301 · flat_genus 0.291 · tau_norm 0.275 · ldam_drw 0.272 · balanced_softmax 0.267 · ltic2 0.261.

**Baselines on the strong encoders (session 3; test macro-F1, mean of 3 seeds; [V-select macro-F1]):**
- dinov2_b14: crt 0.503 [0.535] · la 0.500 [0.532] · tau_norm 0.500 [0.502] · ldam_drw 0.496 [0.542] · ltic3 0.493 [0.522] ·
  ltic3@γ=0 0.481 · balanced_softmax 0.480 · reslt 0.470 (best few-shot, pooled 0.512) · flat_genus 0.465 · ce 0.454 · ltic2 0.424
- bioclip2: tau_norm 0.553 [0.591] · crt 0.540 [0.564] · la 0.536 [0.581] · ltic3@γ=0 0.526 · reslt 0.523 · ldam_drw 0.518 [0.569] ·
  balanced_softmax 0.518 · ltic3 0.515 · ce 0.511 · flat_genus 0.510 · ltic2 0.413
- LTIC-3's lead on clip_b32 (+0.014 over cRT) disappears on the strong encoders, where simple linear probes with a
  frequency correction (cRT, τ-norm, LA) are as good or better. LTIC-2 (IJCNN code) is last on every encoder.
- clip_b16 (session 4): ltic3 0.359 [0.393] · ltic3@γ=0 0.356 [0.392] · crt 0.343 [0.369] · reslt 0.340 [0.382] ·
  tau_norm 0.311 · ce 0.303 · la 0.296 · flat_genus 0.286 · ltic2 0.284 · ldam_drw 0.284 · balanced_softmax 0.279.
  As on clip_b32, LTIC-3 leads on the CLIP encoders (+0.016 over cRT); best few-shot (pooled): ldam_drw 0.292, ltic2 0.286.
- Best baseline by V-select macro-F1 (the bar for the week-2 gate): clip_b32 ltic3@γ=0 (0.373), clip_b16 ltic3 (0.393), dinov2_b14 ldam_drw
  (0.542; cRT 0.535), bioclip2 tau_norm (0.591).

**Conformal (test, α = 0.1), average set size in species:** CP-1: clip_b32 141.5, clip_b16 157.6, dinov2_b14 5.9,
bioclip2 8.5 (median 3 for both). CP-3 (K = 5): coverage 0.94–0.95, fallback 0.97 (CLIP) vs 0.35 (DINOv2, BioCLIP 2).

**Run time:** 12–15 s per linear-probe run, 33–52 s for LTIC/ResLT/LDAM/flat-genus, up to 112 s; 201 GPU-minutes of
head training in session 2.

## Week-2 gate (session 5, V-select): FAIL on all three encoders

Recorded files: `week2_gate_plantnet_<encoder>.txt`. Rule fixed on Nov 16 (PLAN_NEXT §5): best `+c2` method vs the
best non-C2 baseline (both by V-select macro-F1), on macro-F1 and pooled few-shot accuracy, then A9.

| Encoder | Best baseline | Best C2 | Macro-F1 Δ [95% CI] | Pooled few-shot Δ [95% CI] | A9 | Verdict |
|---|---|---|---|---|---|---|
| clip_b32 | ltic3@γ=0 0.373 | ltic3+c2 0.355 | −0.018 [−0.031, −0.004] | −1.3 pts [−5.9, +3.3] | fails (C2 below its own base) | **FAIL** |
| bioclip2 | tau_norm 0.591 | la+c2 0.588 | −0.004 [−0.008, +0.020] | −4.6 pts [−8.2, −1.0] | not run | **FAIL** |
| dinov2_b14 | ldam_drw 0.542 | tau_norm+c2 0.547 | +0.006 [−0.017, +0.008] | +6.6 pts [+3.3, +9.8] | not run | **FAIL** (file says "INCOMPLETE": see correction) |

**Correction (2026-11-18): the DINOv2 macro-F1 "pass" in the recorded file is invalid.** It came from McNemar
p = 6.1e-10, but McNemar is two-sided on top-1 and here favours the **baseline**: V-select top-1 is 0.842 for
ldam_drw vs 0.828 for tau_norm+c2 (`all_runs.csv`, stage `vselect`). The script counted any p < 0.05 with a
positive macro-F1 delta as a C2 pass, without checking which method was ahead (bug fixed in
`tools/week2_gate.py`). The bootstrap CI on macro-F1 includes 0, so criterion 1 fails on DINOv2 and the A9 run
cannot change the verdict. **All three encoders fail.** The recorded file is kept unchanged as the record; the
gate is not re-run (the verdict cannot change, and re-running with a different test after seeing the result
would look like fishing).

Other known weaknesses of this gate, fixed in the code for later runs (H22) but not re-run on Pl@ntNet:
- The macro-F1 bootstrap resampled images, so classes with one V-select image dropped out of resamples and
  biased the difference (point estimates sit at the edge of their own CIs). Now: class-stratified bootstrap for
  reported intervals and a paired randomization test for the decision (`lt/metrics.py`).
- `+c2` and `+c2shuf` were searched on different random candidate configurations (seed from the method name),
  so A9 mixed the taxonomy effect with search luck. Now they share candidates (`lt/harness.search_seed_key`);
  the `+c2shuf` search draws of sessions 1–5 are reproducible only with code up to `98ba06b`. The chosen
  configurations themselves do not depend on this.
- C2 re-searched all base hyperparameters with the same 20 trials over a larger space. Now `+c2f` / `+c2fshuf`
  freeze the tuned base and search only C2's settings and the post-hoc knobs.
- V-select has 305 tail images over 291 classes; on the strong encoders V-select and test rank methods
  differently (Spearman 0.48 DINOv2, 0.46 BioCLIP 2; 0.84 / 0.91 on the CLIP encoders).

**Descriptive only (test set, already seen, not evidence for the method):** C2 raises test macro-F1 over the same
base for cRT (+0.007 / +0.015 / +0.007 on clip_b32 / dinov2_b14 / bioclip2) and τ-norm (+0.035 / +0.034 /
+0.007), but the shuffled-taxonomy control keeps most of cRT's gain on DINOv2 and BioCLIP 2 (+0.013 and +0.006),
so on strong encoders the gain is mostly generic tail augmentation, not taxonomy. LA + C2 loses 0.041 on
clip_b32 (the real prior plus synthetic tail samples corrects the tail twice; `c2_prior='augmented'` now exists).

## H22-clean feature extraction (2026-11-18)

Notebook `h22_extract` (Kaggle, dgpro06), Version 1, GPU T4 x2, competition data `herbarium-2022-fgvc9`,
code `208f9a1`, `ENCODERS=dinov2_b14,bioclip2,clip_b32`, `PART=0/1`, no baselines. Log 12:55 → 16:30
(3 h 35 min; 65.5 img/s for the three encoders together). **ALL CHUNKS PRESENT**; output `feats/` 3.3 GB
(stays on Kaggle), `splits/` 37 MB, `taxonomy/` 1 MB. No H22 model has been trained yet.

Split (plan 5.2): 839,772 images, 15,501 classes (none without images), largest class 80 images,
no classes with ≤ 2 images. Train 670,254 · val 84,759 · test 84,759. Shot groups on clean train
(few < 20, many ≥ 50): few 2,833 · medium 4,273 · many 8,395 classes.

| Part | Images | Classes | Few-shot images (classes) |
|---|---|---|---|
| V-tune | 33,883 | 14,342 | 1,755 (1,714) |
| V-select | 25,514 | 13,663 | 1,305 (1,297) |
| V-cal | 25,362 | 13,678 | 1,336 (1,326) |

Plan 8.0: kept 40/30/30 (V-select few-shot images ≥ 500). Split-file sha1 (every account must reproduce these):

```
ed57110a0acb707e50b597721d110793907eb22d  splits/h22clean_train.txt
ff065459598762ce9d605f4a99456ec61d476744  splits/h22clean_vtune.txt
e65f8c6bafc73007e40f28849dbbdcb22a6cfec1  splits/h22clean_vselect.txt
9b00dea5c9b324e82f358f3feb149a00cd3603f5  splits/h22clean_vcal.txt
62fe0f7ba0243209dd9cf29eec3d3fcc6ae2c9f0  splits/h22clean_test.txt
```

## H22-clean rules agreed (2026-11-21)
gates/h22_confirmatory.json marked AGREED by the team and supervisor, unchanged from the 2026-11-18 draft (208f9a1).
Committed before any H22-clean model run. Since the draft, the configs only gained "pred_format": "topk" and
"pred_topk": 20 (a0d2d83): how predictions are stored, not what is run or how it is judged.
