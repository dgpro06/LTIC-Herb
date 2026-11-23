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

## H22-clean training, session 1 (2026-11-21)
Kaggle notebook dgpro06/h22-train Version 1 (pushed through the Kaggle API), code b8cf6f4 (rules AGREED), MODE=full,
T4 x2, 02:57 -> 11:52 UTC (535 min), both workers exit 0, stopped by the 600-min deadline as designed.
Every run takes 85-102 min (10 trials x 30 epochs + V-select + 3 test/V-cal seeds; ~87 min for A9 controls).

| encoder    | done (6 + 4 C2/A9 needed)                         | todo                    |
|------------|---------------------------------------------------|-------------------------|
| dinov2_b14 | tau_norm, tau_norm+c2f, 3 x c2fshuf, la           | crt, ce, ltic3, ltic2   |
| bioclip2   | tau_norm, tau_norm+c2f, 3 x c2fshuf               | la, crt, ce, ltic3, ltic2 |

V-select macro-F1 (selects the best baseline; the gate needs all 6 baselines, so NO verdict yet):
dinov2_b14 la 0.5226, tau_norm 0.5137, tau_norm+c2f 0.5138; bioclip2 tau_norm 0.6903, tau_norm+c2f 0.6914.
Committed: all_runs_h22clean*.csv, configs/, c2/ donor shares, h22_logs/session1_*.log.
Prediction files (results/preds/, 53 files, 278 MB) are gitignored as usual; copies are on this machine and in the
notebook's Version 1 output. The official gate (tools/confirmatory_gate.py) has NOT been run.

## H22-clean training, session 2, and the confirmatory gate (2026-11-22)
Session 2 (h22-train Version 2, queued 17:00 -> started 20:30 UTC Nov 21, done 05:32 UTC Nov 22, exit 0 / 0):
crt, ce, ltic3, ltic2 for dinov2_b14; la, crt, ce, ltic3, ltic2 for bioclip2. 10/10 methods for both encoders.
GPU used for H22 training: ~20 h of quota (2 sessions).

Confirmatory gate (tools/confirmatory_gate.py, rules gates/h22_confirmatory.json AGREED b8cf6f4), test seed 0:

| encoder    | best baseline (V-select) | macro-F1 C2 / base | randomization p | few-shot C2 / base | McNemar | A9 retained | result |
|------------|--------------------------|--------------------|-----------------|--------------------|---------|-------------|--------|
| dinov2_b14 | crt (0.5227 vs la 0.5226) | 0.5395 / 0.5382   | 0.138 FAIL      | 0.472 / 0.406      | 342:52 PASS | 45% PASS | FAIL |
| bioclip2   | la (0.6959)              | 0.7132 / 0.7120    | 0.105 FAIL      | 0.639 / 0.617      | 165:67 PASS | 13% PASS | FAIL |

Verdict (rules["verdict"]["no_primary_pass"]): Analysis paper (docs/PLAN_ANALYSIS.md); C2 reported as a negative
result on both datasets. Files: results/confirmatory_h22clean_{dinov2_b14,bioclip2,summary}.json.
Not decisive, reported: the class-stratified bootstrap CI of the macro-F1 delta is [+0.0004, +0.0036] (DINOv2) and
[-0.0000, +0.0024] (BioCLIP 2); C2's few-shot gain is large and taxonomy-specific (A9 passes on both encoders).

## H22-clean analyses on CPU, and session 3 started (2026-11-22)
Session 3 (h22-train Version 3, code 9886be9): clip_b32 alone, split over both GPUs (GPU 0: tau_norm, C2,
IJCNN configuration, A9; GPU 1: la, crt, ce, ltic3, ltic2), DEADLINE_MIN 480 (9.3 GPU-h left this week).
IJCNN configuration = LTIC-2 with the sh/R50.sh settings (lr 0.1, wd 1e-4, batch 256, 200 epochs, alpha 0.85,
gamma 0.3, epoch chosen by V-tune top-1), untuned, training seed 0: the corrected H22-clean number (B4).

Error taxonomy (tools/cf_report.py, test seed 0; compact files give the exact prediction, no tau sweep).
Cross-genus share of tail false positives, H22-clean vs Pl@ntNet (LA):

| encoder    | H22 la | H22 crt | H22 tau_norm | H22 tau_norm+c2f | H22 ltic3 | Pl@ntNet la |
|------------|--------|---------|--------------|------------------|-----------|-------------|
| dinov2_b14 | 58.2%  | 60.9%   | 57.6%        | 58.6%            | 63.3%     | 37.6%       |
| bioclip2   | 36.8%  | 37.5%   | 31.7%        | 33.9%            | 45.2%     | 38.8%       |

On herbarium sheets DINOv2's tail errors are mostly cross-genus (unlike on Pl@ntNet field photos); BioCLIP 2's
stay mostly within the genus on both datasets. Files: results/cf/h22clean_*.

Conformal sets (tools/conformal_eval.py with lt/conformal_topk.py: classes outside the stored top 20 get the
worst score, so coverage stays exact and sets can only grow). Test, alpha 0.1, PAS score:

| encoder / method      | CP-1 coverage, avg size | CP-2 (genus) coverage, size | true species in top 20 |
|-----------------------|-------------------------|-----------------------------|------------------------|
| bioclip2 la           | 0.900, 2.9 species      | 0.899, 31 leaves            | 96.8%                  |
| bioclip2 tau_norm+c2f | 0.898, 2.6              | 0.899, 30                   | 96.9%                  |
| dinov2_b14 la         | full set (truncated)    | 0.900, 65                   | 89.9%                  |
| dinov2_b14 tau_norm+c2f | 0.898, 16.8           | 0.899, 65                   | 89.9%                  |

DINOv2's true species is in the stored top 20 for only ~90% of images, so species-level sets at alpha <= 0.1 hit
the truncation. To do: re-predict la / tau_norm+c2f seed 0 with pred_topk 200 for DINOv2 (short GPU job).

## H22-clean session 3: CLIP B/32 and the IJCNN configuration (2026-11-22, 15:16-22:41 UTC)
h22-train Version 3 (code 9886be9), 445 min, stopped by DEADLINE_MIN 480. Done: 8/10 methods for clip_b32 plus
the IJCNN configuration. Not done (next session, PREV=auto): tau_norm+c2fshuf@c2_shuffle_seed=2 (~82 min) and
ltic2 (~158 min). GPU quota left this week: 0.42 h (resets 2026-11-28 00:00 UTC).
Logs: results/h22_logs/session3_gpu{0_clip_b32_c2,1_clip_b32_base}.log.

clip_b32, H22-clean (V-select macro-F1 picks; test is training seed 0):

| method                | V-select F1 | test F1 | test top-1 | test few (pooled) |
|-----------------------|-------------|---------|------------|-------------------|
| ltic3                 | 0.3043      | 0.3275  | 0.3610     | 0.3055            |
| tau_norm+c2f          | 0.2681      | 0.2815  | 0.3230     | 0.1944            |
| tau_norm+c2fshuf (s0) | 0.2674      | 0.2803  | 0.3223     | 0.1899            |
| tau_norm+c2fshuf (s1) | 0.2671      | 0.2808  | 0.3223     | 0.1910            |
| tau_norm              | 0.2656      | 0.2770  | 0.3183     | 0.1651            |
| ce                    | 0.2409      | 0.2400  | 0.3114     | 0.0343            |
| crt                   | 0.2240      | 0.2461  | 0.2785     | 0.2616            |
| la (tau 1.7)          | 0.1713      | 0.1739  | 0.2310     | 0.0541            |
| IJCNN config (LTIC-2) | -           | 0.2713  | 0.3011     | 0.3082            |

On CLIP B/32 the best method is LTIC-3 (+0.046 test macro-F1 over C2). Shuffled genera keep 73% / 84% of C2's
gain over tau_norm, so on CLIP the C2 gain is not taxonomic (A9 would fail at 0.5). Not part of the confirmatory
gate (DINOv2 and BioCLIP 2 only); reported as the third encoder. The IJCNN configuration gives the corrected
H22-clean number for the conference method: test macro-F1 0.2713, top-1 0.3011 (CLIP B/32, untuned, seed 0).

## H22-clean session 4: the last two CLIP B/32 methods (2026-11-23, 07:42-09:32 UTC)
Run on a groupmate's Kaggle account (prosenjitdasgupta/h22-train Version 1, code 383f5bc), 110 min; it attached
the session 3 output and skipped the 8 finished methods. clip_b32 now has 10/10 methods.

| method                | V-select F1 | test F1 (seed 0) | test top-1 | test few (pooled) |
|-----------------------|-------------|------------------|------------|-------------------|
| tau_norm+c2fshuf (s2) | 0.2639      | 0.2836           | 0.3178     | 0.2320            |
| ltic2                 | 0.2479      | 0.2730 (seeds 1, 2: 0.2751, 0.2640) | 0.2957 | 0.3227 |

Reported (not decisive) gate for clip_b32 (results/confirmatory_h22clean_clip_b32.json): best baseline ltic3;
C2 is 0.046 macro-F1 and 0.111 pooled few-shot accuracy behind it (FAIL); A9: the three shuffled runs keep 101%
of C2's gain over tau_norm and shuffle seed 2 beats real C2 (FAIL). On CLIP B/32 the C2 gain is not taxonomic.
Logs: results/h22_logs/session4_gpu{0_clip_b32_c2,1_clip_b32_base}.log.

## H22-clean session 5: top-200 re-prediction for conformal sets (2026-11-23, 16:12-16:39 UTC)
prosenjitdasgupta/h22-repred Version 1 (code be1129e, `MODE=repred`), 27 min on 2 T4s. Re-trained the chosen
configs of la, crt and tau_norm+c2f (seed 0, final stage) for dinov2_b14 and bioclip2 and saved the top 200
adjusted logits per image (results/preds_top200/, ~690 MB, gitignored; kept locally). All 12 re-trained rows
(3 methods x 2 encoders x vcal/test) reproduce the macro-F1 in all_runs_h22clean.csv exactly, so the earlier
results are deterministic for a fixed seed. Rows: results/repred_top200_{dinov2_b14,bioclip2}.csv.
Logs: results/h22_logs/session5_repred_gpu{0,1}.log.

True species in the stored top K (test): dinov2_b14 89.9% (top 20) -> 96.9% (top 200); bioclip2 96.8% -> 99.0%.

Conformal sets on test (PAS score, calibrated on V-cal; results/conformal/h22clean_*_test_seed0_top200_cp.csv).
Coverage / average set size; CP-1 = species sets, CP-2 = genus fallback (size in species):

| encoder / method        | CP-1 a=0.10     | CP-1 a=0.05     | CP-2 a=0.10    | CP-2 a=0.05    |
|-------------------------|-----------------|-----------------|----------------|----------------|
| dinov2_b14 la           | 0.899, 16.8     | 0.948, 54.6     | 0.899, 64.5    | 0.949, 114.0   |
| dinov2_b14 crt          | 0.900, 17.0     | 0.948, 53.4     | 0.899, 69.5    | 0.949, 129.3   |
| dinov2_b14 tau_norm+c2f | 0.899, 16.6     | 0.949, 52.3     | 0.899, 64.6    | 0.949, 115.3   |
| bioclip2 la             | 0.900, 2.9      | 0.950, 7.7      | 0.899, 31.1    | 0.950, 48.5    |
| bioclip2 crt            | 0.900, 2.8      | 0.950, 7.1      | 0.899, 31.6    | 0.950, 49.4    |
| bioclip2 tau_norm+c2f   | 0.898, 2.6      | 0.950, 6.3      | 0.899, 30.4    | 0.950, 46.9    |

With top 200 stored, DINOv2's species-level sets are no longer truncated: they reach the target coverage at both
alphas (the "full set" entry in the top-20 table above is resolved). The species sets of C2 (tau_norm+c2f) are
the smallest for each encoder, but the margin is small (16.6 vs 16.8 for DINOv2 at a=0.10; 2.6 vs 2.9 for
BioCLIP 2). Genus fallback (CP-2) always costs much larger sets for the same coverage. Coverage at a=0.05 for
DINOv2 is 0.948-0.949, slightly under 0.95, because 3.1% of true species lie outside the stored top 200.

Error taxonomy for clip_b32 on H22-clean (tools/cf_report.py, test seed 0, CPU; results/cf/h22clean_clip_b32_*).
Cross-genus share of tail false positives (number of tail FPs out of 84,759 test images):

| clip_b32 | la    | crt   | tau_norm | tau_norm+c2f | ltic3 | ltic2 | Pl@ntNet la |
|----------|-------|-------|----------|--------------|-------|-------|-------------|
| share    | 94.2% | 82.7% | 84.5%    | 81.3%        | 76.7% | 77.5% | 73.3%       |
| tail FPs | 5,723 | 11,446| 5,119    | 4,539        | 6,718 | 12,291| -           |

The order of encoders is the same for every method: CLIP B/32 has the most cross-genus tail errors, then DINOv2,
then BioCLIP 2. C2 has the fewest tail false positives for CLIP, but its cross-genus share stays above 80%, so
most of its remaining errors are not within the genus where the taxonomy could help.

## H22-clean taxonomy geometry and per-class C2 analysis (2026-11-24, CPU only)
dgpro06/h22-geometry Version 1 (code f5c4be3, Kaggle CPU, 4 cores, 72 min, no GPU quota):
tools/feature_geometry.py on the H22-clean train features of each encoder (view 0), with the real taxonomy and the
A9 shuffled taxonomy (seed 0) as control. Then tools/c2_class_analysis.py locally on the saved predictions
(tail classes, test + vcal pooled, training seeds 0-2). Files: results/geometry/, logs results/h22_logs/geometry_*.log.

Taxonomy geometry of the feature space (tail classes; "shuffled" = same measure with the shuffled taxonomy):

| encoder    | genus margin > 0 | nearest centroid in genus | kNN genus purity (k=10) | genus-donor subspace (r=32) vs global pool vs shuffled donors |
|------------|------------------|---------------------------|-------------------------|-----------------------------------------------------|
| bioclip2   | 86.2% (0.3%)     | 80.8% (0.3%)              | 0.590 (0.003)           | 0.558 vs 0.412 vs 0.343                             |
| dinov2_b14 | 66.1% (0.2%)     | 61.9% (0.2%)              | 0.264 (0.003)           | 0.581 vs 0.457 vs 0.387                             |
| clip_b32   | 51.3% (0.4%)     | 48.1% (0.4%)              | 0.107 (0.003)           | 0.659 vs 0.634 vs 0.594                             |

(subspace columns: median share of a class's own within-class variance inside the top-32 eigenvectors of its C2
donors' pooled covariance, 2,000 sampled classes with >= 50 images; random subspace 0.042 for 768-d, 0.062 for
512-d CLIP; the class's own top 32: 0.91.) The order BioCLIP 2 > DINOv2 > CLIP is the same on every measure and
matches the error taxonomy (cross-genus share of tail FPs 36.8% / 58.2% / 94.2% for LA). Genus donors add 0.15
(BioCLIP 2) and 0.12 (DINOv2) of variance over the global pool, but only 0.03 on CLIP.

Per tail class (2,833 classes), Spearman rho with bootstrap 95% CI, train_n partialled out gives the same values:

| encoder    | kNN purity -> base errors within genus | genus margin -> within genus | any geometry -> C2 gain (recall) | mean tail gain: total / taxonomic part |
|------------|----------------------------------------|------------------------------|----------------------------------|----------------------------------------|
| dinov2_b14 | 0.58 [0.55, 0.60]                      | 0.49 [0.46, 0.52]            | abs(rho) <= 0.04                  | +0.102 / +0.046                        |
| bioclip2   | 0.55 [0.51, 0.59]                      | 0.45 [0.41, 0.48]            | abs(rho) <= 0.05                  | +0.029 / +0.026                        |
| clip_b32   | 0.50 [0.47, 0.53]                      | 0.40 [0.37, 0.44]            | abs(rho) <= 0.03                  | +0.035 / -0.010                        |

(taxonomic part = C2 minus the mean of its three shuffled-taxonomy runs, training seed 0.)

Reading for the paper:
- B2 is supported at two levels: across encoders (the geometry order matches the error order) and within an
  encoder (classes whose neighbours are in their genus have their errors in the genus, rho 0.40-0.58 on all three).
- B3, per class: the geometry does NOT predict which tail classes C2 helps (all abs(rho) <= 0.05, several CIs touch
  0). Per encoder, the taxonomic part of C2's gain follows the donor-subspace advantage (largest where genus donors
  transfer, negative on CLIP, where they barely beat the global pool), but that is 3 points. C2 is reported as an
  encoder-dependent component, not as a per-class rule.
- Mistake severity (results/geometry/*_severity.csv, test seed 0): C2 does not make errors less severe; the mean
  taxonomic distance of errors is unchanged against tau_norm (DINOv2 0.901 vs 0.901, BioCLIP 2 0.474 vs 0.472,
  CLIP 1.319 vs 1.323). The encoder sets it: genus-correct share of errors 69% BioCLIP 2, 47% DINOv2, 25% CLIP.
