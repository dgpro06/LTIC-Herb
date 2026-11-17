# Run log

Which Kaggle run produced which file. Notebook: https://www.kaggle.com/code/dgpro06/first-test
(GPU T4 x2, dataset `gabrielneto583/plantnet300k`). All on Pl@ntNet-300K; gate checks on V-select, C-F and
baselines on test (3 seeds) unless noted.

| Session | Kaggle version | Date | Code commit | Settings | Produced |
|---|---|---|---|---|---|
| 1 | Version 1 | 2026-11-15 | `e2dae67` | `ENCODERS` = all four, `RUN_GATE` on (gate on the first encoder only at that code version) | splits, taxonomy (pre-fix), features ×4, LA probe + gate + C-F + conformal on **clip_b32** → `week1_gate_plantnet_clip_b32.json` (the recorded gate result; never overwritten) |
| 2 | Version 2 "session-2" | 2026-11-16 | `ea25be9` | `PREV=auto`, gate on all four, baselines 1–9 on clip_b32, 20 trials, seeds 0–2 | `week1_gate_plantnet_{clip_b16,dinov2_b14,bioclip2}.json`, `week1_gate_plantnet_clip_b32_rerun.json` (adds the secondary analyses), all `cf/` files (clip_b32 redone with the completed taxonomy), `conformal/` ×4, `configs/` ×14, `all_runs.csv` (378 rows, includes session 1's 27) |

Session 1's original C-F files and `all_runs.csv` for clip_b32 are in commit `413fd74`.

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

**Conformal (test, α = 0.1), average set size in species:** CP-1: clip_b32 141.5, clip_b16 157.6, dinov2_b14 5.9,
bioclip2 8.5 (median 3 for both). CP-3 (K = 5): coverage 0.94–0.95, fallback 0.97 (CLIP) vs 0.35 (DINOv2, BioCLIP 2).

**Run time:** 12–15 s per linear-probe run, 33–52 s for LTIC/ResLT/LDAM/flat-genus, up to 112 s; 201 GPU-minutes of
head training in session 2.
