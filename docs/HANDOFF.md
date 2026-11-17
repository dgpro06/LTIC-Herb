# Handoff: LTIC-Herb journal extension, state on 2026-11-17

Read this first in a new session. Attach the project plan too: `LTIC_journal_plan_v9_final.md`
(plan v9.1; it is **not in the repo**, the user has it). Section numbers below (§) refer to that plan.
The forward plan from here is in [PLAN_NEXT.md](PLAN_NEXT.md).

---

## 1. Project in one paragraph
Journal extension of the IJCNN 2025 paper *LTIC-Herb* (long-tailed plant classification, frozen encoder +
3-branch residual head). The journal paper ("TaxoRes-LTIC") claims: **C-F** a finding (logit adjustment's
tail false positives often come from the wrong genus), **C-M** a method (C1 = frequency-dependent taxonomic
gate G-LP on log-probabilities; C2 = taxonomy-guided feature calibration; A9 shuffled-taxonomy control),
**C-E** evaluation (leakage-free H22 benchmark, corrected metrics, calibrated species-or-genus prediction
sets CP-1..CP-4, LIFT comparison). Datasets: Pl@ntNet-300K (development), H22-clean, iNat-2018.
Everything trains small heads on cached frozen-encoder features.

## 2. People, machines, accounts
- **User:** GitHub `dgpro06`, works on Windows 11 (`E:\LTIC-Herb`), laptop GPU RTX 4050 6 GB (not the team's
  4070; the 4070 machine has not been used yet). System Python 3.11 has **no torch**. Not deeply familiar with
  Kaggle/Git UIs: give click-by-click instructions. The PC sometimes sleeps/turns off.
- **Standing permission:** "push anything by yourself if anything needed to be fixed for the run". Other
  commits: ask first. **Never put Claude's name in a commit** (no `Co-Authored-By: Claude` trailer, no
  "Generated with Claude" line): the user asked for this on 2026-11-17.
- **Git:** branch `Dgpro`, tracks `fork/Dgpro` = https://github.com/dgpro06/LTIC-Herb (public; Kaggle clones it).
  Other remotes: `origin` = fussfreak/LTIC-Herb, `upstream` = Raiyan007-gb/LTIC-Herb (original authors).
  `.gitattributes` forces LF for `*.sh` (needed on Kaggle); `core.autocrlf=true` (CRLF warnings are harmless).
- **Kaggle:** account `dgpro06`, notebook **first_test** (https://www.kaggle.com/code/dgpro06/first-test),
  imported from `kaggle/LTIC_Week1_Kaggle.ipynb`. Dataset: **PlantNet300k by Gabriel Neto**
  (`gabrielneto583/plantnet300k`), mounted at `/kaggle/input/datasets/gabrielneto583/plantnet300k/plantnet_300K/`
  (contains `images/{train,val,test}/<species_id>/` and `plantnet300K_species_id_2_name.json`). The old
  `noahbadoa/plantnet-300k-images` copy is no longer listed. GPU T4 x2, Internet on. Quota ~30 GPU h/week;
  about 1.5 h interactive + Version 1 used so far.
- **Kaggle API token:** not set up (no `kaggle.json` found; the user's account may only show the newer on-screen
  token). Runs are started by the user in the browser.
- **Local test env:** throwaway venv in the session scratchpad (CPU torch, numpy, scipy, scikit-learn, pytest,
  pillow). It does not survive sessions; recreate with
  `python -m venv venv && venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && venv/Scripts/python -m pip install numpy scipy scikit-learn pytest pillow`.
  Then `python -m pytest tests -q` (32 tests, ~1–2 min). No open_clip locally: use the `debug_tiny` backbone.

## 3. Code map (all on `Dgpro`)
| Path | What |
|---|---|
| `lt/classes.py` | `ClassIndex` (stable-sort remap, rarest = 0), `ShotGroups` (few/medium/many + R1 asserts; raises if many-shot group empty), `read_split_file` |
| `lt/features.py` | chunked fp16 feature store: `<root>/<split>/<encoder>/chunk_*.pt` + `manifest.json`; labels stored as ORIGINAL split-file labels |
| `lt/heads.py` | bias-free heads: `LinearHead` (cosine option for LDAM), `LTICHead` (3 slices of 56, `reslt=True` = no MLP), `MultitaskHead` |
| `lt/losses.py` | log prior, post-hoc LA, LA loss / Balanced Softmax, LDAM-DRW, `ltic_loss` with branch masks (2 or 3 branches), `TaxoNetLoss` (dual margin, arXiv 2512.18994; baseline 11 = method `taxonet`) |
| `lt/trainer.py` | GPU-resident trainer; methods `ce, la, balanced_softmax, crt, tau_norm, ldam_drw, reslt, ltic2, ltic3, flat_genus`; epoch chosen by macro-F1 on V-tune |
| `lt/harness.py` | search spaces, 20-trial random search on V-tune, one V-select score, seeds on test+V-cal, `results/all_runs.csv`, saves raw logits to `results/preds/*.npz`; method suffix `@k=v` fixes a setting (e.g. `ltic3@gamma=0`) |
| `lt/metrics.py` | macro-F1 (sklearn semantics), per-class/group acc (per-class mean + pooled), bootstrap, paired bootstrap, exact McNemar |
| `lt/cf.py` | C-F buckets (same genus / same family / diff family / unknown), CIs, τ curve, count curve |
| `lt/diagnostics.py` | donor pooling, genus rescue (check b) + `genus_rescue_detail` (secondary), C2 covariance subspace test (genus-label permutation) |
| `lt/conformal.py` | CP-1..CP-4 with PAS score (checked 2026-11-17 against Ding et al.'s `example.ipynb`: identical species-level score and threshold; the genus-level score is our extension) |
| `lt/calibration.py` | **C2**: donors genus -> family -> global (`n_min`), shrunk mean, donor covariance (diag or rank r, built once at 64 and sliced), k seeded synthetic tail features per epoch. Method names `<base>+c2`, A9 control `<base>+c2shuf`; donor-level shares -> `results/c2/` |
| `lt/taxonomy.py` | `Taxonomy.from_csv` (output-index arrays), `shuffled()` for A9 |
| `tools/make_plantnet_splits.py` | official split files; finds dataset + species-name JSON anywhere under `--root` |
| `tools/make_val_parts.py` | V-tune/V-select/V-cal (stratified, golden-ratio assignment) + §8.0 auto-ratio rule |
| `tools/make_h22_clean.py` | H22-clean split from official `train_metadata.json` (or `--txt` dry run) + `taxonomy/h22.csv` |
| `tools/make_taxonomy_plantnet.py` | genus from name, family from GBIF (retries; `MANUAL_FAMILIES` for 11 unmatched genera; `family_source` column) |
| `tools/extract_features.py` | chunked, resumable, multi-encoder (one decode feeds all encoders), atomic writes; `--part K/N` (split chunks across accounts), `--deadline_min` (stop before Kaggle's 12 h) |
| `tools/rewrite_split_prefix.py` | rewrites the Kaggle path prefix of split files for another machine (e.g. the 4070) |
| `tools/run_baselines.py` | stages `search,vselect,final`; `--skip_done` for resuming |
| `tools/week1_diagnostics.py` | the §8.1 gate; recorded check (b) uses a balanced-softmax genus probe (5 trials); secondary section adds a plain-CE probe etc. |
| `tools/cf_report.py`, `tools/conformal_eval.py` | C-F report/figure; CP tables |
| `tools/week2_gate.py` | week-2 gate on V-select: best `+c2` vs best non-C2 baseline (paired bootstrap / McNemar, macro-F1 and pooled few-shot), A9 share of gain; run automatically by the Kaggle script when `BASELINES` has a `+c2` method; never overwrites |
| `kaggle/week1_kaggle.sh` + `kaggle/LTIC_Week1_Kaggle.ipynb` | whole pipeline on Kaggle; env: `ENCODERS, RUN_GATE, GATE_ENCODERS (default = ENCODERS), BASELINES, BASELINE_ENCODERS, TRIALS, EPOCHS, SEEDS, PREV ("auto" finds an attached earlier output)`; skips finished work; never overwrites a recorded gate JSON (re-run -> `*_rerun.json`) |
| `networks/backbones.py` | `clip_b32` (LAION, 512-d), `clip_b16` (OpenAI, **512-d**, not 768 as plan §11.1 says), `bioclip2` (768), `dinov2_b14` (768), `dinov2_l14` (1024), `debug_tiny` (tests); open_clip imported lazily |
| `CLIP_VIT_LONGTAIL.py` | legacy image-path entry point with all C0 fixes (3 branches, no bias, `--alpha`, explicit thresholds, macro-F1 epoch selection, class_map check, no distributed code) |
| `datasets/h22clean.py` | 15,501 classes, default val = `h22clean_vtune.txt` |
| `configs/*.json` | per dataset × encoder (Pl@ntNet: few 20 / many 100; H22: 20 / 50, 15,501 classes) |
| `gates/week1_thresholds.json` | §8.1 thresholds (committed in `914fac8`, dated 2026-11-07, before any diagnostic ran) |
| `kaggle/h22_kaggle.sh` + `kaggle/LTIC_H22_Kaggle.ipynb` | H22-clean split + features on Kaggle, one `PART` per account; `PREV=auto` merges every attached output |
| `WEEK1.md`, `KAGGLE_WEEK1.md`, `KAGGLE_H22.md` | runbooks (local, Kaggle Pl@ntNet sessions, Kaggle H22 for teammates) |

Commits since the plan started: `914fac8` week-1 code · `e2dae67` Kaggle notebook/script · `e5b585e` GBIF retry +
manual families · `88b6bb6` rebuild old taxonomy on resume · `d40829a` secondary (b) analyses, gate on every
encoder, `--skip_done`, `PREV=auto` · `413fd74` week-1 results · `bffcb21` UTF-8 everywhere · `293807b` splits +
taxonomy.

## 4. What has been run
Every Kaggle session, its settings and the files it produced are logged in [results/RUNS.md](../results/RUNS.md).
- **Session 1 (Nov 15):** splits, taxonomy, features for all 4 encoders; LA probe + gate + C-F + conformal on CLIP B/32.
- **Session 2 (Nov 16):** gate on all 4 encoders (+ secondary analyses of check b), C-F and conformal ×4,
  baselines 1–9 on CLIP B/32.
- **Session 3 (Nov 17):** baselines 1–9 on DINOv2 B/14 and BioCLIP 2. `all_runs.csv` has 918 rows.
- **Session 4 (Nov 17, started by the user):** baselines 1–9 on CLIP B/16.
Speed: ~52–62 img/s extraction (decode-bound); 12–15 s per linear-probe run, 33–52 s for LTIC/ResLT/LDAM.

## 5. Results so far (Pl@ntNet-300K; full tables in results/RUNS.md)
**Data facts.** Splits 243,916 / 31,118 / 31,112, 1,081 classes; 523 few / 266 medium / 292 many; 303 genera
(164 with one species), 112 families. §8.0: ratios 35/40/25, week-2 few-shot criterion = pooled image-level.
- **Gate (V-select), verdict "keep C2, drop C1" on all 4 encoders.** C2 covariance p = 0.001 everywhere. Check (b)
  fails everywhere (0.07–0.27). Check (a) passes on the CLIPs (0.77–0.80) and DINOv2 (0.42) but **fails on
  BioCLIP 2 (0.253 < 0.30)**.
- **C-F (test):** cross-genus share of LA tail FPs 0.73 (CLIP B/32), 0.81 (B/16), **0.38 (DINOv2), 0.39 (BioCLIP 2)**.
  The "3/4 cross-genus" claim holds for CLIP only; strong encoders keep most tail errors inside the genus.
- **Baselines (test macro-F1, 3 seeds):** CLIP B/32: LTIC-3 (γ=0) best, 0.343. DINOv2: cRT 0.503 > LA 0.500 >
  LTIC-3 0.493. BioCLIP 2: τ-norm 0.553 > cRT 0.540 > LTIC-3 0.515. **LTIC-3 does not lead on strong encoders**;
  LTIC-2 (IJCNN code) is last everywhere. So C2 is tested on top of the best base per encoder, not only LTIC-3.
- **Conformal (α = 0.1):** CP-1 average set 141–158 species on CLIP, 5.9 (DINOv2) / 8.5 (BioCLIP 2), median 3.

## 6. Open caveats and known issues
1. **Check (b):** the recorded probe uses the balanced loss; secondary plain-CE probe gives 0.29–0.36 (still < 0.5).
   Report the recorded FAIL as primary.
2. **C-F τ curve:** redraw from the CE baseline. Session 5 does it (`CF_FROM_CE`).
3. **SMART (baseline 10) is not implemented.** Its definition is in plan v9.1 (not in the repo); no paper named
   SMART was found online. Needs the plan's reference before it can be written.
4. **TaxoNet** is reconstructed from the paper: s = 32 fixed, m0 searched (0.05–0.35), λ searched (not given in the
   paper), ρ scaled by sum (as the paper states) or max (option: with 1,000+ classes the sum scaling makes the
   margins tiny). No AugMix / norm-guided sampling (images not available at feature level). Say so in the paper.
5. **C2 on H22:** ~2,500 genera -> ~2,500 donor sets; rank-64 statistics take ~0.2 GB each (float32); the harness
   caches up to 6 (n_min × diag/low-rank × A9). Watch GPU memory on the first H22 C2 run.
6. Plan discrepancy: CLIP B/16 projected features are 512-d (plan §11.1 lists 768).
7. Not done: SMART, LIFT, iNat, H22 runs (extraction ready: KAGGLE_H22.md), ablations, corrected-evaluation table,
   README (still shows the leaky IJCNN numbers and the `presizer.py` step).
8. Schedule (PLAN_NEXT): week-2 gate Sat Nov 21, freeze Sat Nov 28, hand-over Mon Nov 30.

## 7. Pitfalls already hit (avoid repeating)
- Kaggle **interactive** sessions die when the user's PC sleeps: always use **Save Version → Save & Run All (Commit)**;
  then stop the draft session (⏻) to save quota. Clicking Stop on a `!bash` cell may leave the process running.
- In Kaggle's Add Input search, `< >` icons are **notebooks**, not datasets; filter by Datasets.
- Browser downloads from Kaggle often save `.csv`/`.json` as `.txt`: rename after download.
- Windows default encoding is not UTF-8: every text `open()` now passes `encoding='utf-8'` (keep it that way).
- Decoding is the Kaggle bottleneck: extract all encoders in one pass (`ENCODERS` = all four).
- The scratch venv can get corrupted between sessions (Windows temp cleanup): recreate it.

## 8. Immediate next actions
1. **User:** when session 4 (CLIP B/16 baselines) finishes, download `results/all_runs.csv` and
   `results/configs/plantnet_clip_b16_*.json` (rename `.txt` back), commit, add a row to `results/RUNS.md`.
2. **User, Kaggle session 5** (KAGGLE_WEEK1.md "Later sessions"): attach session 4's Output, `PREV = "auto"`,
   `RUN_GATE = False`, `ENCODERS = BASELINE_ENCODERS = "dinov2_b14,bioclip2,clip_b32"`,
   `BASELINES = "crt+c2,crt+c2shuf,tau_norm+c2,la+c2,ltic3+c2,ltic3+c2shuf,taxonet"`, `CF_FROM_CE = True`.
   Gate input for Nov 21: C2 vs the best baseline on V-select (paired bootstrap / McNemar), real vs shuffled C2.
3. **Teammates:** H22 extraction, 3 accounts with `PART = 0/3, 1/3, 2/3` (KAGGLE_H22.md), then one merge notebook.
4. **Supervisor:** decisions D1–D6 (PLAN_NEXT §2) plus the two new facts: BioCLIP 2 fails check (a); LTIC-3 is not
   the best base on strong encoders. Ask for SMART's reference.
5. **Next code:** SMART (once defined), LIFT. The week-2 gate prints at the end of session 5's log (`results/week2_gate_plantnet_<encoder>.json`).
