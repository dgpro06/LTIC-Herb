# Handoff: LTIC-Herb journal extension, state on 2026-11-16

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
  Then `python -m pytest tests -q` (21 tests, ~1 min). No open_clip locally: use the `debug_tiny` backbone.

## 3. Code map (all on `Dgpro`)
| Path | What |
|---|---|
| `lt/classes.py` | `ClassIndex` (stable-sort remap, rarest = 0), `ShotGroups` (few/medium/many + R1 asserts; raises if many-shot group empty), `read_split_file` |
| `lt/features.py` | chunked fp16 feature store: `<root>/<split>/<encoder>/chunk_*.pt` + `manifest.json`; labels stored as ORIGINAL split-file labels |
| `lt/heads.py` | bias-free heads: `LinearHead` (cosine option for LDAM), `LTICHead` (3 slices of 56, `reslt=True` = no MLP), `MultitaskHead` |
| `lt/losses.py` | log prior, post-hoc LA, LA loss / Balanced Softmax, LDAM-DRW, `ltic_loss` with branch masks (2 or 3 branches) |
| `lt/trainer.py` | GPU-resident trainer; methods `ce, la, balanced_softmax, crt, tau_norm, ldam_drw, reslt, ltic2, ltic3, flat_genus`; epoch chosen by macro-F1 on V-tune |
| `lt/harness.py` | search spaces, 20-trial random search on V-tune, one V-select score, seeds on test+V-cal, `results/all_runs.csv`, saves raw logits to `results/preds/*.npz`; method suffix `@k=v` fixes a setting (e.g. `ltic3@gamma=0`) |
| `lt/metrics.py` | macro-F1 (sklearn semantics), per-class/group acc (per-class mean + pooled), bootstrap, paired bootstrap, exact McNemar |
| `lt/cf.py` | C-F buckets (same genus / same family / diff family / unknown), CIs, τ curve, count curve |
| `lt/diagnostics.py` | donor pooling, genus rescue (check b) + `genus_rescue_detail` (secondary), C2 covariance subspace test (genus-label permutation) |
| `lt/conformal.py` | CP-1..CP-4 with PAS score (**PAS formula from memory: TODO verify** against github.com/tiffanyding/long-tail-conformal) |
| `lt/taxonomy.py` | `Taxonomy.from_csv` (output-index arrays), `shuffled()` for A9 |
| `tools/make_plantnet_splits.py` | official split files; finds dataset + species-name JSON anywhere under `--root` |
| `tools/make_val_parts.py` | V-tune/V-select/V-cal (stratified, golden-ratio assignment) + §8.0 auto-ratio rule |
| `tools/make_h22_clean.py` | H22-clean split from official `train_metadata.json` (or `--txt` dry run) + `taxonomy/h22.csv` |
| `tools/make_taxonomy_plantnet.py` | genus from name, family from GBIF (retries; `MANUAL_FAMILIES` for 11 unmatched genera; `family_source` column) |
| `tools/extract_features.py` | chunked, resumable, multi-encoder (one decode feeds all encoders), atomic writes |
| `tools/run_baselines.py` | stages `search,vselect,final`; `--skip_done` for resuming |
| `tools/week1_diagnostics.py` | the §8.1 gate; recorded check (b) uses a balanced-softmax genus probe (5 trials); secondary section adds a plain-CE probe etc. |
| `tools/cf_report.py`, `tools/conformal_eval.py` | C-F report/figure; CP tables |
| `kaggle/week1_kaggle.sh` + `kaggle/LTIC_Week1_Kaggle.ipynb` | whole pipeline on Kaggle; env: `ENCODERS, RUN_GATE, GATE_ENCODERS (default = ENCODERS), BASELINES, BASELINE_ENCODERS, TRIALS, EPOCHS, SEEDS, PREV ("auto" finds an attached earlier output)`; skips finished work; never overwrites a recorded gate JSON (re-run -> `*_rerun.json`) |
| `networks/backbones.py` | `clip_b32` (LAION, 512-d), `clip_b16` (OpenAI, **512-d**, not 768 as plan §11.1 says), `bioclip2` (768), `dinov2_b14` (768), `dinov2_l14` (1024), `debug_tiny` (tests); open_clip imported lazily |
| `CLIP_VIT_LONGTAIL.py` | legacy image-path entry point with all C0 fixes (3 branches, no bias, `--alpha`, explicit thresholds, macro-F1 epoch selection, class_map check, no distributed code) |
| `datasets/h22clean.py` | 15,501 classes, default val = `h22clean_vtune.txt` |
| `configs/*.json` | per dataset × encoder (Pl@ntNet: few 20 / many 100; H22: 20 / 50, 15,501 classes) |
| `gates/week1_thresholds.json` | §8.1 thresholds (committed in `914fac8`, dated 2026-11-07, before any diagnostic ran) |
| `WEEK1.md`, `KAGGLE_WEEK1.md` | runbooks (local and Kaggle) |

Commits since the plan started: `914fac8` week-1 code · `e2dae67` Kaggle notebook/script · `e5b585e` GBIF retry +
manual families · `88b6bb6` rebuild old taxonomy on resume · `d40829a` secondary (b) analyses, gate on every
encoder, `--skip_done`, `PREV=auto` · `413fd74` week-1 results · `bffcb21` UTF-8 everywhere · `293807b` splits +
taxonomy.

## 4. What has been run
**Kaggle Version 1 (2026-11-15, completed):** splits, V-parts, taxonomy, features for
`clip_b32,clip_b16,bioclip2,dinov2_b14` (*verify*: `feats/plantnet/train/` in that output should have 4 encoder
folders; the settings were changed to all four just before Save Version), LA probe + C-F + gate + conformal on
**CLIP B/32 only**. Extraction is CPU-decode-bound on Kaggle: **~52–62 img/s**. Head training: **~14 s per run**
(max 28 s); 20-trial LA search = 8.5 min.

Committed outputs: `results/` (gate JSON, `all_runs.csv`, `cf/`, `conformal/`, `configs/`), `splits/`,
`taxonomy/`. Not committed (stay in the Kaggle output): `feats/`, `results/preds/`.

## 5. Results so far (Pl@ntNet-300K, CLIP B/32)
**Data facts.** Splits 243,916 / 31,118 / 31,112, 1,081 classes; train counts min 2, median 21, max 7,208.
Shot groups (train): **523 few / 266 medium / 292 many**. 303 genera (**164 with one species**), 112 families.
**§8.0:** V-select had 305 few-shot images (< 500) even after switching, so ratios are **35/40/25** (V-tune 10,907 /
V-select 12,453 / V-cal 7,758) and the week-2 few-shot criterion uses **pooled image-level** tail accuracy.

**Week-1 gate (V-select):**
| Check | Value | Threshold | |
|---|---|---|---|
| Donor pooling median | 58.5 (quantiles 10/25/50/75/90: 0, 1.9, 58.5, 627, 1684; 18.5% of tail species have no congeners) | ≥ 3 | PASS |
| (a) cross-genus share of LA tail FPs | 0.772 [0.707, 0.835], n = 167 (diff family 0.653) | ≥ 0.30 | PASS |
| (b) genus probe right on cross-genus tail FPs | 0.070 (9 / 129); genus probe macro-F1 0.377 | ≥ 0.50 | **FAIL** |
| C2 covariance (k = 10, 558 species, 2,631 same-genus pairs) | same-genus 0.385 vs random 0.251, p = 0.001 | < 0.05 | PASS |
Verdict (rule fixed in advance): **only C2 supported: keep C2, drop C1.** Not yet discussed with the supervisor.

**C-F on test:** 386 tail FPs / 31,112 images; cross-genus **0.733 [0.688, 0.777]**; diff family 0.663; the
cross-genus share is 0.70–0.77 in every bin of the predicted tail class's train count (3–5, 6–10, 11–19).

**LA probe (test, seeds 0–2):** top-1 69.8–70.0%, macro-F1 0.304–0.307, few 12.5–13.1% (pooled 15.7–16.7%),
medium 44.7%, many 65.8%. Chosen config: `la_mode = loss`, `loss_tau = 0.64`, post-hoc `tau = 0`, lr 0.155,
batch 256.

**Conformal (test, α = 0.1):** CP-1 coverage 0.901 (macro 0.970) but average set **~141 species**; CP-3 coverage
0.952, fallback to genus ~0.97, set ~120 species in leaves. Guarantees hold; sets too big to be useful yet.

## 6. Open caveats and known issues
1. **Check (b) caveat:** the recorded probe uses the balanced (logit-adjusted) loss, which also favours rare genera,
   so it may share LA's bias; n = 129; CLIP B/32 is the weakest encoder. Secondary analyses (plain-CE probe,
   "same wrong genus" share, true-group split) and the gate on the other 3 encoders are implemented
   (`d40829a`) but **not yet run**. Report the recorded FAIL as primary; secondary analyses as sensitivity only.
2. **C-F τ curve:** the LA probe picked the loss form, so the saved τ is 0 and `cf_tau.csv` shows extra post-hoc τ on
   top of an already-adjusted model. Redraw the paper's "share vs τ" figure from the **CE** baseline
   (`cf_report.py --preds ..._ce_test_seed0.npz --tau_max 2`).
3. **PAS score** in `lt/conformal.py` must be checked against Ding et al.'s code before reporting conformal numbers.
4. **Split paths** carry Kaggle's prefix (`datasets/gabrielneto583/plantnet300k/plantnet_300K/images/...`); using them
   on another machine needs that folder structure or a prefix-rewrite tool (not written).
5. Plan discrepancy: CLIP B/16 projected features are 512-d (plan §11.1 lists 768).
6. Not done from the plan: SMART, TaxoNet loss (baselines 10–11), LIFT, iNat-2018, any H22 run (Kaggle H22 metadata
   split + extraction), C1/C2 implementation, A9, ablations, corrected-evaluation table, README update (still shows
   the leaky IJCNN numbers and tells users to run `presizer.py`).
7. Schedule: the plan's week-1 gate was Nov 7 and week-2 gate Nov 14; the project is ~9 days behind.

## 7. Pitfalls already hit (avoid repeating)
- Kaggle **interactive** sessions die when the user's PC sleeps: always use **Save Version → Save & Run All (Commit)**;
  then stop the draft session (⏻) to save quota. Clicking Stop on a `!bash` cell may leave the process running.
- In Kaggle's Add Input search, `< >` icons are **notebooks**, not datasets; filter by Datasets.
- Browser downloads from Kaggle often save `.csv`/`.json` as `.txt`: rename after download.
- Windows default encoding is not UTF-8: every text `open()` now passes `encoding='utf-8'` (keep it that way).
- Decoding is the Kaggle bottleneck: extract all encoders in one pass (`ENCODERS` = all four).
- The scratch venv can get corrupted between sessions (Windows temp cleanup): recreate it.

## 8. Immediate next action
Start **session 2** on Kaggle (instructions in `KAGGLE_WEEK1.md` §Session 2): attach Version 1's output
(Input → Add Input → Your Work → first_test), keep PlantNet300k attached, settings
`ENCODERS = "clip_b32,clip_b16,bioclip2,dinov2_b14"`, `RUN_GATE = True`,
`BASELINES = "ce,balanced_softmax,crt,tau_norm,ldam_drw,reslt,ltic2,ltic3,ltic3@gamma=0,flat_genus"`,
`BASELINE_ENCODERS = "clip_b32"`, `TRIALS = 20`, `PREV = "auto"`. Expected 2–4 h (more if Version 1 did not extract
all four encoders). Then read the per-encoder gate summary at the end of the log, download the new files
(`results/week1_gate_*`, `*_rerun.json`, `all_runs.csv`, `cf/`, `conformal/`, `configs/`), rename `.txt` →
`.csv`/`.json`, validate, commit.
