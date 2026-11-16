# TaxoRes-LTIC: revised plan for weeks 2–4 (from Mon 2026-11-16)

Revises plan v9.1 (`LTIC_journal_plan_v9_final.md`, § numbers below) after the week-1 results. The state of the
code and every number referred to here are in [HANDOFF.md](HANDOFF.md). Final date unchanged: **hand-over
Mon Nov 30**.

---

## 1. Where we stand
- **Done:** all week-1 code (C0 fixes, extractor, harness, baselines 1–9, C-F, gate, CP-1..CP-4), Pl@ntNet splits and
  taxonomy, features for 4 encoders on Kaggle, and the week-1 gate on CLIP B/32.
- **Week-1 gate (CLIP B/32): keep C2, drop C1.** Donor pooling, cross-genus share and the C2 covariance test pass.
  Check (b) fails: a genus classifier recovers the true genus on only 7% of the cross-genus tail errors.
- **C-F is confirmed:** 73–77% of logit adjustment's tail false positives are cross-genus, two thirds cross-family,
  on V-select and test.
- **Behind schedule by ~9 days:** the week-2 gate (Nov 14) has not happened; C1/C2, SMART, TaxoNet, LIFT, H22 and
  iNat have not started.

## 2. Decisions for the gate meeting (supervisor), Tue Nov 17
| # | Decision | Recommendation | Why |
|---|---|---|---|
| D1 | C1 (G-LP gate) | **Drop as a method**, unless session 2 shows check (b) ≥ 50% on a stronger encoder. Keep a short negative ablation (rows a, d, f of A2) as evidence | The rule fixed in advance says drop; the errors are in the features, so a decision-time veto cannot fix them |
| D2 | Method contribution | **C2 on top of LTIC-3** ("taxonomy-guided feature calibration for long-tailed plant recognition on frozen features"), with A9 | C2's preconditions pass strongly; the story "cross-genus errors are baked into the features, so fix the features" follows from the data |
| D3 | Venue target | §17 row 2 (e.g. *Ecological Informatics*, *Expert Systems with Applications*); revisit after the week-2 gate | One week-1 check failed |
| D4 | New dates | Week-2 gate **Sat Nov 21**, week-3 checkpoint **Wed Nov 25**, freeze **Sat Nov 28**, hand-over **Mon Nov 30** | Keeps the final date by compressing analysis |
| D5 | Cuts now | Cut-list items 1–4 (§14): CP-5, A12 multi-view, LIFT on H22/iNat, **iNat reduced to the Plantae subset with 4 baselines** | The Nov 14 trigger has fired; iNat download and extraction have not started |
| D6 | A2 in "never cut" | Reduce A2 to rows a, d, f, h if C1 is dropped | A2 was required only because C1 was the main method |

Bring: the session-2 gate summary for all 4 encoders, the C-F figure, and this table.

## 3. What the paper claims after the pivot
1. **C-F (finding):** LA's tail gains come with tail false positives, ~3/4 cross-genus and ~2/3 cross-family, across
   encoders and datasets; a genus classifier on the same features does not recover them (negative result, check b).
2. **C-M (method):** LTIC-3 + C2 (genus → family → global donors, shrunk mean, borrowed low-rank covariance), with
   the shuffled-taxonomy control A9 showing the gain comes from taxonomy.
3. **C-E (evaluation):** leakage-free H22-clean benchmark with corrected metrics, CP-1..CP-4 prediction sets with
   genus fallback, all on free/consumer hardware.

## 4. Timeline

### Week 2′: Mon Nov 16 – Sat Nov 21
| Day | Track A (engineering) | Track B (baselines, analysis) | Track C (data, evaluation) |
|---|---|---|---|
| Mon 16 | Start **Kaggle session 2** (gate on 4 encoders + baselines 1–9 on CLIP B/32) | Implement **SMART** (feature level) | Verify the **PAS** score against Ding et al.'s code |
| Tue 17 | **Gate meeting** (D1–D6). Start **C2** (`lt/calibration.py`) | Implement the **TaxoNet loss** (feature level) | H22: accept the competition rules on Kaggle; build the H22-clean split from `train_metadata.json` on a **teammate's account** |
| Wed 18 | C2 unit tests (donor levels, shrinkage, sampling seed); integrate as an option on `ce/la/ltic3` | Baselines 1–11 on **BioCLIP 2 and DINOv2 B/14** (Kaggle session 3) | Start **H22 extraction** (3 encoders, chunked, teammate accounts) |
| Thu 19 | C2 search space in the harness; **A9 for C2** (`Taxonomy.shuffled`) | Redraw the **C-F τ curve from the CE baseline**; C-F on all 4 encoders | Conformal on the strongest encoder; also try the softmax score if sets stay huge |
| Fri 20 | C2 tuning on V-tune (all 4 encoders, Kaggle session 4) | LIFT: official code, 1-epoch smoke test (needs the 4070 or Kaggle) | Correction write-up (§12) draft |
| **Sat 21** | **Week-2 gate** (criteria in §5) | Present the baseline tables | Present C-F across encoders + conformal |

### Week 3′: Sun Nov 22 – Wed Nov 25
| Track A | Track B | Track C |
|---|---|---|
| H22-clean: final method + required baselines, 3 seeds (Kaggle) | Ablations on Pl@ntNet: A1, A2 (reduced), A3, A4–A8, A10 | C-F + conformal on H22; donor analysis on H22 |
| LIFT on Pl@ntNet, CLIP B/16 (3 seeds on the 4070, or 1 seed on Kaggle and disclose) + A11 | iNat **Plantae** subset: LA, SMART, TaxoNet, LTIC-3, final method (if the download is done by Mon 23; else drop and say so) | Corrected-evaluation table: IJCNN configuration (original commit) re-run on H22-clean |
| **Wed 25: week-3 checkpoint**: every required number in `results/all_runs.csv`; else apply cut items 5–6 | | |

### Week 4′: Thu Nov 26 – Mon Nov 30
- Thu 26 – Fri 27: figures and tables generated by script from the CSV (C-F first), significance tests, efficiency
  table, failure gallery (singletons without donors, monotypic genera).
- **Sat 28: results freeze.**
- Sun 29 – Mon 30: reproducibility package (clean branch, configs per table), data card, every number cross-checked,
  README rewritten (corrected numbers, no presizer step). **Hand-over Mon Nov 30.**

## 5. Week-2 gate (Sat Nov 21), adapted to the pivot
On **V-select** (never V-tune, never test), every method tuned with the same budget on V-tune:
1. **LTIC-3 + C2 beats the best tuned baseline** on few-shot accuracy (**pooled image-level**, per §8.0) and on
   macro-F1: paired bootstrap 95% CI excludes 0, or McNemar p < 0.05.
2. **A9:** real-taxonomy C2 beats shuffled-taxonomy C2 (same test), and the shuffled run keeps **< 50%** of the gain
   (threshold `a9_max_gain_retained` in `gates/week1_thresholds.json`).
3. Report the seed spread (not a gate criterion).

If it fails: the paper becomes the C-F + C-E benchmark/analysis paper (§17 row 3), and C2 is reported as a modest
component. Decide this on Nov 21, not later.

## 6. C2 implementation spec (from §3.3; Track A)
- **Where:** raw frozen-encoder feature space, before the head; train split statistics only.
- **Donors for a tail species s:** same-genus species with ≥ `n_min` (20) images → else same family → else global
  pool. Report the share at each level. Pl@ntNet: 18.5% of tail species have no congeners.
- **Mean:** `(n_s μ_s + τ_μ μ_g) / (n_s + τ_μ)`, μ_g = count-weighted donor mean excluding s.
- **Covariance:** count-weighted average of donors' diagonal or rank-r (r ∈ {8, 32, 64}) covariances.
- **Sampling:** k synthetic features per tail species per epoch, re-sampled every epoch, seeded. Singletons without
  any donor: no synthesis (count them).
- **Search space:** τ_μ, k, r (diag / 8 / 32 / 64), `n_min`, plus the base method's own parameters.
- **A9:** the same code with `Taxonomy.shuffled(seed)`; τ and all C2 parameters re-tuned.
- **Done when:** unit tests pass (donor selection, shrinkage, sampling determinism); runnable from a method name
  (e.g. `ltic3+c2`); A4–A6 rows runnable from config.

## 7. Compute budget
| Job | Where | Estimate |
|---|---|---|
| Session 2: gate × 4 encoders + baselines 1–9 on CLIP B/32 | Kaggle (user) | 2–4 h |
| Baselines 1–11 on BioCLIP 2, DINOv2, CLIP B/16 | Kaggle | ~1–3 h per encoder (~14–40 s per run) |
| C2 tuning + A9 + ablations on Pl@ntNet | Kaggle | ~4–8 h in total |
| H22 split + extraction, 3 encoders (~1.05M images) | Kaggle, **teammates' accounts** | 12–20 h (decode-bound, ~60 img/s per session) |
| H22 runs (15,501 classes) | Kaggle | ~4–8 h |
| LIFT on Pl@ntNet, 3 seeds | 4070 (7–9 h) or Kaggle (~20 h, likely 1 seed) | decide by Fri 20 |
| iNat Plantae: download + extraction + runs | 4070 or Kaggle | only if the download finishes by Mon 23 |
The user's own quota (~30 h/week) cannot cover H22 extraction on top of the Pl@ntNet runs: H22 must run on
teammates' accounts (one account per person, §11.2).

## 8. Risks
| Risk | Signal | Response |
|---|---|---|
| C2 gain not significant | Week-2 gate fails on Nov 21 | Switch to the benchmark/analysis paper (§17 row 3); C-F + C-E carry it |
| Too few tail images in V-select (305) | Wide CIs on few-shot | Pooled criterion (already decided); lean on macro-F1 and test-set CIs |
| Conformal sets stay ~100+ species on strong encoders | CP tables on BioCLIP 2 | Report as a limitation; try softmax/APS scores; keep CP-3's guarantee as the contribution |
| H22 extraction too slow | < 50% by Fri 20 | Pre-resize images once; more accounts; one encoder (DINOv2 B/14 is required, §5.1) |
| No 4070 time for LIFT | Not started by Fri 20 | 1 seed on Kaggle, disclosed; A11 on that seed only |
| Kaggle quota exhausted | < 5 h left | Teammates' accounts; `--skip_done` resumes cleanly |

## 9. Small fixes that must happen before results are reported
- PAS score verified (or replaced) in `lt/conformal.py`.
- C-F τ curve from the CE baseline, not the LA-loss model.
- Plan §11.1: CLIP B/16 is 512-d; recompute its parameter counts.
- A prefix-rewrite tool for split files if Pl@ntNet runs move to the 4070.
- README: remove the leaky IJCNN table and the presizer step; add the corrected numbers (§12).
