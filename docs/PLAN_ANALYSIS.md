# TaxoRes-LTIC after the week-2 gate: revised plan, Wed Nov 18 – Mon Nov 30

Replaces the method track of [PLAN_NEXT.md](PLAN_NEXT.md) (kept as the record of the gate rules). Code state and
earlier numbers: [HANDOFF.md](HANDOFF.md), [results/RUNS.md](../results/RUNS.md). Final date unchanged:
**hand-over Mon Nov 30**.

---

## 1. Why the plan changes
The week-2 gate (PLAN_NEXT §5, Kaggle session 5, V-select) **fails**:

| Encoder | Best baseline | Best C2 | Macro-F1 Δ (95% CI) | Pooled few-shot Δ (95% CI) | A9 | Verdict |
|---|---|---|---|---|---|---|
| DINOv2 B/14 | ldam_drw 0.542 | tau_norm+c2 0.547 | +0.006 [−0.017, +0.008] | **+6.6 pts [+3.3, +9.8]** | not run | **FAIL** (corrected, see below) |
| BioCLIP 2 | tau_norm 0.591 | la+c2 0.588 | −0.004 [−0.008, +0.020] | **−4.6 pts [−8.2, −1.0]** | not run | FAIL |
| CLIP B/32 | ltic3@γ=0 0.373 | ltic3+c2 0.355 | **−0.018 [−0.031, −0.004]** | −1.3 pts [−5.9, +3.3] | fails | FAIL |

- **Corrected 2026-11-18:** on DINOv2 the script marked macro-F1 "pass" only through McNemar, and that McNemar
  result favours the **baseline** (V-select top-1 0.842 ldam_drw vs 0.828 tau_norm+c2): the script did not check
  the direction (bug, fixed in `tools/week2_gate.py`). The bootstrap CI on macro-F1 includes 0, so criterion 1 fails
  and **all three encoders fail**; the missing A9 run cannot change that. Details in `results/RUNS.md`.
- The plan's literal criterion is **LTIC-3 + C2**. LTIC-3 is not the best base on DINOv2 or BioCLIP 2, so that
  reading fails as well (to be confirmed from `all_runs.csv`).
- By the rule fixed on Nov 16: **the paper becomes the benchmark/analysis paper; C2 is a reported component, not
  the contribution.** We recommend recording this decision now rather than on Nov 21, to free three days.

We do **not** start a new method. With 12 days left and no week for a new gate, a new method would be untested
when the freeze comes. The analysis paper below rests on results we already have, and each remaining run sharpens
it rather than deciding whether there is a paper at all.

## 2. What the data already says
From sessions 1–5 (Pl@ntNet-300K, test macro-F1, mean of 3 seeds, 20-trial search per method):

1. **The encoder matters far more than the long-tail method.** The best method per encoder scores 0.343 (CLIP
   B/32), 0.359 (CLIP B/16), 0.503 (DINOv2), 0.553 (BioCLIP 2), a gap of **+0.21** from encoder alone. Among the top
   five methods on one encoder the spread is 0.01–0.04.
2. **On strong encoders, simple wins.** A tuned logit-adjusted linear probe is within 0.003 (DINOv2) and 0.017
   (BioCLIP 2) of the best of 11 methods. cRT / τ-norm / LA lead; specialised heads (LTIC-2/3, ResLT, LDAM-DRW)
   do not pay off. LTIC-3 leads only on the CLIP encoders (+0.014–0.016 over cRT). LTIC-2 (IJCNN code) is last on
   every encoder.
3. **Where the errors fall in the taxonomy depends on the encoder.** Cross-genus share of LA's tail false positives:
   0.73 / 0.81 (CLIP B/32 / B/16) vs 0.38 / 0.39 (DINOv2 / BioCLIP 2). On a genus probe, those errors are not
   recoverable (check b: 0.07–0.27). The errors are in the features, and strong encoders move them **inside the genus**.
4. **Genus fallback is useful exactly where errors are cross-genus.** Conformal CP-1 sets: 141–158 species (CLIP)
   vs 5.9 / 8.5 (DINOv2 / BioCLIP 2). CP-3 falls back to genus for 97% of images on CLIP and 35% on strong
   encoders.
5. **Taxonomy-guided borrowing (C2) is encoder-dependent.** It lifts tail accuracy on DINOv2 (+6.6 pts pooled;
   tau_norm 0.502 → 0.547 on V-select). It lowers it on BioCLIP 2 and lowers macro-F1 on CLIP B/32.

The common thread: **how a frozen encoder arranges species relative to the taxonomy decides which
long-tail fix helps.** That is the new paper.

## 3. The new paper
**Working title:** *Encoder over method: a taxonomy-aware benchmark of long-tailed plant recognition on frozen
features.*

| # | Claim | Evidence | Status |
|---|---|---|---|
| B1 | On frozen features, encoder choice dominates long-tail method choice; tuned simple probes match specialised heads on strong encoders | 11 methods × 4 encoders × 3 seeds on Pl@ntNet (done); H22-clean on 3 encoders (to run) | Pl@ntNet done |
| B2 | The taxonomic structure of tail errors is a property of the encoder (cross-genus on CLIP, within-genus on DINOv2 / BioCLIP 2), and a genus classifier on the same features cannot recover them | C-F on 4 encoders, check (b), **plus the new taxonomy-geometry measures (§4.1)** | C-F done; geometry done (RUNS.md, 2026-11-24) |
| B3 | Taxonomy-level fixes help in proportion to how the encoder's feature space follows the taxonomy: genus fallback (CP-3) and donor borrowing (C2) | CP-3 fallback rates; C2 per encoder and **per class** (§4.2), A9 | done: per class not predicted (RUNS.md, 2026-11-24) |
| B4 | A leakage-free benchmark: H22-clean, corrected IJCNN numbers, species-or-genus prediction sets | H22-clean split + features (teammates), corrected-evaluation table, CP-1..CP-4 | split done; runs to do |

Reframing of the old claims: C-F becomes B2 (refined: "3/4 cross-genus" holds for CLIP only). C-M (C2) becomes
part of B3, with honest negative results. C-E becomes B4. C1 stays a short negative result (check b).

**Positioning risk.** Reviewers may say foundation models are already known to shrink the benefit of long-tail
tricks (e.g. LIFT). Our answer: plant-specific taxonomy analysis, leakage-free H22, a predictive per-class measure
of when taxonomy helps, and conformal species-or-genus outputs. These are not in that literature.
LIFT stays as the fine-tuning reference point if compute allows.

## 4. New analysis work (code, CPU-only, runs on saved predictions and features)

### 4.1 Taxonomy geometry of each encoder (`tools/feature_geometry.py`)
Per encoder, from train features only:
- **Genus margin** of each species: cosine distance from its centroid to the nearest other-genus centroid minus the
  distance to the nearest same-genus centroid (species in monotypic genera reported separately).
- **Genus kNN purity**: share of the k = 10 nearest train neighbours of a tail image that belong to its genus
  (other species only).
- **Covariance transfer**: for classes with ≥ 50 images, the share of the class's own variance captured by the
  top-32 subspace of its donors (as C2 would pick them), by donor level (genus / family / global). This is the
  effect size behind the week-1 C2 test (which only gave p = 0.001).

### 4.2 Per-class link to errors and to C2 (`tools/error_taxonomy.py`, `tools/c2_class_analysis.py`)
- For every saved prediction file (method × encoder × seed): tail-FP shares by taxonomic level, genus-correct and
  family-correct rates among errors, mean taxonomic distance of errors (mistake severity).
- Per tail class (~500 per encoder, so enough power): does the genus margin predict (i) whether the class's errors
  are within-genus, (ii) the C2 gain over the same base (Spearman, with bootstrap CI)? Grouped by donor level and
  `n_s`.
- Outcome either way is reportable: if geometry predicts C2's gain, B3 is a practical rule ("borrow when donor
  covariance transfers"); if not, C2 is reported as an encoder-dependent component.

### 4.3 Gate record
- `results/RUNS.md`: session-5 row, gate table above, both readings of DINOv2's macro-F1.
- Fix for the future: `week2_gate.py` should refuse to start when the best `+c2` method has no `+c2shuf` run (the
  DINOv2 A9 gap came from queueing controls only for crt and ltic3).

## 5. Runs

| Session | Where | Settings | Purpose | Estimate |
|---|---|---|---|---|
| 6 | Kaggle (user) | ~~`BASELINES = "tau_norm+c2shuf"` on dinov2_b14~~ (dropped 2026-11-18: the DINOv2 gate fails on criterion 1, so A9 cannot change the verdict; new Pl@ntNet runs prove nothing now that test has been seen). Only the §4 tools on the saved predictions | Geometry + error taxonomy tables | ~0.5 h (CPU work) |
| H22 merge | Kaggle (owner) | `PART = "0/1"`, `PREV = "auto"`, teammates' three outputs attached | Full H22-clean feature store | ~0.5 h if all parts are done |
| 7 | Kaggle, **one account per encoder** | H22-clean core set: `la, crt, tau_norm, ltic3` on dinov2_b14, bioclip2, clip_b32, plus `tau_norm+c2, tau_norm+c2shuf` on dinov2_b14; `TRIALS = 10` (same budget for every method; disclosed). Add `ce, ldam_drw` only if time is left | B1/B2/B3 on a second dataset | ~25 GPU-h in total (~7 h per encoder, ~11 h for DINOv2); extrapolated from Pl@ntNet run times (~12× per run: 2.75× images, 14× classes), to be re-checked after the first method finishes |
| 8 | Kaggle | IJCNN configuration (original commit) on H22-clean | Corrected-evaluation table (B4) | ~1 h |
| opt. A | Kaggle | DINOv2 L/14 (already in `networks/backbones.py`) extraction + `la, crt, tau_norm, ltic3` | A 5th encoder point for B1/B2 | ~2 h extraction + ~1 h runs |
| opt. B | 4070 or Kaggle | LIFT on Pl@ntNet, CLIP B/16, 1 seed (disclosed) | Fine-tuning reference point | 7–20 h |

Quota: the user's ~30 h/week has gone mostly to sessions 3–5. H22 runs go on teammates' accounts.

## 6. Timeline
| Day | Work |
|---|---|
| **Wed 18** | Record gate FAIL (§1). Write §4 tools + tests; add the missing-A9 guard. Start session 6 |
| Thu 19 | Supervisor: decisions E1–E5 (§7). Check H22 extraction status with teammates; merge if done |
| Fri 20 | Session 7 (H22) starts. Geometry / error-taxonomy tables from session 6. Draft figure 1: encoder gap vs method gap |
| Sat 21 | (was the gate day) Results section outline from the B1–B4 tables. Optional run A if quota allows |
| Sun 22 – Tue 24 | H22 results; session 8 (corrected evaluation); per-class C2 analysis; conformal on H22 |
| **Wed 25** | Checkpoint: every number for B1–B4 in `results/all_runs.csv` or `results/analysis/`; else cut H22 C2 and opt. A/B |
| Thu 26 – Fri 27 | Figures and tables generated by script; significance tests; failure gallery (monotypic genera, singletons) |
| **Sat 28** | Results freeze |
| Sun 29 – Mon 30 | Reproducibility package, data card, README rewrite (corrected numbers, no presizer step). **Hand-over** |

## 7. Decisions for the supervisor
| # | Decision | Recommendation |
|---|---|---|
| E1 | Accept the week-2 gate FAIL now (Nov 18) instead of Nov 21 | Yes: the result will not change, and three days matter |
| E2 | Paper type | Benchmark/analysis paper, claims B1–B4 (§3) |
| E3 | Cuts | SMART (no definition found; drop and say so unless the reference arrives by Nov 20), iNat (drop), A2 rows beyond a/d/f, C2 ablations A4–A8 except donor level and rank |
| E4 | Keep | H22-clean (required for B4), A9 on DINOv2, C1 as a one-paragraph negative result, CP-1..CP-4 |
| E5 | Venue | §17 row 3 (benchmark/analysis), e.g. *Ecological Informatics* |

## 8. Risks
| Risk | Signal | Response |
|---|---|---|
| H22 features not ready by Fri 20 | merge notebook shows missing chunks | Owner account extracts the missing parts; if not done by Mon 23, run B1 on H22 with DINOv2 only (required encoder) |
| Geometry measures do not predict anything | Spearman CIs include 0 | Report as a negative result; B1, B2 and B4 stand without it |
| A9 on DINOv2 shows the gain is not from taxonomy | shuffled keeps ≥ 50% of the gain | Report C2's DINOv2 gain as a generic tail-augmentation effect |
| Novelty concern (foundation models known to shrink LT gains) | supervisor / reviewer feedback | Lean on plant taxonomy analysis, leakage-free H22 and species-or-genus sets; add LIFT if possible |
| Kaggle quota | < 5 h left on the user's account | Teammates' accounts; `--skip_done` resumes |
