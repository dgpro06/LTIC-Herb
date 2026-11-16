# Week 1 on Kaggle (Pl@ntNet-300K)

This guide runs the week-1 pipeline from [WEEK1.md](WEEK1.md) in a free Kaggle GPU notebook. It uses the
notebook [kaggle/LTIC_Week1_Kaggle.ipynb](kaggle/LTIC_Week1_Kaggle.ipynb) and the script
[kaggle/week1_kaggle.sh](kaggle/week1_kaggle.sh).

It is split into two sessions:
- **Session 1, the gate (~2–4 h on a T4):** splits and the §8.0 count, taxonomy, CLIP B/32 features, LA probe,
  C-F, the week-1 gate diagnostics and conformal sets.
- **Session 2:** the gate on the other three encoders and baselines 1–9 on CLIP B/32. It continues from session 1's output.

Times are estimates; a T4 is roughly 2–3× slower than the 4070. Kaggle gives about 30 GPU hours per week
per account, and one session can run for up to 12 hours.

## Once: account and code
1. **Push the code.** Kaggle clones the `Dgpro` branch of `github.com/dgpro06/LTIC-Herb`, so commit and push first:
   `git add -A`, `git commit -m "..."`, then `git push`.
2. **Verify your phone.** On kaggle.com, go to profile → **Settings** → **Phone verification**. Without it,
   notebooks get no GPU and no internet.

## Session 1: the gate
1. **Create the notebook.** On kaggle.com, click **+ Create → New Notebook**, then **File → Import Notebook**,
   and upload `kaggle/LTIC_Week1_Kaggle.ipynb`.
2. **Set the right-hand panel:**
   - **Session options → Accelerator:** `GPU T4 x2`. Pick T4, not P100.
   - **Session options → Internet:** On. Encoder weights, DINOv2 and GBIF are downloaded.
   - **Input → + Add Input:** set the filter to **Datasets**, search `plantnet`, and add **PlantNet300k** by
     **Gabriel Neto** (`gabrielneto583/plantnet300k`, 306,149 files: the images plus the JSON files with species
     names). It contains `plantnet_300K/images/{train,val,test}/<species_id>/`. Results marked with a `< >` icon
     are notebooks, not datasets. (The Noahbadoa copy used by the older guide is no longer listed.)
3. **Check the Settings cell.** The defaults are already right for session 1:
   `GITHUB_USER = "dgpro06"`, `BRANCH = "Dgpro"`, `ENCODERS = "clip_b32,clip_b16,bioclip2,dinov2_b14"` (one decode pass extracts all four; decoding is the slow part), `RUN_GATE = True`, `BASELINES = ""`.
4. **Quick check (~5 min).** Click **Run All** and watch step 3. It should print:
   - `train: 243916 images`, `val: 31118`, `test: 31112`, `classes: 1081`;
   - `species names: .../plantnet300K_species_id_2_name.json`. If it says `NOT FOUND`, see Troubleshooting;
   - the `plan 8.0 decision` line and the V-select few-shot count;
   - `1081 species, ~303 genera`, then the feature chunks starting.

   Once the first `=> train chunk 1/5 saved (... img/s ...)` line appears, note the img/s (that's your
   throughput test) and click **Stop**.
5. **The real run.** Click **Save Version → Save & Run All (Commit) → Save**. You can close the browser.
   Progress is under **Code → Your Work → the notebook → the running version**.
6. **Read the result.** When the version shows **Complete**, the last cell prints PASS/FAIL for each gate
   check and the verdict. The **Output** tab holds:
   - `results/week1_gate_plantnet_clip_b32.json`: every gate number, with CIs;
   - `results/cf/`: the C-F breakdown, the τ curve, the count curve and a figure;
   - `results/conformal/`: CP-1 to CP-4;
   - `results/all_runs.csv`: every trial and run;
   - `splits/` and `taxonomy/`: download these and commit them to the repo (plan deliverables);
   - `feats/plantnet/`: the features, reused by session 2.

## Session 2: gate on the other encoders + baselines (continues from session 1)
Session 1 already extracted all four encoders, so session 2 only trains small heads on cached features.
1. Open the notebook (**Code → Your Work → first_test**) and click **Edit**.
2. **Input → + Add Input → Your Work**: add **first_test** (the session-1 version's Output). Keep the
   PlantNet300k dataset attached too.
3. Set the Settings cell:
   ```python
   ENCODERS          = "clip_b32,clip_b16,bioclip2,dinov2_b14"
   RUN_GATE          = True       # gate on every encoder; CLIP B/32's LA probe is reused, not redone
   BASELINES         = "ce,balanced_softmax,crt,tau_norm,ldam_drw,reslt,ltic2,ltic3,ltic3@gamma=0,flat_genus"
   BASELINE_ENCODERS = "clip_b32"
   TRIALS            = 20
   PREV              = "auto"     # finds the attached session-1 output by itself
   ```
4. **Save Version → Save & Run All (Commit) → Save**, then stop the draft session with ⏻.

What it does: copies session 1's splits, taxonomy, features and results; skips everything already finished;
runs the gate (LA probe, C-F, diagnostics, conformal) on CLIP B/16, BioCLIP 2 and DINOv2 B/14; re-runs the
CLIP B/32 diagnostics for the secondary analyses into `week1_gate_plantnet_clip_b32_rerun.json` (the recorded
`week1_gate_plantnet_clip_b32.json` is never overwritten); then baselines 1–9 on CLIP B/32. The last lines
of the log are a one-line gate summary per encoder.

If a session hits the 12-hour limit, start another one the same way, attaching the newest version's Output:
finished LA probes and baselines are skipped (`--skip_done`), so it continues where it stopped.

## Troubleshooting

| Message | Fix |
|---|---|
| `No GPU` | Accelerator → **GPU T4 x2** |
| `Could not resolve host` / pip or download fails | Internet is off, or the phone isn't verified |
| `Remote branch Dgpro not found` | The code isn't pushed; run `git push` |
| `no images/train or images_train folder found` | The Pl@ntNet dataset isn't attached (Input panel), or a notebook was added instead of the dataset |
| `species names: NOT FOUND` | The dataset copy lacks `plantnet300K_species_id_2_name.json`. Upload that file (from the official Zenodo zip) as a small private Kaggle dataset, attach it and run again; the file is found anywhere under `/kaggle/input` |
| `genera without a family ...` | GBIF didn't match these genera. Add them to `MANUAL_FAMILIES` in `tools/make_taxonomy_plantnet.py`; only the C-F family breakdown uses families, not the gate's cross-genus share |
| Interactive session died / "kernel died" | The PC slept or lost internet. Use **Save Version → Save & Run All (Commit)**, which runs on Kaggle's servers |
| `CUDA out of memory` during extraction | Run fewer encoders per session, e.g. `ENCODERS = "bioclip2"` alone |
| A baseline run takes > 2 min | Set `TRIALS = 10`, for **every** method (plan §11.2) |
| `existing features were made from a different split file` | `PREV` points at an output with other splits; use the matching version |
| `PREV=auto: no earlier output with feats/plantnet` | Session 1's output isn't attached: **Input → + Add Input → Your Work → first_test** |
