# Week 1 on Kaggle (Pl@ntNet-300K)

This guide runs the week-1 pipeline from [WEEK1.md](WEEK1.md) in a free Kaggle GPU notebook. It uses the
notebook [kaggle/LTIC_Week1_Kaggle.ipynb](kaggle/LTIC_Week1_Kaggle.ipynb) and the script
[kaggle/week1_kaggle.sh](kaggle/week1_kaggle.sh).

It is split into two sessions:
- **Session 1, the gate (~2–4 h on a T4):** splits and the §8.0 count, taxonomy, CLIP B/32 features, LA probe,
  C-F, the week-1 gate diagnostics and conformal sets.
- **Session 2, overnight:** the other three encoders and baselines 1–9. It continues from session 1's output.

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
   - **Input → + Add Input:** search `plantnet 300k images` and add the dataset by **Noahbadoa**
     (`noahbadoa/plantnet-300k-images`, 31.98 GB).
3. **Check the Settings cell.** The defaults are already right for session 1:
   `GITHUB_USER = "dgpro06"`, `BRANCH = "Dgpro"`, `ENCODERS = "clip_b32"`, `RUN_GATE = True`, `BASELINES = ""`.
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

## Session 2: more encoders and baselines (continues from session 1)
1. Open the notebook and click **Edit**.
2. **Input → + Add Input → Your Work** (notebook outputs): add this notebook's session-1 version.
3. Run Step 1 once and note the folder it lists under `/kaggle/input` for that output, for example
   `/kaggle/input/ltic-week-1`.
4. Set the Settings cell:
   ```python
   ENCODERS          = "clip_b32,clip_b16,bioclip2,dinov2_b14"   # clip_b32 is already done and is skipped
   RUN_GATE          = False
   BASELINES         = "la,ce,balanced_softmax,crt,tau_norm,ldam_drw,reslt,ltic2,ltic3,ltic3@gamma=0,flat_genus"
   BASELINE_ENCODERS = "clip_b32"
   PREV              = "/kaggle/input/ltic-week-1"            # the folder from step 3
   ```
5. **Save Version → Save & Run All**.

If a session hits the 12-hour limit, start another the same way. Set `PREV` to the newest version's output;
finished feature chunks and splits are copied back and skipped. Baseline runs for the other encoders are
separate sessions with `BASELINE_ENCODERS` set to that encoder.

## Troubleshooting

| Message | Fix |
|---|---|
| `No GPU` | Accelerator → **GPU T4 x2** |
| `Could not resolve host` / pip or download fails | Internet is off, or the phone isn't verified |
| `Remote branch Dgpro not found` | The code isn't pushed; run `git push` |
| `no images/train or images_train folder found` | The Pl@ntNet dataset isn't attached |
| `species names: NOT FOUND` | The dataset copy lacks `plantnet300K_species_id_2_name.json`. Upload that file (from the official Zenodo zip) as a small private Kaggle dataset, attach it and run again; the file is found anywhere under `/kaggle/input` |
| `genera without a family: [...]` | GBIF didn't match these genera. Fill the family by hand in `taxonomy/plantnet.csv`; only the C-F family breakdown uses it, not the gate's cross-genus share |
| `CUDA out of memory` during extraction | Run fewer encoders per session, e.g. `ENCODERS = "bioclip2"` alone |
| A baseline run takes > 2 min | Set `TRIALS = 10`, for **every** method (plan §11.2) |
| `existing features were made from a different split file` | `PREV` points at an output with other splits; use the matching version |
