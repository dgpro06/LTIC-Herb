# Running LTIC on Pl@ntNet-300K with a Kaggle GPU

> **For the week-1 plan (splits, features, LA probe, gate), use [KAGGLE_WEEK1.md](KAGGLE_WEEK1.md) instead.** This page describes the older single LTIC run.

This guide goes from "code on my PC" to "test results on Kaggle", one click at a time.

**What runs:** LTIC with the BioCLIP 2 encoder on Pl@ntNet-300K (1,081 species). The run:
- extracts the encoder's features once;
- trains the LTIC head for 200 epochs;
- evaluates the best checkpoint on the official test split.

**What you need:**
- a GitHub account;
- a Kaggle account with a verified phone number;
- about **3–5 hours** of Kaggle GPU time per run (my estimate, not yet measured).

Kaggle gives about 30 GPU hours per week, and the remaining amount is shown in Kaggle.

---

## Part A: put the code on GitHub (on your PC)

The new code lives on the branch `plantnet-bioclip2` of your local repo `F:\LTIC-Herb`. It is committed but not pushed yet.

1. Open a terminal in VS Code (**Terminal → New Terminal**) and check the state:
   ```bash
   cd F:\LTIC-Herb
   git status          # "On branch plantnet-bioclip2 ... nothing to commit, working tree clean"
   git log --oneline -3
   ```
2. Push the branch. Choose **one** option:

   **Option 1: you have write access to `Raiyan007-gb/LTIC-Herb`**
   ```bash
   git push -u origin plantnet-bioclip2
   ```
   If this fails with `Permission denied` or `403`, you don't have write access. Use Option 2.

   **Option 2: push to your own fork**
   1. On github.com, logged in as your account, open https://github.com/Raiyan007-gb/LTIC-Herb and click **Fork → Create fork**.
   2. Then run:
      ```bash
      git remote add mine https://github.com/moon-moon708/LTIC-Herb.git
      git push -u mine plantnet-bioclip2
      ```

   The first push may open a browser window to sign in to GitHub. Sign in with the account that owns the repo you push to.

3. Check that the push worked. Open `https://github.com/<account>/LTIC-Herb/tree/plantnet-bioclip2`, where `<account>` is `Raiyan007-gb` or your username. You should see a `kaggle` folder and a `KAGGLE.md` file.

---

## Part B: one-time Kaggle account setup

1. Sign in at https://www.kaggle.com.
2. Click your profile picture → **Settings**, and complete **Phone verification**. Without it, notebooks cannot use a GPU or the internet.

---

## Part C: create the notebook

1. On kaggle.com, click **+ Create → New Notebook** in the left sidebar.
2. In the notebook editor menu, click **File → Import Notebook**. Upload `F:\LTIC-Herb\kaggle\LTIC_PlantNet_Kaggle.ipynb` and click **Import**.
   - If you can't find Import, create the cells by hand. Copy the code cells from `kaggle/LTIC_PlantNet_Kaggle.ipynb` on GitHub; there are only 5.
3. Open the **right-hand panel**. If it's hidden, click the arrow at the top right of the editor.
   - **Session options → Accelerator:** choose **GPU T4 x2** and confirm.
     - Choose T4 rather than P100. The T4 has fast half-precision units that the encoder uses, and the code uses both GPUs automatically.
   - **Session options → Internet:** switch **On**. This is needed to download the code and the BioCLIP 2 weights.
4. Still in the right-hand panel, go to **Input → + Add Input**:
   - search `plantnet 300k images`;
   - pick the dataset by **Noahbadoa** (31.98 GB, https://www.kaggle.com/datasets/noahbadoa/plantnet-300k-images);
   - click the **+** button.

   This is a re-upload of the official Zenodo files, so you don't have to upload 32 GB yourself.
5. In the **first code cell (Settings)**, set `GITHUB_USER` to the account you pushed to in Part A: `"Raiyan007-gb"` for Option 1, or `"moon-moon708"` for Option 2. Leave the rest as it is.

---

## Part D: a quick test run (about 10–15 minutes)

This catches setup mistakes before you commit hours of GPU time.

1. Click **Run All** in the top toolbar.
2. Watch the output of each cell. You should see:
   - **Step 1:** two lines `Tesla T4, 15360 MiB`, and `plantnet-300k-images` listed under `/kaggle/input`.
   - **Step 2:** the latest commit, e.g. `xxxxxxx feat: Kaggle notebook and guide for Pl@ntNet-300K`.
   - **Step 3:**
     - `=== STEP 3/5 ...` followed by `train: 243916 images`, `val: 31118 images`, `test: 31112 images`, `classes: 1081`;
     - then `=== STEP 4/5 ...`, `=> backbone bioclip2 | classes 1081 | few [...] medium [...] many [...]`;
     - then `=> extracting train view 1/5 [0/953]`.
3. Wait for `=> extracting train view 1/5 [100/953]`, then click **Stop** (the ■ button) to end the test.
4. Note how long the step from `[0/953]` to `[100/953]` took. That time × 9.5 is one pass over the training set, and the run makes 6 passes (5 training views + 1 validation). Add about 1 hour for training and testing.
   - If the total is well under 12 hours, you're good.
   - If not, set `CACHE_VIEWS = 3` in the Settings cell.

---

## Part E: the real run (about 3–5 hours, runs without you)

1. Click **Save Version** at the top right.
2. Keep **Save & Run All (Commit)** selected and click **Save**.
3. You can now close the browser. Kaggle runs the whole notebook in the background, with a maximum of 12 hours.
4. To check progress, go to kaggle.com → **Code → Your Work** → your notebook. A running version shows **Running**; open it to see the live log.

---

## Part F: get the results

When the version shows **Complete**, open it.

- **Last cell output:** the `=== DONE` summary shows the best validation line and the **test** line. For example:
  ```
   * Acc@1 ... Acc@5 ... HAcc ... MAcc ... TAcc ... MacroAcc ...
  ```
  - `Acc@1` / `Acc@5`: top-1 / top-5 accuracy in %.
  - `HAcc` / `MAcc` / `TAcc`: accuracy on many-shot (more than 100 training images), medium-shot (20–100) and few-shot (fewer than 20) species, as a fraction from 0 to 1.
  - `MacroAcc`: accuracy averaged over species (0–1), the main Pl@ntNet-300K metric.
- **Output tab:** the files under `runs/plantnetDataset/`, which you can download:
  - `plantnet_bioclip2_bt256/train.log`: the full training log;
  - `plantnet_bioclip2_bt256/model_best.pth.tar`: the best checkpoint (the LTIC head only, a few MB);
  - `plantnet_bioclip2_bt256_test/train.log`: the test-set evaluation.

Copy the test line into the results table in `changes.md`.

---

## Part G: run the other encoders (for the comparison table)

For each encoder:
1. Open the notebook and click **Edit**.
2. In the Settings cell, set `BACKBONE = "dinov2_l14"`, and in a later run `BACKBONE = "clip_b32"`.
3. Click **Save Version → Save & Run All**.

Each version keeps its own outputs. `clip_b32` is the fastest because it's a smaller model.

---

## Troubleshooting

| Message | Fix |
|---|---|
| `No GPU found` | Right-hand panel → Accelerator → **GPU T4 x2**. |
| `Could not resolve host: github.com`, or pip cannot download | Internet is off. Turn it on in the right-hand panel; this needs a verified phone number (Part B). |
| `Remote branch plantnet-bioclip2 not found` | The branch isn't on GitHub yet, or `GITHUB_USER` is wrong. Redo Part A step 3, then check the Settings cell. |
| `no images/train or images_train folder found below '/kaggle/input'` | The dataset isn't attached. Redo Part C step 4. |
| `no kernel image is available for execution on the device` | You picked P100. Switch to **GPU T4 x2**. |
| `CUDA out of memory` | Set `BATCH = 128` in the Settings cell. |
| The run is stopped at 12 hours | Set `CACHE_VIEWS = 3` (fewer feature passes) or `EPOCHS = 100`. |
| Red pip warnings about dependency conflicts | Harmless; ignore them. |
| `Your notebook tried to allocate more memory than is available` | Set `CACHE_VIEWS = 3`. |

---

## What happens under the hood

`kaggle/run_kaggle.sh` runs five steps, prints `=== STEP n/5` before each, and stops at the first error:

1. **GPU check:** `nvidia-smi`.
2. **Install:** `open_clip_torch`, which loads BioCLIP 2 and CLIP. Everything else is already on Kaggle.
3. **Split files:** `tools/make_plantnet_splits.py` finds the dataset under `/kaggle/input` and writes `plantnet_{train,val,test}.txt` to `/kaggle/working/splits`. `/kaggle/input` is read-only, so they go there instead.
4. **Training:** `sh/plantnet.sh`, with the feature cache in `/kaggle/working/feat_cache` and logs and checkpoints in `/kaggle/working/runs`.
5. **Test evaluation:** `sh/plantnet_eval.sh` loads `model_best.pth.tar` and scores the test split.

Everything in `/kaggle/working` becomes the notebook's saved Output. That is about 2–3 GB per run, well under Kaggle's 20 GB limit.
