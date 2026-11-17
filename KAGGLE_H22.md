# H22-clean on Kaggle (Herbarium 2022)

This guide builds the leakage-free **H22-clean** split (plan §5.2) and extracts frozen-encoder features on
free Kaggle GPUs. It uses the notebook [kaggle/LTIC_H22_Kaggle.ipynb](kaggle/LTIC_H22_Kaggle.ipynb) and the
script [kaggle/h22_kaggle.sh](kaggle/h22_kaggle.sh).

H22 has about 840,000 images. Decoding the images is the slow part (about 50 img/s on a Kaggle notebook), so
the work is **split across three Kaggle accounts**. Each one extracts a third of the images for all three
encoders (DINOv2 B/14, BioCLIP 2, CLIP B/32). One notebook then merges the three outputs.

| Account | `PART` | Work | Expected time |
|---|---|---|---|
| Teammate 1 | `"0/3"` | chunks 0, 3, 6, … | ~2–5 h |
| Teammate 2 | `"1/3"` | chunks 1, 4, 7, … | ~2–5 h |
| Teammate 3 | `"2/3"` | chunks 2, 5, 8, … | ~2–5 h |

With only one account, use `PART = "0/1"`. That takes ~6–15 h, so plan for two sessions; see "If a session
stops early".

## For each teammate (once per account)

1. **Verify your phone.** On kaggle.com, open your profile picture → **Settings** → **Phone verification**.
   Without it, notebooks get no GPU and no internet.
2. **Join the competition.** Open https://www.kaggle.com/competitions/herbarium-2022-fgvc9, click
   **Join Competition** (or **Late Submission**), and click **I Understand and Accept** on the rules.
   Kaggle refuses to attach the data until you do this.

## For each teammate: the run

1. **Create the notebook.** On kaggle.com, click **+ Create → New Notebook**, then **File → Import Notebook**.
   Choose the **GitHub** tab, enter `dgpro06/LTIC-Herb`, pick branch `Dgpro`, and select
   `kaggle/LTIC_H22_Kaggle.ipynb`. You can also download that file from GitHub and upload it with the
   **File** tab.
2. **Set the right-hand panel:**
   - **Session options → Accelerator:** `GPU T4 x2`.
   - **Session options → Internet:** On.
   - **Input → + Add Input:** set the filter to **Competitions**, search `herbarium 2022`, and add
     **Herbarium 2022 - FGVC9**.
3. **Edit the Settings cell.** Change only `PART`, to your row in the table above (`"0/3"`, `"1/3"` or
   `"2/3"`). Leave everything else as it is.
4. **Quick check (~5–10 min).** Click **Run All** and watch step 3. It should print:
   - `839772 images, 15501 categories with images` (or very close);
   - the **split-file sha1** lines. Send them to the group: they must be the same on all three accounts;
   - `=> loaded dinov2_b14 (768-d)`, `=> loaded bioclip2 (768-d)`, `=> loaded clip_b32 (512-d)`;
   - batch progress lines for the first chunk.

   Then click **Stop** (■) at the top.
5. **The real run.** Click **Save Version** (top right) → **Save & Run All (Commit)** → **Save**. You can close
   the browser now; the run keeps going. Then stop the editing session with the ⏻ button so it doesn't use
   your GPU hours.
6. **Check it finished.** Open **Code → Your Work → the notebook**. The version should show **Complete**. In its
   log, near the end, the **Feature store status** lists each encoder and split. With a third of the work
   done, about a third of the chunks are present, which is expected. `NOT COMPLETE` is normal for one part.
7. **Share the output with the project owner.** In the notebook, click **Share** (top right) → add
   **dgpro06** as a collaborator. Keep the notebook **private**: the features come from competition data.

## Project owner: merge the three parts

1. Create one more notebook from `kaggle/LTIC_H22_Kaggle.ipynb` (same import as above), with the same panel
   settings, and keep the Herbarium 2022 competition attached.
2. **Input → + Add Input**: add the three teammates' notebooks. Look under **Your Work** → **Shared with you**,
   or search their notebook names.
3. Settings cell: `PART = "0/1"`, `PREV = "auto"`. Leave `BASELINES = ""` for the first merge.
4. **Save Version → Save & Run All (Commit)**. The script copies all three outputs, finds that every chunk is
   already there, and prints `ALL CHUNKS PRESENT`. If chunks are missing, it extracts them itself (that's
   also how one account finishes alone).
5. This notebook's Output is the full H22-clean feature store. Later sessions attach it with `PREV = "auto"`
   and set `BASELINES`, e.g. `"la,crt,tau_norm,ltic3,crt+c2,crt+c2shuf"`. Baselines refuse to start until
   all chunks are present.
6. Download `splits/` and `taxonomy/h22.csv` from the Output and commit them, as with Pl@ntNet. Don't commit
   `feats/`.

## If a session stops early

The script stops starting new chunks after ~10.7 h (`DEADLINE_MIN`), before Kaggle's 12 h limit, so the
finished chunks are saved in the Output. To continue: open the notebook → **Edit** → **Input → + Add Input →
Your Work** → add this notebook (its newest version) → set `PREV = "auto"` (keep the same `PART`) → **Save
Version → Save & Run All**. Finished chunks are skipped.

## Troubleshooting

| Message | Fix |
|---|---|
| `train_metadata.json not found` | The competition isn't attached, or you haven't accepted its rules (step 2 of "once per account"). |
| `No GPU` | Accelerator is off, or the phone isn't verified. |
| `existing features were made from a different split file or settings` | The split files differ between accounts. Compare the sha1 lines; everyone must use the same code commit. |
| The quick check shows < 20 img/s | Normal for the first minute (encoder downloads). If it stays low, check that the accelerator is a GPU. |
| Output too large | Should not happen: ~3–4 GB in total for three encoders. Each part is about a third of that. |
