# Changes: new long-tailed dataset and a plant-specific feature extractor

## Summary

LTIC was built and evaluated on Herbarium 2021/2022, using a general-purpose CLIP image encoder. This update makes two changes:

1. The code can now run on a **second long-tailed plant dataset, Pl@ntNet-300K**, and more generally on any dataset in the same format.
2. The general-purpose encoder is replaced by **BioCLIP 2**, an encoder pretrained on images of living organisms. The encoder is now selectable, so we can compare it with other options.

**The LTIC method itself is unchanged.** The classification head, the three-branch long-tail loss and the test-time weight normalization are the same as in the paper, and the original paper setup can still be run from the same code.

---

## 1. Quick recap: how LTIC works

LTIC has three stages:

1. **Feature extraction.** A large pretrained image model (the *encoder*) turns each image into a *feature vector*, a list of a few hundred numbers that describes the image. The encoder is **frozen**: its weights are never changed during our training.
2. **Parameter specialization.** A small trainable network (the *head*) processes the feature vector and splits it into three branches that share one classifier. One branch learns from all classes. Another concentrates on the medium and rare classes, so they are not drowned out by the common ones.
3. **Residual fusion.** The branch outputs are added together to give the final prediction. At test time the classifier weights are rescaled so that frequent classes do not dominate.

Only the head is trained. That has two consequences:
- **Swapping the encoder is easy.** The head just receives different feature vectors.
- **The encoder matters a lot.** The head can only be as good as the features it receives.

---

## 2. What changed at a glance

| Area | Before | Now |
|---|---|---|
| Dataset | Herbarium 2022 only. The class count, file paths and class-group boundaries were hard-coded. | Any dataset in the "image path + label" format. Pl@ntNet-300K is added, and settings come from the data or from command-line flags. |
| Feature extractor | CLIP ViT-B/32 (general internet images), fixed | Selectable: **BioCLIP 2** (default), DINOv2, or the original CLIP |
| Image colour normalization | Herbarium-specific values | The values each encoder was trained with. The original values are still available via a flag. |
| Many / medium / few-shot groups | Fixed index numbers that only fit Herbarium 2022 | Computed from the training data (more than 100, 20 to 100, fewer than 20 images per class) |
| Training speed | The encoder processes every image in every epoch | Optional **feature cache**: the encoder processes the data once |
| Metrics | Top-1/5 accuracy, precision/recall/F1, per-group accuracy (which could become NaN) | Same, plus **mean per-class accuracy**. The NaN problem is fixed. |
| Saved checkpoints | The whole model, including the frozen encoder | Only the trainable head (a few MB instead of more than 1 GB) |
| Hardware | GPU required | Falls back to CPU for quick tests |

---

## 3. Why Pl@ntNet-300K

[Pl@ntNet-300K](https://github.com/plantnet/PlantNet-300K) (NeurIPS 2021, Datasets & Benchmarks track) was built from observations shared by users of the Pl@ntNet plant-identification app.

- **Long-tailed by nature.** It has 306,146 images of 1,081 species, and **80% of the species account for only 11% of the images**. A few species have thousands of photos, and many have only a handful. This is exactly the imbalance LTIC is designed for.
- **Official train / validation / test splits** (243,916 / 31,118 / 31,112 images). Our results can be compared directly with published work. We choose the best model on the validation set and report on the test set.
- **A different kind of image.** These are field photographs (leaves, flowers, fruit and bark in natural settings), not dried herbarium sheets. This tests whether LTIC **generalizes beyond herbarium data**, which is a natural reviewer question for the paper.
- **Hard.** Many species look very similar, and the photos vary in angle, plant part and lighting.
- **Practical size.** The download is 31.7 GB, which fits on one machine, and 1,081 classes train quickly.

Two alternatives were considered:
- **Herbarium 2019** is the same image type as the paper, but it is small and its imbalance is milder, so it adds less new evidence.
- **The iNaturalist 2018 plant subset** is widely used, but it needs a ~120 GB download plus filtering to extract the plants.

---

## 4. Why the feature extractor was replaced

**We still use a Vision Transformer (ViT).** What changed is *which* pretrained ViT we use, and what data it learned from.

### Before: CLIP ViT-B/32
- Trained on **about 2 billion general internet images** with captions (LAION-2B). Its features are good in general, but plants are a tiny fraction of that data.
- "B/32" means a base-size model that cuts the image into **32×32-pixel patches**. That is coarse for fine botanical details such as leaf venation or flower parts.

### Now: BioCLIP 2 (default)
- **[BioCLIP 2](https://huggingface.co/imageomics/bioclip-2)** (NeurIPS 2025) is a **ViT-L/14**:
  - a larger model, about 300M parameters in the image encoder versus about 88M;
  - smaller **14×14-pixel patches**, so it sees finer detail.
- It was trained on **TreeOfLife-200M**: about **214 million images of organisms covering about 952,000 taxa**. Its training labels follow the biological classification (kingdom → … → species), so its features are already organised to separate species.
- The BioCLIP 2 paper reports clear gains over general-purpose models (CLIP, SigLIP, DINO) on species-classification benchmarks, including plant datasets. See Table 1 of the [paper](https://arxiv.org/abs/2505.23883).
- It loads through the same library as before (`open_clip`), so it is a drop-in replacement. Its feature vector has 768 numbers instead of 512, and the head's first layer adjusts automatically.

### Two other encoders we can select
- **DINOv2 ViT-L/14.** A strong general-purpose encoder trained on 142M web images **without any species labels**. It serves as a control; see the caveat below.
- **The original CLIP ViT-B/32**, to reproduce the paper's numbers.

### Caveat: possible overlap between training and test data
Most of BioCLIP 2's training data comes from GBIF, a global biodiversity database that **also stores Pl@ntNet observations and herbarium specimen images**. We could not confirm that Pl@ntNet-300K or the Herbarium competition images were removed from its training data. Some of our test images may therefore have been seen during BioCLIP 2's pretraining, which could make its results look better than they really are.

To stay transparent, we will **report DINOv2 results alongside BioCLIP 2**. DINOv2 never saw species labels, so it cannot have this advantage.

---

## 5. How the code works now

```
 Pl@ntNet-300K images            images/train/<species_id>/*.jpg  (+ val, test)
          │
          │  tools/make_plantnet_splits.py            (run once)
          ▼
 plantnet_train.txt / plantnet_val.txt / plantnet_test.txt
 (one line per image: "path  label")
          │
          │  datasets/plantnet.py → reuses the loader in datasets/herbarium.py
          ▼
 Data loader: random crop + flip (training), resize + centre crop (testing),
              colour normalization, classes renumbered rarest → most common
          │
          ▼
 networks/backbones.py      FROZEN ENCODER   bioclip2 | dinov2_l14 | clip_b32
          │                 → one feature vector per image (768 / 1024 / 512 numbers)
          │
          │  [optional] utils/feature_cache.py saves these vectors to disk once
          ▼
 networks/ltic.py           LTIC HEAD (trainable, same as the paper)
                            Linear → ReLU → BatchNorm → Linear  (170 numbers)
                            split into 3 chunks of 56 → shared classifier
                            → three branch scores: H, M, T
          │
          ▼
 CLIP_VIT_LONGTAIL.py       training loop, evaluation, logging, checkpoints
```

### Step by step

1. **Preparing the data** ([tools/make_plantnet_splits.py](tools/make_plantnet_splits.py))
   - Walks through the dataset folders, gives each species a number from 0 to 1080, and writes one text file per split.
   - The format matches the existing Herbarium files, so the original data loader is reused unchanged.
   - It also prints how many species are many-, medium- and few-shot.

2. **Loading the data** ([datasets/herbarium.py](datasets/herbarium.py), [datasets/plantnet.py](datasets/plantnet.py))
   - A key idea kept from the original code: classes are **renumbered by how many training images they have**. Class 0 is the rarest, and the last class is the most common.
   - The "few / medium / many-shot" groups therefore become simple ranges of class numbers.
   - Previously the class count and file paths were fixed for Herbarium 2022. They are now settings, and the Pl@ntNet version only declares its own values.

3. **Defining the class groups** (`set_shot_splits` in [CLIP_VIT_LONGTAIL.py](CLIP_VIT_LONGTAIL.py))
   - Counts the training images per class and applies the standard long-tail convention: **few-shot** below 20 images, **medium** 20 to 100, **many-shot** above 100.
   - The resulting boundaries are written to the training log. The paper's original Herbarium boundaries can still be passed explicitly.

4. **Feature extraction** ([networks/backbones.py](networks/backbones.py))
   - One small function per encoder, collected in a dictionary called `BACKBONES`. Adding another encoder later means adding one entry.
   - Each encoder is wrapped in `FrozenEncoder`, which:
     - keeps only the image part (the text part of CLIP-style models is dropped to save memory);
     - freezes the weights;
     - remembers the colour normalization the encoder expects.

5. **The LTIC head** ([networks/ltic.py](networks/ltic.py), class `LTICNet`)
   - The same head as in the paper. Only its input size adapts to the chosen encoder.
   - It accepts either images or pre-computed feature vectors, which is what makes the feature cache possible.

6. **Training** ([CLIP_VIT_LONGTAIL.py](CLIP_VIT_LONGTAIL.py))
   - Loss = (1 − β) × *combined loss* + β × *branch loss*, with β = 0.85:
     - the combined loss is the standard cross-entropy on the sum of branches H and M;
     - in the branch loss, branch H learns from every image, while branch M only learns from images of non-many-shot classes, which shifts capacity toward rare species.
   - Optimizer: SGD with a cosine learning-rate schedule, as in the paper.

7. **Evaluation**
   - The prediction is branch H + branch M.
   - The classifier weights are divided by their size raised to the power γ, so that frequent classes, which tend to get larger weights, do not dominate.
   - Reported: top-1/top-5 accuracy; precision, recall and F1; accuracy on the many-, medium- and few-shot groups; and **mean per-class accuracy**, the metric used by the Pl@ntNet-300K benchmark.

8. **Feature cache** ([utils/feature_cache.py](utils/feature_cache.py), optional flag `--cache_features`)
   - Because the encoder is frozen, it would produce the same vectors every epoch for the same image crop. So we compute them **once**:
     - 5 randomly augmented versions of each training image;
     - 1 version of each validation image.
   - These are stored on disk, and the 200 training epochs then run only the small head: **minutes instead of roughly a day** (rough estimate for a single GPU).
   - Trade-off: the model sees 5 fixed augmented views per image instead of a fresh random crop every epoch.

### Files

- **New:**
  - `networks/backbones.py`, `networks/ltic.py`
  - `datasets/plantnet.py`
  - `utils/feature_cache.py`
  - `tools/make_plantnet_splits.py`
  - `sh/plantnet.sh`, `sh/plantnet_eval.sh`
  - `requirements.txt`, `.gitignore`
- **Modified:**
  - `CLIP_VIT_LONGTAIL.py`
  - `datasets/herbarium.py`, `datasets/dataset.py`
  - `networks/nets.py`
  - `sh/R50.sh`, `sh/R50_eval.sh`
  - `README.md`
- **Unchanged, kept for reference:** `networks/clip.py`, `networks/clip_large_block.py`, `presizer.py`

---

## 6. How to run it

```bash
pip install -r requirements.txt

# 1. Download plantnet_300K.zip (31.7 GB) from https://zenodo.org/records/5645731 and unzip it
# 2. Create the split files (run once)
python tools/make_plantnet_splits.py --root /path/to/plantnet_300K

# 3. Train with BioCLIP 2 (model selected on the validation set)
DATA=/path/to/plantnet_300K bash sh/plantnet.sh
# 4. Evaluate the best model on the official test set
DATA=/path/to/plantnet_300K bash sh/plantnet_eval.sh

# Same experiment with the other encoders
DATA=/path/to/plantnet_300K BACKBONE=dinov2_l14 bash sh/plantnet.sh
DATA=/path/to/plantnet_300K BACKBONE=clip_b32   bash sh/plantnet.sh
```

**On a free Kaggle GPU:** see [KAGGLE.md](KAGGLE.md). A ready-made notebook (`kaggle/LTIC_PlantNet_Kaggle.ipynb`) runs all of the above in one go on 2× T4 GPUs.

**Reproducing the paper:**
- `sh/R50.sh` still runs the original setup: CLIP ViT-B/32, Herbarium colour normalization, and the original class-group boundaries.
- Two small differences remain:
  - the head now computes in full precision (float32) instead of half precision;
  - checkpoints saved by the old code cannot be loaded by the new code.

---

## 7. Testing status

**Done:**
- All three encoders were downloaded and loaded with their real pretrained weights.
  - They produce feature vectors of the expected size (512 / 768 / 1024).
  - Only the head is trainable.
- The whole pipeline ran end to end on a tiny synthetic dataset (6 species) on CPU:
  - data preparation;
  - training with and without the feature cache;
  - reusing a saved cache;
  - resuming from a checkpoint and evaluating on the test split.

**Not done yet:**
- A full training run on the real Pl@ntNet-300K data (this needs a GPU).
- Multi-GPU runs.
- **There are no accuracy numbers yet.**

### Planned results table (Pl@ntNet-300K, test set)

| Encoder | Many-shot | Medium-shot | Few-shot | All (top-1) | Mean per-class |
|---|---|---|---|---|---|
| CLIP ViT-B/32 (paper setup) | | | | | |
| DINOv2 ViT-L/14 | | | | | |
| BioCLIP 2 ViT-L/14 | | | | | |

---

## Glossary

- **Long-tailed dataset:** a few classes have many images and most classes have very few.
- **Many / medium / few-shot classes:** classes with many (more than 100), a moderate number (20 to 100), or few (fewer than 20) training images.
- **Encoder / feature extractor:** a pretrained model that turns an image into a feature vector.
- **Frozen:** the model's weights are not updated during our training.
- **ViT (Vision Transformer):** an image model that cuts the image into small square patches and relates them to each other. "B/32" and "L/14" mean base/large size with 32- or 14-pixel patches.
- **Head:** the small trainable network on top of the encoder that makes the final prediction.
- **Mean per-class accuracy:** the accuracy computed for each species separately and then averaged, so rare species count as much as common ones.
