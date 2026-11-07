<p align="center">
  <img src="logo.svg" alt="LTIC-Herb logo" width="600"/>
</p>

# LTIC-Herb : Long-tailed Image Classification on the Specimens of Herbarium Dataset (IJCNN 2025)

[![IJCNN 2025](https://img.shields.io/badge/IJCNN-2025-blue.svg)](https://ieeexplore.ieee.org/abstract/document/11228538)
[![IEEE Xplore](https://img.shields.io/badge/IEEE-Xplore-red.svg)](https://ieeexplore.ieee.org/abstract/document/11228538)
[![GitHub stars](https://img.shields.io/github/stars/Raiyan007-gb/LTIC-Herb.svg?style=social&label=Star)](https://github.com/Raiyan007-gb/LTIC-Herb)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![CLIP-ViT](https://img.shields.io/badge/Backbone-CLIP--ViT-orange.svg)](https://github.com/openai/CLIP)

> Official implementation of **"Long-tailed Image Classification on the Specimens of Herbarium Dataset"**, accepted at **2025 International Joint Conference on Neural Networks (IJCNN)**.

**📄 Paper:** https://ieeexplore.ieee.org/abstract/document/11228538  
**💻 Repository:** https://github.com/Raiyan007-gb/LTIC-Herb

## Overview

Modern transformer-based image encoders have revolutionized computer vision, consistently achieving new benchmarks in various tasks. However, challenges like the long-tail distribution of species in botanical datasets and incomplete representation of rare specimens in herbarium collections hinder accurate classification of diverse plant species. We propose a three-stage approach: leveraging pre-trained transformers for feature extraction, introducing parameter specialization to address class imbalance, and employing a residual fusion mechanism for unified predictions. Evaluations on the Herbarium 2021 and Herbarium 2022 datasets demonstrate state-of-the-art performance across few-shot, medium-shot, and many-shot settings. This work highlights the potential of transformer-based encoders with targeted improvements in advancing biodiversity monitoring and botanical research.

![LTIC Overview](LTIC.png)

## Results

### Performance on Herbarium Datasets

| Dataset Name   | Epochs | Many-shot | Medium-shot | Few-shot | All   |
|----------------|--------|-----------|-------------|----------|-------|
| Herbarium 2022 | 200    | 77.70     | 75.83       | 75.19    | 76.56 |
| Herbarium 2021 | 200    | 84.20     | 81.24       | 80.19    | 81.03 |

### Comparison with Conviformer-B

| Model          | Dataset        | Test F1 Score |
|----------------|----------------|---------------|
| LTIC           | Herbarium 2022 | 89.31         |
| LTIC           | Herbarium 2021 | 81.88         |
| Conviformer-B  | Herbarium 2022 | 86.8          |
| Conviformer-B  | Herbarium 2021 | 71.9          |

## Usage

To use this model, please adjust the model datapath accordingly in the code (e.g., update the `--data_path` argument in `R50.sh` or `CLIP_VIT_LONGTAIL.py`). Additionally, ensure that the dataset is presized using the `presizer.py` script before training or evaluation. The `presizer.py` script processes images to a target resolution (default 512x512) with central cropping (default 448x448), preparing the Herbarium dataset for optimal model input.

Install the dependencies with `pip install -r requirements.txt`. Split files are read from `--data_path` (`train_hbm.txt` / `val_hbm.txt` for Herbarium); pass `--train_txt` / `--val_txt` to use other locations.

### Feature extractors

The image encoder is frozen and selected with `--backbone`:

| `--backbone` | Encoder | Pretraining data | Feature dim |
|---|---|---|---|
| `bioclip2` (default) | [BioCLIP 2](https://huggingface.co/imageomics/bioclip-2) ViT-L/14 | TreeOfLife-200M (biology images) | 768 |
| `dinov2_l14` | [DINOv2](https://github.com/facebookresearch/dinov2) ViT-L/14 | LVD-142M, self-supervised | 1024 |
| `clip_b32` | open_clip ViT-B/32 (paper setup) | LAION-2B | 512 |

`sh/R50.sh` keeps the paper setup (`--backbone clip_b32 --dataset_norm` and the Herbarium 2022 class cuts). Other runs normalize images with the encoder's own mean/std and derive the many/medium/few-shot cuts from the training counts (`--many_thr 100`, `--few_thr 20`).

> **Note on BioCLIP 2:** TreeOfLife-200M is built from GBIF, which also hosts Pl@ntNet observations and herbarium specimens, and its de-duplication was reported against iNat21 and Rare Species only. Some test images of Pl@ntNet-300K or Herbarium may have been seen during pretraining, so report `dinov2_l14` alongside it.

Because the encoder is frozen, `--cache_features` extracts its features once (`--cache_views` augmented views per training image, stored in `--cache_dir`) and trains the head on them, which turns a 200-epoch run into minutes after the one-time extraction.

### Pl@ntNet-300K

[Pl@ntNet-300K](https://github.com/plantnet/PlantNet-300K) is a long-tailed plant dataset: 306,146 images of 1,081 species, where 80% of the species account for only 11% of the images.

```bash
# 1. Download plantnet_300K.zip (31.7 GB) from https://zenodo.org/records/5645731 and unzip it
# 2. Write the split files (plantnet_{train,val,test}.txt) into the dataset folder
python tools/make_plantnet_splits.py --root /path/to/plantnet_300K
# 3. Train (select the model on val), then evaluate on the official test split
DATA=/path/to/plantnet_300K bash sh/plantnet.sh
DATA=/path/to/plantnet_300K bash sh/plantnet_eval.sh
# Same run with another encoder
DATA=/path/to/plantnet_300K BACKBONE=dinov2_l14 bash sh/plantnet.sh
```

To run on a free Kaggle GPU instead, follow [KAGGLE.md](KAGGLE.md) (notebook: `kaggle/LTIC_PlantNet_Kaggle.ipynb`).

Pl@ntNet images are field photos, so `presizer.py` (which trims herbarium sheet borders) is not needed. Besides top-1/top-5, the log reports `MacroAcc`, the mean per-class accuracy used by the Pl@ntNet-300K benchmark.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@INPROCEEDINGS{11228538,
  author={Ahmed, Raiyan and Naheen, Intisar Tahmid and Haque, Yashfinul and Abir, Md. Towsif and Farazi, Moshiur and Rahman, Shafin},
  booktitle={2025 International Joint Conference on Neural Networks (IJCNN)},
  title={Long-tailed Image Classification on the Specimens of Herbarium Dataset},
  year={2025},
  volume={},
  number={},
  pages={1-8},
  keywords={Computer vision;Adaptation models;Heavily-tailed distribution;Biological system modeling;Neural networks;Transformers;Feature extraction;Labeling;Monitoring;Image classification;CLIP;long-tail distribution;botanical classification;Herbarium dataset;vision transformer},
  doi={10.1109/IJCNN64981.2025.11228538}}
```

## Acknowledgements

We thank the providers of the Herbarium 2021 and Herbarium 2022 datasets for their contributions to botanical research. We also acknowledge the support of the open-source community, particularly the developers of CLIP and PyTorch, which enabled this work.

## Contact

If you have any questions, feel free to contact us through email (raiyan2025@gmail.com) or GitHub issues. Enjoy!

⭐ If you find this repository helpful, please consider starring it!
thank you

