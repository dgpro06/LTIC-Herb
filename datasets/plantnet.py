from datasets.herbarium import herbariumDataset


class plantnetDataset(herbariumDataset):
    """Pl@ntNet-300K (Garcin et al., NeurIPS 2021 Datasets & Benchmarks): 1,081 species, long-tailed.

    Split files are written by tools/make_plantnet_splits.py into the dataset root.
    """
    num_classes = 1081
    train_file = "plantnet_train.txt"
    val_file = "plantnet_val.txt"
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
