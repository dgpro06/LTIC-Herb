from datasets.herbarium import herbariumDataset


class h22cleanDataset(herbariumDataset):
    """Leakage-free Herbarium 2022 split (plan §5.2), written by tools/make_h22_clean.py.

    15,501 classes: the official category ids that have images, renumbered 0..15,500.
    The default val file is V-tune (tools/make_val_parts.py --prefix h22clean); never train on V-select.
    Shot groups: --few_thr 20 --many_thr 50 (the largest class has ~80 images).
    """
    num_classes = 15501
    train_file = "h22clean_train.txt"
    val_file = "h22clean_vtune.txt"
