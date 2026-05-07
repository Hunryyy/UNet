"""Shared helpers for Task 2 tests."""
import os
import random
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

TEST_DIR = Path(__file__).resolve().parent
REPO_DIR = TEST_DIR.parent.parent
CODE_DIR = REPO_DIR / "code"
DATASET_DIR = CODE_DIR / "dataset"

DATA_MEAN = [0.43782742, 0.44557303, 0.41160695]
DATA_STD = [0.19686149, 0.18481555, 0.19296625]
SEED = 42

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def split_dir(split, kind):
    return DATASET_DIR / split / kind


def make_dataset_subset(root, train_count=4, val_count=2):
    """Copy a tiny deterministic dataset subset for fast integration tests."""
    root = Path(root)
    subset_root = root / "dataset"
    selected = {}

    for split, count in (("train", train_count), ("val", val_count)):
        img_src = split_dir(split, "image")
        lbl_src = split_dir(split, "label")
        names = sorted(os.listdir(img_src))[:count]
        selected[split] = names

        for kind, src_dir in (("image", img_src), ("label", lbl_src)):
            dst_dir = subset_root / split / kind
            dst_dir.mkdir(parents=True, exist_ok=True)
            for name in names:
                shutil.copy2(src_dir / name, dst_dir / name)

    return {
        "root": subset_root,
        "train_img": subset_root / "train" / "image",
        "train_lbl": subset_root / "train" / "label",
        "val_img": subset_root / "val" / "image",
        "val_lbl": subset_root / "val" / "label",
        "selected": selected,
    }


@contextmanager
def pushd(path):
    old_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)
