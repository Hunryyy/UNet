"""Shared helpers for Task 2 tests."""
import os
import random
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image

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


def _dir_has_pngs(path):
    path = Path(path)
    return path.is_dir() and any(path.glob("*.png"))


def _expected_baseline_patch_counts():
    from step1_crop_dataset import TILE_SIZE, TRAIN_RATIO, compute_grid, split_positions

    img_path = CODE_DIR / "img_trainval.png"
    with Image.open(img_path) as img:
        width, height = img.size

    positions, y_positions, x_positions = compute_grid(height, width, TILE_SIZE, TILE_SIZE)
    train_positions, val_positions, _ = split_positions(
        positions, y_positions, x_positions, TILE_SIZE, TRAIN_RATIO
    )
    return len(train_positions), len(val_positions)


def baseline_dataset_ready():
    required = [
        split_dir("train", "image"),
        split_dir("train", "label"),
        split_dir("val", "image"),
        split_dir("val", "label"),
    ]
    if not all(_dir_has_pngs(path) for path in required):
        return False

    train_img = split_dir("train", "image")
    train_lbl = split_dir("train", "label")
    val_img = split_dir("val", "image")
    val_lbl = split_dir("val", "label")
    train_names = sorted(p.name for p in train_img.glob("*.png"))
    val_names = sorted(p.name for p in val_img.glob("*.png"))
    if train_names != sorted(p.name for p in train_lbl.glob("*.png")):
        return False
    if val_names != sorted(p.name for p in val_lbl.glob("*.png")):
        return False
    if not train_names or not val_names:
        return False
    if any(name.startswith("patch_y") for name in train_names):
        return False
    if any(name.startswith("patch_y") for name in val_names):
        return False

    expected_train, expected_val = _expected_baseline_patch_counts()
    return len(train_names) == expected_train and len(val_names) == expected_val


def ensure_dataset_ready(train_stride=512, timeout=1800):
    """Generate the default cropped dataset once if it is missing."""
    if baseline_dataset_ready():
        return
    ensure_baseline_dataset_rebuilt(timeout=timeout)


def ensure_baseline_dataset_rebuilt(timeout=1800):
    """Force a clean baseline crop so Task 2 tests are not contaminated by overlap runs."""
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR, ignore_errors=True)
    cmd = [
        sys.executable,
        str(CODE_DIR / "step1_crop_dataset.py"),
        "--train-stride",
        "512",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(CODE_DIR),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to rebuild baseline Task 2 dataset via step1_crop_dataset.py\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    if not baseline_dataset_ready():
        raise RuntimeError("Baseline dataset rebuild finished, but cropped dataset is still incomplete.")


def make_dataset_subset(root, train_count=4, val_count=2):
    """Copy a tiny deterministic dataset subset for fast integration tests."""
    ensure_dataset_ready()
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
