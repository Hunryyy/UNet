"""
Test 10: Training augmentation should vary across epochs but stay reproducible.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TEST_DIR)

import torch

from _task2_common import CODE_DIR, DATA_MEAN, DATA_STD, ensure_baseline_dataset_rebuilt
from dataset import MyDataset


def main():
    os.chdir(CODE_DIR)
    ensure_baseline_dataset_rebuilt()

    print("=" * 60)
    print("TEST 10: Epoch-aware augmentation reproducibility")
    print("=" * 60)

    ds = MyDataset(
        "./dataset/train/image/",
        "./dataset/train/label/",
        mean=DATA_MEAN,
        std=DATA_STD,
        is_train=True,
        seed=42,
    )

    ds.set_epoch(0)
    sample_epoch0_a = ds[0]
    sample_epoch0_b = ds[0]
    assert torch.equal(sample_epoch0_a["image"], sample_epoch0_b["image"])
    assert torch.equal(sample_epoch0_a["mask"], sample_epoch0_b["mask"])

    changed = False
    for idx in range(min(16, len(ds))):
        ds.set_epoch(0)
        sample0 = ds[idx]
        ds.set_epoch(1)
        sample1 = ds[idx]
        if not torch.equal(sample0["image"], sample1["image"]) or not torch.equal(
            sample0["mask"], sample1["mask"]
        ):
            changed = True
            break

    assert changed, (
        "Training augmentation is identical across epochs for every checked sample; "
        "augmentation diversity did not improve."
    )

    ds_recreated = MyDataset(
        "./dataset/train/image/",
        "./dataset/train/label/",
        mean=DATA_MEAN,
        std=DATA_STD,
        is_train=True,
        seed=42,
    )
    ds.set_epoch(3)
    ds_recreated.set_epoch(3)
    sample_ref = ds[0]
    sample_recreated = ds_recreated[0]
    assert torch.equal(sample_ref["image"], sample_recreated["image"])
    assert torch.equal(sample_ref["mask"], sample_recreated["mask"])

    print("  Same epoch gives deterministic samples; different epochs can differ  OK")
    print("=" * 60)
    print("TEST 10 PASSED — Epoch-aware augmentation works and is reproducible")
    print("=" * 60)
    return True


if __name__ == "__main__":
    main()
