"""Verify Task 1 outputs can be consumed by dataset.MyDataset."""
import os
import sys

import torch

from _task1_common import (
    CODE_DIR,
    SPLIT_DIRS,
    TILE_SIZE,
    collect_patch_records,
    expected_split_positions,
)

sys.path.insert(0, CODE_DIR)
os.chdir(CODE_DIR)

from dataset import MyDataset

DATA_MEAN = [0.43782742, 0.44557303, 0.41160695]
DATA_STD = [0.19686149, 0.18481555, 0.19296625]


def build_datasets():
    train_ds = MyDataset(
        "./dataset/train/image/",
        "./dataset/train/label/",
        mean=DATA_MEAN,
        std=DATA_STD,
        is_train=True,
    )
    val_ds = MyDataset(
        "./dataset/val/image/",
        "./dataset/val/label/",
        mean=DATA_MEAN,
        std=DATA_STD,
        is_train=False,
    )
    return train_ds, val_ds


def assert_dataset_lengths(train_ds, val_ds):
    expected_train, expected_val = expected_split_positions()
    assert len(train_ds) == len(expected_train), (
        f"Train len mismatch: got {len(train_ds)}, expected {len(expected_train)}"
    )
    assert len(val_ds) == len(expected_val), (
        f"Val len mismatch: got {len(val_ds)}, expected {len(expected_val)}"
    )


def assert_dataset_name_sets(train_ds, val_ds):
    train_names = set(train_ds.ids)
    val_names = set(val_ds.ids)
    disk_train = {record.name for record in collect_patch_records("train")}
    disk_val = {record.name for record in collect_patch_records("val")}

    assert train_names == disk_train, "Train dataset ids differ from train directory"
    assert val_names == disk_val, "Val dataset ids differ from val directory"
    assert train_names.isdisjoint(val_names), "Train/val ids overlap inside MyDataset"


def assert_sample_structure(sample):
    assert set(sample.keys()) == {"image", "mask", "name"}
    image = sample["image"]
    mask = sample["mask"]
    assert image.ndim == 3, f"Image dims: {image.ndim}"
    assert mask.ndim == 2, f"Mask dims: {mask.ndim}"
    assert image.shape[0] == 3, f"Image channels: {image.shape[0]}"
    assert image.shape[1:] == (TILE_SIZE, TILE_SIZE), f"Wrong image shape: {image.shape}"
    assert mask.shape == (TILE_SIZE, TILE_SIZE), f"Wrong mask shape: {mask.shape}"
    assert image.dtype == torch.float32, f"Unexpected image dtype: {image.dtype}"
    assert mask.dtype == torch.float32, f"Unexpected mask dtype: {mask.dtype}"
    assert torch.all((mask == 0) | (mask == 1)), "Mask is not binary after MyDataset"
    assert torch.isfinite(image).all(), "Image contains NaN/Inf after normalization"


def assert_training_augmentation_preserves_name_coverage(train_ds):
    sample_names = {train_ds[i]["name"] for i in range(min(32, len(train_ds)))}
    assert sample_names.issubset(set(train_ds.ids))
    assert sample_names, "No train samples could be read"


def assert_non_train_loader_is_stable(val_ds):
    sample_a = val_ds[0]
    sample_b = val_ds[0]
    assert torch.equal(sample_a["image"], sample_b["image"]), "Val image should be deterministic"
    assert torch.equal(sample_a["mask"], sample_b["mask"]), "Val mask should be deterministic"
    assert sample_a["name"] == sample_b["name"]


def main():
    train_ds, val_ds = build_datasets()
    print(f"Train dataset: {len(train_ds)} samples")
    print(f"Val dataset:   {len(val_ds)} samples")

    assert_dataset_lengths(train_ds, val_ds)
    assert_dataset_name_sets(train_ds, val_ds)

    train_sample = train_ds[0]
    val_sample = val_ds[0]
    assert_sample_structure(train_sample)
    assert_sample_structure(val_sample)
    assert_training_augmentation_preserves_name_coverage(train_ds)
    assert_non_train_loader_is_stable(val_ds)

    building_count = 0
    for i in range(min(50, len(train_ds))):
        if train_ds[i]["mask"].max() > 0:
            building_count += 1
    print(f"Train sample: image={train_sample['image'].shape}, mask={train_sample['mask'].shape}")
    print(f"Val sample:   image={val_sample['image'].shape}, mask={val_sample['mask'].shape}")
    print(f"Patches with buildings in first 50 train samples: {building_count}/50")
    print("\n=== Dataset compatibility tests passed. ===")


if __name__ == "__main__":
    main()
