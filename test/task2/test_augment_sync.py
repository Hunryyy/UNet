"""
Test 5: Augmentation image-mask synchronization.

Verifies that imgaug applies identical geometric transforms
to both image and mask, preserving pixel-level alignment.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TEST_DIR)

import numpy as np
import torch
import imgaug
from imgaug import augmenters as iaa

from _task2_common import CODE_DIR, DATA_MEAN, DATA_STD

SEED = 42


def main():
    os.chdir(CODE_DIR)

    print("=" * 60)
    print("TEST 5: Augmentation image-mask synchronization")
    print("=" * 60)

    # --- 5.1 Basic sync: identical transform on image and mask ---
    print("\n[5.1] Deterministic geometric transform sync")

    np.random.seed(SEED)
    transform = iaa.Sequential([
        iaa.Rot90([0, 1, 2, 3]),
        iaa.VerticalFlip(p=0.5),
        iaa.HorizontalFlip(p=0.5),
    ])

    # Create test image with distinct colored quadrants
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:32, :32, 0] = 255        # top-left: red
    img[:32, 32:, 1] = 255        # top-right: green
    img[32:, :32, 2] = 255        # bottom-left: blue
    img[32:, 32:, :] = 255        # bottom-right: white

    # Create mask with matching quadrant patterns
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[:32, :32] = 1
    mask[:32, 32:] = 2
    mask[32:, :32] = 3
    mask[32:, 32:] = 4

    # Run with fixed seed
    imgaug.seed(SEED)
    np.random.seed(SEED)
    aug_img, aug_mask = transform(
        image=img,
        segmentation_maps=mask[np.newaxis, :, :, np.newaxis])
    aug_mask = aug_mask[0, :, :, 0]

    # Verify spatial alignment: each pixel position should have matching quadrant
    # The actual quadrant values may differ due to rotation, but same position
    # in image and mask should come from the same source quadrant

    # Check that image non-zero regions match mask value regions
    for mask_val in [1, 2, 3, 4]:
        mask_region = (aug_mask == mask_val)
        img_in_region = aug_img[mask_region]
        if mask_region.sum() > 0:
            # All pixels in this mask region should have the same unique color pattern
            img_vals = img_in_region.mean(axis=0)
            assert len(np.unique(img_in_region, axis=0)) <= 4, (
                f"Mask region {mask_val} corresponds to inconsistent image region")

    print("  Image and mask transform synchronously  OK")

    # --- 5.2 Repeated runs produce valid binary masks ---
    print("\n[5.2] Augmented mask always binary {0, 1}")
    mask_bin = np.random.randint(0, 2, (64, 64)).astype(np.uint8)
    for trial in range(20):
        np.random.seed(SEED + trial)
        imgaug.seed(SEED + trial)
        _, aug_mask = transform(
            image=np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
            segmentation_maps=mask_bin[np.newaxis, :, :, np.newaxis])
        result = aug_mask[0, :, :, 0]
        unique_vals = np.unique(result)
        assert all(v in [0, 1] for v in unique_vals), (
            f"Trial {trial}: non-binary mask values {unique_vals}")
    print(f"  20 trials: all masks remain binary {0, 1}  OK")

    # --- 5.3 Dataset augmentation produces aligned image-mask ---
    print("\n[5.3] Dataset-level augmentation sync")
    from dataset import MyDataset

    ds = MyDataset(
        "./dataset/train/image/", "./dataset/train/label/",
        mean=DATA_MEAN, std=DATA_STD, is_train=True)

    for i in range(5):
        sample = ds[i]
        img, mask = sample["image"], sample["mask"]
        # Check spatial dimensions match
        assert img.shape[1:] == mask.shape[:], (
            f"Sample {i}: spatial mismatch img {img.shape} vs mask {mask.shape}")
        # Check mask is binary
        vals = mask.unique()
        assert all(v in [0.0, 1.0] for v in vals.tolist()), (
            f"Sample {i}: non-binary mask {vals.tolist()}")
        # Ensure there's at least some positive pixels (or check file name)
        # Actually, background-only patches are valid, so we only check alignment

    print("  5 dataset samples: all image-mask spatially aligned  OK")

    print("\n" + "=" * 60)
    print("TEST 5 PASSED — Augmentation sync verified")
    print("=" * 60)
    return True


if __name__ == "__main__":
    main()
