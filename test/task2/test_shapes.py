"""
Test 2: Shape/dtype/device consistency verification.

Every component in the pipeline must produce and consume tensors
with consistent shapes, dtypes, and devices.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TEST_DIR)

import numpy as np
import torch
from torch.utils.data import DataLoader

from _task2_common import CODE_DIR, DATA_MEAN, DATA_STD, ensure_baseline_dataset_rebuilt
from dataset import MyDataset
from unet_model import Res34UNet_light

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    os.chdir(CODE_DIR)
    ensure_baseline_dataset_rebuilt()

    print("=" * 60)
    print("TEST 2: Shape / dtype / device verification")
    print("=" * 60)

    # --- 2.1 Dataset output shapes ---
    print("\n[2.1] Dataset output shapes")
    for is_train in [True, False]:
        subdir = "train" if is_train else "val"
        ds = MyDataset(
            f"./dataset/{subdir}/image/",
            f"./dataset/{subdir}/label/",
            mean=DATA_MEAN, std=DATA_STD, is_train=is_train)
        sample = ds[0]
        img, mask = sample["image"], sample["mask"]

        assert img.ndim == 3, f"Image should be 3D [C,H,W], got {img.shape}"
        assert mask.ndim == 2, f"Mask should be 2D [H,W], got {mask.shape}"
        assert img.shape[1:] == mask.shape[:], (
            f"Spatial mismatch: img {img.shape[1:]} vs mask {mask.shape}")
        assert img.shape[0] == 3, f"Image should have 3 channels, got {img.shape[0]}"
        assert img.shape[1] == 512 and img.shape[2] == 512, (
            f"Expected 512x512, got {img.shape[1]}x{img.shape[2]}")
        assert img.dtype == torch.float32, f"Image dtype: {img.dtype}"
        assert mask.dtype == torch.float32, f"Mask dtype: {mask.dtype}"
        print(f"  {'train' if is_train else 'val':>5}: "
              f"img {list(img.shape)} {img.dtype}, mask {list(mask.shape)} {mask.dtype}  OK")

    # --- 2.2 DataLoader batching ---
    print("\n[2.2] DataLoader batching")
    ds = MyDataset(
        "./dataset/train/image/", "./dataset/train/label/",
        mean=DATA_MEAN, std=DATA_STD, is_train=True)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    imgs, masks = batch["image"], batch["mask"]

    assert imgs.shape == (4, 3, 512, 512), f"Batch img shape: {imgs.shape}"
    assert masks.shape == (4, 512, 512), f"Batch mask shape: {masks.shape}"
    assert imgs.dtype == torch.float32
    print(f"  Batch shapes: imgs {list(imgs.shape)}, masks {list(masks.shape)}  OK")

    # --- 2.3 Model forward shapes (train mode) ---
    print("\n[2.3] Model forward — train mode")
    model = Res34UNet_light().to(DEVICE)
    model.train()
    imgs_dev = imgs.to(DEVICE)
    masks_dev = masks.to(DEVICE)
    loss, out = model(imgs_dev, masks_dev)

    assert out.shape == (4, 1, 512, 512), f"Train out shape: {out.shape}"
    assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
    print(f"  loss: {loss.item():.4f}, out shape: {list(out.shape)}  OK")

    # --- 2.4 Model forward shapes (eval mode) ---
    print("\n[2.4] Model forward — eval mode")
    model.eval()
    with torch.no_grad():
        out = model(imgs_dev)
    assert out.shape == (4, 1, 512, 512), f"Eval out shape: {out.shape}"
    print(f"  out shape: {list(out.shape)}  OK")

    # --- 2.5 Device consistency ---
    print("\n[2.5] Device consistency")
    imgs_dev2 = imgs.to(DEVICE)
    masks_dev2 = masks.to(DEVICE)
    model.train()
    _, out = model(imgs_dev2, masks_dev2)
    assert out.device.type == DEVICE.type, f"Output device {out.device} != {DEVICE}"
    print(f"  Input device: {imgs_dev2.device}, Output device: {out.device}  OK")

    # --- 2.6 Label binarization ---
    print("\n[2.6] Label binarization")
    ds_val = MyDataset(
        "./dataset/val/image/", "./dataset/val/label/",
        mean=DATA_MEAN, std=DATA_STD, is_train=False)
    for i in range(min(10, len(ds_val))):
        sample = ds_val[i]
        mask_vals = sample["mask"].unique()
        assert all(v in [0.0, 1.0] for v in mask_vals.tolist()), (
            f"Mask should be binary 0/1, got unique values: {mask_vals.tolist()}")
    print("  All checked masks are binary {0, 1}  OK")

    # --- 2.7 Normalization parameter propagation ---
    print("\n[2.7] Normalization consistency")
    ds1 = MyDataset(
        "./dataset/train/image/", "./dataset/train/label/",
        mean=DATA_MEAN, std=DATA_STD, is_train=False)
    ds2 = MyDataset(
        "./dataset/train/image/", "./dataset/train/label/",
        mean=DATA_MEAN, std=DATA_STD, is_train=False)
    s1, s2 = ds1[0]["image"], ds2[0]["image"]
    assert torch.allclose(s1, s2, atol=1e-6), (
        "Same index with same normalization should produce identical tensor")
    print("  Identical normalization for identical inputs  OK")

    print("\n" + "=" * 60)
    print("TEST 2 PASSED — All shape/dtype/device checks")
    print("=" * 60)
    return True


if __name__ == "__main__":
    main()
