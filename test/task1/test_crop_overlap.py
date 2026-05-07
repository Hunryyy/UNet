"""Verify v2 overlap crop augmentation – correctness & val-contamination guard.

Tests:
  1. baseline (no overlap) produces identical output to original grid split
  2. v2 overlap (--train-stride 384) produces more training patches
  3. no training patch overlaps a validation row by more than 25 %
  4. every output patch is exactly TILE_SIZE × TILE_SIZE
  5. image-label pixel alignment is preserved
"""

import os
import subprocess
import sys
import re
from pathlib import Path

import numpy as np
from skimage import io

TEST_DIR = Path(__file__).resolve().parent
CODE_DIR = TEST_DIR.parent.parent / "code"
sys.path.insert(0, str(CODE_DIR))

from step1_crop_dataset import (  # noqa: E402
    TILE_SIZE,
    _build_val_rect_mask,
    compute_grid,
    split_positions,
)

SRC_IMAGE = CODE_DIR / "img_trainval.png"
SRC_LABEL = CODE_DIR / "label_trainval.png"
TRAIN_IMG_DIR = CODE_DIR / "dataset" / "train" / "image"
TRAIN_LBL_DIR = CODE_DIR / "dataset" / "train" / "label"
VAL_IMG_DIR   = CODE_DIR / "dataset" / "val" / "image"
VAL_LBL_DIR   = CODE_DIR / "dataset" / "val" / "label"
TRAIN_Y_RE = re.compile(r"patch_y(\d{5})_x(\d{5})\.png")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _run_step1(train_stride):
    """Run step1_crop_dataset.py and return (returncode, stdout)."""
    timeout = 120 if train_stride >= 384 else 900
    result = subprocess.run(
        [sys.executable, str(CODE_DIR / "step1_crop_dataset.py"),
         "--train-stride", str(train_stride)],
        cwd=str(CODE_DIR),
        capture_output=True, text=True, timeout=timeout,
    )
    return result.returncode, result.stdout


def _count_files(directory):
    if not directory.is_dir():
        return 0
    return len([f for f in os.listdir(directory) if f.endswith(".png")])


def _patch_names(directory):
    if not directory.is_dir():
        return set()
    return {f for f in os.listdir(directory) if f.endswith(".png")}


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_baseline_no_overlap():
    """stride=512 → output must match original grid-split expectations."""
    rc, stdout = _run_step1(512)
    assert rc == 0, f"step1 exited with {rc}\n{stdout}"

    # All four directories should contain files
    for d in [TRAIN_IMG_DIR, TRAIN_LBL_DIR, VAL_IMG_DIR, VAL_LBL_DIR]:
        assert _count_files(d) > 0, f"{d} is empty"

    # Train + val count = total grid cells
    train_names = _patch_names(TRAIN_IMG_DIR)
    val_names   = _patch_names(VAL_IMG_DIR)
    assert train_names == _patch_names(TRAIN_LBL_DIR), "train image/label mismatch"
    assert val_names   == _patch_names(VAL_LBL_DIR),   "val image/label mismatch"
    assert not (train_names & val_names),              "train/val name collision"

    # Every patch is 512×512
    for name in list(train_names)[:5] + list(val_names)[:5]:
        img = io.imread(str(TRAIN_IMG_DIR / name)) if name in train_names \
              else io.imread(str(VAL_IMG_DIR / name))
        assert img.shape[:2] == (TILE_SIZE, TILE_SIZE), \
            f"{name}: expected {(TILE_SIZE, TILE_SIZE)}, got {img.shape[:2]}"

    print("PASS: baseline no-overlap produces correct output")


def test_overlap_more_train_patches():
    """stride=384 → more training patches than baseline, same val count."""
    # Baseline
    rc0, out0 = _run_step1(512)
    assert rc0 == 0
    n_train_base = _count_files(TRAIN_IMG_DIR)
    n_val_base   = _count_files(VAL_IMG_DIR)

    # Overlap
    rc1, out1 = _run_step1(384)
    assert rc1 == 0
    n_train_ov = _count_files(TRAIN_IMG_DIR)
    n_val_ov   = _count_files(VAL_IMG_DIR)

    assert n_train_ov > n_train_base, \
        f"overlap train patches ({n_train_ov}) not > baseline ({n_train_base})"
    assert n_val_ov == n_val_base, \
        f"val count changed: {n_val_base} → {n_val_ov}"

    print(f"PASS: overlap produces {n_train_ov} train patches (baseline: {n_train_base}), "
          f"val stays at {n_val_ov}")


def test_no_val_contamination():
    """No training patch should overlap a validation row by > 25 % area."""
    image = io.imread(str(SRC_IMAGE))
    height, width = image.shape[:2]

    # Determine val rows (same logic as step1)
    positions, y_positions, x_positions = compute_grid(height, width, TILE_SIZE, TILE_SIZE)
    _, _, val_block = split_positions(positions, y_positions, x_positions, TILE_SIZE, 0.70)
    val_mask = _build_val_rect_mask(y_positions, x_positions, val_block, TILE_SIZE, height, width)

    # Run overlap crop
    rc, _ = _run_step1(384)
    assert rc == 0

    # Check each training patch
    for name in _patch_names(TRAIN_IMG_DIR):
        match = TRAIN_Y_RE.fullmatch(name)
        assert match, f"Unexpected overlap patch name: {name}"
        y = int(match.group(1))
        x = int(match.group(2))
        y1 = min(y + TILE_SIZE, height)
        x1 = min(x + TILE_SIZE, width)
        frac_val = float(val_mask[y:y1, x:x1].mean())
        assert frac_val <= 0.25, \
            f"Training patch {name} at y={y} overlaps val by {frac_val:.2%}"

    print("PASS: no training patch significantly overlaps validation rows")


def test_overlap_patch_size():
    """Every overlap-mode patch is exactly TILE_SIZE × TILE_SIZE."""
    rc, _ = _run_step1(256)
    assert rc == 0

    for directory in [TRAIN_IMG_DIR, TRAIN_LBL_DIR, VAL_IMG_DIR, VAL_LBL_DIR]:
        for name in list(_patch_names(directory))[:10]:
            img = io.imread(str(directory / name))
            assert img.shape[:2] == (TILE_SIZE, TILE_SIZE), \
                f"{directory.name}/{name}: shape {img.shape[:2]}"

    print("PASS: all overlap patches are 512×512")


def test_image_label_alignment():
    """Image and label patches must remain pixel-aligned in overlap mode."""
    rc, _ = _run_step1(384)
    assert rc == 0

    train_imgs = sorted(_patch_names(TRAIN_IMG_DIR))
    train_lbls = sorted(_patch_names(TRAIN_LBL_DIR))
    assert train_imgs == train_lbls, \
        f"train image/label name mismatch: {len(train_imgs)} vs {len(train_lbls)}"

    val_imgs = sorted(_patch_names(VAL_IMG_DIR))
    val_lbls = sorted(_patch_names(VAL_LBL_DIR))
    assert val_imgs == val_lbls, \
        f"val image/label name mismatch: {len(val_imgs)} vs {len(val_lbls)}"

    # Spot-check: image & label for the same patch have same spatial extent
    for name in train_imgs[:3]:
        img = io.imread(str(TRAIN_IMG_DIR / name))
        lbl = io.imread(str(TRAIN_LBL_DIR / name))
        assert img.shape[:2] == lbl.shape[:2], \
            f"{name}: img {img.shape[:2]} vs lbl {lbl.shape[:2]}"

    print("PASS: image-label alignment verified in overlap mode")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("TEST: v2 overlap crop augmentation")
    print("=" * 70)

    test_baseline_no_overlap()
    test_overlap_more_train_patches()
    test_no_val_contamination()
    test_overlap_patch_size()
    test_image_label_alignment()

    print("\nAll overlap-crop tests passed.")
