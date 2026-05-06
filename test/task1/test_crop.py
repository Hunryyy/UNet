"""Task 1 verification with dynamic checks against real source imagery."""
import os

import numpy as np
from PIL import Image
from skimage import io

from _task1_common import (
    SPLIT_DIRS,
    TILE_SIZE,
    TRAIN_RATIO,
    actual_split_positions,
    collect_patch_records,
    expected_positions,
    expected_split_positions,
    grid_shape,
    load_source_image,
    load_source_label,
    parse_patch_name,
    patch_window_overlap,
    source_shape,
)


def test_source_dimensions_match():
    image = load_source_image()
    label = load_source_label()
    assert image.shape[:2] == label.shape[:2], (
        f"Dimension mismatch: image {image.shape[:2]} vs label {label.shape[:2]}"
    )
    print("[PASS] test_source_dimensions_match")


def test_all_patches_512x512():
    errors = []
    for split in ["train", "val"]:
        for record in collect_patch_records(split):
            img = io.imread(record.image_path)
            lbl = io.imread(record.label_path)
            if img.shape[:2] != (TILE_SIZE, TILE_SIZE):
                errors.append(f"{split}/image/{record.name}: {img.shape}")
            if lbl.shape[:2] != (TILE_SIZE, TILE_SIZE):
                errors.append(f"{split}/label/{record.name}: {lbl.shape}")
    assert not errors, "Non-512x512 patches:\n" + "\n".join(errors)
    print("[PASS] test_all_patches_512x512")


def test_image_label_pairing():
    for split in ["train", "val"]:
        images = set(os.listdir(SPLIT_DIRS[split]["image"]))
        labels = set(os.listdir(SPLIT_DIRS[split]["label"]))
        assert images == labels, (
            f"{split}: image/label mismatch, diff={sorted(images ^ labels)[:10]}"
        )
    print("[PASS] test_image_label_pairing")


def test_patch_names_are_traceable():
    n_rows, n_cols = grid_shape()
    for split in ["train", "val"]:
        for record in collect_patch_records(split):
            parse_patch_name(record.name)
            assert 0 <= record.row < n_rows, f"Row out of range: {record.name}"
            assert 0 <= record.col < n_cols, f"Col out of range: {record.name}"
            assert record.y >= 0 and record.x >= 0, f"Negative coord: {record.name}"
    print("[PASS] test_patch_names_are_traceable")


def test_grid_coverage_is_complete_without_duplicates():
    expected = {(row, col) for row, col, _, _ in expected_positions()}
    actual_train = actual_split_positions("train")
    actual_val = actual_split_positions("val")
    actual_all = actual_train | actual_val

    assert actual_train.isdisjoint(actual_val), "Train/val share duplicated positions"
    assert actual_all == expected, (
        f"Grid mismatch: missing={sorted(expected - actual_all)[:10]}, "
        f"extra={sorted(actual_all - expected)[:10]}"
    )
    print("[PASS] test_grid_coverage_is_complete_without_duplicates")


def test_pixel_alignment():
    src_img = load_source_image()
    src_lbl = load_source_label()

    for split in ["train", "val"]:
        for record in collect_patch_records(split):
            patch_img = io.imread(record.image_path)
            patch_lbl = io.imread(record.label_path)

            src_roi_img = src_img[record.y:record.y + TILE_SIZE, record.x:record.x + TILE_SIZE]
            src_roi_lbl = src_lbl[record.y:record.y + TILE_SIZE, record.x:record.x + TILE_SIZE]

            assert np.array_equal(patch_img, src_roi_img), (
                f"Image mismatch: {record.name} at y={record.y}, x={record.x}"
            )
            assert np.array_equal(patch_lbl, src_roi_lbl), (
                f"Label mismatch: {record.name} at y={record.y}, x={record.x}"
            )
    print("[PASS] test_pixel_alignment")


def test_boundary_coverage():
    h, w = source_shape()
    records = collect_patch_records("train") + collect_patch_records("val")

    assert any(record.x + TILE_SIZE >= w for record in records), "Right edge not covered"
    assert any(record.y + TILE_SIZE >= h for record in records), "Bottom edge not covered"
    assert any(record.x == w - TILE_SIZE for record in records), "Missing rightmost flush tile"
    assert any(record.y == h - TILE_SIZE for record in records), "Missing bottom flush tile"
    print("[PASS] test_boundary_coverage")


def test_label_semantics_preserved():
    src_unique = set(np.unique(load_source_label()).tolist())
    observed = set()
    for split in ["train", "val"]:
        for record in collect_patch_records(split):
            observed.update(np.unique(io.imread(record.label_path)).tolist())
    assert observed.issubset(src_unique), (
        f"Label values changed: observed={sorted(observed)}, source={sorted(src_unique)}"
    )
    print("[PASS] test_label_semantics_preserved")


def test_split_counts_and_assignment_match_strategy():
    expected_train, expected_val = expected_split_positions()
    actual_train = actual_split_positions("train")
    actual_val = actual_split_positions("val")

    total = len(expected_train) + len(expected_val)
    assert total == len(expected_positions())
    assert len(expected_train) == int(total * TRAIN_RATIO)
    assert len(expected_val) == total - len(expected_train)
    assert actual_train == expected_train, (
        "Train assignment differs from expected deterministic split strategy: "
        f"missing={sorted(expected_train - actual_train)[:10]}, "
        f"extra={sorted(actual_train - expected_train)[:10]}"
    )
    assert actual_val == expected_val, (
        "Val assignment differs from expected deterministic split strategy: "
        f"missing={sorted(expected_val - actual_val)[:10]}, "
        f"extra={sorted(actual_val - expected_val)[:10]}"
    )
    print("[PASS] test_split_counts_and_assignment_match_strategy")


def test_saved_patches_are_pil_loadable():
    for split in ["train", "val"]:
        for record in collect_patch_records(split):
            img = Image.open(record.image_path)
            img.load()
            lbl = Image.open(record.label_path)
            lbl.load()
    print("[PASS] test_saved_patches_are_pil_loadable")


def test_no_cross_split_physical_overlap():
    train_records = collect_patch_records("train")
    val_records = collect_patch_records("val")
    overlaps = []

    for train_record in train_records:
        for val_record in val_records:
            if abs(train_record.row - val_record.row) > 1:
                continue
            if abs(train_record.col - val_record.col) > 1:
                continue
            y_overlap, x_overlap = patch_window_overlap(train_record, val_record)
            if y_overlap > 0 and x_overlap > 0:
                overlaps.append(
                    (
                        train_record.name,
                        val_record.name,
                        y_overlap,
                        x_overlap,
                    )
                )

    assert not overlaps, (
        "Train/val contain physically overlapping patches, causing spatial leakage. "
        f"Examples: {overlaps[:10]}"
    )
    print("[PASS] test_no_cross_split_physical_overlap")


if __name__ == "__main__":
    test_source_dimensions_match()
    test_all_patches_512x512()
    test_image_label_pairing()
    test_patch_names_are_traceable()
    test_grid_coverage_is_complete_without_duplicates()
    test_pixel_alignment()
    test_boundary_coverage()
    test_label_semantics_preserved()
    test_split_counts_and_assignment_match_strategy()
    test_saved_patches_are_pil_loadable()
    test_no_cross_split_physical_overlap()
    print("\n=== Task 1 crop tests completed. ===")
