"""Crop training/validation patches from the full-slide image and label.

Baseline (Task 1):
    python step1_crop_dataset.py                      # stride=512, no overlap

v2 overlap augmentation (Task 3 improvement):
    python step1_crop_dataset.py --train-stride 384   # 25% overlap, training only
    python step1_crop_dataset.py --train-stride 256   # 50% overlap, training only

The validation set is always non-overlapping (stride = tile_size) to keep the
evaluation unbiased.  Training patches are filtered so that no patch overlaps a
validation row by more than 25 % of its area.
"""

import argparse
import os
import shutil
import sys

import numpy as np
from skimage import io

RANDOM_SEED = 42
TILE_SIZE = 512
TRAIN_RATIO = 0.70

SRC_IMAGE = "img_trainval.png"
SRC_LABEL = "label_trainval.png"

DIR_TRAIN_IMAGE = "./dataset/train/image/"
DIR_TRAIN_LABEL = "./dataset/train/label/"
DIR_VAL_IMAGE = "./dataset/val/image/"
DIR_VAL_LABEL = "./dataset/val/label/"

OUTPUT_DIRS = [DIR_TRAIN_IMAGE, DIR_TRAIN_LABEL,
               DIR_VAL_IMAGE, DIR_VAL_LABEL]


# ---------------------------------------------------------------------------
# utility
# ---------------------------------------------------------------------------

def reset_output_directories():
    for directory in OUTPUT_DIRS:
        if os.path.isdir(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)


def compute_axis_positions(length, tile_size, stride):
    """Return every valid start position along one axis so that the last
    window is flush with the boundary."""
    if length < tile_size:
        raise ValueError(
            f"Image side {length} is smaller than tile size {tile_size}"
        )
    last_start = length - tile_size
    positions = list(range(0, last_start + 1, stride))
    if not positions or positions[-1] != last_start:
        positions.append(last_start)
    # Deduplicate in case stride divides evenly
    if len(positions) >= 2 and positions[-1] == positions[-2]:
        positions.pop()
    return positions


# ---------------------------------------------------------------------------
# grid & split (baseline – non-overlapping)
# ---------------------------------------------------------------------------

def compute_grid(height, width, tile_size, stride):
    y_positions = compute_axis_positions(height, tile_size, stride)
    x_positions = compute_axis_positions(width, tile_size, stride)
    positions = []
    for row, y in enumerate(y_positions):
        for col, x in enumerate(x_positions):
            positions.append((row, col, y, x))
    return positions, y_positions, x_positions


def _contiguous_ranges(indices):
    indices = sorted(indices)
    if not indices:
        return []

    ranges = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append((start, prev))
        start = prev = idx
    ranges.append((start, prev))
    return ranges


def _rows_overlap(y_positions, tile_size):
    overlap_rows = set()
    for row in range(len(y_positions) - 1):
        if y_positions[row] + tile_size > y_positions[row + 1]:
            overlap_rows.add(row)
            overlap_rows.add(row + 1)
    return overlap_rows


def choose_val_block(y_positions, x_positions, tile_size, train_ratio):
    """Pick one compact 2D validation block instead of mixing all columns in a
    few rows. This reduces spatial leakage more strictly than row-only splits."""
    total_rows = len(y_positions)
    total_cols = len(x_positions)
    total_tiles = total_rows * total_cols
    target_val = int(round(total_tiles * (1.0 - train_ratio)))
    target_val = max(1, min(total_tiles, target_val))

    overlap_rows = _rows_overlap(y_positions, tile_size)

    best_mask = None
    best_score = None
    for row_count in range(1, total_rows + 1):
        col_count_center = int(round(target_val / float(row_count)))
        candidate_cols = sorted({
            max(1, min(total_cols, col_count_center - 1)),
            max(1, min(total_cols, col_count_center)),
            max(1, min(total_cols, col_count_center + 1)),
        })
        for col_count in candidate_cols:
            area = row_count * col_count
            if area <= 0:
                continue
            for row_start in range(0, total_rows - row_count + 1):
                row_end = row_start + row_count - 1
                row_set = set(range(row_start, row_end + 1))
                overlap_hit = len(row_set & overlap_rows)

                for col_start in range(0, total_cols - col_count + 1):
                    col_end = col_start + col_count - 1
                    centre_row = row_start + (row_count - 1) / 2.0
                    centre_col = col_start + (col_count - 1) / 2.0
                    row_dist = abs(centre_row - (total_rows - 1) / 2.0)
                    col_dist = abs(centre_col - (total_cols - 1) / 2.0)
                    area_penalty = abs(area - target_val)
                    aspect_penalty = abs(np.log((row_count + 1e-8) / (col_count + 1e-8)))
                    full_width_penalty = int(col_count == total_cols)
                    full_height_penalty = int(row_count == total_rows)

                    score = (
                        full_width_penalty,
                        full_height_penalty,
                        area_penalty,
                        overlap_hit,
                        aspect_penalty,
                        row_dist + col_dist,
                        row_start,
                        col_start,
                    )
                    if best_score is None or score < best_score:
                        best_score = score
                        best_mask = {
                            "rows": row_set,
                            "cols": set(range(col_start, col_end + 1)),
                            "row_range": (row_start, row_end),
                            "col_range": (col_start, col_end),
                            "tile_count": area,
                        }

    return best_mask


def split_positions(positions, y_positions, x_positions, tile_size, train_ratio):
    val_block = choose_val_block(y_positions, x_positions, tile_size, train_ratio)
    train_positions = []
    val_positions = []
    for item in positions:
        row, col = item[0], item[1]
        if row in val_block["rows"] and col in val_block["cols"]:
            val_positions.append(item)
        else:
            train_positions.append(item)
    return train_positions, val_positions, val_block


# ---------------------------------------------------------------------------
# v2: overlapping training patches (training only, val always non-overlapping)
# ---------------------------------------------------------------------------

def _build_val_mask(y_positions, val_rows, tile_size, height):
    """Boolean mask of shape (height,) – True where a pixel belongs to a
    validation row."""
    mask = np.zeros(height, dtype=bool)
    for row in val_rows:
        y0 = y_positions[row]
        y1 = min(y0 + tile_size, height)
        mask[y0:y1] = True
    return mask


def _build_val_rect_mask(y_positions, x_positions, val_block, tile_size, height, width):
    mask = np.zeros((height, width), dtype=bool)
    for row in val_block["rows"]:
        y0 = y_positions[row]
        y1 = min(y0 + tile_size, height)
        for col in val_block["cols"]:
            x0 = x_positions[col]
            x1 = min(x0 + tile_size, width)
            mask[y0:y1, x0:x1] = True
    return mask


def _generate_train_positions_overlap(y_positions_base, x_positions_base,
                                       val_block, height, width,
                                       tile_size, train_stride,
                                       max_val_overlap_frac=0.25):
    """Generate overlapping training positions filtered against validation rows.

    Returns a flat list of (row, col, y, x) where row/col are indices in the
    *training* grid (computed with *train_stride*).
    """
    y_pos = compute_axis_positions(height, tile_size, train_stride)
    x_pos = compute_axis_positions(width,  tile_size, train_stride)

    val_mask = _build_val_rect_mask(
        y_positions_base, x_positions_base, val_block, tile_size, height, width
    )

    positions = []
    for row, y in enumerate(y_pos):
        for col, x in enumerate(x_pos):
            y1 = min(y + tile_size, height)
            x1 = min(x + tile_size, width)
            frac_val = float(val_mask[y:y1, x:x1].mean())
            if frac_val > max_val_overlap_frac:
                continue
            positions.append((row, col, y, x))

    return positions, y_pos, x_pos


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _name_patch(row, col, y, x, stride):
    """Use coordinate-based naming when training stride differs from tile_size,
    so overlapping patches are clearly distinguishable from baseline ones."""
    if stride == TILE_SIZE:
        return f"patch_r{row:04d}_c{col:04d}.png"
    else:
        return f"patch_y{y:05d}_x{x:05d}.png"


def save_patch(image, label, y, x, tile_size, row, col, stride,
               img_dir, lbl_dir):
    img_patch = image[y:y + tile_size, x:x + tile_size]
    lbl_patch = label[y:y + tile_size, x:x + tile_size]
    name = _name_patch(row, col, y, x, stride)
    io.imsave(os.path.join(img_dir, name), img_patch, check_contrast=False)
    io.imsave(os.path.join(lbl_dir, name), lbl_patch, check_contrast=False)


def save_split(patches, image, label, img_dir, lbl_dir, stride):
    for row, col, y, x in patches:
        save_patch(image, label, y, x, TILE_SIZE, row, col, stride,
                   img_dir, lbl_dir)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Crop training/validation patches (baseline & v2 overlap)"
    )
    parser.add_argument(
        "--train-stride", type=int, default=TILE_SIZE,
        help="Stride for training patches.  <512 enables overlap (v2).  "
             f"Default: {TILE_SIZE} (baseline no-overlap)."
    )
    args = parser.parse_args()

    train_stride = args.train_stride
    if train_stride <= 0:
        raise ValueError("--train-stride must be positive")
    if train_stride > TILE_SIZE:
        print("WARNING: train-stride > tile_size leaves uncovered gaps. "
              "Falling back to tile_size.",
              file=sys.stderr)
        train_stride = TILE_SIZE

    # ------------------------------------------------------------------ #
    reset_output_directories()

    image = io.imread(SRC_IMAGE)
    label = io.imread(SRC_LABEL)
    height, width = image.shape[:2]

    assert image.shape[:2] == label.shape[:2], (
        f"Dimension mismatch: image {image.shape[:2]}, label {label.shape[:2]}"
    )

    # --- validation set (always non-overlapping) ----------------------- #
    positions, y_positions, x_positions = compute_grid(
        height, width, TILE_SIZE, TILE_SIZE,
    )
    _, val_positions, val_block = split_positions(
        positions, y_positions, x_positions, TILE_SIZE, TRAIN_RATIO,
    )

    save_split(val_positions, image, label,
               DIR_VAL_IMAGE, DIR_VAL_LABEL, TILE_SIZE)

    # --- training set (overlapping when train_stride < TILE_SIZE) ------ #
    if train_stride < TILE_SIZE:
        # v2: overlap augmentation — more training samples, same val
        train_positions, train_y_pos, train_x_pos = \
            _generate_train_positions_overlap(
                y_positions, x_positions, val_block,
                height, width, TILE_SIZE, train_stride,
            )
        save_split(train_positions, image, label,
                   DIR_TRAIN_IMAGE, DIR_TRAIN_LABEL, train_stride)
        train_grid_label = (f"{len(train_y_pos)} rows x "
                            f"{len(train_x_pos)} cols")
    else:
        # baseline: same non-overlapping grid for train and val
        train_positions, _, _ = split_positions(
            positions, y_positions, x_positions, TILE_SIZE, TRAIN_RATIO,
        )
        save_split(train_positions, image, label,
                   DIR_TRAIN_IMAGE, DIR_TRAIN_LABEL, TILE_SIZE)
        train_grid_label = (f"{len(y_positions)} rows x "
                            f"{len(x_positions)} cols  (baseline no-overlap)")

    # ------------------------------------------------------------------ #
    train_count = len(train_positions)
    val_count   = len(val_positions)
    total       = train_count + val_count

    print(f"Image dimensions (H x W): {height} x {width}")
    print(f"Tile size:               {TILE_SIZE}")
    print(f"Train stride:            {train_stride}")
    print(f"Train grid:              {train_grid_label}")
    print(f"Total patches:           {total}")
    print(f"Train patches:           {train_count}")
    print(f"Val patches:             {val_count}")
    if train_count > 0:
        print(f"Train ratio:             {train_count / total:.4f}")
    print(
        "Val block (baseline grid):  "
        f"rows {_contiguous_ranges(val_block['rows'])}, "
        f"cols {_contiguous_ranges(val_block['cols'])}"
    )
    print(f"Baseline grid:            {len(y_positions)} rows x {len(x_positions)} cols")
    print(f"Random seed:              {RANDOM_SEED}")
    print("Done.")


if __name__ == "__main__":
    main()
