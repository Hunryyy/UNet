import os
import shutil

import numpy as np
from skimage import io

RANDOM_SEED = 42
TILE_SIZE = 512
STRIDE = 384
TRAIN_RATIO = 0.70

SRC_IMAGE = "img_trainval.png"
SRC_LABEL = "label_trainval.png"

DIR_TRAIN_IMAGE = "./dataset/train_overlap/image/"
DIR_TRAIN_LABEL = "./dataset/train_overlap/label/"
DIR_VAL_IMAGE = "./dataset/val_overlap/image/"
DIR_VAL_LABEL = "./dataset/val_overlap/label/"

OUTPUT_DIRS = [DIR_TRAIN_IMAGE, DIR_TRAIN_LABEL, DIR_VAL_IMAGE, DIR_VAL_LABEL]


def reset_output_directories():
    for directory in OUTPUT_DIRS:
        if os.path.isdir(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)


def compute_axis_positions(length, tile_size, stride):
    if length < tile_size:
        return [0]
    n_full = (length - tile_size) // stride + 1
    positions = [i * stride for i in range(n_full)]
    last_pos = n_full * stride
    if last_pos + tile_size < length:
        positions.append(length - tile_size)
    elif last_pos < length:
        rightmost = length - tile_size
        if rightmost != positions[-1]:
            positions.append(rightmost)
    return sorted(set(positions))


def compute_grid(height, width, tile_size, stride):
    y_positions = compute_axis_positions(height, tile_size, stride)
    x_positions = compute_axis_positions(width, tile_size, stride)
    positions = []
    for row, y in enumerate(y_positions):
        for col, x in enumerate(x_positions):
            positions.append((row, col, y, x))
    return positions, y_positions, x_positions


def split_train_val_by_stripe(positions, y_positions, train_ratio):
    rng = np.random.default_rng(RANDOM_SEED)
    total_rows = len(y_positions)
    n_val_rows = max(1, int(round(total_rows * (1.0 - train_ratio))))
    val_row_indices = set(rng.choice(total_rows, size=n_val_rows, replace=False))

    train_positions = []
    val_positions = []
    for item in positions:
        row = item[0]
        if row in val_row_indices:
            val_positions.append(item)
        else:
            train_positions.append(item)

    return train_positions, val_positions, sorted(val_row_indices)


def save_patch(image, label, y, x, tile_size, row, col, img_dir, lbl_dir, prefix):
    img_patch = image[y:y + tile_size, x:x + tile_size]
    lbl_patch = label[y:y + tile_size, x:x + tile_size]
    name = f"{prefix}_r{row:04d}_c{col:04d}.png"
    io.imsave(os.path.join(img_dir, name), img_patch, check_contrast=False)
    io.imsave(os.path.join(lbl_dir, name), lbl_patch, check_contrast=False)


def save_split(patches, image, label, img_dir, lbl_dir, prefix):
    for row, col, y, x in patches:
        save_patch(image, label, y, x, TILE_SIZE, row, col, img_dir, lbl_dir, prefix)


def main(stride=None):
    if stride is None:
        stride = STRIDE
    reset_output_directories()

    image = io.imread(SRC_IMAGE)
    label = io.imread(SRC_LABEL)

    assert image.shape[:2] == label.shape[:2], (
        f"Dimension mismatch: image {image.shape[:2]}, label {label.shape[:2]}"
    )
    height, width = image.shape[:2]

    positions, y_positions, x_positions = compute_grid(height, width, TILE_SIZE, stride)

    np.random.seed(RANDOM_SEED)
    train_positions, val_positions, val_rows = split_train_val_by_stripe(
        positions, y_positions, TRAIN_RATIO
    )

    save_split(train_positions, image, label, DIR_TRAIN_IMAGE, DIR_TRAIN_LABEL, "t")
    save_split(val_positions, image, label, DIR_VAL_IMAGE, DIR_VAL_LABEL, "v")

    train_count = len(train_positions)
    val_count = len(val_positions)
    total = train_count + val_count

    print(f"Image dimensions (H x W): {height} x {width}")
    print(f"Tile size: {TILE_SIZE}  Stride: {stride}  Overlap: {TILE_SIZE - stride}px")
    print(f"Tile grid: {len(y_positions)} rows x {len(x_positions)} cols")
    print(f"Total patches: {total}")
    print(f"Train patches: {train_count}")
    print(f"Val patches:   {val_count}")
    print(f"Val rows (stripe): {val_rows}")
    print(f"Expansion factor vs non-overlap: {total / (len(compute_axis_positions(height, TILE_SIZE, TILE_SIZE)) * len(compute_axis_positions(width, TILE_SIZE, TILE_SIZE))):.1f}x")
    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=STRIDE)
    args = parser.parse_args()
    main(stride=args.stride)
