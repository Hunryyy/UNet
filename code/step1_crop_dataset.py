import os
import shutil

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

OUTPUT_DIRS = [DIR_TRAIN_IMAGE, DIR_TRAIN_LABEL, DIR_VAL_IMAGE, DIR_VAL_LABEL]


def reset_output_directories():
    for directory in OUTPUT_DIRS:
        if os.path.isdir(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)


def compute_axis_positions(length, tile_size):
    if length < tile_size:
        raise ValueError(f"Image side {length} is smaller than tile size {tile_size}")

    n_full = length // tile_size
    has_edge = (length % tile_size) != 0
    count = n_full + (1 if has_edge else 0)

    positions = []
    for idx in range(count):
        pos = idx * tile_size if idx < n_full else length - tile_size
        if not positions or pos != positions[-1]:
            positions.append(pos)
    return positions


def compute_grid(height, width, tile_size):
    y_positions = compute_axis_positions(height, tile_size)
    x_positions = compute_axis_positions(width, tile_size)

    positions = []
    for row, y in enumerate(y_positions):
        for col, x in enumerate(x_positions):
            positions.append((row, col, y, x))
    return positions, y_positions, x_positions


def choose_val_rows(y_positions, tile_size, train_ratio):
    total_rows = len(y_positions)
    target_val = int(round(total_rows * (1.0 - train_ratio)))
    target_val = max(1, min(total_rows, target_val))

    overlap_rows = set()
    for row in range(total_rows - 1):
        if y_positions[row] + tile_size > y_positions[row + 1]:
            overlap_rows.add(row)
            overlap_rows.add(row + 1)

    best_rows = None
    best_score = None
    for start in range(0, total_rows - target_val + 1):
        rows = set(range(start, start + target_val))
        overlap_hit = len(rows & overlap_rows)
        center_distance = abs((start + (target_val - 1) / 2.0) - (total_rows - 1) / 2.0)
        score = (overlap_hit, center_distance, start)
        if best_score is None or score < best_score:
            best_score = score
            best_rows = rows

    return best_rows


def split_positions(positions, y_positions, tile_size, train_ratio):
    val_rows = choose_val_rows(y_positions, tile_size, train_ratio)
    train_positions = []
    val_positions = []

    for item in positions:
        row = item[0]
        if row in val_rows:
            val_positions.append(item)
        else:
            train_positions.append(item)

    return train_positions, val_positions, val_rows


def save_patch(image, label, y, x, tile_size, row, col, img_dir, lbl_dir):
    img_patch = image[y:y + tile_size, x:x + tile_size]
    lbl_patch = label[y:y + tile_size, x:x + tile_size]
    name = f"patch_r{row:04d}_c{col:04d}.png"
    io.imsave(os.path.join(img_dir, name), img_patch, check_contrast=False)
    io.imsave(os.path.join(lbl_dir, name), lbl_patch, check_contrast=False)


def save_split(patches, image, label, img_dir, lbl_dir):
    for row, col, y, x in patches:
        save_patch(image, label, y, x, TILE_SIZE, row, col, img_dir, lbl_dir)


def main():
    reset_output_directories()

    image = io.imread(SRC_IMAGE)
    label = io.imread(SRC_LABEL)
    height, width = image.shape[:2]

    assert image.shape[:2] == label.shape[:2], (
        f"Dimension mismatch: image {image.shape[:2]}, label {label.shape[:2]}"
    )

    positions, y_positions, x_positions = compute_grid(height, width, TILE_SIZE)
    total_patches = len(positions)
    assert total_patches == len(y_positions) * len(x_positions)

    train_positions, val_positions, val_rows = split_positions(
        positions, y_positions, TILE_SIZE, TRAIN_RATIO
    )

    save_split(train_positions, image, label, DIR_TRAIN_IMAGE, DIR_TRAIN_LABEL)
    save_split(val_positions, image, label, DIR_VAL_IMAGE, DIR_VAL_LABEL)

    train_count = len(train_positions)
    val_count = len(val_positions)
    assert train_count + val_count == total_patches

    print(f"Image dimensions (H x W): {height} x {width}")
    print(f"Tile size: {TILE_SIZE}")
    print(f"Tile grid: {len(y_positions)} rows x {len(x_positions)} cols")
    print(f"Total patches: {total_patches}")
    print(f"Train patches: {train_count}")
    print(f"Val patches:   {val_count}")
    print(f"Train ratio:   {train_count / total_patches:.4f}")
    print(f"Val rows:      {sorted(val_rows)}")
    print(f"Last-row y:    {y_positions[-1]}")
    print(f"Last-col x:    {x_positions[-1]}")
    print(f"Random seed:   {RANDOM_SEED} (reserved for future randomized train-only expansion)")
    print("Done.")


if __name__ == "__main__":
    main()
