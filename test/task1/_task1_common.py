"""Shared helpers for Task 1 output verification."""
import os
import re
import warnings
from dataclasses import dataclass
from functools import lru_cache

from PIL import Image
from skimage import io

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(TEST_DIR, "..", "..", "code")

TILE_SIZE = 512
TRAIN_RATIO = 0.70
RANDOM_SEED = 42

SRC_IMAGE = os.path.join(CODE_DIR, "img_trainval.png")
SRC_LABEL = os.path.join(CODE_DIR, "label_trainval.png")

TRAIN_IMG = os.path.join(CODE_DIR, "dataset", "train", "image")
TRAIN_LBL = os.path.join(CODE_DIR, "dataset", "train", "label")
VAL_IMG = os.path.join(CODE_DIR, "dataset", "val", "image")
VAL_LBL = os.path.join(CODE_DIR, "dataset", "val", "label")

PATCH_RE = re.compile(r"patch_r(\d{4})_c(\d{4})\.png")

SPLIT_DIRS = {
    "train": {"image": TRAIN_IMG, "label": TRAIN_LBL},
    "val": {"image": VAL_IMG, "label": VAL_LBL},
}


@dataclass(frozen=True)
class PatchRecord:
    split: str
    name: str
    row: int
    col: int
    y: int
    x: int
    image_path: str
    label_path: str


@lru_cache(maxsize=1)
def load_source_image():
    return io.imread(SRC_IMAGE)


@lru_cache(maxsize=1)
def load_source_label():
    return io.imread(SRC_LABEL)


def source_shape():
    return load_source_image().shape[:2]


def source_channels():
    return load_source_image().shape[2:] or (1,)


def grid_shape():
    h, w = source_shape()
    n_full_rows = h // TILE_SIZE
    n_full_cols = w // TILE_SIZE
    n_rows = n_full_rows + int(h % TILE_SIZE != 0)
    n_cols = n_full_cols + int(w % TILE_SIZE != 0)
    return n_rows, n_cols


def window_for_position(row, col):
    h, w = source_shape()
    n_full_rows = h // TILE_SIZE
    n_full_cols = w // TILE_SIZE
    y = row * TILE_SIZE if row < n_full_rows else h - TILE_SIZE
    x = col * TILE_SIZE if col < n_full_cols else w - TILE_SIZE
    return y, x


def expected_positions():
    n_rows, n_cols = grid_shape()
    positions = []
    for row in range(n_rows):
        for col in range(n_cols):
            positions.append((row, col, *window_for_position(row, col)))
    return positions


def choose_val_rows():
    h, _ = source_shape()
    n_rows, _ = grid_shape()
    y_positions = [window_for_position(row, 0)[0] for row in range(n_rows)]
    target_val = int(round(n_rows * (1.0 - TRAIN_RATIO)))
    target_val = max(1, min(n_rows, target_val))

    overlap_rows = set()
    for row in range(n_rows - 1):
        if y_positions[row] + TILE_SIZE > y_positions[row + 1]:
            overlap_rows.add(row)
            overlap_rows.add(row + 1)

    best_rows = None
    best_score = None
    for start in range(0, n_rows - target_val + 1):
        rows = set(range(start, start + target_val))
        overlap_hit = len(rows & overlap_rows)
        center_distance = abs((start + (target_val - 1) / 2.0) - (n_rows - 1) / 2.0)
        score = (overlap_hit, center_distance, start)
        if best_score is None or score < best_score:
            best_score = score
            best_rows = rows
    return best_rows


def expected_split_positions():
    val_rows = choose_val_rows()
    train_positions = set()
    val_positions = set()
    for row, col, _, _ in expected_positions():
        if row in val_rows:
            val_positions.add((row, col))
        else:
            train_positions.add((row, col))
    return train_positions, val_positions


def parse_patch_name(name):
    match = PATCH_RE.fullmatch(name)
    if not match:
        raise AssertionError(f"Invalid patch name: {name}")
    return int(match.group(1)), int(match.group(2))


def split_names(split, kind):
    return sorted(os.listdir(SPLIT_DIRS[split][kind]))


def collect_patch_records(split):
    names = split_names(split, "image")
    records = []
    for name in names:
        row, col = parse_patch_name(name)
        y, x = window_for_position(row, col)
        records.append(
            PatchRecord(
                split=split,
                name=name,
                row=row,
                col=col,
                y=y,
                x=x,
                image_path=os.path.join(SPLIT_DIRS[split]["image"], name),
                label_path=os.path.join(SPLIT_DIRS[split]["label"], name),
            )
        )
    return records


def actual_split_positions(split):
    return {(record.row, record.col) for record in collect_patch_records(split)}


def patch_window_overlap(a, b):
    y_overlap = max(0, min(a.y + TILE_SIZE, b.y + TILE_SIZE) - max(a.y, b.y))
    x_overlap = max(0, min(a.x + TILE_SIZE, b.x + TILE_SIZE) - max(a.x, b.x))
    return y_overlap, x_overlap
