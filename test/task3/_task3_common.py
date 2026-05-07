"""Shared helpers for Task 3 review, validation, and reproduction."""
import json
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import io

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

TEST_DIR = Path(__file__).resolve().parent
REPO_DIR = TEST_DIR.parent.parent
CODE_DIR = REPO_DIR / "code"

TEST_IMAGE_PATH = CODE_DIR / "img_test.png"
TEST_LABEL_PATH = CODE_DIR / "label_test.png"
CHECKPOINT_PATH = CODE_DIR / "checkpoints" / "UNet_res34_best.pth"
WEIGHT_PATH = CODE_DIR / "weight" / "resnet34-b627a593.pth"

OFFICIAL_EXPERIMENTS = [
    ("baseline", "Baseline (no-overlap inference)"),
    ("overlap_uniform", "Overlap inference + uniform fusion"),
    ("overlap_gaussian", "Overlap inference + gaussian fusion"),
]


def add_code_dir_to_path():
    code_dir_str = str(CODE_DIR)
    if code_dir_str not in sys.path:
        sys.path.insert(0, code_dir_str)


def print_header(title):
    print("=" * 70)
    print(title)
    print(f"Repo: {REPO_DIR}")
    print(f"Code: {CODE_DIR}")
    print("=" * 70)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_test_label():
    return io.imread(str(TEST_LABEL_PATH))


@lru_cache(maxsize=1)
def load_test_image_shape():
    with Image.open(TEST_IMAGE_PATH) as img:
        width, height = img.size
    return height, width


@lru_cache(maxsize=1)
def load_experiment_records():
    path = TEST_DIR / "experiment_records.json"
    if not path.is_file():
        return []
    return load_json(path)


def official_dir(exp_id):
    return TEST_DIR / exp_id


def load_official_artifacts(exp_id):
    exp_dir = official_dir(exp_id)
    results_path = exp_dir / "predict_results.json"
    predict_path = exp_dir / "predict.png"
    prob_path = exp_dir / "predict_prob.npy"
    vis_path = exp_dir / "visualization.png"
    if not (results_path.is_file() and predict_path.is_file() and prob_path.is_file() and vis_path.is_file()):
        return None
    return {
        "dir": exp_dir,
        "results_path": results_path,
        "predict_path": predict_path,
        "prob_path": prob_path,
        "vis_path": vis_path,
        "results": load_json(results_path),
        "predict": io.imread(str(predict_path)),
        "prob": np.load(str(prob_path)),
        "vis": io.imread(str(vis_path)),
    }


def threshold_from_prob(prob_map, threshold):
    return (prob_map > threshold).astype(np.uint8) * 255


def confusion_from_binary(pred_binary, label_binary):
    pred01 = (pred_binary > 127).astype(np.uint8)
    label01 = (label_binary > 0).astype(np.uint8)
    tn = int(np.sum((pred01 == 0) & (label01 == 0)))
    fp = int(np.sum((pred01 == 1) & (label01 == 0)))
    fn = int(np.sum((pred01 == 0) & (label01 == 1)))
    tp = int(np.sum((pred01 == 1) & (label01 == 1)))
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


def pixel_stats_from_confusion(hist):
    tn = int(hist[0, 0])
    fp = int(hist[0, 1])
    fn = int(hist[1, 0])
    tp = int(hist[1, 1])
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "errors": fp + fn}


def assert_close(actual, expected, atol=1e-9, label="value"):
    if abs(float(actual) - float(expected)) > atol:
        raise AssertionError(
            f"{label} mismatch: actual={actual!r}, expected={expected!r}, atol={atol}"
        )


def select_positive_crop(size=640):
    label = load_test_label()
    ys, xs = np.nonzero(label > 0)
    if len(ys) == 0:
        raise RuntimeError("label_test.png contains no positive building pixels.")

    anchor_idx = len(ys) // 2
    center_y = int(ys[anchor_idx])
    center_x = int(xs[anchor_idx])
    height, width = label.shape

    top = max(0, min(center_y - size // 2, height - size))
    left = max(0, min(center_x - size // 2, width - size))
    return top, left


def load_crop(size=640):
    top, left = select_positive_crop(size=size)
    box = (left, top, left + size, top + size)

    with Image.open(TEST_IMAGE_PATH) as img:
        image_crop = np.array(img.crop(box).convert("RGB"))

    with Image.open(TEST_LABEL_PATH) as label:
        label_crop = np.array(label.crop(box).convert("L"))

    return image_crop, label_crop, top, left
