"""
Test: best-model selection must align with calibrated validation IoU.
"""
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_DIR))

import torch

from _task2_common import CODE_DIR, ensure_baseline_dataset_rebuilt, make_dataset_subset, pushd
import step2_train
from unet_model import Res34UNet_light


def main():
    print("=" * 60)
    print("TEST: Best-model selection uses calibrated validation IoU")
    print("=" * 60)

    ensure_baseline_dataset_rebuilt()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        subset = make_dataset_subset(tmpdir, train_count=2, val_count=1)
        ckpt_dir = tmpdir / "checkpoints"

        config = copy.deepcopy(step2_train.CONFIG)
        config.update({
            "epochs": 2,
            "batch_size": 1,
            "num_workers": 0,
            "save_name": "selection_smoke",
            "read_name": "",
            "traindir_img": str(subset["train_img"]),
            "traindir_mask": str(subset["train_lbl"]),
            "valdir_img": str(subset["val_img"]),
            "valdir_mask": str(subset["val_lbl"]),
            "dir_checkpoint": str(ckpt_dir),
        })

        with pushd(CODE_DIR):
            model = Res34UNet_light().to("cpu")
            history = step2_train.train_net(model, "cpu", config)

        assert history["selection_metric"] == "calibrated_miou"
        assert len(history["val_iou"]) == 2
        assert len(history["val_iou_fixed"]) == 2
        assert len(history["val_best_threshold"]) == 2
        assert history["best_epoch"] == int(max(range(2), key=lambda i: history["val_iou"][i]) + 1)

        hist_path = ckpt_dir / "selection_smoke_res34_history.json"
        with open(hist_path, "r", encoding="utf-8") as f:
            hist_disk = json.load(f)

        assert hist_disk["selection_metric"] == "calibrated_miou"
        assert hist_disk["best_val_iou"] == history["best_val_iou"]
        assert hist_disk["best_val_iou_fixed"] == history["best_val_iou_fixed"]
        print(f"  best_epoch={history['best_epoch']}")
        print(f"  best_val_iou={history['best_val_iou']:.6f}")
        print(f"  best_val_iou_fixed={history['best_val_iou_fixed']:.6f}")
        print(f"  best_threshold={history['best_threshold']:.4f}")

    print("=" * 60)
    print("TEST PASSED — best-model selection is calibrated and reproducible")
    print("=" * 60)
    return True


if __name__ == "__main__":
    main()
