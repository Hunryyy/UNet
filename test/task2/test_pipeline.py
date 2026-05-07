"""
Test 6: Real step2_train integration test.

Runs the actual step2_train.train_net entrypoint on a tiny copied subset,
verifying that the baseline training script can save and reload its best
checkpoint with consistent validation IoU.
"""
import copy
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TEST_DIR)

import torch

from _task2_common import CODE_DIR, ensure_baseline_dataset_rebuilt, make_dataset_subset, pushd, set_seed
import step2_train
from eval import eval_net
from unet_model import Res34UNet_light

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    set_seed()
    ensure_baseline_dataset_rebuilt()

    print("=" * 60)
    print("TEST 6: Real step2_train integration")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        subset = make_dataset_subset(tmpdir, train_count=2, val_count=1)
        ckpt_dir = tmpdir / "checkpoints"

        config = copy.deepcopy(step2_train.CONFIG)
        config.update({
            "epochs": 1,
            "batch_size": 1,
            "num_workers": 0,
            "save_name": "task2_smoke",
            "read_name": "",
            "traindir_img": str(subset["train_img"]),
            "traindir_mask": str(subset["train_lbl"]),
            "valdir_img": str(subset["val_img"]),
            "valdir_mask": str(subset["val_lbl"]),
            "dir_checkpoint": str(ckpt_dir),
        })

        print("\n[6.1] Running step2_train.train_net on tiny subset")
        with pushd(CODE_DIR):
            model = Res34UNet_light().to(DEVICE)
            history = step2_train.train_net(model, DEVICE, config)

        best_ckpt = ckpt_dir / "task2_smoke_res34_best.pth"
        history_json = ckpt_dir / "task2_smoke_res34_history.json"

        print("\n[6.2] Output artifact checks")
        assert best_ckpt.exists(), f"Best checkpoint not found: {best_ckpt}"
        assert history_json.exists(), f"History json not found: {history_json}"
        print(f"  Best checkpoint: {best_ckpt}")
        print(f"  History json:    {history_json}")

        print("\n[6.3] History schema checks")
        with open(history_json, "r") as f:
            history_disk = json.load(f)

        for key in [
            "train_loss", "val_iou", "val_iou_fixed", "val_best_threshold", "lr",
            "best_val_iou", "best_val_iou_fixed", "final_val_iou", "final_val_iou_fixed",
            "best_epoch", "best_threshold", "selection_metric",
        ]:
            assert key in history, f"Missing history key in return value: {key}"
            assert key in history_disk, f"Missing history key on disk: {key}"

        assert len(history["train_loss"]) == 1, f"Expected 1 epoch, got {len(history['train_loss'])}"
        assert len(history["val_iou"]) == 1, f"Expected 1 val score, got {len(history['val_iou'])}"
        assert len(history["val_iou_fixed"]) == 1
        assert len(history["val_best_threshold"]) == 1
        assert history["best_epoch"] == 1, f"Single epoch run should have best_epoch=1, got {history['best_epoch']}"
        assert history["selection_metric"] == "calibrated_miou"
        print(f"  Returned history: {history}")

        print("\n[6.4] Reload best checkpoint and re-evaluate")
        with pushd(CODE_DIR):
            val_ds = step2_train.MyDataset(
                config["valdir_img"],
                config["valdir_mask"],
                mean=config["data_mean"],
                std=config["data_std"],
                is_train=False,
            )
            val_loader = torch.utils.data.DataLoader(
                val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0
            )

            reloaded = Res34UNet_light().to(DEVICE)
            reloaded.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
            reloaded_iou = eval_net(reloaded, val_loader, DEVICE)

        print(f"  Saved best IoU@0.5: {history['best_val_iou_fixed']:.6f}")
        print(f"  Reloaded IoU@0.5:   {reloaded_iou:.6f}")
        assert abs(reloaded_iou - history["best_val_iou_fixed"]) < 1e-6, (
            "Reloaded checkpoint does not reproduce saved fixed-threshold IoU"
        )

    print("\n" + "=" * 60)
    print("TEST 6 PASSED — Real training entrypoint verified")
    print("=" * 60)
    return True


if __name__ == "__main__":
    main()
