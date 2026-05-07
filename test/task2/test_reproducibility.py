"""
Test 9: Reproducibility of the real training entrypoint.

Task 2 explicitly requires a reproducible baseline. Two runs with the same
seed and the same tiny dataset subset should produce identical history.
"""
import copy
import os
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TEST_DIR)

from _task2_common import CODE_DIR, make_dataset_subset, pushd
import step2_train
from unet_model import Res34UNet_light


def run_once(config):
    with pushd(CODE_DIR):
        model = Res34UNet_light().to("cpu")
        return step2_train.train_net(model, "cpu", config)


def main():
    print("=" * 60)
    print("TEST 9: Training reproducibility")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        subset = make_dataset_subset(tmpdir, train_count=1, val_count=1)

        histories = []
        for run_idx in range(2):
            ckpt_dir = tmpdir / f"checkpoints_{run_idx}"
            config = copy.deepcopy(step2_train.CONFIG)
            config.update({
                "epochs": 1,
                "batch_size": 1,
                "num_workers": 0,
                "save_name": f"repro_{run_idx}",
                "read_name": "",
                "traindir_img": str(subset["train_img"]),
                "traindir_mask": str(subset["train_lbl"]),
                "valdir_img": str(subset["val_img"]),
                "valdir_mask": str(subset["val_lbl"]),
                "dir_checkpoint": str(ckpt_dir),
            })
            histories.append(run_once(config))

    print(f"  Run 1 history: {histories[0]}")
    print(f"  Run 2 history: {histories[1]}")

    assert histories[0] == histories[1], (
        "Same-seed runs produced different histories. "
        "The baseline is not reproducible yet."
    )

    print("\n" + "=" * 60)
    print("TEST 9 PASSED — Same-seed training is reproducible")
    print("=" * 60)
    return True


if __name__ == "__main__":
    main()
