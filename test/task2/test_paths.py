"""
Test 7: Path and working-directory assumptions.

Task 2 currently depends on relative paths inside code/step2_train.py. This
test makes that behavior explicit so future fixes can be verified cleanly.
"""
import copy
import os
import sys
import warnings

warnings.filterwarnings("ignore")

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TEST_DIR)

from _task2_common import CODE_DIR, REPO_DIR, pushd, set_seed
import step2_train
from unet_model import Res34UNet_light


def main():
    set_seed()

    print("=" * 60)
    print("TEST 7: Path / cwd assumptions")
    print("=" * 60)

    cfg = copy.deepcopy(step2_train.CONFIG)
    cfg.update({
        "epochs": 1,
        "batch_size": 1,
        "num_workers": 0,
        "save_name": "path_smoke",
    })

    print("\n[7.1] Default config should work from code/ cwd")
    with pushd(CODE_DIR):
        model = Res34UNet_light().to("cpu")
        step2_train._check_paths_exist(cfg)
    print("  code/ cwd path resolution  OK")

    print("\n[7.2] Default config currently fails from repo root cwd")
    try:
        with pushd(REPO_DIR):
            step2_train._check_paths_exist(cfg)
    except FileNotFoundError as exc:
        msg = str(exc)
        print(f"  Root cwd failure captured: {msg}")
        assert "./dataset/train/image/" in msg, "Failure message should mention missing relative path"
    else:
        raise AssertionError(
            "Expected FileNotFoundError from repo-root cwd, but default config unexpectedly worked"
        )

    print("\n" + "=" * 60)
    print("TEST 7 PASSED — Current cwd dependency is explicitly covered")
    print("=" * 60)
    return True


if __name__ == "__main__":
    main()
