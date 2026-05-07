"""Smoke test: verify v2 multi-model training pipeline works end-to-end.

Builds a tiny synthetic dataset, trains each model type for a few batches,
and checks that checkpoints are saved with the correct model-specific names.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

TEST_DIR = Path(__file__).resolve().parent
CODE_DIR = TEST_DIR.parent.parent / "code"
sys.path.insert(0, str(CODE_DIR))

from step2_train import (  # noqa: E402
    MODEL_REGISTRY,
    CONFIG,
    build_model,
    resolve_config_paths,
    set_seed,
    train_net,
)


# ---------------------------------------------------------------------------
# synthetic dataset
# ---------------------------------------------------------------------------

def _make_synthetic_patch(path, h=512, w=512):
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    lbl = np.random.randint(0, 2, (h, w), dtype=np.uint8) * 255
    Image.fromarray(img).save(path)
    lbl_path = path.replace("/image/", "/label/")
    os.makedirs(os.path.dirname(lbl_path), exist_ok=True)
    Image.fromarray(lbl).save(lbl_path)


def _build_synthetic_dataset(root, n_train=8, n_val=4):
    for split, count in [("train", n_train), ("val", n_val)]:
        img_dir = os.path.join(root, split, "image")
        lbl_dir = os.path.join(root, split, "label")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for i in range(count):
            name = f"patch_r0000_c{i:04d}.png"
            _make_synthetic_patch(os.path.join(img_dir, name))
    return root


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_v2_training_smoke():
    """Train each model type for 2 epochs on synthetic data, verify ckpt."""
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_types = ["res34"]  # always test baseline
    # Add CBAM variants if available
    for k in ["res34_cbam_enc_dec", "res34_cbam_enc",
              "res34_cbam_dec", "res34_cbam_skip"]:
        if k in MODEL_REGISTRY:
            model_types.append(k)
    # Add one EfficientNet variant
    if "efficientnet_b0" in MODEL_REGISTRY:
        model_types.append("efficientnet_b0")

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_root = os.path.join(tmpdir, "dataset")
        _build_synthetic_dataset(dataset_root, n_train=8, n_val=4)
        ckpt_dir = os.path.join(tmpdir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        for model_type in model_types:
            config = dict(CONFIG)
            config.update({
                "model_type": model_type,
                "model_kwargs": {},
                "epochs": 2,
                "batch_size": 2,
                "save_name": "v2_smoke",
                "seed": 42,
                "traindir_img": os.path.join(dataset_root, "train", "image"),
                "traindir_mask": os.path.join(dataset_root, "train", "label"),
                "valdir_img": os.path.join(dataset_root, "val", "image"),
                "valdir_mask": os.path.join(dataset_root, "val", "label"),
                "dir_checkpoint": ckpt_dir,
            })
            config = resolve_config_paths(config)

            model = build_model(config)
            model.to(device)

            history = train_net(model, device, config)

            # Verify checkpoint saved
            ckpt_name = f"v2_smoke_{model_type}_best.pth"
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            assert os.path.isfile(ckpt_path), \
                f"{model_type}: checkpoint not found at {ckpt_path}"

            # Verify history saved
            hist_path = ckpt_path.replace("_best.pth", "_history.json")
            assert os.path.isfile(hist_path), \
                f"{model_type}: history not found at {hist_path}"

            with open(hist_path) as f:
                hist = json.load(f)
            assert len(hist["train_loss"]) == 2, \
                f"{model_type}: expected 2 loss entries, got {len(hist['train_loss'])}"
            assert len(hist["val_iou"]) == 2

            # Verify best checkpoint can be loaded
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()

            print(f"  {model_type}: loss={hist['train_loss'][-1]:.4f}, "
                  f"val_iou={hist['val_iou'][-1]:.4f}")

    print(f"PASS: v2 training pipeline works for {len(model_types)} model types")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("TEST: v2 multi-model training smoke")
    print("=" * 70)

    test_v2_training_smoke()

    print("\nV2 training smoke test passed.")
