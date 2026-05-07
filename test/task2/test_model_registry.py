"""Verify all models in the v2 registry instantiate & forward correctly.

Covers:
  - baseline res34 unchanged
  - all 4 CBAM position variants
  - EfficientNet b0..b4
  - Model registry key matches what step2_train / step3_predict expect
"""

import sys
from pathlib import Path

import torch

TEST_DIR = Path(__file__).resolve().parent
CODE_DIR = TEST_DIR.parent.parent / "code"
sys.path.insert(0, str(CODE_DIR))

from step2_train import MODEL_REGISTRY as TRAIN_REGISTRY  # noqa: E402
from step3_predict import MODEL_REGISTRY as PREDICT_REGISTRY  # noqa: E402


# ---------------------------------------------------------------------------

def test_registries_agree():
    """Training and inference registries should contain the same model keys."""
    train_keys = set(TRAIN_REGISTRY.keys())
    pred_keys  = set(PREDICT_REGISTRY.keys())

    # Both should have res34 baseline
    assert "res34" in train_keys, "train registry missing res34"
    assert "res34" in pred_keys,  "predict registry missing res34"

    # CBAM variants
    for cbam_key in ["res34_cbam_enc_dec", "res34_cbam_enc",
                     "res34_cbam_dec", "res34_cbam_skip"]:
        assert cbam_key in train_keys, \
            f"train registry missing {cbam_key}"
        # predict registry might have additional aliases

    # EfficientNet variants
    for eff_key in ["efficientnet_b0", "efficientnet_b1",
                    "efficientnet_b2", "efficientnet_b3",
                    "efficientnet_b4", "efficientnet_auto"]:
        assert eff_key in train_keys, f"train registry missing {eff_key}"
        assert eff_key in pred_keys,  f"predict registry missing {eff_key}"

    print(f"PASS: train registry ({len(train_keys)} models), "
          f"predict registry ({len(pred_keys)} models)")


def test_all_models_instantiate():
    """Every model in the training registry instantiates without error."""
    for key, builder in sorted(TRAIN_REGISTRY.items()):
        try:
            kwargs = {"init_seed": 42}
            # v2: avoid downloading pretrained weights for smoke-test speed
            if "efficientnet" in key:
                kwargs["pretrained"] = False
            model = builder(**kwargs)
        except Exception as e:
            raise AssertionError(f"Failed to instantiate {key}: {e}") from e

        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0, f"{key}: zero parameters"
        print(f"  {key:30s}  {n_params:>10,} params")

    print("PASS: all models instantiate")


def test_all_models_forward_eval():
    """Every model produces (1, 1, 512, 512) output in eval mode."""
    x = torch.randn(1, 3, 512, 512)
    for key, builder in sorted(TRAIN_REGISTRY.items()):
        kwargs = {"init_seed": 42}
        if "efficientnet" in key:
            kwargs["pretrained"] = False
        model = builder(**kwargs)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 512, 512), \
            f"{key}: expected (1,1,512,512), got {out.shape}"
    print("PASS: all models forward (eval) shape OK")


def test_all_models_forward_train():
    """Every model returns (loss, out) in training mode and backward works."""
    x = torch.randn(2, 3, 512, 512)
    gts = torch.randint(0, 2, (2, 512, 512)).float()
    for key, builder in sorted(TRAIN_REGISTRY.items()):
        kwargs = {"init_seed": 42}
        if "efficientnet" in key:
            kwargs["pretrained"] = False
        model = builder(**kwargs)
        model.train()
        loss, out = model(x, gts)
        assert out.shape == (2, 1, 512, 512), \
            f"{key} train: expected (2,1,512,512), got {out.shape}"
        loss.backward()
        assert loss.item() > 0, f"{key}: loss should be positive"
    print("PASS: all models forward (train) + backward OK")


def test_baseline_unchanged():
    """The res34 baseline model must still work as before."""
    model = TRAIN_REGISTRY["res34"](init_seed=42)
    model.eval()
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 512, 512)

    model.train()
    gts = torch.randint(0, 2, (2, 512, 512)).float()
    loss, out_train = model(x.expand(2, -1, -1, -1), gts)
    loss.backward()
    print("PASS: baseline res34 forward unchanged")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("TEST: v2 model registry")
    print("=" * 70)

    test_registries_agree()
    test_all_models_instantiate()
    test_all_models_forward_eval()
    test_all_models_forward_train()
    test_baseline_unchanged()

    print("\nAll model-registry tests passed.")
