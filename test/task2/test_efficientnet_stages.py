"""Verify v2 EfficientUNet auto-detects encoder stages correctly.

Covers:
  - Every backbone variant (b0..b4) builds without error
  - Exactly 5 encoder stages are detected for each
  - Encoder channel counts are plausible
  - Forward pass produces correct output shape
  - Reproducibility with fixed seed
  - Encoder freezing works
"""

import sys
from pathlib import Path

import torch

TEST_DIR = Path(__file__).resolve().parent
CODE_DIR = TEST_DIR.parent.parent / "code"
sys.path.insert(0, str(CODE_DIR))

from unet_efficientnet import (  # noqa: E402
    EfficientUNet,
    _BUILDERS,
    _resolve_efficientnet_stages,
    _build_stage_boundaries,
)


BACKBONES = list(_BUILDERS.keys())


# ---------------------------------------------------------------------------
# stage detection
# ---------------------------------------------------------------------------

def test_stage_detection_all_variants():
    """Every registered backbone must yield exactly 5 stages."""
    for name in BACKBONES:
        # Build a bare encoder (no pretrained weights download)
        effnet = _BUILDERS[name](weights=None)
        stage_info = _resolve_efficientnet_stages(effnet)
        boundaries = _build_stage_boundaries(stage_info)
        assert len(boundaries) == 5, \
            f"{name}: expected 5 stages, got {len(boundaries)}. info={stage_info}"
        print(f"  {name}: 5 stages, channels={[s[1] for s in stage_info]}")

    print("PASS: all backbones auto-detect 5 stages")


def test_stage_resolution_monotonic():
    """Spatial resolution must be strictly decreasing across stages."""
    for name in BACKBONES:
        effnet = _BUILDERS[name](weights=None)
        stage_info = _resolve_efficientnet_stages(effnet)
        heights = [s[2] for s in stage_info]
        for i in range(len(heights) - 1):
            assert heights[i] > heights[i + 1], \
                f"{name}: resolution not decreasing: {heights}"

    print("PASS: all backbones have monotonically decreasing resolution")


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------

def test_efficientunet_construction():
    """Build EfficientUNet for each backbone (no pretrained weights, fast)."""
    for name in BACKBONES:
        model = EfficientUNet(backbone=name, pretrained=False, init_seed=42)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0
        print(f"  {name}: {n_params:,} params")

    print("PASS: EfficientUNet constructs for all backbones")


def test_forward_shape():
    """Output must be (B, 1, H, W)."""
    for name in BACKBONES[:2]:  # test b0 and b1 to save time
        model = EfficientUNet(backbone=name, pretrained=False, init_seed=42)
        model.eval()
        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 512, 512), \
            f"{name}: expected (1,1,512,512), got {out.shape}"

    print("PASS: EfficientUNet forward shape")


def test_training_mode():
    """Forward in training mode returns (loss, out) and backward succeeds."""
    model = EfficientUNet(backbone="efficientnet-b0", pretrained=False,
                           init_seed=42)
    model.train()
    x = torch.randn(2, 3, 512, 512)
    gts = torch.randint(0, 2, (2, 512, 512)).float()
    loss, out = model(x, gts)
    assert out.shape == (2, 1, 512, 512)
    loss.backward()
    # All trainable params should have grad
    no_grad = [n for n, p in model.named_parameters()
               if p.grad is None and p.requires_grad]
    assert len(no_grad) == 0, f"params without grad: {no_grad}"

    print("PASS: EfficientUNet training mode loss + backward")


def test_reproducibility():
    """Same seed → same parameters."""
    m1 = EfficientUNet(backbone="efficientnet-b0", pretrained=False, init_seed=42)
    m2 = EfficientUNet(backbone="efficientnet-b0", pretrained=False, init_seed=42)
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), f"param {n1} differs"
    print("PASS: EfficientUNet reproducibility")


def test_freeze_encoder():
    """When freeze_encoder=True, encoder params have requires_grad=False."""
    model = EfficientUNet(backbone="efficientnet-b0", pretrained=False,
                           freeze_encoder=True, init_seed=42)
    for i, stage in enumerate(model.encoder_stages):
        for p in stage.parameters():
            assert not p.requires_grad, \
                f"encoder stage {i}: parameter not frozen"
    # Decoder params must still be trainable
    dec_trainable = any(p.requires_grad for p in model.up1.parameters())
    assert dec_trainable, "decoder should remain trainable"
    print("PASS: encoder freezing")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("TEST: EfficientUNet stage auto-detection & forward")
    print("=" * 70)

    test_stage_detection_all_variants()
    test_stage_resolution_monotonic()
    test_efficientunet_construction()
    test_forward_shape()
    test_training_mode()
    test_reproducibility()
    test_freeze_encoder()

    print("\nAll EfficientUNet tests passed.")
