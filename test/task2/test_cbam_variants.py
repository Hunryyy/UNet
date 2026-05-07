"""Verify v2 CBAM-UNet variant forward passes and gradient flow.

Covers:
  - All 4 insertion-position variants produce correct output shapes
  - Each variant's attention modules are correctly placed
  - Loss can be computed & backward pass succeeds (training mode)
  - Reproducibility: same seed → same output
"""

import sys
from pathlib import Path

import torch

TEST_DIR = Path(__file__).resolve().parent
CODE_DIR = TEST_DIR.parent.parent / "code"
sys.path.insert(0, str(CODE_DIR))

from cbam import CBAM, ChannelAttention, SpatialAttention  # noqa: E402
from unet_cbam import Res34UNet_CBAM  # noqa: E402
from unet_model import Res34UNet_light  # noqa: E402


# ---------------------------------------------------------------------------
# CBAM module unit tests
# ---------------------------------------------------------------------------

def test_cbam_channel_attention():
    ca = ChannelAttention(64, reduction=16)
    x = torch.randn(2, 64, 32, 32)
    att = ca(x)
    assert att.shape == (2, 64, 1, 1), f"channel att shape: {att.shape}"
    assert torch.all((att >= 0) & (att <= 1)), "channel att not in [0,1]"
    print("PASS: ChannelAttention output shape & range")


def test_cbam_spatial_attention():
    sa = SpatialAttention(kernel_size=7)
    x = torch.randn(2, 64, 32, 32)
    att = sa(x)
    assert att.shape == (2, 1, 32, 32), f"spatial att shape: {att.shape}"
    assert torch.all((att >= 0) & (att <= 1)), "spatial att not in [0,1]"
    print("PASS: SpatialAttention output shape & range")


def test_cbam_full_module():
    cbam = CBAM(128, reduction=16)
    x = torch.randn(2, 128, 64, 64)
    y = cbam(x)
    assert y.shape == x.shape, f"CBAM changed shape: {x.shape} → {y.shape}"
    # CBAM should modify the input (not return identity)
    assert not torch.allclose(y, x), "CBAM returned identity"
    print("PASS: CBAM forward preserves shape, modifies content")


# ---------------------------------------------------------------------------
# CBAM-UNet variant tests
# ---------------------------------------------------------------------------

VARIANTS = {
    "adaptive": {"cbam_mode": "adaptive"},
    "enc_dec": {"cbam_mode": "enc_dec"},
    "enc":     {"cbam_mode": "encoder"},
    "dec":     {"cbam_mode": "decoder"},
    "skip":    {"cbam_mode": "skip"},
}


def _forward_one(variant_name, kwargs, device="cpu"):
    model = Res34UNet_CBAM(init_seed=42, **kwargs).to(device)
    x = torch.randn(1, 3, 512, 512, device=device)

    # Eval mode
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 512, 512), \
        f"{variant_name} eval: expected (1,1,512,512), got {out.shape}"

    # Training mode — loss + backward
    model.train()
    gts = torch.randint(0, 2, (1, 512, 512), device=device).float()
    loss, out_train = model(x, gts)
    assert out_train.shape == (1, 1, 512, 512), \
        f"{variant_name} train output: {out_train.shape}"
    loss.backward()

    # Every parameter should have a gradient (or be part of the encoder with
    # frozen BN running stats that don't receive grad)
    no_grad = [n for n, p in model.named_parameters()
               if p.grad is None and p.requires_grad]
    assert len(no_grad) == 0, \
        f"{variant_name}: parameters without grad: {no_grad}"

    return model


def test_all_cbam_variants_forward():
    for name, kwargs in VARIANTS.items():
        _forward_one(name, kwargs)
    print("PASS: all CBAM variants forward + backward OK")


def test_cbam_variant_reproducibility():
    """Same seed → identical parameters."""
    m1 = Res34UNet_CBAM(cbam_mode="adaptive", init_seed=42)
    m2 = Res34UNet_CBAM(cbam_mode="adaptive", init_seed=42)
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), f"parameter {n1} differs"
    print("PASS: CBAM variant reproducibility (same seed → same params)")


def test_cbam_module_presence():
    """Verify CBAM modules are attached only when requested."""
    m_all = Res34UNet_CBAM(cbam_mode="adaptive", init_seed=42)
    m_enc = Res34UNet_CBAM(cbam_mode="encoder", init_seed=42)
    m_skip = Res34UNet_CBAM(cbam_mode="skip", init_seed=42)
    m_dec = Res34UNet_CBAM(cbam_mode="decoder", init_seed=42)

    assert not hasattr(m_all, "cbam_enc"),  "adaptive: should not enable cbam_enc by default"
    assert hasattr(m_all, "cbam_dec"),  "all: missing cbam_dec"
    assert hasattr(m_all, "cbam_skip"), "all: missing cbam_skip"
    assert hasattr(m_enc, "cbam_enc"),  "enc: missing cbam_enc"
    assert not hasattr(m_enc, "cbam_dec"),  "enc: should not have cbam_dec"
    assert not hasattr(m_enc, "cbam_skip"), "enc: should not have cbam_skip"
    assert hasattr(m_skip, "cbam_skip"), "skip: missing cbam_skip"
    assert not hasattr(m_skip, "cbam_dec"), "skip: should not have cbam_dec"
    assert hasattr(m_dec, "cbam_dec"), "dec: missing cbam_dec"
    assert not hasattr(m_dec, "cbam_skip"), "dec: should not have cbam_skip"
    print("PASS: CBAM module presence matches flags")


def test_cbam_param_count():
    """CBAM variants should have more parameters than baseline."""
    base = Res34UNet_light(init_seed=42)
    n_base = sum(p.numel() for p in base.parameters())

    for name, kwargs in VARIANTS.items():
        model = Res34UNet_CBAM(init_seed=42, **kwargs)
        n_model = sum(p.numel() for p in model.parameters())
        assert n_model > n_base, \
            f"{name}: {n_model} params not > baseline {n_base}"
        print(f"  {name}: {n_model:,} params (baseline: {n_base:,})")

    print("PASS: all CBAM variants have more parameters than baseline")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("TEST: CBAM-UNet variants")
    print("=" * 70)

    test_cbam_channel_attention()
    test_cbam_spatial_attention()
    test_cbam_full_module()
    test_all_cbam_variants_forward()
    test_cbam_variant_reproducibility()
    test_cbam_module_presence()
    test_cbam_param_count()

    print("\nAll CBAM variant tests passed.")
