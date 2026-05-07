"""
Test 3: Checkpoint save/load consistency.

Verifies that model weights can be saved and loaded correctly,
and produce identical outputs on the same input after loading.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TEST_DIR)

import tempfile
import numpy as np
import torch

from _task2_common import CODE_DIR, ensure_baseline_dataset_rebuilt
from unet_model import Res34UNet_light

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    os.chdir(CODE_DIR)
    ensure_baseline_dataset_rebuilt()

    print("=" * 60)
    print("TEST 3: Checkpoint save/load consistency")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    set_seed(SEED)

    # --- 3.1 Create model and get reference output ---
    print("\n[3.1] Reference model forward")
    model = Res34UNet_light().to(DEVICE)
    model.eval()
    x = torch.randn(2, 3, 512, 512, device=DEVICE)

    with torch.no_grad():
        ref_out = model(x).clone()
    print(f"  Reference output shape: {list(ref_out.shape)}")
    print(f"  Reference output stats: min={ref_out.min().item():.6f}, "
          f"max={ref_out.max().item():.6f}, mean={ref_out.mean().item():.6f}")

    # --- 3.2 Save and reload ---
    print("\n[3.2] Save → delete → reload")
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        ckpt_path = tmp.name

    torch.save(model.state_dict(), ckpt_path)
    print(f"  Saved to: {ckpt_path}")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    model2 = Res34UNet_light().to(DEVICE)
    model2.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model2.eval()
    print("  Model reloaded successfully")

    # --- 3.3 Compare outputs ---
    print("\n[3.3] Output comparison")
    with torch.no_grad():
        new_out = model2(x).clone()

    diff = (ref_out - new_out).abs().max().item()
    print(f"  Max absolute difference: {diff:.10f}")

    assert diff < 1e-6, (
        f"Reloaded model produces different output! Max diff = {diff}")
    assert torch.allclose(ref_out, new_out, atol=1e-6), (
        "Outputs not allclose after reload")

    # --- 3.4 Verify state_dict integrity ---
    print("\n[3.4] State dict parameter check")
    state = torch.load(ckpt_path, map_location="cpu")
    model3 = Res34UNet_light()
    model3.load_state_dict(state)
    for name, param in model3.named_parameters():
        assert name in state, f"Missing: {name}"
        assert torch.equal(param.data, state[name].to(param.device)), (
            f"Parameter mismatch: {name}")

    print(f"  All {len(list(model3.parameters()))} parameters verified  OK")

    # --- Cleanup ---
    os.unlink(ckpt_path)
    print(f"\n  Cleaned up temp file: {ckpt_path}")

    print("\n" + "=" * 60)
    print("TEST 3 PASSED — Checkpoint save/load consistent")
    print("=" * 60)
    return True


if __name__ == "__main__":
    main()
