"""Smoke-check Task 3 variant model definitions without touching training code."""
import torch

from _task3_common import add_code_dir_to_path, print_header

add_code_dir_to_path()

from unet_cbam import Res34UNet_CBAM
from unet_efficientnet import EfficientUNet


def run_one(name, model):
    x = torch.randn(1, 3, 512, 512)
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 512, 512), f"{name} output shape mismatch: {out.shape}"
    print(f"  {name:24s} output shape={tuple(out.shape)}  OK")


def main():
    print_header("TASK 3 DIAGNOSTIC: Variant model forward smoke")

    print("\n[diag-v1] CBAM UNet forward")
    cbam_model = Res34UNet_CBAM(
        cbam_encoder=True,
        cbam_decoder=True,
        cbam_skip=True,
        init_seed=42,
    )
    run_one("Res34UNet_CBAM", cbam_model)

    print("\n[diag-v2] EfficientNet-B0 UNet forward (no download)")
    eff_model = EfficientUNet(
        backbone="efficientnet-b0",
        pretrained=False,
        freeze_encoder=False,
        init_seed=42,
    )
    run_one("EfficientUNet-B0", eff_model)

    print("\nConclusion: variant architectures can complete a forward pass,")
    print("but end-to-end Task 3 evaluation still requires their own trained checkpoints.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
