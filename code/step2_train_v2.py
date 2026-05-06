import argparse
import logging
import os
import sys
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from step2_train import train_net, resolve_config_paths, set_seed, CONFIG, _log_config


MODEL_BUILDERS = {}


def _register():
    from unet_model import Res34UNet_light
    MODEL_BUILDERS["res34"] = lambda cfg: Res34UNet_light(init_seed=int(cfg["seed"]))

    try:
        from unet_cbam import Res34UNet_CBAM
        MODEL_BUILDERS["res34_cbam"] = lambda cfg: Res34UNet_CBAM(
            cbam_encoder=True, cbam_decoder=True, cbam_skip=True, init_seed=int(cfg["seed"]))
        MODEL_BUILDERS["res34_cbam_enc"] = lambda cfg: Res34UNet_CBAM(
            cbam_encoder=True, cbam_decoder=False, cbam_skip=False, init_seed=int(cfg["seed"]))
        MODEL_BUILDERS["res34_cbam_dec"] = lambda cfg: Res34UNet_CBAM(
            cbam_encoder=False, cbam_decoder=True, cbam_skip=False, init_seed=int(cfg["seed"]))
        MODEL_BUILDERS["res34_cbam_skip"] = lambda cfg: Res34UNet_CBAM(
            cbam_encoder=False, cbam_decoder=False, cbam_skip=True, init_seed=int(cfg["seed"]))
    except ImportError:
        pass

    try:
        from unet_efficientnet import EfficientUNet
        for b in ["efficientnet-b0", "efficientnet-b1", "efficientnet-b2",
                   "efficientnet-b3", "efficientnet-b4"]:
            key = b.replace("-", "_").replace(".", "_")
            def _mk(backbone=b):
                return lambda cfg: EfficientUNet(backbone=backbone, init_seed=int(cfg["seed"]))
            MODEL_BUILDERS[key] = _mk()
    except ImportError:
        pass


_register()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Train UNet variant")
    parser.add_argument("--model", default="res34",
                        help=f"Model type: {list(MODEL_BUILDERS.keys())}")
    parser.add_argument("--save-name", default=None,
                        help="Override checkpoint save name")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--traindir-img", default=None)
    parser.add_argument("--traindir-mask", default=None)
    parser.add_argument("--valdir-img", default=None)
    parser.add_argument("--valdir-mask", default=None)
    args = parser.parse_args()

    if args.model not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model: {args.model}. Available: {list(MODEL_BUILDERS.keys())}")

    config = dict(CONFIG)
    if args.save_name:
        config["save_name"] = args.save_name
    else:
        config["save_name"] = args.model

    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.lr is not None:
        config["lr"] = args.lr
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.traindir_img:
        config["traindir_img"] = args.traindir_img
    if args.traindir_mask:
        config["traindir_mask"] = args.traindir_mask
    if args.valdir_img:
        config["valdir_img"] = args.valdir_img
    if args.valdir_mask:
        config["valdir_mask"] = args.valdir_mask

    config = resolve_config_paths(config)
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)

    model = MODEL_BUILDERS[args.model](config)
    _log_config(config, model, device)
    net = model.to(device)

    if config["read_name"]:
        read_path = os.path.join(config["dir_checkpoint"], config["read_name"] + ".pth")
        if os.path.exists(read_path):
            from step2_train import _load_state_dict
            net.load_state_dict(_load_state_dict(read_path, device))
            logging.info("Loaded checkpoint from %s", read_path)

    train_net(net, device, config)


if __name__ == "__main__":
    main()
