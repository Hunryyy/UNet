import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MyDataset
from eval import eval_net
from unet_model import Res34UNet_light


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SEED = 42

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


CONFIG = {
    "seed": DEFAULT_SEED,
    "lr": 1e-3,
    "batch_size": 8,
    "epochs": 15,
    "num_workers": 0,
    "weight_decay": 1e-5,
    "grad_clip_norm": 1.0,
    "lr_step_size": 8,
    "lr_gamma": 0.3,
    "read_name": "",
    "save_name": "UNet",
    "traindir_img": "./dataset/train/image/",
    "traindir_mask": "./dataset/train/label/",
    "valdir_img": "./dataset/val/image/",
    "valdir_mask": "./dataset/val/label/",
    "dir_checkpoint": "./checkpoints/",
    "data_mean": [0.43782742, 0.44557303, 0.41160695],
    "data_std": [0.19686149, 0.18481555, 0.19296625],
}


def _resolve_path(path_like):
    path = Path(path_like)
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    return str(path)


def resolve_config_paths(config):
    resolved = dict(config)
    for key in [
        "traindir_img",
        "traindir_mask",
        "valdir_img",
        "valdir_mask",
        "dir_checkpoint",
    ]:
        resolved[key] = _resolve_path(resolved[key])
    return resolved


def _load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _check_paths_exist(config):
    for key in ["traindir_img", "traindir_mask", "valdir_img", "valdir_mask"]:
        path = config[key]
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Required directory '{path}' ({key}) not found. "
                f"Run step1_crop_dataset.py first. CWD: {os.getcwd()}"
            )
        if len(os.listdir(path)) == 0:
            raise RuntimeError(f"Directory '{path}' is empty.")


def _build_train_loader(dataset, config):
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=False,
        generator=generator,
    )


def _build_val_loader(dataset, config):
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )


def _log_config(config, model, device):
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(
        "Baseline training configuration:\n"
        f"  Model: Res34UNet_light  (params: {n_params:,})\n"
        f"  Seed: {config['seed']}\n"
        f"  Epochs: {config['epochs']}\n"
        f"  Batch size: {config['batch_size']}\n"
        f"  Learning rate: {config['lr']}\n"
        f"  Weight decay: {config['weight_decay']}\n"
        f"  Grad clip norm: {config['grad_clip_norm']}\n"
        f"  LR schedule: StepLR(step={config['lr_step_size']}, gamma={config['lr_gamma']})\n"
        f"  Device: {device}\n"
        f"  Save name: {config['save_name']}\n"
        f"  Train image dir: {config['traindir_img']}\n"
        f"  Train label dir: {config['traindir_mask']}\n"
        f"  Val image dir: {config['valdir_img']}\n"
        f"  Val label dir: {config['valdir_mask']}\n"
        f"  Checkpoint dir: {config['dir_checkpoint']}\n"
    )


def build_model(config=None):
    if config is None:
        config = CONFIG
    return Res34UNet_light(init_seed=int(config["seed"]))


def train_net(net, device, config):
    config = resolve_config_paths(config)
    set_seed(config["seed"])
    _check_paths_exist(config)

    os.makedirs(config["dir_checkpoint"], exist_ok=True)

    d_mean, d_std = config["data_mean"], config["data_std"]
    traindataset = MyDataset(
        config["traindir_img"],
        config["traindir_mask"],
        mean=d_mean,
        std=d_std,
        is_train=True,
        seed=config["seed"],
    )
    valdataset = MyDataset(
        config["valdir_img"],
        config["valdir_mask"],
        mean=d_mean,
        std=d_std,
        is_train=False,
        seed=config["seed"],
    )

    train_loader = _build_train_loader(traindataset, config)
    val_loader = _build_val_loader(valdataset, config)

    logging.info(f"Data loaded - train: {len(traindataset)}, val: {len(valdataset)}")

    optimizer = optim.Adam(
        net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, config["lr_step_size"], config["lr_gamma"]
    )

    checkpoint_path = os.path.join(
        config["dir_checkpoint"], config["save_name"] + "_best.pth"
    )

    if config["read_name"]:
        read_path = os.path.join(config["dir_checkpoint"], config["read_name"] + ".pth")
        if os.path.exists(read_path):
            logging.info(f"Loading checkpoint: {read_path}")
            net.load_state_dict(_load_state_dict(read_path, device))
            best_val_score = eval_net(net, val_loader, device)
            logging.info(f"Loaded model best IoU: {best_val_score:.6f}")
        else:
            logging.warning(f"Checkpoint not found: {read_path}, starting fresh.")
            best_val_score = -1.0
    else:
        logging.info("Training new model from scratch.")
        best_val_score = -1.0

    history = {"train_loss": [], "val_iou": [], "lr": []}

    for epoch in range(config["epochs"]):
        net.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config['epochs']}",
            unit="batch",
            leave=False,
        )

        for batch in pbar:
            imgs = batch["image"].to(device=device, dtype=torch.float32)
            true_masks = batch["mask"].to(device=device, dtype=torch.float32)

            loss, _ = net(imgs, true_masks)
            loss_val = loss.mean()

            optimizer.zero_grad(set_to_none=True)
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(
                net.parameters(), config["grad_clip_norm"]
            )
            optimizer.step()

            epoch_loss += loss_val.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss_val.item():.4f}")

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        avg_train_loss = epoch_loss / max(n_batches, 1)

        val_score = eval_net(net, val_loader, device)
        is_best = val_score > best_val_score

        history["train_loss"].append(float(avg_train_loss))
        history["val_iou"].append(float(val_score))
        history["lr"].append(float(current_lr))

        logging.info(
            f"Epoch {epoch + 1:3d}/{config['epochs']} | "
            f"loss: {avg_train_loss:.6f} | "
            f"val IoU: {val_score:.6f} | "
            f"lr: {current_lr:.2e} | "
            f"{'BEST' if is_best else ''}"
        )

        if is_best:
            best_val_score = val_score
            torch.save(net.state_dict(), checkpoint_path)
            logging.info(f"Checkpoint saved -> {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise RuntimeError("Training finished without writing a best checkpoint.")

    logging.info("Loading best checkpoint for final validation...")
    net.load_state_dict(_load_state_dict(checkpoint_path, device))
    final_score = eval_net(net, val_loader, device)
    logging.info(f"Final validation IoU (best ckpt): {final_score:.6f}")

    history["best_val_iou"] = float(best_val_score)
    history["final_val_iou"] = float(final_score)
    history["best_epoch"] = int(np.argmax(history["val_iou"]) + 1)
    history_path = os.path.join(
        config["dir_checkpoint"], config["save_name"] + "_history.json"
    )
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logging.info(f"Training history saved -> {history_path}")

    return history


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = resolve_config_paths(CONFIG)
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = build_model(config)
    _log_config(config, model, device)
    net = model.to(device)

    if config["read_name"]:
        read_path = os.path.join(config["dir_checkpoint"], config["read_name"] + ".pth")
        if os.path.exists(read_path):
            net.load_state_dict(_load_state_dict(read_path, device))
            logging.info(f"Model loaded from {read_path}")

    train_net(net, device, config)
