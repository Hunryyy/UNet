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
from eval import compute_iou, fast_hist
from eval import eval_net
from unet_model import Res34UNet_light

# ---------------------------------------------------------------------------
# v2 model registry – extendable with CBAM / EfficientNet variants
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {}


def _register_models():
    MODEL_REGISTRY["res34"] = Res34UNet_light

    try:
        from unet_cbam import Res34UNet_CBAM  # noqa: F811

        MODEL_REGISTRY["res34_cbam_enc_dec"] = lambda **kw: Res34UNet_CBAM(
            cbam_mode="adaptive", **kw,
        )
        MODEL_REGISTRY["res34_cbam_enc"] = lambda **kw: Res34UNet_CBAM(
            cbam_mode="encoder", **kw,
        )
        MODEL_REGISTRY["res34_cbam_dec"] = lambda **kw: Res34UNet_CBAM(
            cbam_mode="decoder", **kw,
        )
        MODEL_REGISTRY["res34_cbam_skip"] = lambda **kw: Res34UNet_CBAM(
            cbam_mode="skip", **kw,
        )
    except ImportError:
        pass

    try:
        from unet_efficientnet import EfficientUNet  # noqa: F811

        for _backbone in [
            "efficientnet-b0",
            "efficientnet-b1",
            "efficientnet-b2",
            "efficientnet-b3",
            "efficientnet-b4",
        ]:
            _key = _backbone.replace("-", "_").replace(".", "_")

            def _make(_b=_backbone):
                return lambda **kw: EfficientUNet(
                    backbone=kw.pop("backbone", _b), **kw
                )

            MODEL_REGISTRY[_key] = _make()
        MODEL_REGISTRY["efficientnet_auto"] = lambda **kw: EfficientUNet(
            backbone=kw.pop("backbone", "adaptive"), **kw
        )
    except ImportError:
        pass


_register_models()


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
    # v2: select model from MODEL_REGISTRY  (baseline: "res34")
    "model_type": "res34",
    "model_kwargs": {},
    "traindir_img": "./dataset/train/image/",
    "traindir_mask": "./dataset/train/label/",
    "valdir_img": "./dataset/val/image/",
    "valdir_mask": "./dataset/val/label/",
    "dir_checkpoint": "./checkpoints/",
    "data_mean": [0.43782742, 0.44557303, 0.41160695],
    "data_std": [0.19686149, 0.18481555, 0.19296625],
    "threshold_search_min": 0.2,
    "threshold_search_max": 0.8,
    "threshold_search_steps": 601,
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


@torch.inference_mode()
def find_best_threshold(net, loader, device, threshold_min=0.2, threshold_max=0.8, steps=601):
    thresholds = np.linspace(float(threshold_min), float(threshold_max), int(steps))
    histograms = np.zeros((len(thresholds), 2, 2), dtype=np.int64)

    was_training = net.training
    net.eval()

    for batch in loader:
        imgs = batch["image"].to(device=device, dtype=torch.float32)
        true_masks = batch["mask"].cpu().numpy().astype(np.uint8)
        logits = net(imgs)
        probs = torch.sigmoid(logits[:, 0]).float().cpu().numpy()

        flat_true = true_masks.reshape(true_masks.shape[0], -1)
        flat_prob = probs.reshape(probs.shape[0], -1)

        for sample_true, sample_prob in zip(flat_true, flat_prob):
            pos_mask = sample_true == 1
            neg_mask = ~pos_mask
            pos_prob = sample_prob[pos_mask]
            neg_prob = sample_prob[neg_mask]

            if pos_prob.size:
                pos_sorted = np.sort(pos_prob)
                pos_counts = pos_sorted.size - np.searchsorted(
                    pos_sorted, thresholds, side="right"
                )
            else:
                pos_counts = np.zeros(len(thresholds), dtype=np.int64)

            if neg_prob.size:
                neg_sorted = np.sort(neg_prob)
                fp_counts = neg_sorted.size - np.searchsorted(
                    neg_sorted, thresholds, side="right"
                )
            else:
                fp_counts = np.zeros(len(thresholds), dtype=np.int64)

            tp = pos_counts
            fn = int(pos_mask.sum()) - tp
            fp = fp_counts
            tn = int(neg_mask.sum()) - fp

            histograms[:, 0, 0] += tn
            histograms[:, 0, 1] += fp
            histograms[:, 1, 0] += fn
            histograms[:, 1, 1] += tp

    intersections = np.diagonal(histograms, axis1=1, axis2=2)
    unions = histograms.sum(axis=1) + histograms.sum(axis=2) - intersections
    iou_per_class = intersections / (unions + 1e-8)
    mean_iou = iou_per_class.mean(axis=1)
    best_idx = int(np.argmax(mean_iou))

    if was_training:
        net.train()

    return {
        "best_threshold": float(thresholds[best_idx]),
        "best_iou": float(mean_iou[best_idx]),
        "best_iou_background": float(iou_per_class[best_idx, 0]),
        "best_iou_building": float(iou_per_class[best_idx, 1]),
        "search_min": float(threshold_min),
        "search_max": float(threshold_max),
        "search_steps": int(steps),
    }


@torch.inference_mode()
def evaluate_validation(
    net,
    loader,
    device,
    threshold_min=0.2,
    threshold_max=0.8,
    steps=601,
):
    thresholds = np.linspace(float(threshold_min), float(threshold_max), int(steps))
    histograms = np.zeros((len(thresholds), 2, 2), dtype=np.int64)
    hist_fixed = np.zeros((2, 2), dtype=np.int64)

    was_training = net.training
    net.eval()

    for batch in loader:
        imgs = batch["image"].to(device=device, dtype=torch.float32)
        true_masks = batch["mask"].cpu().numpy().astype(np.uint8)
        logits = net(imgs)
        probs = torch.sigmoid(logits[:, 0]).float().cpu().numpy()
        pred_fixed = (probs > 0.5).astype(np.uint8)

        hist_fixed += fast_hist(
            pred_fixed.ravel(),
            true_masks.ravel(),
            num_classes=2,
        )

        flat_true = true_masks.reshape(true_masks.shape[0], -1)
        flat_prob = probs.reshape(probs.shape[0], -1)
        for sample_true, sample_prob in zip(flat_true, flat_prob):
            pos_mask = sample_true == 1
            neg_mask = ~pos_mask
            pos_prob = sample_prob[pos_mask]
            neg_prob = sample_prob[neg_mask]

            if pos_prob.size:
                pos_sorted = np.sort(pos_prob)
                pos_counts = pos_sorted.size - np.searchsorted(
                    pos_sorted, thresholds, side="right"
                )
            else:
                pos_counts = np.zeros(len(thresholds), dtype=np.int64)

            if neg_prob.size:
                neg_sorted = np.sort(neg_prob)
                fp_counts = neg_sorted.size - np.searchsorted(
                    neg_sorted, thresholds, side="right"
                )
            else:
                fp_counts = np.zeros(len(thresholds), dtype=np.int64)

            tp = pos_counts
            fn = int(pos_mask.sum()) - tp
            fp = fp_counts
            tn = int(neg_mask.sum()) - fp

            histograms[:, 0, 0] += tn
            histograms[:, 0, 1] += fp
            histograms[:, 1, 0] += fn
            histograms[:, 1, 1] += tp

    intersections = np.diagonal(histograms, axis1=1, axis2=2)
    unions = histograms.sum(axis=1) + histograms.sum(axis=2) - intersections
    iou_per_class = intersections / (unions + 1e-8)
    mean_iou = iou_per_class.mean(axis=1)
    best_idx = int(np.argmax(mean_iou))

    fixed_mean_iou, fixed_iou_per_class = compute_iou(hist_fixed)

    if was_training:
        net.train()

    return {
        "iou_at_0_5": float(fixed_mean_iou),
        "iou_at_0_5_background": float(fixed_iou_per_class[0]),
        "iou_at_0_5_building": float(fixed_iou_per_class[1]),
        "best_threshold": float(thresholds[best_idx]),
        "best_iou": float(mean_iou[best_idx]),
        "best_iou_background": float(iou_per_class[best_idx, 0]),
        "best_iou_building": float(iou_per_class[best_idx, 1]),
        "search_min": float(threshold_min),
        "search_max": float(threshold_max),
        "search_steps": int(steps),
    }


def _log_config(config, model, device):
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(
        "Baseline training configuration:\n"
        f"  Model: {config.get('model_type', 'res34')}  (params: {n_params:,})\n"
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
    model_type = config.get("model_type", "res34")
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    kwargs = dict(config.get("model_kwargs", {}))
    kwargs.setdefault("init_seed", int(config["seed"]))
    return MODEL_REGISTRY[model_type](**kwargs)


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

    model_tag = config.get("model_type", "res34")
    checkpoint_path = os.path.join(
        config["dir_checkpoint"],
        f"{config['save_name']}_{model_tag}_best.pth",
    )

    if config["read_name"]:
        read_path = os.path.join(config["dir_checkpoint"], config["read_name"] + ".pth")
        if os.path.exists(read_path):
            logging.info(f"Loading checkpoint: {read_path}")
            net.load_state_dict(_load_state_dict(read_path, device))
            loaded_metrics = evaluate_validation(
                net,
                val_loader,
                device,
                threshold_min=config["threshold_search_min"],
                threshold_max=config["threshold_search_max"],
                steps=config["threshold_search_steps"],
            )
            best_val_score = loaded_metrics["best_iou"]
            logging.info(
                "Loaded model validation: mIoU@0.5=%.6f  calibrated mIoU=%.6f  threshold=%.4f",
                loaded_metrics["iou_at_0_5"],
                loaded_metrics["best_iou"],
                loaded_metrics["best_threshold"],
            )
        else:
            logging.warning(f"Checkpoint not found: {read_path}, starting fresh.")
            best_val_score = -1.0
    else:
        logging.info("Training new model from scratch.")
        best_val_score = -1.0

    best_val_iou_fixed = -1.0
    best_threshold = 0.5
    history = {
        "train_loss": [],
        "val_iou": [],
        "val_iou_fixed": [],
        "val_best_threshold": [],
        "lr": [],
        "selection_metric": "calibrated_miou",
    }

    for epoch in range(config["epochs"]):
        if hasattr(traindataset, "set_epoch"):
            traindataset.set_epoch(epoch)
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

        val_metrics = evaluate_validation(
            net,
            val_loader,
            device,
            threshold_min=config["threshold_search_min"],
            threshold_max=config["threshold_search_max"],
            steps=config["threshold_search_steps"],
        )
        val_score = val_metrics["best_iou"]
        val_fixed = val_metrics["iou_at_0_5"]
        epoch_threshold = val_metrics["best_threshold"]
        is_best = (
            val_score > best_val_score + 1e-12
            or (
                abs(val_score - best_val_score) <= 1e-12
                and val_fixed > best_val_iou_fixed + 1e-12
            )
        )

        history["train_loss"].append(float(avg_train_loss))
        history["val_iou"].append(float(val_score))
        history["val_iou_fixed"].append(float(val_fixed))
        history["val_best_threshold"].append(float(epoch_threshold))
        history["lr"].append(float(current_lr))

        logging.info(
            f"Epoch {epoch + 1:3d}/{config['epochs']} | "
            f"loss: {avg_train_loss:.6f} | "
            f"val mIoU@0.5: {val_fixed:.6f} | "
            f"val calibrated mIoU: {val_score:.6f} | "
            f"best thr: {epoch_threshold:.4f} | "
            f"lr: {current_lr:.2e} | "
            f"{'BEST' if is_best else ''}"
        )

        if is_best:
            best_val_score = val_score
            best_val_iou_fixed = val_fixed
            best_threshold = epoch_threshold
            torch.save(net.state_dict(), checkpoint_path)
            logging.info(f"Checkpoint saved -> {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise RuntimeError("Training finished without writing a best checkpoint.")

    logging.info("Loading best checkpoint for final validation...")
    net.load_state_dict(_load_state_dict(checkpoint_path, device))
    final_metrics = evaluate_validation(
        net,
        val_loader,
        device,
        threshold_min=config["threshold_search_min"],
        threshold_max=config["threshold_search_max"],
        steps=config["threshold_search_steps"],
    )
    logging.info(
        "Best checkpoint validation: mIoU@0.5=%.6f  calibrated mIoU=%.6f  threshold=%.4f",
        final_metrics["iou_at_0_5"],
        final_metrics["best_iou"],
        final_metrics["best_threshold"],
    )

    history["best_val_iou"] = float(best_val_score)
    history["best_val_iou_fixed"] = float(best_val_iou_fixed)
    history["final_val_iou"] = float(final_metrics["best_iou"])
    history["final_val_iou_fixed"] = float(final_metrics["iou_at_0_5"])
    history["best_epoch"] = int(np.argmax(history["val_iou"]) + 1)
    history["best_threshold"] = float(final_metrics["best_threshold"])
    history["best_threshold_epoch_value"] = float(best_threshold)
    history["best_threshold_val_iou"] = float(final_metrics["best_iou"])
    history["best_threshold_iou_background"] = float(
        final_metrics["best_iou_background"]
    )
    history["best_threshold_iou_building"] = float(
        final_metrics["best_iou_building"]
    )
    history_path = os.path.join(
        config["dir_checkpoint"],
        f"{config['save_name']}_{model_tag}_history.json",
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
