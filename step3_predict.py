import argparse
import json
import logging
import os
import sys
import time
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from skimage import io

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from eval import compute_iou, fast_hist  # noqa: E402
from step2_train import CONFIG as TRAIN_CONFIG  # noqa: E402
from unet_model import Res34UNet_light  # noqa: E402

MODEL_REGISTRY = {}


def _register_models():
    MODEL_REGISTRY["res34"] = Res34UNet_light

    try:
        from unet_cbam import Res34UNet_CBAM  # noqa: E402

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
        from unet_efficientnet import EfficientUNet  # noqa: E402

        for backbone in [
            "efficientnet-b0",
            "efficientnet-b1",
            "efficientnet-b2",
            "efficientnet-b3",
            "efficientnet-b4",
        ]:
            key = backbone.replace("-", "_").replace(".", "_")

            def _make(backbone_name=backbone):
                # pretrained=False: weights come from the checkpoint, not ImageNet
                return lambda **kw: EfficientUNet(
                    backbone=kw.pop("backbone", backbone_name), pretrained=False, **kw,
                )

            MODEL_REGISTRY[key] = _make()
        MODEL_REGISTRY["efficientnet_auto"] = lambda **kw: EfficientUNet(
            backbone=kw.pop("backbone", "adaptive"), pretrained=False, **kw,
        )
    except ImportError:
        pass


_register_models()


TILE_SIZE = 512
DEFAULT_STRIDE_NONOVERLAP = TILE_SIZE
DEFAULT_STRIDE_OVERLAP = 384
DEFAULT_BATCH_SIZE_CPU = 4
DEFAULT_BATCH_SIZE_CUDA = 8
GAUSSIAN_SIGMA_FRAC = 0.25
THRESHOLD = 0.5

DATA_MEAN = list(TRAIN_CONFIG["data_mean"])
DATA_STD = list(TRAIN_CONFIG["data_std"])
_MEAN_ARRAY = np.asarray(DATA_MEAN, dtype=np.float32).reshape(1, 1, 1, 3)
_STD_ARRAY = np.asarray(DATA_STD, dtype=np.float32).reshape(1, 1, 1, 3)

TEST_IMAGE = "img_test.png"
TEST_LABEL = "label_test.png"
DEFAULT_CHECKPOINT = "./checkpoints/UNet_res34_best.pth"

OUTPUT_PREDICT = "predict.png"
OUTPUT_VIS = "visualization.png"
OUTPUT_PROB = "predict_prob.npy"

EXPERIMENT_PRESETS = {
    "baseline": {
        "name": "Baseline (no-overlap train, no-overlap inference)",
        "mode": "no_overlap",
        "fusion": "none",
        "stride": 512,
        "key_change": "None (baseline reference)",
    },
    "overlap_uniform": {
        "name": "Overlap inference + uniform fusion",
        "mode": "overlap",
        "fusion": "uniform",
        "stride": 384,
        "key_change": "Overlapping test windows (stride=384), uniform averaging",
    },
    "overlap_gaussian": {
        "name": "Overlap inference + gaussian fusion",
        "mode": "overlap",
        "fusion": "gaussian",
        "stride": 384,
        "key_change": "Overlapping test windows (stride=384), gaussian center-weighted fusion",
    },
    "overlap_stride256": {
        "name": "Overlap inference + gaussian fusion (stride=256)",
        "mode": "overlap",
        "fusion": "gaussian",
        "stride": 256,
        "key_change": "Higher overlap (stride=256), gaussian center-weighted fusion",
    },
}


def _resolve(path_str):
    path = Path(path_str)
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    return str(path)


def _load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _history_path_from_checkpoint(checkpoint_path):
    ckpt = Path(checkpoint_path)
    name = ckpt.name
    if name.endswith("_best.pth"):
        history_name = name[:-len("_best.pth")] + "_history.json"
        return ckpt.with_name(history_name)
    return None


def load_threshold_from_history(checkpoint_path, fallback=THRESHOLD):
    history_path = _history_path_from_checkpoint(checkpoint_path)
    if history_path is None or not history_path.is_file():
        return float(fallback)

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    threshold = history.get("best_threshold", fallback)
    return float(threshold)


def build_model(model_type, device, **kwargs):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}"
        )
    model = MODEL_REGISTRY[model_type](**kwargs)
    return model.to(device)


def load_checkpoint(model, checkpoint_path, device):
    ckpt = _load_state_dict(checkpoint_path, device)
    model.load_state_dict(ckpt)
    logging.info("Checkpoint loaded from %s", checkpoint_path)
    return model


def compute_axis_positions(length, tile_size, stride):
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    last_start = max(length - tile_size, 0)
    positions = list(range(0, last_start + 1, stride))
    if not positions or positions[-1] != last_start:
        positions.append(last_start)
    return positions


def compute_grid(height, width, tile_size, stride):
    y_positions = compute_axis_positions(height, tile_size, stride)
    x_positions = compute_axis_positions(width, tile_size, stride)
    patches = [(y, x) for y in y_positions for x in x_positions]
    return patches, y_positions, x_positions


def ensure_rgb_uint8(image):
    image = np.asarray(image)

    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.ndim == 3 and image.shape[2] >= 4:
        image = image[:, :, :3]
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got shape={image.shape}")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(image)


def normalize_patch(patch_np):
    batch = np.expand_dims(ensure_rgb_uint8(patch_np), axis=0)
    normalized = normalize_batch(batch)
    return np.transpose(normalized[0], (1, 2, 0))


def normalize_batch(batch_np):
    batch = ensure_rgb_uint8(batch_np[0])[None, ...] if batch_np.ndim == 3 else batch_np
    batch = batch.astype(np.float32) / 255.0
    batch = (batch - _MEAN_ARRAY) / _STD_ARRAY
    batch = np.transpose(batch, (0, 3, 1, 2))
    return np.ascontiguousarray(batch)


def _choose_batch_size(device, batch_size):
    if batch_size is not None:
        return max(1, int(batch_size))
    if getattr(device, "type", str(device)) == "cuda":
        return DEFAULT_BATCH_SIZE_CUDA
    return DEFAULT_BATCH_SIZE_CPU


def _inference_autocast(device):
    if getattr(device, "type", str(device)) == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@torch.inference_mode()
def predict_batch(model, batch_rgb, device):
    batch_norm = normalize_batch(batch_rgb)
    tensor = torch.from_numpy(batch_norm).to(device=device, dtype=torch.float32)
    with _inference_autocast(device):
        logits = model(tensor)
    probs = torch.sigmoid(logits[:, 0]).float().cpu().numpy()
    return probs.astype(np.float32, copy=False)


@torch.inference_mode()
def predict_patch(model, patch_rgb, device):
    batch = ensure_rgb_uint8(patch_rgb)[None, ...]
    return predict_batch(model, batch, device)[0]


@lru_cache(maxsize=None)
def _fusion_weight(tile_size, fusion, sigma_frac=GAUSSIAN_SIGMA_FRAC):
    if fusion == "uniform":
        return np.ones((tile_size, tile_size), dtype=np.float32)
    if fusion != "gaussian":
        raise ValueError(f"Unsupported fusion mode: {fusion}")

    sigma = float(tile_size) * float(sigma_frac)
    center = tile_size / 2.0
    ys, xs = np.ogrid[:tile_size, :tile_size]
    weight = np.exp(
        -((ys - center + 0.5) ** 2 + (xs - center + 0.5) ** 2) / (2.0 * sigma * sigma)
    ).astype(np.float32)
    weight /= np.max(weight)
    return weight


def _gaussian_weight(tile_size, sigma_frac=GAUSSIAN_SIGMA_FRAC):
    return _fusion_weight(tile_size, "gaussian", sigma_frac)


def _stack_patches(image, patch_positions, tile_size):
    patches = [image[y:y + tile_size, x:x + tile_size] for y, x in patch_positions]
    if not patches:
        raise ValueError("No patches to stack.")
    return np.stack(patches, axis=0)


def sliding_window_infer(
    model,
    image,
    device,
    tile_size,
    stride,
    fusion="uniform",
    batch_size=None,
):
    image = ensure_rgb_uint8(image)
    height, width = image.shape[:2]
    patch_positions, _, _ = compute_grid(height, width, tile_size, stride)
    batch_size = _choose_batch_size(device, batch_size)

    weight_full = _fusion_weight(tile_size, fusion)
    accum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)

    n_total = len(patch_positions)
    msg = "Sliding window" if fusion == "uniform" else f"Sliding window ({fusion} fusion)"
    logging.info(
        "%s: %d patches (tile=%d, stride=%d, batch=%d)",
        msg,
        n_total,
        tile_size,
        stride,
        batch_size,
    )

    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        batch_positions = patch_positions[start:end]
        batch_rgb = _stack_patches(image, batch_positions, tile_size)
        batch_prob = predict_batch(model, batch_rgb, device)

        for (y, x), prob in zip(batch_positions, batch_prob):
            patch_h, patch_w = prob.shape
            weight = weight_full[:patch_h, :patch_w]
            y_end = y + patch_h
            x_end = x + patch_w

            accum[y:y_end, x:x_end] += prob * weight
            weight_sum[y:y_end, x:x_end] += weight

        if end == n_total or end % 50 == 0 or start == 0:
            logging.info("  patch %d/%d", end, n_total)

    prob_map = accum / np.maximum(weight_sum, 1e-8)
    return prob_map.astype(np.float32, copy=False)


def sliding_window_predict(model, image, device, tile_size, stride, batch_size=None):
    return sliding_window_infer(
        model,
        image,
        device,
        tile_size=tile_size,
        stride=stride,
        fusion="uniform",
        batch_size=batch_size,
    )


def sliding_window_predict_weighted(model, image, device, tile_size, stride, batch_size=None):
    return sliding_window_infer(
        model,
        image,
        device,
        tile_size=tile_size,
        stride=stride,
        fusion="gaussian",
        batch_size=batch_size,
    )


def threshold_map(prob_map, threshold=THRESHOLD):
    return (np.asarray(prob_map) > float(threshold)).astype(np.uint8) * 255


def compute_test_iou(pred_binary, label, num_classes=2):
    pred_binary = np.asarray(pred_binary)
    label = np.asarray(label)
    if pred_binary.shape[:2] != label.shape[:2]:
        raise ValueError(
            f"Shape mismatch: pred {pred_binary.shape[:2]} vs label {label.shape[:2]}"
        )

    pred_flat = (
        (pred_binary > 127).astype(np.uint8).ravel()
        if pred_binary.max() > 1
        else pred_binary.astype(np.uint8).ravel()
    )
    label_flat = (label > 0).astype(np.uint8).ravel()
    hist = fast_hist(pred_flat, label_flat, num_classes)
    mean_iou, per_class_iou = compute_iou(hist)
    return float(mean_iou), per_class_iou, hist


def error_visualization(pred_binary, label):
    pred_binary = np.asarray(pred_binary)
    label = np.asarray(label)

    if pred_binary.max() > 1:
        pred_binary = (pred_binary > 127).astype(np.uint8)
    else:
        pred_binary = pred_binary.astype(np.uint8)

    label_binary = (label > 0).astype(np.uint8)

    vis = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    fp_mask = (pred_binary == 1) & (label_binary == 0)
    fn_mask = (pred_binary == 0) & (label_binary == 1)
    tp_mask = (pred_binary == 1) & (label_binary == 1)

    vis[fp_mask] = [255, 0, 0]
    vis[fn_mask] = [0, 255, 0]
    vis[tp_mask] = [255, 255, 255]
    return vis


def _load_image_and_label(test_image, test_label):
    image_path = _resolve(test_image)
    label_path = _resolve(test_label)

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Test image not found: {image_path}")

    image = ensure_rgb_uint8(io.imread(image_path))
    label = io.imread(label_path) if os.path.isfile(label_path) else None

    if label is not None and image.shape[:2] != label.shape[:2]:
        raise ValueError(
            f"Size mismatch: image {image.shape[:2]} vs label {label.shape[:2]}"
        )

    return image, label, image_path, label_path


def _run_inference_for_mode(model, image, device, mode, fusion, tile_size, stride, batch_size):
    if mode == "overlap" and fusion == "gaussian":
        return sliding_window_predict_weighted(
            model,
            image,
            device,
            tile_size=tile_size,
            stride=stride,
            batch_size=batch_size,
        )
    return sliding_window_predict(
        model,
        image,
        device,
        tile_size=tile_size,
        stride=stride,
        batch_size=batch_size,
    )


def _save_prediction_outputs(out_dir, pred_binary, prob_map, label, save_prob, save_vis):
    predict_path = out_dir / OUTPUT_PREDICT
    io.imsave(str(predict_path), pred_binary, check_contrast=False)
    logging.info("predict.png saved -> %s  [shape=%s]", predict_path, pred_binary.shape)

    if save_prob:
        prob_path = out_dir / OUTPUT_PROB
        np.save(str(prob_path), prob_map)
        logging.info("Probability map saved -> %s", prob_path)

    if label is not None and save_vis:
        vis = error_visualization(pred_binary, label)
        vis_path = out_dir / OUTPUT_VIS
        io.imsave(str(vis_path), vis, check_contrast=False)
        logging.info("visualization.png saved -> %s  [shape=%s]", vis_path, vis.shape)


def _build_results(
    *,
    model_type,
    mode,
    fusion,
    tile_size,
    stride,
    threshold,
    batch_size,
    elapsed,
    pred_binary,
    label,
    metadata=None,
):
    results = {
        "model_type": model_type,
        "mode": mode,
        "fusion": fusion if mode == "overlap" else "none",
        "tile_size": int(tile_size),
        "stride": int(stride),
        "threshold": float(threshold),
        "batch_size": int(batch_size),
        "inference_time_s": round(float(elapsed), 1),
        "predict_shape": list(pred_binary.shape),
    }

    if metadata:
        results.update(metadata)

    if label is None:
        return results

    mean_iou, per_class_iou, hist = compute_test_iou(pred_binary, label)
    fp_count = int(np.sum((pred_binary > 127) & (label == 0)))
    fn_count = int(np.sum((pred_binary == 0) & (label > 0)))
    tp_count = int(np.sum((pred_binary > 127) & (label > 0)))

    results.update(
        {
            "mean_iou": float(mean_iou),
            "iou_background": float(per_class_iou[0]),
            "iou_building": float(per_class_iou[1]),
            "confusion_matrix": hist.tolist(),
            "fp_pixels": fp_count,
            "fn_pixels": fn_count,
            "tp_pixels": tp_count,
        }
    )

    logging.info(
        "Test IoU: mean=%.6f  bg=%.6f  building=%.6f",
        mean_iou,
        per_class_iou[0],
        per_class_iou[1],
    )
    logging.info("Confusion matrix:\n%s", hist)
    logging.info("Error pixels: FP=%d  FN=%d  TP=%d", fp_count, fn_count, tp_count)
    return results


def _save_results_json(out_dir, results):
    json_path = out_dir / "predict_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logging.info("Results saved -> %s", json_path)


def run_prediction(
    model,
    image,
    label,
    device,
    *,
    model_type,
    output_dir,
    mode,
    fusion,
    tile_size,
    stride,
    threshold,
    batch_size,
    save_prob=True,
    save_vis=True,
    metadata=None,
):
    out_dir = Path(output_dir)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    prob_map = _run_inference_for_mode(
        model=model,
        image=image,
        device=device,
        mode=mode,
        fusion=fusion,
        tile_size=tile_size,
        stride=stride,
        batch_size=batch_size,
    )
    elapsed = time.time() - t0
    logging.info("Inference completed in %.1f s (%.2f min)", elapsed, elapsed / 60.0)

    pred_binary = threshold_map(prob_map, threshold)
    _save_prediction_outputs(out_dir, pred_binary, prob_map, label, save_prob, save_vis)

    results = _build_results(
        model_type=model_type,
        mode=mode,
        fusion=fusion,
        tile_size=tile_size,
        stride=stride,
        threshold=threshold,
        batch_size=batch_size,
        elapsed=elapsed,
        pred_binary=pred_binary,
        label=label,
        metadata=metadata,
    )
    _save_results_json(out_dir, results)
    return results


def _selected_experiments(exp_ids):
    if not exp_ids:
        return list(EXPERIMENT_PRESETS.keys())

    requested = [item.strip() for item in exp_ids.split(",") if item.strip()]
    unknown = [item for item in requested if item not in EXPERIMENT_PRESETS]
    if unknown:
        raise ValueError(
            f"Unknown experiment ids: {unknown}. Available: {list(EXPERIMENT_PRESETS.keys())}"
        )
    return requested


def print_comparison_table(records):
    if not records:
        return

    header = (
        f"{'Experiment':<45s} {'mIoU':>8s} {'Bld IoU':>8s} "
        f"{'Time(s)':>8s} {'FP':>8s} {'FN':>8s}  Key Change"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    for record in records:
        print(
            f"{record.get('name', record.get('exp_id', 'unknown')):<45s} "
            f"{record.get('mean_iou', float('nan')):8.4f} "
            f"{record.get('iou_building', float('nan')):8.4f} "
            f"{record.get('inference_time_s', float('nan')):8.1f} "
            f"{record.get('fp_pixels', -1):8d} "
            f"{record.get('fn_pixels', -1):8d}  "
            f"{record.get('key_change', '')}"
        )
    print(sep)

    baseline = records[0]
    if "mean_iou" not in baseline:
        return

    for record in records[1:]:
        if "mean_iou" not in record:
            continue
        delta = record["mean_iou"] - baseline["mean_iou"]
        delta_bld = record["iou_building"] - baseline["iou_building"]
        print(
            f"  {record.get('exp_id', 'unknown')}: "
            f"delta_mIoU={delta:+.4f}  delta_bldIoU={delta_bld:+.4f}"
        )


def run_experiment_suite(args, model, image, label, device):
    output_root = Path(_resolve(args.output_dir))
    os.makedirs(output_root, exist_ok=True)

    records = []
    for exp_id in _selected_experiments(args.exp_ids):
        preset = dict(EXPERIMENT_PRESETS[exp_id])
        exp_dir = output_root / exp_id

        logging.info("=" * 70)
        logging.info("Running experiment: %s", preset["name"])
        logging.info("=" * 70)

        results = run_prediction(
            model=model,
            image=image,
            label=label,
            device=device,
            model_type=args.model_type,
            output_dir=exp_dir,
            mode=preset["mode"],
            fusion=preset["fusion"],
            tile_size=args.tile_size,
            stride=preset["stride"],
            threshold=args.threshold,
            batch_size=args.batch_size,
            save_prob=not args.no_save_prob,
            save_vis=not args.no_save_vis,
            metadata={
                "exp_id": exp_id,
                "name": preset["name"],
                "key_change": preset["key_change"],
            },
        )
        records.append(results)

    summary_path = output_root / "experiment_records.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    logging.info("Experiment records saved -> %s", summary_path)
    print_comparison_table(records)
    return records


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.batch_size = _choose_batch_size(device, args.batch_size)
    logging.info("Device: %s", device)

    test_img_path = _resolve(args.test_image)
    ckpt_path = _resolve(args.checkpoint)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if args.use_history_threshold and args.threshold is None:
        args.threshold = load_threshold_from_history(ckpt_path, fallback=THRESHOLD)
        logging.info("Threshold loaded from training history: %.4f", args.threshold)
    elif args.threshold is None:
        args.threshold = float(THRESHOLD)
        logging.info("Using default threshold: %.4f", args.threshold)
    else:
        logging.info("Threshold set explicitly: %.4f", args.threshold)

    image, label, _, _ = _load_image_and_label(args.test_image, args.test_label)
    logging.info("Test image: %s  dtype=%s", image.shape, image.dtype)
    if label is not None:
        logging.info("Test label: %s  dtype=%s", label.shape, label.dtype)

    model = build_model(args.model_type, device)
    model = load_checkpoint(model, ckpt_path, device)
    model.eval()

    if args.run_suite:
        return run_experiment_suite(args, model, image, label, device)

    output_dir = Path(_resolve(args.output_dir))
    metadata = {"source_image": test_img_path}
    if label is not None:
        metadata["source_label"] = _resolve(args.test_label)

    return run_prediction(
        model=model,
        image=image,
        label=label,
        device=device,
        model_type=args.model_type,
        output_dir=output_dir,
        mode=args.mode,
        fusion=args.fusion,
        tile_size=args.tile_size,
        stride=args.stride,
        threshold=args.threshold,
        batch_size=args.batch_size,
        save_prob=not args.no_save_prob,
        save_vis=not args.no_save_vis,
        metadata=metadata,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UNet sliding-window test inference and overlap-fusion evaluation"
    )
    parser.add_argument("--mode", choices=["no_overlap", "overlap"], default="no_overlap")
    parser.add_argument("--fusion", choices=["uniform", "gaussian"], default="gaussian")
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--model-type",
        default="res34",
        help=f"Model type: {list(MODEL_REGISTRY.keys())}",
    )
    parser.add_argument("--test-image", default=TEST_IMAGE)
    parser.add_argument("--test-label", default=TEST_LABEL)
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--use-history-threshold",
        action="store_true",
        help="Load best_threshold from the paired *_history.json next to the checkpoint.",
    )
    parser.add_argument("--no-save-prob", action="store_true")
    parser.add_argument("--no-save-vis", action="store_true")
    parser.add_argument(
        "--run-suite",
        action="store_true",
        help="Run the built-in baseline/overlap comparison suite into subdirectories.",
    )
    parser.add_argument(
        "--exp-ids",
        default=None,
        help=(
            "Comma-separated experiment ids for --run-suite. "
            f"Available: {list(EXPERIMENT_PRESETS.keys())}"
        ),
    )

    args = parser.parse_args()
    if args.stride is None:
        args.stride = (
            DEFAULT_STRIDE_NONOVERLAP
            if args.mode == "no_overlap"
            else DEFAULT_STRIDE_OVERLAP
        )

    main(args)
