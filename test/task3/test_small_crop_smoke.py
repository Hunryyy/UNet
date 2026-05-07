"""Real-model smoke test on a small crop to keep Task 3 reproducible and cheap."""
import numpy as np
import torch

from _task3_common import (
    CHECKPOINT_PATH,
    WEIGHT_PATH,
    add_code_dir_to_path,
    load_crop,
    print_header,
)

add_code_dir_to_path()

import step3_predict


def summarize_prediction(name, pred_binary, label):
    positives = int((pred_binary > 127).sum())
    label_positives = int((label > 0).sum())
    mean_iou, per_class_iou, _ = step3_predict.compute_test_iou(pred_binary, label)
    print(
        f"  {name:14s} positives={positives:7d}, label={label_positives:7d}, "
        f"mIoU={float(mean_iou):.6f}, bldIoU={float(per_class_iou[1]):.6f}"
    )
    return mean_iou, per_class_iou


def main():
    print_header("TASK 3 TEST 4: Small-crop end-to-end smoke test")

    assert CHECKPOINT_PATH.exists(), f"Missing checkpoint: {CHECKPOINT_PATH}"
    assert WEIGHT_PATH.exists(), f"Missing bundled pretrained weight: {WEIGHT_PATH}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_crop, label_crop, top, left = load_crop(size=640)
    label_positive_pixels = int((label_crop > 0).sum())

    print(f"\n[4.1] Crop location: top={top}, left={left}, shape={image_crop.shape}")
    print(f"  Device: {device}")
    assert label_positive_pixels > 0, "Selected smoke-test crop should contain building pixels"
    print(f"  Crop building pixels: {label_positive_pixels}")

    print("\n[4.2] Build model and load checkpoint")
    model = step3_predict.build_model("res34", device)
    model = step3_predict.load_checkpoint(model, str(CHECKPOINT_PATH), device)
    model.eval()
    print("  Model build + checkpoint load succeeded  OK")

    print("\n[4.3] No-overlap inference is deterministic on the same crop")
    prob_a = step3_predict.sliding_window_predict(
        model, image_crop, device, step3_predict.TILE_SIZE, step3_predict.DEFAULT_STRIDE_NONOVERLAP
    )
    prob_b = step3_predict.sliding_window_predict(
        model, image_crop, device, step3_predict.TILE_SIZE, step3_predict.DEFAULT_STRIDE_NONOVERLAP
    )
    assert prob_a.shape == label_crop.shape
    assert np.allclose(prob_a, prob_b, atol=1e-7)
    assert np.isfinite(prob_a).all()
    assert 0.0 <= float(prob_a.min()) <= 1.0
    assert 0.0 <= float(prob_a.max()) <= 1.0
    pred_a = step3_predict.threshold_map(prob_a)
    assert set(np.unique(pred_a).tolist()).issubset({0, 255})
    summarize_prediction("no_overlap", pred_a, label_crop)

    print("\n[4.4] Gaussian overlap inference runs successfully on the same crop")
    prob_g = step3_predict.sliding_window_predict_weighted(
        model, image_crop, device, step3_predict.TILE_SIZE, step3_predict.DEFAULT_STRIDE_OVERLAP
    )
    assert prob_g.shape == label_crop.shape
    assert np.isfinite(prob_g).all()
    assert 0.0 <= float(prob_g.min()) <= 1.0
    assert 0.0 <= float(prob_g.max()) <= 1.0
    pred_g = step3_predict.threshold_map(prob_g)
    assert set(np.unique(pred_g).tolist()).issubset({0, 255})
    summarize_prediction("overlap_gauss", pred_g, label_crop)

    print("\n" + "=" * 70)
    print("TEST 4 PASSED - Real checkpointed model can complete Task 3 inference on a small crop")
    print("=" * 70)
    return True


if __name__ == "__main__":
    main()
