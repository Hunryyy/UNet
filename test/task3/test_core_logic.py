"""Fast semantic checks for Task 3 sliding-window inference primitives."""
import numpy as np

from _task3_common import add_code_dir_to_path, load_test_image_shape, print_header

add_code_dir_to_path()

import step2_train
import step3_predict


def assert_axis_is_fully_covered(length, positions, tile_size):
    covered = np.zeros(length, dtype=bool)
    for start in positions:
        end = min(length, start + tile_size)
        covered[start:end] = True
    assert covered.all(), (
        f"Axis coverage is incomplete: uncovered={int((~covered).sum())} pixels"
    )


def main():
    print_header("TASK 3 TEST 1: Core logic checks")

    print("\n[1.1] Inference normalization matches training config")
    assert np.allclose(step3_predict.DATA_MEAN, step2_train.CONFIG["data_mean"])
    assert np.allclose(step3_predict.DATA_STD, step2_train.CONFIG["data_std"])
    print("  DATA_MEAN / DATA_STD are synchronized with Task 2 config  OK")

    print("\n[1.2] Sliding-window axis positions cover both regular and edge cases")
    cases = [
        (512, 512),
        (513, 512),
        (1024, 512),
        (5354, 512),
        (5354, 384),
        (11265, 512),
        (11265, 384),
    ]
    for length, stride in cases:
        positions = step3_predict.compute_axis_positions(length, 512, stride)
        assert positions[0] == 0, f"First position should be 0, got {positions[0]}"
        assert positions[-1] == max(0, length - 512), (
            f"Last position should flush to the edge for length={length}, stride={stride}"
        )
        assert positions == sorted(positions), "Positions must be sorted"
        assert len(set(positions)) == len(positions), "Positions must not duplicate"
        assert_axis_is_fully_covered(length, positions, 512)
        print(
            f"  length={length:5d}, stride={stride:3d}, tiles={len(positions):2d}, "
            f"last={positions[-1]:5d}  OK"
        )

    print("\n[1.3] Official test-image grid sizes match recorded patch counts")
    height, width = load_test_image_shape()
    patches, _, _ = step3_predict.compute_grid(height, width, 512, 512)
    overlap_patches, _, _ = step3_predict.compute_grid(height, width, 512, 384)
    assert len(patches) == 253, f"Expected 253 baseline patches, got {len(patches)}"
    assert len(overlap_patches) == 420, (
        f"Expected 420 overlap patches, got {len(overlap_patches)}"
    )
    print(f"  baseline patches={len(patches)}, overlap patches={len(overlap_patches)}  OK")

    print("\n[1.4] Gaussian fusion weights are symmetric and center-weighted")
    weight = step3_predict._gaussian_weight(512)
    assert weight.shape == (512, 512)
    assert abs(float(weight.max()) - 1.0) < 1e-12
    assert np.allclose(weight, np.flipud(weight))
    assert np.allclose(weight, np.fliplr(weight))
    assert float(weight[256, 256]) > float(weight[0, 0])
    print(
        f"  center={float(weight[256, 256]):.6f}, corner={float(weight[0, 0]):.6f}  OK"
    )

    print("\n[1.5] Uniform and gaussian fusion preserve constant patch probabilities")
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    old_predict_batch = step3_predict.predict_batch

    def fake_predict_batch(model, batch_rgb, device):
        batch = np.asarray(batch_rgb)
        return np.full((batch.shape[0], batch.shape[1], batch.shape[2]), 0.37, dtype=np.float32)

    step3_predict.predict_batch = fake_predict_batch
    try:
        prob_uniform = step3_predict.sliding_window_predict(
            object(), image, "cpu", 512, 384
        )
        prob_gaussian = step3_predict.sliding_window_predict_weighted(
            object(), image, "cpu", 512, 384
        )
    finally:
        step3_predict.predict_batch = old_predict_batch

    assert np.allclose(prob_uniform, 0.37, atol=1e-7)
    assert np.allclose(prob_gaussian, 0.37, atol=1e-7)
    print("  Both fusion modes keep constant predictions unchanged  OK")

    print("\n[1.6] Threshold / IoU / error-visualization semantics are consistent")
    prob_map = np.array([[0.40, 0.60], [0.49, 0.90]], dtype=np.float32)
    pred_binary = step3_predict.threshold_map(prob_map, threshold=0.5)
    label = np.array([[0, 255], [255, 0]], dtype=np.uint8)

    assert pred_binary.tolist() == [[0, 255], [0, 255]]
    mean_iou, per_class_iou, hist = step3_predict.compute_test_iou(pred_binary, label)
    expected_hist = np.array([[1, 1], [1, 1]], dtype=np.int64)
    assert np.array_equal(hist, expected_hist), f"Unexpected confusion matrix: {hist}"

    vis = step3_predict.error_visualization(pred_binary, label)
    assert np.array_equal(vis[0, 0], [0, 0, 0]), "TN should stay black"
    assert np.array_equal(vis[0, 1], [255, 255, 255]), "TP should be white"
    assert np.array_equal(vis[1, 0], [0, 255, 0]), "FN should be green"
    assert np.array_equal(vis[1, 1], [255, 0, 0]), "FP should be red"
    print(
        f"  mean_iou={float(mean_iou):.6f}, "
        f"bg_iou={float(per_class_iou[0]):.6f}, "
        f"bld_iou={float(per_class_iou[1]):.6f}  OK"
    )

    print("\n" + "=" * 70)
    print("TEST 1 PASSED - Core Task 3 inference logic is internally consistent")
    print("=" * 70)
    return True


if __name__ == "__main__":
    main()
