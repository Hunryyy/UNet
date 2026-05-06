"""Validate saved Task 3 outputs against labels and result JSON."""
import numpy as np

from _task3_common import (
    OFFICIAL_EXPERIMENTS,
    add_code_dir_to_path,
    assert_close,
    confusion_from_binary,
    load_experiment_records,
    load_official_artifacts,
    load_test_label,
    pixel_stats_from_confusion,
    print_header,
    threshold_from_prob,
)

add_code_dir_to_path()

import step3_predict


def main():
    print_header("TASK 3 TEST 2: Saved artifact integrity")

    label = load_test_label()
    record_index = {
        record["exp_id"]: record
        for record in load_experiment_records()
        if isinstance(record, dict) and "exp_id" in record
    }

    for exp_id, exp_name in OFFICIAL_EXPERIMENTS:
        print(f"\n[2.{OFFICIAL_EXPERIMENTS.index((exp_id, exp_name)) + 1}] {exp_name}")
        artifacts = load_official_artifacts(exp_id)
        results = artifacts["results"]
        pred_binary = artifacts["predict"]
        prob_map = artifacts["prob"]
        vis = artifacts["vis"]

        assert pred_binary.shape == label.shape
        assert prob_map.shape == label.shape
        assert vis.shape == label.shape + (3,)
        assert list(pred_binary.shape) == results["predict_shape"]
        assert np.issubdtype(prob_map.dtype, np.floating)
        assert set(np.unique(pred_binary).tolist()).issubset({0, 255})
        assert 0.0 <= float(prob_map.min()) <= 1.0
        assert 0.0 <= float(prob_map.max()) <= 1.0

        recreated_pred = threshold_from_prob(prob_map, results["threshold"])
        assert np.array_equal(recreated_pred, pred_binary), (
            "predict.png is not the thresholded form of predict_prob.npy"
        )

        mean_iou, per_class_iou, hist = step3_predict.compute_test_iou(pred_binary, label)
        hist_expected = np.array(results["confusion_matrix"], dtype=np.int64)
        assert np.array_equal(hist, hist_expected), (
            f"Confusion matrix mismatch for {exp_id}: {hist.tolist()} vs {hist_expected.tolist()}"
        )
        assert_close(mean_iou, results["mean_iou"], atol=1e-12, label=f"{exp_id}.mean_iou")
        assert_close(
            per_class_iou[0], results["iou_background"], atol=1e-12,
            label=f"{exp_id}.iou_background"
        )
        assert_close(
            per_class_iou[1], results["iou_building"], atol=1e-12,
            label=f"{exp_id}.iou_building"
        )

        stats = pixel_stats_from_confusion(hist)
        assert stats["fp"] == results["fp_pixels"]
        assert stats["fn"] == results["fn_pixels"]
        assert stats["tp"] == results["tp_pixels"]

        red = int(np.sum(np.all(vis == [255, 0, 0], axis=-1)))
        green = int(np.sum(np.all(vis == [0, 255, 0], axis=-1)))
        white = int(np.sum(np.all(vis == [255, 255, 255], axis=-1)))
        black = int(np.sum(np.all(vis == [0, 0, 0], axis=-1)))

        assert red == stats["fp"], f"{exp_id}: red pixels must equal FP count"
        assert green == stats["fn"], f"{exp_id}: green pixels must equal FN count"
        assert white == stats["tp"], f"{exp_id}: white pixels must equal TP count"
        assert red + green + white + black == label.size

        summary = record_index.get(exp_id)
        if summary is not None:
            for key in [
                "mean_iou",
                "iou_background",
                "iou_building",
                "fp_pixels",
                "fn_pixels",
                "tp_pixels",
            ]:
                if isinstance(results[key], float):
                    assert_close(summary[key], results[key], atol=1e-12, label=f"{exp_id}.{key}")
                else:
                    assert summary[key] == results[key]

        print(
            f"  shape={tuple(pred_binary.shape)}, prob=[{float(prob_map.min()):.3e}, "
            f"{float(prob_map.max()):.6f}], FP={stats['fp']}, FN={stats['fn']}, TP={stats['tp']}  OK"
        )

    print("\n" + "=" * 70)
    print("TEST 2 PASSED - Saved Task 3 outputs are self-consistent and reproducible")
    print("=" * 70)
    return True


if __name__ == "__main__":
    main()
