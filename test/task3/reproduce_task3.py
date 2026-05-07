"""Reproduce official Task 3 full-image inference results via step3_predict.py."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from _task3_common import TEST_DIR, load_json, print_header

OFFICIAL_EXP_IDS = "baseline,overlap_uniform,overlap_gaussian"


def required_paths(output_root):
    root = Path(output_root)
    paths = [root / "experiment_records.json"]
    for exp_id in OFFICIAL_EXP_IDS.split(","):
        paths.append(root / exp_id / "predict_results.json")
        paths.append(root / exp_id / "predict.png")
        paths.append(root / exp_id / "predict_prob.npy")
        paths.append(root / exp_id / "visualization.png")
    return paths


def build_command(output_root, exp_ids, batch_size):
    cmd = [
        sys.executable,
        str((TEST_DIR.parent.parent / "code" / "step3_predict.py").resolve()),
        "--run-suite",
        "--exp-ids",
        exp_ids,
        "--checkpoint",
        "./checkpoints/UNet_best.pth",
        "--test-image",
        "img_test.png",
        "--test-label",
        "label_test.png",
        "--output-dir",
        str(Path(output_root).resolve()),
    ]
    if batch_size is not None:
        cmd.extend(["--batch-size", str(int(batch_size))])
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Reproduce Task 3 official inference runs")
    parser.add_argument("--output-root", default=str(TEST_DIR))
    parser.add_argument("--exp-ids", default=OFFICIAL_EXP_IDS)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()

    print_header("TASK 3 REPRODUCTION RUNNER")
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and all(path.exists() for path in required_paths(output_root)):
        print(f"[SKIP] Existing official artifacts detected under {output_root}")
    else:
        cmd = build_command(output_root, args.exp_ids, args.batch_size)
        print("[RUN ] " + " ".join(cmd))
        t0 = time.time()
        result = subprocess.run(
            cmd,
            cwd=(TEST_DIR.parent.parent / "code").resolve(),
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
        elapsed = time.time() - t0

        log_path = output_root / "suite.log"
        log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
        print(f"  suite.log -> {log_path}")
        print(f"  wall_time_s = {elapsed:.1f}")

        if result.returncode != 0:
            raise RuntimeError(
                f"Task 3 reproduction failed with exit code {result.returncode}. "
                f"See log: {log_path}"
            )

    records = load_json(output_root / "experiment_records.json")
    for record in records:
        print(
            f"  {record['exp_id']:16s} "
            f"mIoU={record.get('mean_iou', float('nan')):.6f} "
            f"bldIoU={record.get('iou_building', float('nan')):.6f} "
            f"time={record.get('inference_time_s', float('nan'))}"
        )

    summary_path = output_root / "experiment_records.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"\nSaved summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
