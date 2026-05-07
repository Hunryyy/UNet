"""Task 3 should support opt-in reuse of the calibrated threshold saved by Task 2."""
import json
import tempfile
from pathlib import Path

from _task3_common import add_code_dir_to_path, print_header

add_code_dir_to_path()

import step3_predict


def main():
    print_header("TASK 3 TEST 5: Threshold-history loading")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        ckpt = tmpdir / "demo_best.pth"
        hist = tmpdir / "demo_history.json"
        ckpt.write_bytes(b"placeholder")
        hist.write_text(
            json.dumps({"best_threshold": 0.681, "best_threshold_val_iou": 0.93}),
            encoding="utf-8",
        )

        threshold = step3_predict.load_threshold_from_history(str(ckpt))
        assert abs(threshold - 0.681) < 1e-12
        print(f"  Loaded calibrated threshold: {threshold:.3f}  OK")

        fallback = step3_predict.load_threshold_from_history(
            str(tmpdir / "missing_best.pth"), fallback=0.5
        )
        assert abs(fallback - 0.5) < 1e-12
        print(f"  Missing history falls back to default: {fallback:.3f}  OK")

    assert step3_predict.THRESHOLD == 0.5
    print("  Default inference threshold remains the explicit baseline 0.500  OK")

    print("\n" + "=" * 70)
    print("TEST 5 PASSED - Task 3 can reuse Task 2 threshold calibration")
    print("=" * 70)
    return True


if __name__ == "__main__":
    main()
