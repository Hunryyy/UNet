"""
Test runner for Task 2: runs all tests and reports results.
"""
import subprocess
import sys
import os
import time

TESTS = [
    ("test_threshold.py",   "Threshold & IoU logic"),
    ("test_shapes.py",      "Shape / dtype / device consistency"),
    ("test_paths.py",       "Path / cwd assumptions"),
    ("test_pretrained_weight_source.py", "Pretrained weight source behavior"),
    ("test_checkpoint.py",  "Checkpoint save/load"),
    ("test_augment_sync.py","Augmentation image-mask sync"),
    ("test_overfit.py",     "Small-sample overfitting"),
    ("test_pipeline.py",    "Real step2_train integration"),
    ("test_reproducibility.py", "Same-seed reproducibility"),
]


def run_one(script, description):
    path = os.path.join(os.path.dirname(__file__), script)
    print(f"\n{'#' * 70}")
    print(f"# {description}")
    print(f"# {script}")
    print(f"{'#' * 70}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, path],
        capture_output=True, text=True, timeout=1200)
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(result.stdout)
        print(f"  [{description}] PASSED ({elapsed:.1f}s)")
        return True
    else:
        print(result.stdout)
        print(result.stderr)
        print(f"  [{description}] FAILED (exit {result.returncode}, {elapsed:.1f}s)")
        return False


def main():
    print("=" * 70)
    print("TASK 2 BASELINE TEST SUITE")
    print(f"Python: {sys.executable}")
    print(f"CWD: {os.getcwd()}")
    print("=" * 70)

    results = {}
    for script, desc in TESTS:
        results[desc] = run_one(script, desc)

    print("\n\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    passed = sum(results.values())
    total = len(results)
    for desc, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]  {desc}")
    print(f"\n  {passed}/{total} tests passed")
    print("=" * 70)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
