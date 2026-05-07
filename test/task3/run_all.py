"""Run the default fast Task 3 validation suite."""
import os
import subprocess
import sys
import time

TESTS = [
    ("test_core_logic.py", "Core sliding-window semantics"),
    ("diagnose_variant_models.py", "Variant model forward smoke"),
    ("test_saved_artifacts.py", "Saved artifact integrity"),
    ("test_effectiveness.py", "Official-result effectiveness evidence"),
    ("test_small_crop_smoke.py", "Small-crop real-model smoke test"),
]


def ensure_official_artifacts():
    test_dir = os.path.dirname(__file__)
    required = [
        os.path.join(test_dir, "experiment_records.json"),
        os.path.join(test_dir, "baseline", "predict_results.json"),
        os.path.join(test_dir, "overlap_uniform", "predict_results.json"),
        os.path.join(test_dir, "overlap_gaussian", "predict_results.json"),
    ]
    if all(os.path.exists(path) for path in required):
        print("Official Task 3 artifacts already present.")
        return

    print("Official Task 3 artifacts missing. Reproducing them now...")
    path = os.path.join(test_dir, "reproduce_task3.py")
    result = subprocess.run(
        [sys.executable, path, "--output-root", test_dir],
        capture_output=True,
        text=True,
        timeout=7200,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("Failed to reproduce official Task 3 artifacts.")
    print(result.stdout)


def run_one(script, description):
    path = os.path.join(os.path.dirname(__file__), script)
    print(f"\n{'#' * 70}")
    print(f"# {description}")
    print(f"# {script}")
    print(f"{'#' * 70}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, path],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(result.stdout)
        print(f"  [{description}] PASSED ({elapsed:.1f}s)")
        return True

    print(result.stdout)
    print(result.stderr)
    print(f"  [{description}] FAILED (exit {result.returncode}, {elapsed:.1f}s)")
    return False


def main():
    print("=" * 70)
    print("TASK 3 VALIDATION SUITE")
    print(f"Python: {sys.executable}")
    print(f"CWD: {os.getcwd()}")
    print("=" * 70)

    ensure_official_artifacts()

    results = {}
    for script, description in TESTS:
        results[description] = run_one(script, description)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    passed = sum(results.values())
    total = len(results)
    for description, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]  {description}")
    print(f"\n  {passed}/{total} tests passed")
    print("=" * 70)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
