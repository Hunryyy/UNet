"""Check that saved Task 3 experiments support the claimed improvements."""
from _task3_common import load_official_artifacts, print_header


def total_errors(results):
    return int(results["fp_pixels"]) + int(results["fn_pixels"])


def main():
    print_header("TASK 3 TEST 3: Effectiveness evidence")

    baseline = load_official_artifacts("baseline")["results"]
    overlap_uniform = load_official_artifacts("overlap_uniform")["results"]
    overlap_gaussian = load_official_artifacts("overlap_gaussian")["results"]

    print("\n[3.1] Overlap inference must improve over baseline on the official test image")
    assert overlap_uniform["mean_iou"] > baseline["mean_iou"]
    assert overlap_uniform["iou_building"] > baseline["iou_building"]
    assert total_errors(overlap_uniform) < total_errors(baseline)
    print(
        f"  baseline      mIoU={baseline['mean_iou']:.6f}, "
        f"bldIoU={baseline['iou_building']:.6f}, errors={total_errors(baseline)}"
    )
    print(
        f"  overlap+avg   mIoU={overlap_uniform['mean_iou']:.6f}, "
        f"bldIoU={overlap_uniform['iou_building']:.6f}, errors={total_errors(overlap_uniform)}"
    )

    print("\n[3.2] Gaussian fusion must further improve over uniform averaging")
    assert overlap_gaussian["mean_iou"] > overlap_uniform["mean_iou"]
    assert overlap_gaussian["iou_building"] > overlap_uniform["iou_building"]
    assert total_errors(overlap_gaussian) < total_errors(overlap_uniform)
    print(
        f"  overlap+gauss mIoU={overlap_gaussian['mean_iou']:.6f}, "
        f"bldIoU={overlap_gaussian['iou_building']:.6f}, "
        f"errors={total_errors(overlap_gaussian)}"
    )

    print("\n[3.3] Improvement deltas")
    delta_uniform = overlap_uniform["mean_iou"] - baseline["mean_iou"]
    delta_gaussian = overlap_gaussian["mean_iou"] - baseline["mean_iou"]
    delta_bld = overlap_gaussian["iou_building"] - baseline["iou_building"]
    error_reduction = total_errors(baseline) - total_errors(overlap_gaussian)
    print(f"  overlap+avg vs baseline     delta mIoU = {delta_uniform:+.6f}")
    print(f"  overlap+gauss vs baseline   delta mIoU = {delta_gaussian:+.6f}")
    print(f"  overlap+gauss vs baseline   delta bldIoU = {delta_bld:+.6f}")
    print(f"  overlap+gauss error reduction = {error_reduction:,} pixels")

    print("\n[3.4] Timing fields are recorded, but should not be used as proof of speedup")
    for name, record in [
        ("baseline", baseline),
        ("overlap_uniform", overlap_uniform),
        ("overlap_gaussian", overlap_gaussian),
    ]:
        print(f"  {name:16s} inference_time_s = {record['inference_time_s']}")
    print("  Timing is preserved for audit, but accuracy deltas are the reliable comparison axis.")

    print("\n" + "=" * 70)
    print("TEST 3 PASSED - Official Task 3 results support the overlap-fusion improvement claim")
    print("=" * 70)
    return True


if __name__ == "__main__":
    main()
