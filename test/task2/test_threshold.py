"""
Test 4: Threshold logic consistency across training and evaluation.

The sigmoid + threshold(0.5) pipeline must be consistent:
- Model output = logits (not probabilities)
- eval_net applies sigmoid → threshold 0.5
- BCEWithLogitsLoss expects logits (not sigmoid'd)

This test verifies the threshold logic on known logit values.
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TEST_DIR)

import numpy as np
import torch

from _task2_common import CODE_DIR
from eval import fast_hist, compute_iou

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    os.chdir(CODE_DIR)

    print("=" * 60)
    print("TEST 4: Threshold logic consistency")
    print("=" * 60)

    # --- 4.1 Sigmoid + threshold on known values ---
    print("\n[4.1] Sigmoid + threshold(0.5) semantics")
    logits = torch.tensor([-10.0, -1.0, 0.0, 0.5, 1.0, 10.0])
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).int()

    expected_probs = torch.tensor([0.0000, 0.2689, 0.5000, 0.6225, 0.7311, 1.0000])
    expected_preds = torch.tensor([0, 0, 0, 1, 1, 1])

    for i in range(len(logits)):
        print(f"  logit={logits[i].item():+.1f}  prob={probs[i].item():.4f}  "
              f"pred={preds[i].item()}  (expected pred={expected_preds[i].item()})")

    assert torch.allclose(probs, expected_probs, atol=1e-3), "Sigmoid mismatch"
    assert torch.equal(preds, expected_preds), "Threshold mismatch at 0.5"
    print("  Sigmoid and threshold(0.5) behave as expected  OK")

    # --- 4.2 BCEWithLogitsLoss vs manual sigmoid+BCE ---
    print("\n[4.2] BCEWithLogitsLoss consistency check")
    logits_test = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0]).unsqueeze(1)
    targets = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]).unsqueeze(1)

    bce_logits = torch.nn.BCEWithLogitsLoss(reduction="none")
    loss_auto = bce_logits(logits_test, targets)

    bce_manual = torch.nn.BCELoss(reduction="none")
    loss_manual = bce_manual(torch.sigmoid(logits_test), targets)

    for i in range(len(logits_test)):
        print(f"  BCEWithLogits={loss_auto[i].item():.6f}  "
              f"BCE(sigmoid(logit))={loss_manual[i].item():.6f}")

    assert torch.allclose(loss_auto, loss_manual, atol=1e-5), (
        "BCEWithLogitsLoss != BCELoss(sigmoid(logits)) — numerical instability?")
    print("  BCEWithLogitsLoss ≡ BCELoss ∘ sigmoid  OK")

    # --- 4.3 IoU computation on known confusion matrix ---
    print("\n[4.3] IoU computation correctness")
    hist = np.array([[90, 10],
                      [5, 95]], dtype=np.int64)

    mean_iou, per_class = compute_iou(hist)
    # Background IoU = 90/(90+10+5) = 90/105 = 0.8571
    # Building IoU  = 95/(95+10+5) = 95/110 = 0.8636
    # Mean IoU = (0.8571 + 0.8636)/2 = 0.8604
    expected_mean = (90 / 105 + 95 / 110) / 2

    print(f"  Confusion matrix: TP_bg={hist[0,0]}, FP={hist[0,1]}, FN={hist[1,0]}, TP_bld={hist[1,1]}")
    print(f"  Background IoU: {per_class[0]:.6f} (expected {90/105:.6f})")
    print(f"  Building IoU:   {per_class[1]:.6f} (expected {95/110:.6f})")
    print(f"  Mean IoU:       {mean_iou:.6f} (expected {expected_mean:.6f})")

    assert abs(mean_iou - expected_mean) < 1e-5, "IoU computation error"
    print("  IoU computation correct  OK")

    # --- 4.4 fast_hist identity check ---
    print("\n[4.4] fast_hist identity check")
    pred = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int64)
    true = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int64)
    hist_perfect = fast_hist(pred, true, 2)
    assert hist_perfect[0, 0] == 4 and hist_perfect[1, 1] == 4, (
        f"Perfect prediction should have only diagonals, got {hist_perfect}")
    assert hist_perfect[0, 1] == 0 and hist_perfect[1, 0] == 0, "No off-diagonals"
    print(f"  Perfect prediction hist: {hist_perfect.tolist()}  OK")

    # --- 4.5 fast_hist all-wrong check ---
    print("\n[4.5] fast_hist all-wrong check")
    pred_wrong = np.array([1, 1, 0, 0, 1, 0, 1, 0], dtype=np.int64)
    hist_wrong = fast_hist(pred_wrong, true, 2)
    assert hist_wrong[0, 0] == 0 and hist_wrong[1, 1] == 0, (
        f"All-wrong should have zero diagonals, got {hist_wrong}")
    print(f"  All-wrong prediction hist: {hist_wrong.tolist()}  OK")

    print("\n" + "=" * 60)
    print("TEST 4 PASSED — Threshold and IoU logic consistent")
    print("=" * 60)
    return True


if __name__ == "__main__":
    main()
