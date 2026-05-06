"""
Test 1: Small-sample overfitting test.

This is the highest-value sanity check for Task 2: with one fixed sample and
no augmentation noise, the model should rapidly memorize the example. If it
cannot, the data / model / loss wiring is likely wrong.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TEST_DIR)

import torch
from torch.utils.data import DataLoader, Subset

from _task2_common import CODE_DIR, DATA_MEAN, DATA_STD, pushd, set_seed
from dataset import MyDataset
from unet_model import Res34UNet_light

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES = 1
OVERFIT_EPOCHS = 6
OVERFIT_LR = 1e-3
MAX_FINAL_LOSS = 0.10
MIN_LOSS_DROP_RATIO = 0.90


def main():
    set_seed()

    print("=" * 60)
    print("TEST 1: Small-sample overfitting")
    print(f"  Device: {DEVICE}")
    print(f"  Samples: {N_SAMPLES}, epochs: {OVERFIT_EPOCHS}, lr: {OVERFIT_LR}")
    print("=" * 60)

    with pushd(CODE_DIR):
        full_ds = MyDataset(
            "./dataset/train/image/",
            "./dataset/train/label/",
            mean=DATA_MEAN,
            std=DATA_STD,
            is_train=False,
        )
        small_ds = Subset(full_ds, list(range(N_SAMPLES)))
        loader = DataLoader(small_ds, batch_size=1, shuffle=False)

        model = Res34UNet_light().to(DEVICE)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=OVERFIT_LR)

        losses = []
        for epoch in range(OVERFIT_EPOCHS):
            epoch_loss = 0.0
            for batch in loader:
                imgs = batch["image"].to(DEVICE, dtype=torch.float32)
                masks = batch["mask"].to(DEVICE, dtype=torch.float32)

                loss, _ = model(imgs, masks)
                loss_val = loss.mean()

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

                epoch_loss += loss_val.item()

            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)
            print(f"  Epoch {epoch + 1:2d}/{OVERFIT_EPOCHS}  loss={avg_loss:.6f}")

    initial_loss = losses[0]
    final_loss = losses[-1]
    loss_drop_ratio = (initial_loss - final_loss) / initial_loss

    print(f"\n  Initial loss: {initial_loss:.6f}")
    print(f"  Final   loss: {final_loss:.6f}")
    print(f"  Loss drop:    {loss_drop_ratio:.2%}")

    assert final_loss < initial_loss, (
        f"Loss did not decrease: {initial_loss:.6f} -> {final_loss:.6f}"
    )
    assert final_loss < MAX_FINAL_LOSS, (
        f"Failed to overfit fixed sample. Final loss {final_loss:.6f} >= {MAX_FINAL_LOSS}"
    )
    assert loss_drop_ratio > MIN_LOSS_DROP_RATIO, (
        f"Loss did not collapse enough for a memorization test: "
        f"drop ratio {loss_drop_ratio:.2%} <= {MIN_LOSS_DROP_RATIO:.0%}"
    )

    print("=" * 60)
    print("TEST 1 PASSED — Small-sample memorization confirmed")
    print("=" * 60)
    return True


if __name__ == "__main__":
    main()
