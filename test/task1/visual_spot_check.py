"""Visual spot-check for Task 1 patch/image alignment."""
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

from _task1_common import CODE_DIR, SPLIT_DIRS, collect_patch_records

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visual_check")
os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.RandomState(123)


def has_building(label_path):
    return io.imread(label_path).max() > 0


def choose_records(split, count):
    records = collect_patch_records(split)
    records_with_building = [record for record in records if has_building(record.label_path)]
    candidates = records_with_building if len(records_with_building) >= count else records
    if not candidates:
        return []
    chosen_idx = rng.choice(len(candidates), min(count, len(candidates)), replace=False)
    return [candidates[i] for i in chosen_idx]


def label_boundary(mask):
    mask_i = mask.astype(np.int32)
    b0 = np.abs(np.diff(mask_i, axis=0))
    b0 = np.pad(b0, ((0, 1), (0, 0))) > 0
    b1 = np.abs(np.diff(mask_i, axis=1))
    b1 = np.pad(b1, ((0, 0), (0, 1))) > 0
    return b0 | b1


train_records = choose_records("train", 6)
val_records = choose_records("val", 4)
records = train_records + val_records

fig, axes = plt.subplots(5, 4, figsize=(16, 18))
axes = np.asarray(axes)

for i, record in enumerate(records):
    img = io.imread(record.image_path)
    lbl = io.imread(record.label_path)
    boundary = label_boundary(lbl)
    tag = record.split.upper()
    title = f"{tag} {record.name} @ y={record.y}, x={record.x}"

    ax_img = axes[i // 2, (i % 2) * 2]
    ax_ovl = axes[i // 2, (i % 2) * 2 + 1]

    ax_img.imshow(img)
    ax_img.set_title(f"{title} (img)", fontsize=7)
    ax_img.axis("off")

    ax_ovl.imshow(img)
    ax_ovl.imshow(boundary, cmap="hot", alpha=0.5)
    ax_ovl.set_title(f"{title} (overlay)", fontsize=7)
    ax_ovl.axis("off")

src_img = io.imread(os.path.join(CODE_DIR, "img_trainval.png"))
h, w = src_img.shape[:2]
axes[4, 0].imshow(src_img[0:512, 0:512])
axes[4, 0].set_title("Source direct: y=0, x=0", fontsize=7)
axes[4, 0].axis("off")

axes[4, 2].imshow(src_img[h - 512:h, w - 512:w])
axes[4, 2].set_title(f"Source direct: y={h - 512}, x={w - 512}", fontsize=7)
axes[4, 2].axis("off")

for i in range(10, 20):
    axes[i // 4, i % 4].axis("off")

plt.tight_layout(pad=1.0)
out_path = os.path.join(OUT_DIR, "spot_check.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Visual spot-check saved to {out_path}")
