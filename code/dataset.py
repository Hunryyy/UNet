import logging
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as transF


class MyDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, mean, std, is_train, seed=42):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.mean = mean
        self.std = std
        self.is_train = is_train
        self.seed = int(seed)
        self.ids = sorted(os.listdir(imgs_dir))

        logging.info(f"Creating dataset with {len(self.ids)} examples")

    def __len__(self):
        return len(self.ids)

    def _augment(self, img_np, mask_np, index):
        # Deterministic per-sample geometric augmentation keeps image/mask
        # perfectly synchronized and makes the baseline reproducible.
        rng = np.random.default_rng(self.seed + int(index))
        rot_k = int(rng.integers(0, 4))
        do_vflip = bool(rng.integers(0, 2))
        do_hflip = bool(rng.integers(0, 2))

        if rot_k:
            img_np = np.rot90(img_np, k=rot_k, axes=(0, 1))
            mask_np = np.rot90(mask_np, k=rot_k, axes=(0, 1))
        if do_vflip:
            img_np = np.flip(img_np, axis=0)
            mask_np = np.flip(mask_np, axis=0)
        if do_hflip:
            img_np = np.flip(img_np, axis=1)
            mask_np = np.flip(mask_np, axis=1)

        return np.ascontiguousarray(img_np), np.ascontiguousarray(mask_np)

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = os.path.join(self.masks_dir, idx)
        img_file = os.path.join(self.imgs_dir, idx)
        mask = Image.open(mask_file)
        img = Image.open(img_file)

        img_np = np.array(img)
        mask_np = (np.array(mask) > 0).astype(np.uint8)

        if self.is_train:
            img_np, mask_np = self._augment(img_np, mask_np, i)

        img_tensor = transF.to_tensor(img_np.copy())
        mask_tensor = (transF.to_tensor(mask_np.copy()) > 0).int()
        img_tensor = transF.normalize(img_tensor, self.mean, self.std)

        return {
            "image": img_tensor.float(),
            "mask": mask_tensor[0].float(),
            "name": idx,
        }
