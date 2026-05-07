import logging

import numpy as np
import torch


def fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) + label_pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return hist


def compute_iou(hist):
    intersection = np.diag(hist)
    union = hist.sum(axis=1) + hist.sum(axis=0) - intersection
    iou_per_class = intersection / (union + 1e-8)
    return iou_per_class.mean(), iou_per_class


def eval_net(net, loader, device):
    was_training = net.training
    net.eval()
    hist = np.zeros((2, 2), dtype=np.int64)
    n_batches = len(loader)

    with torch.no_grad():
        for num, batch in enumerate(loader):
            if num % 20 == 19:
                logging.info(f"Validating {num + 1}/{n_batches}")

            imgs = batch["image"].to(device=device, dtype=torch.float32)
            true_masks = batch["mask"].to(device=device, dtype=torch.float32)
            mask_pred = net(imgs)

            if mask_pred.shape[1] != 1:
                pred = torch.argmax(mask_pred, dim=1)
            else:
                pred = torch.sigmoid(mask_pred[:, 0])
                pred = (pred > 0.5).int()

            hist += fast_hist(
                pred.cpu().numpy().ravel(),
                true_masks.cpu().int().numpy().ravel(),
                num_classes=2,
            )

    mean_iou, per_class_iou = compute_iou(hist)
    logging.info(
        "IOU: %.6f  (bg: %.6f, bld: %.6f)",
        mean_iou,
        per_class_iou[0],
        per_class_iou[1],
    )
    logging.info("%s", hist)

    if was_training:
        net.train()

    return float(mean_iou)
