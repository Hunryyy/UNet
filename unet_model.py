import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


BASE_DIR = Path(__file__).resolve().parent
LOCAL_RESNET34_WEIGHT = BASE_DIR / "weight" / "resnet34-b627a593.pth"


def _load_state_dict(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_pretrained_resnet34():
    if not LOCAL_RESNET34_WEIGHT.is_file():
        raise FileNotFoundError(
            "Missing required pretrained weight file: "
            f"{LOCAL_RESNET34_WEIGHT}. Download resnet34-b627a593.pth and place it "
            "under code/weight/ before running Task 2."
        )

    logging.info("Loading pretrained ResNet34 from repo-local weight file.")
    res34 = models.resnet34(weights=None)
    state_dict = _load_state_dict(LOCAL_RESNET34_WEIGHT)
    res34.load_state_dict(state_dict)
    return res34


def _init_modules_deterministically(modules, seed):
    # Keep initialization reproducible without leaking RNG changes outward.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        for module in modules:
            for submodule in module.modules():
                if isinstance(submodule, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        submodule.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if submodule.bias is not None:
                        nn.init.zeros_(submodule.bias)
                elif isinstance(submodule, nn.BatchNorm2d):
                    nn.init.ones_(submodule.weight)
                    nn.init.zeros_(submodule.bias)


class Res34UNet_light(nn.Module):
    def __init__(self, init_seed=42):
        super().__init__()
        res34 = _load_pretrained_resnet34()

        self.inc = nn.Sequential(res34.conv1, res34.bn1, res34.relu)
        self.maxpool = res34.maxpool
        self.down1 = res34.layer1
        self.down2 = res34.layer2
        self.down3 = res34.layer3
        self.down4 = res34.layer4

        self.state = [64, 64, 128, 256, 512]

        self.up1 = nn.Sequential(
            nn.Conv2d(self.state[4], self.state[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.state[3]),
            nn.ReLU(),
            nn.Conv2d(self.state[3], self.state[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.state[3]),
            nn.ReLU(),
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(self.state[3] * 2, self.state[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.state[2]),
            nn.ReLU(),
            nn.Conv2d(self.state[2], self.state[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.state[2]),
            nn.ReLU(),
        )

        self.up3 = nn.Sequential(
            nn.Conv2d(self.state[2] * 2, self.state[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.state[1]),
            nn.ReLU(),
            nn.Conv2d(self.state[1], self.state[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.state[1]),
            nn.ReLU(),
        )

        self.up4 = nn.Sequential(
            nn.Conv2d(self.state[1] * 2, self.state[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.state[0]),
            nn.ReLU(),
            nn.Conv2d(self.state[0], self.state[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.state[0]),
            nn.ReLU(),
        )

        self.up5 = nn.Sequential(
            nn.Conv2d(self.state[0] * 2, self.state[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.state[0]),
            nn.ReLU(),
            nn.Conv2d(self.state[0], self.state[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.state[0]),
            nn.ReLU(),
        )

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.classifier = nn.Sequential(
            nn.Conv2d(self.state[0], 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1, 1),
        )
        self.criterion = nn.BCEWithLogitsLoss()

        _init_modules_deterministically(
            [self.up1, self.up2, self.up3, self.up4, self.up5, self.classifier],
            seed=init_seed,
        )

    def forward(self, x, gts=None):
        x1 = self.inc(x)
        x2 = self.down1(self.maxpool(x1))
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = torch.cat([self.up(x), x4], dim=1)
        x = self.up2(x)
        x = torch.cat([self.up(x), x3], dim=1)
        x = self.up3(x)
        x = torch.cat([self.up(x), x2], dim=1)
        x = self.up4(x)
        x = torch.cat([self.up(x), x1], dim=1)
        x = self.up5(x)
        out = self.classifier(self.up(x))

        if self.training:
            if gts is None:
                raise ValueError("Ground-truth masks are required when model is in training mode.")
            loss = self.criterion(out[:, 0], gts.float())
            return loss, out

        return out
