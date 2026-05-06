import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


BASE_DIR = Path(__file__).resolve().parent


class EfficientUNet(nn.Module):
    def __init__(self, backbone="efficientnet-b0", pretrained=True, freeze_encoder=False,
                 init_seed=42):
        super().__init__()

        if backbone not in _BACKBONE_CONFIG:
            raise ValueError(f"Unknown backbone: {backbone}. Choose from {list(_BACKBONE_CONFIG.keys())}")
        cfg = _BACKBONE_CONFIG[backbone]

        builder_fn = cfg["builder"]
        if pretrained:
            try:
                effnet = builder_fn(weights="IMAGENET1K_V1")
            except (TypeError, ValueError):
                effnet = builder_fn(weights=None)
        else:
            effnet = builder_fn(weights=None)

        self.encoder_stages = nn.ModuleList()
        stage_boundaries = cfg["stages"]
        for start, end in stage_boundaries:
            stage = nn.Sequential(*effnet.features[start:end])
            self.encoder_stages.append(stage)

        self.enc_channels = self._probe_channels()
        logging.info("EfficientUNet encoder channels: %s", self.enc_channels)

        self.dec_state = self._build_decoder_channels()
        self.up1, self.up2, self.up3, self.up4, self.up5 = self._build_decoder()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.classifier = nn.Sequential(
            nn.Conv2d(self.dec_state[0], 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1, 1),
        )
        self.criterion = nn.BCEWithLogitsLoss()

        if freeze_encoder:
            for stage in self.encoder_stages:
                for p in stage.parameters():
                    p.requires_grad = False

        self._init_decoder(seed=init_seed)

    def _probe_channels(self):
        x = torch.randn(1, 3, 512, 512)
        channels = []
        for stage in self.encoder_stages:
            x = stage(x)
            channels.append(x.shape[1])
        return channels

    def _build_decoder_channels(self):
        ch = list(self.enc_channels)
        return [ch[0], ch[1], ch[2], ch[3], ch[4]]

    def _build_decoder(self):
        s = self.dec_state
        up1 = _conv_block(s[4], s[3])
        up2 = _conv_block(s[3] * 2, s[2])
        up3 = _conv_block(s[2] * 2, s[1])
        up4 = _conv_block(s[1] * 2, s[0])
        up5 = _conv_block(s[0] * 2, s[0])
        return up1, up2, up3, up4, up5

    def _init_decoder(self, seed):
        modules = [self.up1, self.up2, self.up3, self.up4, self.up5, self.classifier]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            for module in modules:
                for m in module.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.ones_(m.weight)
                        nn.init.zeros_(m.bias)

    def forward(self, x, gts=None):
        skips = []
        for stage in self.encoder_stages:
            x = stage(x)
            skips.append(x)

        enc0, enc1, enc2, enc3, enc4 = skips

        x = self.up1(enc4)
        x = self.up(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.up2(x)
        x = self.up(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.up3(x)
        x = self.up(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.up4(x)
        x = self.up(x)
        x = torch.cat([x, enc0], dim=1)
        x = self.up5(x)
        out = self.classifier(self.up(x))

        if self.training:
            if gts is None:
                raise ValueError("Ground-truth masks are required when model is in training mode.")
            loss = self.criterion(out[:, 0], gts.float())
            return loss, out

        return out


def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
    )


def _resolve_efficientnet_stages(model, input_size=512):
    x = torch.randn(1, 3, input_size, input_size)
    stage_info = []
    current_h = input_size
    with torch.no_grad():
        for idx, layer in enumerate(model.features):
            x = layer(x)
            h = x.shape[2]
            c = x.shape[1]
            if h != current_h or idx == 0:
                stage_info.append((idx, c, h))
                current_h = h
    return stage_info


def _build_stage_boundaries(stage_info):
    boundaries = []
    for i in range(len(stage_info)):
        start = stage_info[i][0]
        end = stage_info[i + 1][0] if i + 1 < len(stage_info) else None
        boundaries.append((start, end))
    return boundaries


_BACKBONE_CONFIG = {}

for _name, _builder, _stages_key in [
    ("efficientnet-b0", models.efficientnet_b0,
     [(0, 1), (1, 3), (3, 4), (4, 6), (6, 8)]),
    ("efficientnet-b1", models.efficientnet_b1,
     [(0, 1), (1, 3), (3, 4), (4, 6), (6, 8)]),
    ("efficientnet-b2", models.efficientnet_b2,
     [(0, 1), (1, 3), (3, 4), (4, 6), (6, 8)]),
    ("efficientnet-b3", models.efficientnet_b3,
     [(0, 1), (1, 3), (3, 4), (4, 6), (6, 8)]),
    ("efficientnet-b4", models.efficientnet_b4,
     [(0, 1), (1, 3), (3, 4), (4, 6), (6, 8)]),
]:
    _BACKBONE_CONFIG[_name] = {"builder": _builder, "stages": _stages_key}
