# v2 improvement: EfficientNet-b0..b4 encoder backbone replacing ResNet34.
# Stage boundaries are auto-detected from spatial resolution changes at init
# time, so the skip-connection channels are always correct per variant.

import logging

import torch
import torch.nn as nn
from torchvision import models


def _resolve_efficientnet_stages(effnet):
    """Trace through effnet.features and record (block_idx, channels, height)
    at every point where spatial resolution changes."""
    x = torch.randn(1, 3, 512, 512)
    stage_info = []
    current_h = 512
    with torch.no_grad():
        for idx, layer in enumerate(effnet.features):
            x = layer(x)
            h = x.shape[2]
            if h != current_h or idx == 0:
                stage_info.append((idx, x.shape[1], h))
                current_h = h
    return stage_info


def _build_stage_boundaries(stage_info):
    """Convert stage-info list into (start, end) slice boundaries."""
    boundaries = []
    for i in range(len(stage_info)):
        start = stage_info[i][0]
        end = stage_info[i + 1][0] if i + 1 < len(stage_info) else None
        boundaries.append((start, end))
    return boundaries


def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, 1, 1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, 3, 1, 1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
    )


# ---------------------------------------------------------------------------
# Builder registry (no hardcoded stage boundaries)
# ---------------------------------------------------------------------------

_BUILDERS = {
    "efficientnet-b0": models.efficientnet_b0,
    "efficientnet-b1": models.efficientnet_b1,
    "efficientnet-b2": models.efficientnet_b2,
    "efficientnet-b3": models.efficientnet_b3,
    "efficientnet-b4": models.efficientnet_b4,
}


class EfficientUNet(nn.Module):
    """UNet with an EfficientNet encoder.

    v2 over baseline Res34UNet_light:
      - Auto-detects encoder stages from the actual backbone, so every variant
        (b0..b4) wires skip connections correctly without manual tuning.
      - Supports optional encoder freezing for transfer-learning experiments.
    """

    def __init__(self, backbone="efficientnet-b0", pretrained=True,
                 freeze_encoder=False, init_seed=42):
        super().__init__()

        if backbone == "adaptive":
            backbone = self.select_adaptive_backbone()

        if backbone not in _BUILDERS:
            raise ValueError(
                f"Unknown backbone: {backbone}. Choose from {list(_BUILDERS)}."
            )
        self.backbone = backbone

        # --- build encoder -------------------------------------------------
        builder = _BUILDERS[backbone]
        # v2: fork RNG so encoder init is reproducible without leaking state
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(init_seed))
            try:
                effnet = builder(weights="IMAGENET1K_V1" if pretrained else None)
            except (TypeError, ValueError):
                logging.warning(
                    "IMAGENET1K_V1 weights unavailable for %s, "
                    "falling back to random init.",
                    backbone,
                )
                effnet = builder(weights=None)

        # v2: dynamic stage detection — no hardcoded slice indices
        stage_info = _resolve_efficientnet_stages(effnet)
        boundaries = _build_stage_boundaries(stage_info)
        if len(boundaries) != 5:
            raise RuntimeError(
                f"Expected 5 encoder stages for UNet, got {len(boundaries)}. "
                f"Stage info: {stage_info}"
            )

        self.encoder_stages = nn.ModuleList([
            nn.Sequential(*effnet.features[start:end])
            for start, end in boundaries
        ])

        self.enc_channels = self._probe_channels()
        logging.info(
            "EfficientUNet (%s) encoder channels: %s",
            backbone, self.enc_channels,
        )

        # --- build decoder -------------------------------------------------
        s = self.enc_channels  # [c0, c1, c2, c3, c4]
        self.up1 = _conv_block(s[4], s[3])
        self.up2 = _conv_block(s[3] * 2, s[2])
        self.up3 = _conv_block(s[2] * 2, s[1])
        self.up4 = _conv_block(s[1] * 2, s[0])
        self.up5 = _conv_block(s[0] * 2, s[0])
        self.dec_state = [s[0], s[1], s[2], s[3], s[4]]

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.classifier = nn.Sequential(
            nn.Conv2d(s[0], 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1, 1),
        )
        self.criterion = nn.BCEWithLogitsLoss()

        if freeze_encoder:
            for stage in self.encoder_stages:
                for p in stage.parameters():
                    p.requires_grad = False

        self._init_decoder(seed=init_seed)

    # ------------------------------------------------------------------ #

    @staticmethod
    def select_adaptive_backbone():
        # Start from the smallest strong baseline; later ablations can override.
        return "efficientnet-b0"

    def _probe_channels(self):
        x = torch.randn(1, 3, 512, 512)
        channels = []
        for stage in self.encoder_stages:
            x = stage(x)
            channels.append(x.shape[1])
        return channels

    def _init_decoder(self, seed):
        modules = [self.up1, self.up2, self.up3, self.up4, self.up5,
                   self.classifier]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            for module in modules:
                for m in module.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(
                            m.weight, mode="fan_out", nonlinearity="relu",
                        )
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.ones_(m.weight)
                        nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #

    def forward(self, x, gts=None):
        skips = []
        for stage in self.encoder_stages:
            x = stage(x)
            skips.append(x)

        enc0, enc1, enc2, enc3, enc4 = skips

        d = self.up1(enc4)
        d = self.up(d)
        d = torch.cat([d, enc3], dim=1)
        d = self.up2(d)

        d = self.up(d)
        d = torch.cat([d, enc2], dim=1)
        d = self.up3(d)

        d = self.up(d)
        d = torch.cat([d, enc1], dim=1)
        d = self.up4(d)

        d = self.up(d)
        d = torch.cat([d, enc0], dim=1)
        d = self.up5(d)

        out = self.classifier(self.up(d))

        if self.training:
            if gts is None:
                raise ValueError(
                    "Ground-truth masks are required when model is in training mode."
                )
            loss = self.criterion(out[:, 0], gts.float())
            return loss, out

        return out
