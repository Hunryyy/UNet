import torch
import torch.nn as nn

from cbam import CBAM
from unet_model import Res34UNet_light


class Res34UNet_CBAM(Res34UNet_light):
    def __init__(self, cbam_mode="adaptive",
                 cbam_encoder=True, cbam_decoder=True, cbam_skip=True,
                 cbam_reduction=16, init_seed=42):
        cbam_encoder, cbam_decoder, cbam_skip = self._resolve_cbam_mode(
            cbam_mode, cbam_encoder, cbam_decoder, cbam_skip
        )
        self.cbam_mode = cbam_mode
        self._cbam_flags = (cbam_encoder, cbam_decoder, cbam_skip)
        self.init_seed = int(init_seed)
        super().__init__(init_seed=init_seed)
        self.cbam_reduction = cbam_reduction
        self._build_cbam_modules(cbam_encoder, cbam_decoder, cbam_skip, cbam_reduction)

    @staticmethod
    def _resolve_cbam_mode(mode, cbam_encoder, cbam_decoder, cbam_skip):
        if mode == "adaptive":
            return False, True, True
        if mode == "skip":
            return False, False, True
        if mode == "decoder":
            return False, True, False
        if mode == "encoder":
            return True, False, False
        if mode == "enc_dec":
            return True, True, True
        if mode == "manual":
            return cbam_encoder, cbam_decoder, cbam_skip
        raise ValueError(
            "Unsupported cbam_mode. Choose from "
            "['adaptive', 'skip', 'decoder', 'encoder', 'enc_dec', 'manual']."
        )

    def _build_cbam_modules(self, encoder, decoder, skip, reduction):
        ch = self.state  # [64, 64, 128, 256, 512]
        # v2: fork RNG so CBAM init is reproducible without leaking state
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.init_seed)

            if encoder:
                self.cbam_enc = nn.ModuleList([
                    CBAM(ch[0], reduction),  # enc0:  64ch  H/2
                    CBAM(ch[1], reduction),  # enc1:  64ch  H/4
                    CBAM(ch[2], reduction),  # enc2: 128ch  H/8
                    CBAM(ch[3], reduction),  # enc3: 256ch  H/16
                    CBAM(ch[4], reduction),  # enc4: 512ch  H/32
                ])
            if skip:
                c_skip = [ch[3], ch[2], ch[1], ch[0]]  # deep → shallow
                self.cbam_skip = nn.ModuleList([
                    CBAM(c_skip[0], reduction),  # skip x4: 256ch
                    CBAM(c_skip[1], reduction),  # skip x3: 128ch
                    CBAM(c_skip[2], reduction),  # skip x2:  64ch
                    CBAM(c_skip[3], reduction),  # skip x1:  64ch
                ])
            if decoder:
                c_dec = [ch[2], ch[1], ch[1], ch[0]]  # 128, 64, 64, 64
                self.cbam_dec = nn.ModuleList([
                    CBAM(c_dec[0], reduction),  # after up2 → 128ch
                    CBAM(c_dec[1], reduction),  # after up3 →  64ch
                    CBAM(c_dec[2], reduction),  # after up4 →  64ch
                    CBAM(c_dec[3], reduction),  # after up5 →  64ch
                ])

    def forward(self, x, gts=None):
        cbam_enc, cbam_dec, cbam_skip = self._cbam_flags

        x1 = self.inc(x)
        if cbam_enc:
            x1 = self.cbam_enc[0](x1)

        x2 = self.down1(self.maxpool(x1))
        if cbam_enc:
            x2 = self.cbam_enc[1](x2)

        x3 = self.down2(x2)
        if cbam_enc:
            x3 = self.cbam_enc[2](x3)

        x4 = self.down3(x3)
        if cbam_enc:
            x4 = self.cbam_enc[3](x4)

        x5 = self.down4(x4)
        if cbam_enc:
            x5 = self.cbam_enc[4](x5)

        # Decoder stage 0
        d = self.up1(x5)
        d = self.up(d)
        s4 = self.cbam_skip[0](x4) if cbam_skip else x4
        d = torch.cat([d, s4], dim=1)
        d = self.up2(d)
        if cbam_dec:
            d = self.cbam_dec[0](d)

        # Decoder stage 1
        d = self.up(d)
        s3 = self.cbam_skip[1](x3) if cbam_skip else x3
        d = torch.cat([d, s3], dim=1)
        d = self.up3(d)
        if cbam_dec:
            d = self.cbam_dec[1](d)

        # Decoder stage 2
        d = self.up(d)
        s2 = self.cbam_skip[2](x2) if cbam_skip else x2
        d = torch.cat([d, s2], dim=1)
        d = self.up4(d)
        if cbam_dec:
            d = self.cbam_dec[2](d)

        # Decoder stage 3
        d = self.up(d)
        s1 = self.cbam_skip[3](x1) if cbam_skip else x1
        d = torch.cat([d, s1], dim=1)
        d = self.up5(d)
        if cbam_dec:
            d = self.cbam_dec[3](d)

        out = self.classifier(self.up(d))

        if self.training:
            if gts is None:
                raise ValueError(
                    "Ground-truth masks are required when model is in training mode."
                )
            loss = self.criterion(out[:, 0], gts.float())
            return loss, out

        return out
