"""Swin UNETR image encoder.

Wraps MONAI's ``SwinUNETR``. ``encode`` runs the Swin transformer + CNN encoder
path and returns the flattened bottleneck tokens consumed by the pooling taps.
The U-Net decoder path is kept callable for dense downstream use, but nothing in
the inference API below requires it.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class SwinUNETREncoder(nn.Module):
    """CardioFM image tower. Four input channels = four HU windows."""

    def __init__(
        self,
        in_channels: int = 4,
        feature_size: int = 48,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        spatial_dims: int = 3,
        use_checkpoint: bool = False,
        out_channels: int = 1,
        **_ignore,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.feature_size = feature_size
        self.net = SwinUNETR(
            img_size=(64, 64, 64),  # ignored in monai>=1.3; kept for checkpoint compat
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            depths=tuple(depths),
            num_heads=tuple(num_heads),
            spatial_dims=spatial_dims,
            use_checkpoint=use_checkpoint,
        )
        self.bottleneck_dim = feature_size * 16

    def encode(self, image: torch.Tensor) -> Dict[str, object]:
        net = self.net
        hs = net.swinViT(image, net.normalize)
        enc0 = net.encoder1(image)
        enc1 = net.encoder2(hs[0])
        enc2 = net.encoder3(hs[1])
        enc3 = net.encoder4(hs[2])
        dec4 = net.encoder10(hs[4])                            # [B, C, d, h, w]
        tokens = dec4.flatten(2).transpose(1, 2).contiguous()   # [B, N, C]
        return {
            "tokens": tokens,
            "bottleneck": dec4,
            "feat_pyramid": {"enc0": enc0, "enc1": enc1, "enc2": enc2,
                             "enc3": enc3, "dec4": dec4, "hs3": hs[3]},
        }

    def decode_dense(self, feat_pyramid: Dict[str, torch.Tensor]) -> torch.Tensor:
        """U-Net decoder path -> ``[B, feature_size, D, H, W]`` (pre seg-head)."""
        net, fp = self.net, feat_pyramid
        dec3 = net.decoder5(fp["dec4"], fp["hs3"])
        dec2 = net.decoder4(dec3, fp["enc3"])
        dec1 = net.decoder3(dec2, fp["enc2"])
        dec0 = net.decoder2(dec1, fp["enc1"])
        return net.decoder1(dec0, fp["enc0"])

    def forward(self, image: torch.Tensor) -> Dict[str, object]:
        return self.encode(image)
