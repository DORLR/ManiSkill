"""UNet on the wrist RGB (v4): per-pixel classes (perception/labels.CLASSES), used to place the crops.

Encoder: ResNet-34, ImageNet-pretrained (torchvision), trained. Decoder: UNet skip connections at
strides 16, 8, 4, 2 and a full-resolution head (the pins and openings are a few pixels wide; the
crop positions are taken from these masks, so the output is at the input resolution).
Input: RGB only (no depth), as decided with the user.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .labels import CLASSES


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Up(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.conv = _block(cin + cskip, cout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class UNet(nn.Module):
    def __init__(self, n_classes: int = len(CLASSES), pretrained: bool = True):
        super().__init__()
        r = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)           # /2, 64
        self.pool = r.maxpool                                       # /4
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4   # /4 64, /8 128, /16 256, /32 512
        self.full = _block(3, 32)                                   # full-resolution skip from the image itself
        self.u4 = Up(512, 256, 256)
        self.u3 = Up(256, 128, 128)
        self.u2 = Up(128, 64, 64)
        self.u1 = Up(64, 64, 64)
        self.u0 = Up(64, 32, 32)
        self.head = nn.Conv2d(32, n_classes, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """rgb (B, H, W, 3) uint8 -> logits (B, n_classes, H, W)."""
        x = rgb.permute(0, 3, 1, 2).float() / 255.0
        f0 = self.full(x)
        x = (x - self.mean) / self.std
        s1 = self.stem(x)
        s2 = self.l1(self.pool(s1))
        s3 = self.l2(s2)
        s4 = self.l3(s3)
        s5 = self.l4(s4)
        y = self.u4(s5, s4)
        y = self.u3(y, s3)
        y = self.u2(y, s2)
        y = self.u1(y, s1)
        y = self.u0(y, f0)
        return self.head(y)


def load_unet(path: str, device=None) -> UNet:
    ck = torch.load(path, map_location="cpu")
    m = UNet(pretrained=False)
    m.load_state_dict(ck["model"])
    m.eval()
    return m.to(device) if device is not None else m
