"""
model.py
--------
Attention U-Net (Ronneberger 2015 + Oktay 2018) with a ResNet-style
encoder built from scratch (no ImageNet weights — keeps the demo
self-contained and CPU-trainable in a few minutes).

Input : (B, 3, H, W)  — RGB tile
Output: (B, 1, H, W)  — logits for binary road mask

Key context-aware features:
  - Attention gates on the skip connections (channel + spatial)
  - Deep supervision: auxiliary head at bottleneck + mid level
  - Boundary-aware loss is implemented in losses.py
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class ConvBNAct(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """Residual block: two 3x3 convs with a 1x1 skip."""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv1 = ConvBNAct(in_c, out_c, 3, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.skip = (nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
                     if in_c != out_c else nn.Identity())
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.conv2(y)
        return self.act(y + self.skip(x))


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al. 2018).

    g : gating signal from the coarser decoder level  (B, C_g, H, W)
    x : encoder skip features                          (B, C_x, H, W)
    """

    def __init__(self, c_x: int, c_g: int, c_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(c_g, c_int, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(c_x, c_int, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(c_int, 1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # match spatial dims in case of off-by-one
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear",
                              align_corners=False)
        q = self.W_g(g)
        k = self.W_x(x)
        alpha = self.psi(self.act(q + k))
        return x * alpha


# --------------------------------------------------------------------------- #
# Attention U-Net
# --------------------------------------------------------------------------- #
class AttentionUNet(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 32):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(ResBlock(in_ch, base),
                                  ResBlock(base, base))
        self.enc2 = nn.Sequential(ResBlock(base, base * 2),
                                  ResBlock(base * 2, base * 2))
        self.enc3 = nn.Sequential(ResBlock(base * 2, base * 4),
                                  ResBlock(base * 4, base * 4))
        self.enc4 = nn.Sequential(ResBlock(base * 4, base * 8),
                                  ResBlock(base * 8, base * 8))
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(base * 8, base * 16),
            ResBlock(base * 16, base * 16),
        )
        # Decoder
        # Gating signals come from the coarser decoder level (one level
        # deeper), so they have *more* channels than the skip features.
        self.up3 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        # g = bottleneck b with base*16 channels
        self.ag3 = AttentionGate(c_x=base * 8, c_g=base * 16, c_int=base * 4)
        self.dec3 = nn.Sequential(ResBlock(base * 16, base * 8),
                                  ResBlock(base * 8, base * 8))
        self.up2 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        # g = d3 with base*8 channels
        self.ag2 = AttentionGate(c_x=base * 4, c_g=base * 8, c_int=base * 2)
        self.dec2 = nn.Sequential(ResBlock(base * 8, base * 4),
                                  ResBlock(base * 4, base * 4))
        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        # g = d2 with base*4 channels
        self.ag1 = AttentionGate(c_x=base * 2, c_g=base * 4, c_int=base)
        self.dec1 = nn.Sequential(ResBlock(base * 4, base * 2),
                                  ResBlock(base * 2, base * 2))
        # Final decoder stage: H/2 → H, using the e1 skip connection
        self.up0 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        # g = d1 with base*2 channels
        self.ag0 = AttentionGate(c_x=base, c_g=base * 2, c_int=max(base // 2, 1))
        self.dec0 = nn.Sequential(ResBlock(base * 2, base),
                                  ResBlock(base, base))
        # Output head
        self.head = nn.Conv2d(base, 1, 1, 1, 0)
        # Deep-supervision aux heads
        self.aux_b = nn.Conv2d(base * 16, 1, 1, 1, 0)
        self.aux_m = nn.Conv2d(base * 4, 1, 1, 1, 0)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d3 = self.up3(b)
        # g = b (bottleneck, base*16 channels, H/16 — AG interpolates to H/8)
        e4 = self.ag3(e4, b)
        d3 = self.dec3(torch.cat([d3, e4], dim=1))
        d2 = self.up2(d3)
        # g = d3 (base*8 channels, H/8 — AG interpolates to H/4)
        e3 = self.ag2(e3, d3)
        d2 = self.dec2(torch.cat([d2, e3], dim=1))
        d1 = self.up1(d2)
        # g = d2 (base*4 channels, H/4 — AG interpolates to H/2)
        e2 = self.ag1(e2, d2)
        d1 = self.dec1(torch.cat([d1, e2], dim=1))
        d0 = self.up0(d1)
        # g = d1 (base*2 channels, H/2 — AG interpolates to H)
        e1 = self.ag0(e1, d1)
        d0 = self.dec0(torch.cat([d0, e1], dim=1))

        logits = self.head(d0)
        # aux heads for deep supervision (downsampled to match loss)
        aux_b = F.interpolate(self.aux_b(b), size=logits.shape[-2:],
                              mode="bilinear", align_corners=False)
        aux_m = F.interpolate(self.aux_m(d2), size=logits.shape[-2:],
                              mode="bilinear", align_corners=False)
        return logits, aux_b, aux_m


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    m = AttentionUNet(in_ch=3, base=16)
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y, ab, am = m(x)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"AttentionUNet: {n_params/1e6:.2f}M params")
    print(f"main out:  {y.shape}")
    print(f"aux b out: {ab.shape}")
    print(f"aux m out: {am.shape}")
