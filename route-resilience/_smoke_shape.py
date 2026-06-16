"""Smoke test: confirm model output shape for 256x256 input."""
import torch
from model import AttentionUNet

m = AttentionUNet(in_ch=3, base=16)
x = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    out, d1, d2 = m(x)
print(f"in:  {tuple(x.shape)}")
print(f"out: {tuple(out.shape)}")
print(f"d1:  {tuple(d1.shape)}")
print(f"d2:  {tuple(d2.shape)}")
