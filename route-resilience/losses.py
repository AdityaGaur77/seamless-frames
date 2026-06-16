"""
losses.py
---------
Combined segmentation loss for occlusion-robust road extraction.

Components
----------
1. DiceLoss              — region overlap, robust to class imbalance
2. IoULoss               — same goal as Dice, sharper gradient at high overlap
3. BoundaryLoss          — penalises errors exactly on the road edge
                           (so the network learns to push the boundary onto
                           occluded pixels instead of leaving a hole)
4. FocalBCE              — optional, down-weights easy background pixels
5. DeepSupervision       — applies the same loss to the two aux heads of
                           AttentionUNet so gradients flow through the
                           encoder/bottleneck.

Final loss =  L_main
              + 0.4 * L_bottleneck
              + 0.4 * L_mid
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Core losses
# --------------------------------------------------------------------------- #
def dice_loss(logits: torch.Tensor, target: torch.Tensor,
              eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss. logits, target in [0, 1] (use sigmoid for logits)."""
    p = torch.sigmoid(logits)
    t = target.float()
    num = 2 * (p * t).sum(dim=(1, 2, 3)) + eps
    den = (p + t).sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()


def iou_loss(logits: torch.Tensor, target: torch.Tensor,
             eps: float = 1e-6) -> torch.Tensor:
    p = torch.sigmoid(logits)
    t = target.float()
    inter = (p * t).sum(dim=(1, 2, 3))
    union = (p + t - p * t).sum(dim=(1, 2, 3))
    return (1 - (inter + eps) / (union + eps)).mean()


def focal_bce(logits: torch.Tensor, target: torch.Tensor,
              alpha: float = 0.75, gamma: float = 2.0) -> torch.Tensor:
    """Focal binary cross-entropy. alpha > 0.5 biases toward the positive class
    so the network does not collapse to all-background on sparse road tiles."""
    bce = F.binary_cross_entropy_with_logits(logits, target.float(),
                                             reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * target + (1 - p) * (1 - target)
    w = alpha * target + (1 - alpha) * (1 - target)
    loss = w * (1 - p_t).pow(gamma) * bce
    return loss.mean()


def boundary_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Boundary-aware loss: extract the 1-pixel boundary of the ground-truth
    mask with a fixed 4-neighbour Laplacian, then weight the BCE there
    ~10x higher than interior pixels.

    This is the trick that lets the network "close" gaps caused by canopy
    or shadow: it forces the predicted boundary onto the occluded pixels
    rather than letting them leak into the background class.

    Expected shapes: logits (B, 1, H, W) or (B, C, H, W); target (B, 1, H, W)
    or (B, H, W).
    """
    # 4-neighbour Laplacian kernel
    k = torch.tensor([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]], dtype=logits.dtype,
                     device=logits.device).view(1, 1, 3, 3)
    tgt = target.float()
    if tgt.ndim == 3:                     # (B, H, W) → (B, 1, H, W)
        tgt = tgt.unsqueeze(1)
    # If logits has C>1 channels, take the first (binary segmentation)
    if logits.shape[1] > 1:
        logits = logits[:, :1]
    edges = (F.conv2d(tgt, k, padding=1).abs() > 0).float()
    # Soften with a small Gaussian so we also punish predictions that
    # come within 1 pixel of the true boundary
    k_g = torch.tensor([[1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]], dtype=logits.dtype,
                       device=logits.device).view(1, 1, 3, 3) / 16.0
    soft_edges = F.conv2d(edges, k_g, padding=1) * 10.0  # 0..10
    soft_edges = soft_edges.clamp(0.0, 10.0)

    bce = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
    return (soft_edges * bce).mean()


# --------------------------------------------------------------------------- #
# Combined loss
# --------------------------------------------------------------------------- #
class CombinedLoss(nn.Module):
    """L = Dice + IoU + FocalBCE + Boundary, with optional deep supervision."""

    def __init__(self, w_dice: float = 1.0, w_iou: float = 1.0,
                 w_focal: float = 0.5, w_bnd: float = 2.0,
                 deep_supervision: bool = True):
        super().__init__()
        self.w_dice, self.w_iou = w_dice, w_iou
        self.w_focal, self.w_bnd = w_focal, w_bnd
        self.deep_supervision = deep_supervision

    def _core(self, logits: torch.Tensor,
              target: torch.Tensor) -> torch.Tensor:
        return (self.w_dice * dice_loss(logits, target)
                + self.w_iou * iou_loss(logits, target)
                + self.w_focal * focal_bce(logits, target)
                + self.w_bnd * boundary_loss(logits, target))

    def forward(self, outputs, target: torch.Tensor) -> torch.Tensor:
        if isinstance(outputs, tuple):
            main, aux_b, aux_m = outputs
        else:
            main, aux_b, aux_m = outputs, None, None
        loss = self._core(main, target)
        if self.deep_supervision and aux_b is not None:
            loss = loss + 0.4 * self._core(aux_b, target)
            loss = loss + 0.4 * self._core(aux_m, target)
        return loss


# --------------------------------------------------------------------------- #
# Metrics (no grad)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def dice_iou(logits: torch.Tensor, target: torch.Tensor,
             thresh: float = 0.5, eps: float = 1e-6):
    p = (torch.sigmoid(logits) > thresh).float()
    t = (target > 0.5).float()
    inter = (p * t).sum(dim=(1, 2, 3))
    union = (p + t - p * t).sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (p.sum(dim=(1, 2, 3))
                                + t.sum(dim=(1, 2, 3)) + eps)
    iou = (inter + eps) / (union + eps)
    return dice.mean().item(), iou.mean().item()


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 64, 64)
    tgt = (torch.rand(2, 1, 64, 64) > 0.85).float()
    crit = CombinedLoss()
    loss = crit((logits, logits, logits), tgt)
    d, i = dice_iou(logits, tgt)
    print(f"loss={loss.item():.4f}  dice={d:.3f}  iou={i:.3f}")
