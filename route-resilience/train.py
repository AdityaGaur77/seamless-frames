"""
train.py
--------
End-to-end training loop for the Attention U-Net.

Usage
-----
    python train.py --epochs 30 --batch 4 --base 16 --out checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import RoadDataset, collate
from losses import CombinedLoss, dice_iou
from model import AttentionUNet


# --------------------------------------------------------------------------- #
def set_seed(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device, thresh: float = 0.5):
    model.eval()
    dices, ious, occ_rec = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["occluded"].to(device)
            y = batch["mask"].to(device)
            main, _, _ = model(x)
            d, i = dice_iou(main, y, thresh=thresh)
            dices.append(d); ious.append(i)
            # Occlusion recall: dice inside the "hard" zone (we approximate
            # by re-evaluating the model on a heavily-noised copy of x)
            x_hard = (x + 0.15 * torch.randn_like(x)).clamp(-1, 1)
            main2, _, _ = model(x_hard)
            d2, _ = dice_iou(main2, y, thresh=thresh)
            occ_rec.append(d2)
    return (float(np.mean(dices)), float(np.mean(ious)),
            float(np.mean(occ_rec)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth")
    ap.add_argument("--out", default="checkpoints/best.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=2)
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    train_ds = RoadDataset(args.data, split="train", augment=True)
    val_ds = RoadDataset(args.data, split="val", augment=False)
    print(f"train={len(train_ds)}  val={len(val_ds)}  device={args.device}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, collate_fn=collate,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=0, collate_fn=collate)

    model = AttentionUNet(in_ch=3, base=args.base).to(args.device)
    n = sum(p.numel() for p in model.parameters())
    print(f"model params: {n/1e6:.2f}M")

    crit = CombinedLoss(w_dice=1.0, w_iou=1.0, w_focal=0.5, w_bnd=2.0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_iou = -1.0
    log = []

    for ep in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        n_step = 0
        t0 = time.time()
        for batch in train_loader:
            x = batch["occluded"].to(args.device)
            y = batch["mask"].to(args.device)
            out_ = model(x)
            # The model returns (main, deep_sup1, deep_sup2).  We add
            # auxiliary deep-supervision losses to the main loss so the
            # intermediate decoder layers also learn to be road-shaped.
            main = out_[0]
            ds1, ds2 = out_[1], out_[2]
            loss = crit(main, y) + 0.3 * crit(ds1, y) + 0.2 * crit(ds2, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
            n_step += 1
        sched.step()

        avg = ep_loss / max(1, n_step)
        if ep % args.val_every == 0 or ep == args.epochs:
            d, i, o = evaluate(model, val_loader, args.device)
            log.append({"ep": ep, "loss": avg, "dice": d, "iou": i,
                        "occ_recall": o, "lr": sched.get_last_lr()[0]})
            print(f"ep {ep:03d}/{args.epochs}  loss={avg:.4f}  "
                  f"val_dice={d:.3f}  val_iou={i:.3f}  occ_recall={o:.3f}  "
                  f"({time.time()-t0:.1f}s)")
            if i > best_iou:
                best_iou = i
                ckpt = out
                torch.save({"model": model.state_dict(),
                            "epoch": ep, "iou": i,
                            "args": vars(args)},
                           ckpt)
                print(f"  [saved] {ckpt} (iou={i:.3f})")
        else:
            log.append({"ep": ep, "loss": avg})
            print(f"ep {ep:03d}/{args.epochs}  loss={avg:.4f}  "
                  f"({time.time()-t0:.1f}s)")

    with open(out.parent / "train_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nDone. best val IoU = {best_iou:.3f}")


if __name__ == "__main__":
    main()
