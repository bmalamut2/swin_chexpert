#!/usr/bin/env python3
"""
Train a Swin Transformer on CheXpert.

Examples:
  python train_swin_chexpert.py --data-root chexpert --pretrained
  python train_swin_chexpert.py --data-root chexpert --no-pretrained
  python train_swin_chexpert.py --data-root chexpert --resume
"""

from __future__ import annotations

import argparse
import os
import random
from typing import List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import Dataset, DataLoader
import timm
from timm.data import resolve_data_config, create_transform


LABEL_COLUMNS_DEFAULT = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CheXpertDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        data_root: str,
        label_columns: List[str],
        transform,
        strip_prefix: str,
        uncertain: str,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root
        self.label_columns = label_columns
        self.transform = transform
        self.strip_prefix = strip_prefix
        self.uncertain = uncertain

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, rel_path: str) -> str:
        path = rel_path
        if self.strip_prefix and path.startswith(self.strip_prefix):
            path = path[len(self.strip_prefix) :]
        return os.path.join(self.data_root, path)

    def _process_labels(self, row) -> np.ndarray:
        labels = row[self.label_columns].to_numpy(dtype=np.float32)
        labels = np.nan_to_num(labels, nan=0.0)
        if self.uncertain == "zero":
            labels = np.where(labels == -1, 0, labels)
        elif self.uncertain == "one":
            labels = np.where(labels == -1, 1, labels)
        return labels

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self._resolve_path(row["Path"])
        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"Image not found: {img_path}. Check --data-root or --strip-prefix."
            )

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                img = self.transform(img)

        labels = torch.from_numpy(self._process_labels(row))
        return img, labels


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    return (preds == targets).float().mean().item()


def compute_auc(logits: torch.Tensor, targets: torch.Tensor) -> float:
    probs = torch.sigmoid(logits).cpu().numpy()
    y_true = targets.cpu().numpy()
    per_class = []
    for i in range(y_true.shape[1]):
        y_i = y_true[:, i]
        if np.unique(y_i).size < 2:
            continue
        per_class.append(roc_auc_score(y_i, probs[:, i]))
    if not per_class:
        return float("nan")
    return float(np.mean(per_class))


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_samples = 0
    all_logits = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * batch_size
        total_acc += compute_accuracy(logits.detach(), targets) * batch_size
        num_samples += batch_size
        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())

    epoch_logits = torch.cat(all_logits, dim=0)
    epoch_targets = torch.cat(all_targets, dim=0)
    auc = compute_auc(epoch_logits, epoch_targets)
    return total_loss / num_samples, total_acc / num_samples, auc


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_samples = 0
    all_logits = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = images.size(0)

        logits = model(images)
        loss = criterion(logits, targets)

        total_loss += loss.item() * batch_size
        total_acc += compute_accuracy(logits, targets) * batch_size
        num_samples += batch_size
        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())

    epoch_logits = torch.cat(all_logits, dim=0)
    epoch_targets = torch.cat(all_targets, dim=0)
    auc = compute_auc(epoch_logits, epoch_targets)
    return total_loss / num_samples, total_acc / num_samples, auc


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_loss: float,
    args,
) -> None:
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_loss": best_loss,
        "args": vars(args),
    }
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Swin on CheXpert")
    parser.add_argument("--data-root", default="chexpert", help="Dataset root directory.")
    parser.add_argument("--train-csv", default=None, help="Path to train CSV.")
    parser.add_argument("--val-csv", default=None, help="Path to val CSV.")
    parser.add_argument(
        "--strip-prefix",
        default="CheXpert-v1.0-small/",
        help="Prefix to strip from CSV Path entries. Use '' to disable.",
    )
    parser.add_argument(
        "--label-cols",
        default=",".join(LABEL_COLUMNS_DEFAULT),
        help="Comma-separated label columns to use.",
    )
    parser.add_argument(
        "--uncertain",
        choices=["zero", "one"],
        default="zero",
        help="How to treat uncertain labels (-1).",
    )
    parser.add_argument("--model", default="swin_tiny_patch4_window7_224", help="timm model name.")
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ImageNet pretrained weights.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=None, help="Override input image size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path for resume/last.")
    parser.add_argument("--resume", action="store_true", help="Resume if checkpoint exists.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.train_csv is None:
        args.train_csv = os.path.join(args.data_root, "train.csv")
    if args.val_csv is None:
        args.val_csv = os.path.join(args.data_root, "valid.csv")
    if args.checkpoint is None:
        args.checkpoint = os.path.join(args.output_dir, "last.pt")

    label_columns = [c.strip() for c in args.label_cols.split(",") if c.strip()]
    if not label_columns:
        raise ValueError("No label columns provided.")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    use_amp = args.amp and device.type == "cuda"

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    set_seed(args.seed)

    model = timm.create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=len(label_columns),
    )
    model.to(device)

    data_config = resolve_data_config({}, model=model)
    if args.img_size is not None:
        data_config["input_size"] = (3, args.img_size, args.img_size)
    train_transform = create_transform(**data_config, is_training=True)
    val_transform = create_transform(**data_config, is_training=False)

    train_ds = CheXpertDataset(
        args.train_csv,
        args.data_root,
        label_columns,
        train_transform,
        args.strip_prefix,
        args.uncertain,
    )
    val_ds = CheXpertDataset(
        args.val_csv,
        args.data_root,
        label_columns,
        val_transform,
        args.strip_prefix,
        args.uncertain,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    start_epoch = 0
    best_loss = float("inf")
    if args.resume and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler is not None and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        best_loss = checkpoint.get("best_loss", best_loss)
        start_epoch = checkpoint.get("epoch", -1) + 1
        print(f"Resumed from {args.checkpoint} at epoch {start_epoch}")
    elif args.resume:
        print(f"Checkpoint not found at {args.checkpoint}; starting fresh.")

    os.makedirs(args.output_dir, exist_ok=True)
    last_path = args.checkpoint
    best_path = os.path.join(args.output_dir, "best.pt")

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                best_path, model, optimizer, scheduler, scaler, epoch, best_loss, args
            )

        save_checkpoint(
            last_path, model, optimizer, scheduler, scaler, epoch, best_loss, args
        )

        print(
            f"Epoch {epoch+1}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_auc={train_auc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_auc={val_auc:.4f}"
        )


if __name__ == "__main__":
    main()
