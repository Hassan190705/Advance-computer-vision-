"""
Training Pipeline for Reproducible Semantic Segmentation Experiment
Assignment B - Unit 3: Advanced Computer Vision (CS107203CS)

Records and executes:
- Architecture & mechanism: Baseline vs Dilated Bottleneck vs Additive Skips
- Fixed seed (reproducibility contract)
- Fixed resolution: 192x256
- Augmentation: Horizontal flip (p=0.5), Brightness jitter
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Loss: CrossEntropyLoss with ignore_index=255
- Stopping rule: Fixed max epochs (default: 10) with early stopping on val mIoU
"""

import os
import sys
import time
import json
import random
import argparse
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data_contract import get_dataloaders, CLASS_NAMES, NUM_CLASSES, IGNORE_INDEX
from src.models import build_model, get_model_summary
from src.metrics import StreamSegMetrics


def set_seed(seed: int = 42):
    """Sets seeds for pseudo-random number generators across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class Logger:
    """Simultaneously writes messages to stdout and a log file."""
    def __init__(self, log_path: str):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_file = open(log_path, "w", encoding="utf-8")

    def log(self, message: str = ""):
        self.terminal.write(message + "\n")
        self.terminal.flush()
        self.log_file.write(message + "\n")
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    running_loss = 0.0
    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device, evaluator) -> Dict[str, Any]:
    model.eval()
    running_loss = 0.0
    evaluator.reset()

    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)
        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)
        evaluator.update(preds, masks)

    val_loss = running_loss / len(loader.dataset)
    metrics = evaluator.get_results()
    metrics["Loss"] = val_loss
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Reproducible Semantic Segmentation Training")
    parser.add_argument("--mechanism", type=str, default="baseline",
                        choices=["baseline", "dilated_bottleneck", "additive_skips"],
                        help="Ablation mechanism to run")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--base_channels", type=int, default=16, help="Encoder-decoder base channel width")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--resolution", nargs=2, type=int, default=[192, 256], help="H W input resolution")
    parser.add_argument("--data_dir", type=str, default="data/CamVid", help="Path to CamVid root directory")
    parser.add_argument("--save_dir", type=str, default="experiments", help="Base output directory")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience (epochs)")
    args = parser.parse_args()

    # 1. Enforce strict reproducibility
    set_seed(args.seed)

    # 2. Setup directory and logging
    exp_name = f"{args.mechanism}_seed{args.seed}_res{args.resolution[0]}x{args.resolution[1]}_bc{args.base_channels}"
    exp_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = Logger(os.path.join(exp_dir, "training.log"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.log("=" * 70)
    logger.log("ASSIGNMENT B: REPRODUCIBLE SEGMENTATION TRAINING RUN")
    logger.log("=" * 70)
    logger.log(f"Experiment Name:    {exp_name}")
    logger.log(f"Mechanism:          {args.mechanism}")
    logger.log(f"Base Channels:      {args.base_channels}")
    logger.log(f"Random Seed:        {args.seed}")
    logger.log(f"Input Resolution:   {args.resolution[0]} x {args.resolution[1]}")
    logger.log(f"Target Epochs:      {args.epochs} (Early Stopping Patience: {args.patience})")
    logger.log(f"Batch Size:         {args.batch_size}")
    logger.log(f"Optimizer:          AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
    logger.log(f"Scheduler:          CosineAnnealingLR (T_max={args.epochs}, eta_min=1e-5)")
    logger.log(f"Loss Function:      CrossEntropyLoss (ignore_index={IGNORE_INDEX})")
    logger.log(f"Compute Device:     {device}")
    logger.log("=" * 70)

    # 3. Data Loaders
    resolution = tuple(args.resolution)
    loaders = get_dataloaders(args.data_dir, batch_size=args.batch_size, resolution=resolution)
    logger.log(f"Data Splits: Train={len(loaders['train'].dataset)}, Val={len(loaders['val'].dataset)}, Test={len(loaders['test'].dataset)}")

    # 4. Model Instantiation & Parameter Budget
    model = build_model(mechanism=args.mechanism, num_classes=NUM_CLASSES, base_channels=args.base_channels)
    model.to(device)
    summary = get_model_summary(model, input_size=(1, 3, resolution[0], resolution[1]))
    logger.log(f"Total Parameters:      {summary['total_parameters']:,}")
    logger.log(f"Trainable Parameters:  {summary['trainable_parameters']:,}")
    logger.log("-" * 70)

    # 5. Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    evaluator = StreamSegMetrics(num_classes=NUM_CLASSES, class_names=CLASS_NAMES, ignore_index=IGNORE_INDEX)

    # 6. Training Loop with Stopping Rule
    best_val_miou = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss = train_epoch(model, loaders["train"], optimizer, criterion, device)
        val_metrics = evaluate_epoch(model, loaders["val"], criterion, device, evaluator)
        scheduler.step()

        val_loss = val_metrics["Loss"]
        val_miou = val_metrics["mIoU"]
        val_acc = val_metrics["Pixel_Accuracy"]
        epoch_duration = time.time() - epoch_start

        logger.log(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val mIoU: {val_miou * 100:.2f}% | "
            f"Val Acc: {val_acc * 100:.2f}% | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_duration:.1f}s"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_miou": val_miou,
            "val_acc": val_acc,
            "lr": current_lr,
            "duration_s": epoch_duration,
            "per_class_val_iou": val_metrics["Per_Class_IoU"]
        }
        history.append(epoch_record)

        # Checkpoint: Save Best Model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_miou": val_miou,
                "args": vars(args),
                "summary": summary
            }, os.path.join(exp_dir, "best_model.pth"))
            logger.log(f"  >>> Best model saved (mIoU: {best_val_miou * 100:.2f}%)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logger.log(f"\n[STOPPING RULE TRIGGERED] Validation mIoU did not improve for {args.patience} epochs.")
                break

    total_training_time = time.time() - start_time
    logger.log("=" * 70)
    logger.log(f"TRAINING COMPLETE in {total_training_time:.1f}s ({total_training_time / 60:.2f} mins)")
    logger.log(f"Best Validation mIoU: {best_val_miou * 100:.2f}% at Epoch {best_epoch}")
    logger.log("=" * 70)

    # Save last model checkpoint and full history JSON
    torch.save(model.state_dict(), os.path.join(exp_dir, "last_model.pth"))
    with open(os.path.join(exp_dir, "history.json"), "w") as f:
        json.dump({
            "config": vars(args),
            "summary": summary,
            "best_epoch": best_epoch,
            "best_val_miou": best_val_miou,
            "total_training_time_s": total_training_time,
            "epochs": history
        }, f, indent=2)

    logger.close()


if __name__ == "__main__":
    main()
