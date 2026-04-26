"""
Training Script — Document Forgery Detection
=============================================
Usage:
    python train.py --data_dir data/processed --epochs 30 --batch_size 32

Features:
  - Mixed precision training (AMP)
  - Cosine annealing LR scheduler
  - Early stopping
  - Checkpointing (best model saved)
  - TensorBoard logging
  - Class-weighted loss for imbalanced datasets
"""

import os
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from tqdm import tqdm

from models.model import build_model
from utils.dataset import build_dataloaders


# ─── Argument Parser ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train Forgery Detection Model")
    parser.add_argument("--data_dir",    type=str,   default="data/processed")
    parser.add_argument("--model_type",  type=str,   default="dual",
                        choices=["dual", "single"])
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--image_size",  type=int,   default=224)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--save_dir",    type=str,   default="checkpoints")
    parser.add_argument("--patience",    type=int,   default=7,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--fp16",        action="store_true",
                        help="Enable mixed precision training")
    parser.add_argument("--pretrained",  action="store_true", default=True)
    return parser.parse_args()


# ─── Metric Helpers ─────────────────────────────────────────────────────────────

def compute_metrics(all_labels, all_preds, all_probs):
    report = classification_report(all_labels, all_preds,
                                   target_names=["Authentic", "Forged"],
                                   output_dict=True)
    try:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    except Exception:
        auc = 0.0
    cm = confusion_matrix(all_labels, all_preds)
    return report, auc, cm


# ─── Training Loop ───────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device,
                    scaler, use_ela: bool, is_dual: bool):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in tqdm(loader, desc="  Train", leave=False):
        orig, ela, labels = batch
        orig, ela, labels = orig.to(device), ela.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=scaler is not None):
            if is_dual:
                logits = model(orig, ela)
            else:
                logits = model(ela)
            loss = criterion(logits, labels)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, is_dual: bool):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds, all_probs = [], [], []

    for batch in tqdm(loader, desc="  Eval ", leave=False):
        orig, ela, labels = batch
        orig, ela, labels = orig.to(device), ela.to(device), labels.to(device)

        if is_dual:
            logits = model(orig, ela)
        else:
            logits = model(ela)

        loss   = criterion(logits, labels)
        preds  = logits.argmax(dim=1)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()

        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        total_loss += loss.item() * labels.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs)

    all_probs = np.array(all_probs)
    report, auc, cm = compute_metrics(all_labels, all_preds, all_probs)
    return total_loss / total, correct / total, report, auc, cm


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Train] Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "logs"))

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        use_ela=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    is_dual = args.model_type == "dual"
    model = build_model(args.model_type, pretrained=args.pretrained).to(device)
    print(f"[Train] Model: {args.model_type} | Params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Loss: class-weighted for imbalance ────────────────────────────────────
    labels_all = [s[1] for s in train_loader.dataset.dataset.samples]
    n_auth = labels_all.count(0)
    n_forg = labels_all.count(1)
    weight = torch.tensor([n_forg / (n_auth + 1e-6),
                            n_auth / (n_forg + 1e-6)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    # ── Optimiser + Scheduler ─────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler() if args.fp16 else None

    # ── Training Loop ─────────────────────────────────────────────────────────
    best_val_auc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [],
                "val_acc": [], "val_auc": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, scaler, use_ela=True, is_dual=is_dual,
        )
        val_loss, val_acc, report, val_auc, cm = evaluate(
            model, val_loader, criterion, device, is_dual,
        )
        scheduler.step()

        elapsed = time.time() - t0
        print(f"  Loss  train={train_loss:.4f}  val={val_loss:.4f}")
        print(f"  Acc   train={train_acc:.4f}  val={val_acc:.4f}")
        print(f"  AUC   val={val_auc:.4f}   [{elapsed:.1f}s]")

        # TensorBoard
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)
        writer.add_scalar("AUC/val", val_auc, epoch)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        # Checkpoint best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            ckpt_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "val_acc": val_acc,
                "args": vars(args),
            }, ckpt_path)
            print(f"  ✓ Saved best model (AUC={best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[Train] Early stopping after {epoch} epochs.")
                break

    # ── Test Evaluation ───────────────────────────────────────────────────────
    print("\n[Train] Loading best model for final test evaluation …")
    ckpt = torch.load(os.path.join(args.save_dir, "best_model.pth"),
                      map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_acc, test_report, test_auc, test_cm = evaluate(
        model, test_loader, criterion, device, is_dual,
    )
    print(f"\n{'='*50}")
    print(f"  Test Accuracy : {test_acc:.4f}")
    print(f"  Test AUC      : {test_auc:.4f}")
    print(f"  Confusion Matrix:\n{test_cm}")
    print(f"{'='*50}")

    # Save results
    results = {"test_accuracy": test_acc, "test_auc": test_auc,
               "classification_report": test_report,
               "confusion_matrix": test_cm.tolist(), "history": history}
    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    writer.close()
    print("\n[Train] Done! Results saved to", args.save_dir)


if __name__ == "__main__":
    main()

import torch
import os

os.makedirs("checkpoints", exist_ok=True)

torch.save(model.state_dict(), "checkpoints/best_model.pth")

print("Model saved successfully!")