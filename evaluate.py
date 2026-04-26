"""
Evaluation & Visualisation Script
===================================
Generates comprehensive evaluation reports including:
  - Confusion matrix heatmap
  - ROC curve
  - Precision-Recall curve
  - Per-class metrics
  - Sample predictions grid

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth --data_dir data/processed
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
)
from tqdm import tqdm

from models.model import build_model
from utils.dataset import build_dataloaders, get_val_transforms
from predict import ForgeryPredictor


# ─── Dark theme setup ────────────────────────────────────────────────────────────
plt.style.use("dark_background")
PALETTE = {"bg": "#0a0a0f", "surface": "#12121a", "accent": "#00e5ff",
           "safe": "#00e676", "danger": "#ff3d5a", "text": "#e0e0f0"}


def set_dark_axes(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PALETTE["surface"])
    ax.tick_params(colors=PALETTE["text"])
    ax.spines[:].set_color("#2a2a3e")
    if title:   ax.set_title(title,  color=PALETTE["text"], fontsize=12, pad=10)
    if xlabel:  ax.set_xlabel(xlabel, color=PALETTE["text"])
    if ylabel:  ax.set_ylabel(ylabel, color=PALETTE["text"])


# ─── Inference pass ──────────────────────────────────────────────────────────────

@torch.no_grad()
def run_evaluation(model, loader, device, is_dual):
    model.eval()
    all_labels, all_probs = [], []

    for orig, ela, labels in tqdm(loader, desc="Evaluating"):
        orig, ela = orig.to(device), ela.to(device)
        if is_dual:
            logits = model(orig, ela)
        else:
            logits = model(ela)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)

    return np.array(all_labels), np.array(all_probs)


# ─── Plot Functions ──────────────────────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, save_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=PALETTE["bg"])
    set_dark_axes(ax, "Confusion Matrix")
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=["Authentic", "Forged"],
                yticklabels=["Authentic", "Forged"],
                cmap="Blues", ax=ax, linewidths=0.5,
                linecolor="#2a2a3e", cbar_kws={"shrink": 0.8})
    ax.set_xlabel("Predicted", color=PALETTE["text"])
    ax.set_ylabel("True",      color=PALETTE["text"])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"[Eval] Saved: {save_path}")


def plot_roc_curve(labels, probs, save_path):
    fpr, tpr, _ = roc_curve(labels, probs[:, 1])
    roc_auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5), facecolor=PALETTE["bg"])
    set_dark_axes(ax, f"ROC Curve  (AUC = {roc_auc_val:.4f})",
                  "False Positive Rate", "True Positive Rate")
    ax.plot(fpr, tpr, color=PALETTE["accent"], lw=2.5, label=f"AUC = {roc_auc_val:.4f}")
    ax.plot([0, 1], [0, 1], color=PALETTE["danger"], lw=1.5,
            linestyle="--", label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE["accent"])
    ax.legend(facecolor=PALETTE["surface"], edgecolor="#2a2a3e",
              labelcolor=PALETTE["text"])
    ax.grid(True, color="#1e1e2e", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"[Eval] Saved: {save_path}")


def plot_pr_curve(labels, probs, save_path):
    precision, recall, _ = precision_recall_curve(labels, probs[:, 1])
    ap = average_precision_score(labels, probs[:, 1])

    fig, ax = plt.subplots(figsize=(6, 5), facecolor=PALETTE["bg"])
    set_dark_axes(ax, f"Precision-Recall Curve  (AP = {ap:.4f})",
                  "Recall", "Precision")
    ax.plot(recall, precision, color=PALETTE["safe"], lw=2.5,
            label=f"AP = {ap:.4f}")
    ax.fill_between(recall, precision, alpha=0.08, color=PALETTE["safe"])
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.legend(facecolor=PALETTE["surface"], edgecolor="#2a2a3e",
              labelcolor=PALETTE["text"])
    ax.grid(True, color="#1e1e2e", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"[Eval] Saved: {save_path}")


def plot_score_distribution(labels, probs, save_path):
    auth_scores = probs[labels == 0, 1]
    forg_scores = probs[labels == 1, 1]

    fig, ax = plt.subplots(figsize=(7, 4), facecolor=PALETTE["bg"])
    set_dark_axes(ax, "Forged-Class Score Distribution",
                  "P(Forged)", "Density")
    ax.hist(auth_scores, bins=40, alpha=0.7, color=PALETTE["safe"],
            label="Authentic", density=True)
    ax.hist(forg_scores, bins=40, alpha=0.7, color=PALETTE["danger"],
            label="Forged",    density=True)
    ax.axvline(0.5, color="white", linestyle="--", lw=1.5, label="Threshold=0.5")
    ax.legend(facecolor=PALETTE["surface"], edgecolor="#2a2a3e",
              labelcolor=PALETTE["text"])
    ax.grid(True, color="#1e1e2e", linewidth=0.5, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"[Eval] Saved: {save_path}")


def print_classification_report(labels, preds):
    report = classification_report(labels, preds,
                                   target_names=["Authentic", "Forged"])
    print("\n" + "="*56)
    print("CLASSIFICATION REPORT")
    print("="*56)
    print(report)
    print("="*56 + "\n")


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate forgery detection model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir",   type=str, default="data/processed")
    parser.add_argument("--model_type", type=str, default="dual",
                        choices=["dual", "single"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_dual = args.model_type == "dual"

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"[Eval] Loading model from {args.checkpoint} …")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(args.model_type, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    # ── Load test data ─────────────────────────────────────────────────────────
    _, _, test_loader = build_dataloaders(
        root_dir=args.data_dir, batch_size=args.batch_size,
        image_size=args.image_size, num_workers=2,
    )

    # ── Run inference ──────────────────────────────────────────────────────────
    labels, probs = run_evaluation(model, test_loader, device, is_dual)
    preds = (probs[:, 1] >= 0.5).astype(int)

    # ── Metrics ────────────────────────────────────────────────────────────────
    print_classification_report(labels, preds)

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_confusion_matrix(labels, preds,
                          os.path.join(args.output_dir, "confusion_matrix.png"))
    plot_roc_curve(labels, probs,
                   os.path.join(args.output_dir, "roc_curve.png"))
    plot_pr_curve(labels, probs,
                  os.path.join(args.output_dir, "pr_curve.png"))
    plot_score_distribution(labels, probs,
                             os.path.join(args.output_dir, "score_distribution.png"))

    print(f"[Eval] All plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
