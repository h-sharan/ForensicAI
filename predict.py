"""
Inference Script — Document Forgery Detection
==============================================
Usage:
    # Single image
    python predict.py --image path/to/doc.jpg --checkpoint checkpoints/best_model.pth

    # Folder of images
    python predict.py --image_dir path/to/docs/ --checkpoint checkpoints/best_model.pth

    # With Grad-CAM visualization
    python predict.py --image path/to/doc.jpg --checkpoint checkpoints/best_model.pth --gradcam
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from models.model import build_model
from utils.ela import compute_ela, overlay_ela_on_original
from utils.dataset import get_val_transforms


# ─── Argument Parser ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Run forgery detection inference")
    parser.add_argument("--image",      type=str, default=None,
                        help="Path to a single image")
    parser.add_argument("--image_dir",  type=str, default=None,
                        help="Directory of images to process")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pth checkpoint file")
    parser.add_argument("--model_type", type=str, default="dual",
                        choices=["dual", "single"])
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--threshold",  type=float, default=0.5,
                        help="Decision threshold for 'forged' class (default 0.5)")
    parser.add_argument("--gradcam",    action="store_true",
                        help="Generate Grad-CAM visualisations")
    parser.add_argument("--output_dir", type=str, default="outputs/predictions")
    return parser.parse_args()


# ─── Predictor Class ─────────────────────────────────────────────────────────────

class ForgeryPredictor:
    """Wraps model loading and single/batch inference."""

    LABELS = {0: "Authentic", 1: "Forged"}
    COLORS = {0: (0, 200, 0), 1: (0, 0, 255)}   # green / red (BGR)

    def __init__(self, checkpoint_path: str, model_type: str = "dual",
                 image_size: int = 224, threshold: float = 0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.image_size = image_size
        self.threshold  = threshold
        self.transform  = get_val_transforms(image_size)

        self.model = self._load_model(checkpoint_path)
        print(f"[Predictor] Model loaded on {self.device}")

    def _load_model(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        model = build_model(self.model_type, pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device).eval()
        return model

    def _preprocess(self, image_path: str):
        """Returns (orig_tensor, ela_tensor, orig_np_rgb, ela_np_rgb)."""
        orig_np = np.array(Image.open(image_path).convert("RGB"))
        ela_np  = compute_ela(image_path)

        orig_t = self.transform(image=orig_np)["image"].unsqueeze(0).to(self.device)
        ela_t  = self.transform(image=ela_np)["image"].unsqueeze(0).to(self.device)

        return orig_t, ela_t, orig_np, ela_np

    @torch.no_grad()
    def predict(self, image_path: str) -> dict:
        """
        Run inference on a single image.

        Returns:
            dict with keys: label, label_str, confidence,
                            prob_authentic, prob_forged
        """
        orig_t, ela_t, orig_np, ela_np = self._preprocess(image_path)

        if self.model_type == "dual":
            logits = self.model(orig_t, ela_t)
        else:
            logits = self.model(ela_t)

        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred_label = int(probs[1] >= self.threshold)

        return {
            "image_path":     image_path,
            "label":          pred_label,
            "label_str":      self.LABELS[pred_label],
            "confidence":     float(probs[pred_label]),
            "prob_authentic": float(probs[0]),
            "prob_forged":    float(probs[1]),
        }

    def predict_batch(self, image_paths: list) -> list:
        """Run inference on a list of image paths."""
        return [self.predict(p) for p in image_paths]


# ─── Visualisation ───────────────────────────────────────────────────────────────

def visualize_prediction(image_path: str, result: dict,
                          output_path: str = None, show: bool = False):
    """
    Create a three-panel figure:
      [Original]  [ELA image]  [Result banner]
    """
    orig   = np.array(Image.open(image_path).convert("RGB"))
    ela    = compute_ela(image_path)
    overlay = overlay_ela_on_original(image_path, alpha=0.5)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    label   = result["label_str"]
    conf    = result["confidence"] * 100
    color   = "green" if label == "Authentic" else "red"

    fig = plt.figure(figsize=(15, 5), facecolor="#1a1a2e")
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.05)

    titles = ["Original Document", "ELA Analysis", "ELA Overlay"]
    images = [orig, ela, overlay]

    for i, (img, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(gs[i])
        ax.imshow(img)
        ax.set_title(title, color="white", fontsize=11, pad=8,
                     fontweight="bold")
        ax.axis("off")

    fig.suptitle(
        f"Verdict: {label}   |   Confidence: {conf:.1f}%",
        fontsize=16, color=color, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor="#1a1a2e")
    if show:
        plt.show()
    plt.close()


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    predictor = ForgeryPredictor(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        image_size=args.image_size,
        threshold=args.threshold,
    )

    # Collect image paths
    if args.image:
        image_paths = [args.image]
    elif args.image_dir:
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        image_paths = [
            os.path.join(args.image_dir, f)
            for f in os.listdir(args.image_dir)
            if f.lower().endswith(exts)
        ]
    else:
        raise ValueError("Provide --image or --image_dir")

    print(f"[Predict] Running inference on {len(image_paths)} image(s) …\n")

    results = []
    for path in image_paths:
        result = predictor.predict(path)
        results.append(result)

        print(f"  {os.path.basename(path):<40}  "
              f"{result['label_str']:<12}  "
              f"conf={result['confidence']:.3f}  "
              f"(auth={result['prob_authentic']:.3f}, forged={result['prob_forged']:.3f})")

        # Visualise
        out_name = os.path.splitext(os.path.basename(path))[0] + "_result.png"
        visualize_prediction(
            path, result,
            output_path=os.path.join(args.output_dir, out_name),
        )

    # Save JSON summary
    summary_path = os.path.join(args.output_dir, "predictions.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Predict] Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
