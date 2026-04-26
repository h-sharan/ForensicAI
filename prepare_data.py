"""
prepare_data.py - Download and prepare datasets for training

Supported datasets:
  - CASIA v2.0 (most commonly used for image forgery)
  - Custom document dataset preparation

Usage:
    python prepare_data.py --dataset casia --output data/raw
    python prepare_data.py --dataset custom --input_dir /path/to/images --output data/raw
"""

import os
import shutil
import argparse
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm


def create_folder_structure(base_dir: str):
    """Create required directory structure."""
    dirs = [
        os.path.join(base_dir, "authentic"),
        os.path.join(base_dir, "forged"),
        "data/processed/authentic",
        "data/processed/forged",
        "data/ela/authentic",
        "data/ela/forged",
        "checkpoints",
        "evaluation",
        "static/uploads",
        "static/reports",
        "logs"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"[SETUP] Folder structure created.")


def prepare_custom_dataset(input_dir: str, output_dir: str,
                            authentic_subfolder: str = "authentic",
                            forged_subfolder: str = "forged"):
    """
    Organize a custom dataset from input_dir into the required structure.
    
    Expects:
        input_dir/authentic/  <- real documents
        input_dir/forged/     <- tampered documents
    """
    create_folder_structure(output_dir)

    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    stats = {"authentic": 0, "forged": 0}

    for cls in ["authentic", "forged"]:
        src = os.path.join(input_dir, cls if cls == authentic_subfolder else forged_subfolder)
        dst = os.path.join(output_dir, cls)

        if not os.path.exists(src):
            print(f"[WARN] Skipping {src} (not found)")
            continue

        files = [f for f in os.listdir(src) if Path(f).suffix.lower() in exts]
        for fname in tqdm(files, desc=f"Copying {cls}"):
            shutil.copy2(os.path.join(src, fname), os.path.join(dst, fname))
            stats[cls] += 1

    print(f"\n[DATA] Prepared: {stats['authentic']} authentic, {stats['forged']} forged")
    return stats


def generate_synthetic_forgeries(authentic_dir: str, forged_dir: str,
                                   n_samples: int = 100):
    """
    Generate synthetic forged documents for testing/demo purposes.
    Applies copy-move, text addition, and splice operations.
    NOT for production training - use real labeled datasets.
    """
    os.makedirs(forged_dir, exist_ok=True)
    auth_files = [f for f in os.listdir(authentic_dir)
                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if not auth_files:
        print("[WARN] No authentic images found to generate forgeries from.")
        return

    print(f"[SYNTH] Generating {min(n_samples, len(auth_files))} synthetic forgeries...")

    for i, fname in enumerate(tqdm(auth_files[:n_samples])):
        src_path = os.path.join(authentic_dir, fname)
        img = cv2.imread(src_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        forgery_type = i % 3  # Rotate through forgery types

        if forgery_type == 0:
            # Copy-move: duplicate a patch to another location
            ry, rx = np.random.randint(0, h // 2), np.random.randint(0, w // 2)
            patch  = img[ry:ry+50, rx:rx+50].copy()
            dy, dx = np.random.randint(h // 2, h - 60), np.random.randint(w // 2, w - 60)
            img[dy:dy+50, dx:dx+50] = patch

        elif forgery_type == 1:
            # Brightness splice: alter a rectangular region
            ry, rx = np.random.randint(10, h // 3), np.random.randint(10, w // 3)
            rh, rw = np.random.randint(30, 80), np.random.randint(30, 80)
            region = img[ry:ry+rh, rx:rx+rw].astype(np.float32)
            factor = np.random.uniform(1.3, 1.8)
            img[ry:ry+rh, rx:rx+rw] = np.clip(region * factor, 0, 255).astype(np.uint8)

        elif forgery_type == 2:
            # Add random text overlay (simulates text insertion forgery)
            text   = str(np.random.randint(1000, 9999))
            pos    = (np.random.randint(10, w - 80), np.random.randint(20, h - 20))
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 2, cv2.LINE_AA)

        out_name = f"forged_{i:04d}_{fname}"
        cv2.imwrite(os.path.join(forged_dir, out_name), img)

    print(f"[SYNTH] Done. Forgeries saved to {forged_dir}")


def validate_dataset(data_dir: str):
    """Validate dataset integrity and report statistics."""
    print("\n" + "="*50)
    print("  DATASET VALIDATION")
    print("="*50)
    for cls in ["authentic", "forged"]:
        folder = os.path.join(data_dir, cls)
        if not os.path.exists(folder):
            print(f"  {cls}: MISSING")
            continue

        files = [f for f in os.listdir(folder)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        corrupt = 0
        sizes   = []

        for f in files:
            try:
                img = Image.open(os.path.join(folder, f))
                img.verify()
                sizes.append(os.path.getsize(os.path.join(folder, f)))
            except Exception:
                corrupt += 1

        print(f"  {cls.upper()}:")
        print(f"    Count:    {len(files)}")
        print(f"    Corrupt:  {corrupt}")
        if sizes:
            print(f"    Avg size: {np.mean(sizes)/1024:.1f} KB")

    print("="*50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for forgery detection")
    parser.add_argument("--dataset",  type=str, default="demo",
                        choices=["demo", "custom"])
    parser.add_argument("--input_dir",type=str, default=None,
                        help="Input directory for custom dataset")
    parser.add_argument("--output",   type=str, default="data/raw")
    parser.add_argument("--n_synth",  type=int, default=50,
                        help="Number of synthetic forgeries to generate (demo mode)")
    args = parser.parse_args()

    create_folder_structure(args.output)

    if args.dataset == "demo":
        print("[DEMO] Creating synthetic demo dataset...")
        # Create some blank authentic docs for demo
        auth_dir   = os.path.join(args.output, "authentic")
        forged_dir = os.path.join(args.output, "forged")

        # Generate blank authentic document images
        for i in range(args.n_synth):
            img = np.ones((400, 600, 3), dtype=np.uint8) * 250
            cv2.rectangle(img, (20, 20), (580, 380), (200, 200, 200), 2)
            cv2.putText(img, f"DOCUMENT {i:04d}", (160, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
            for j in range(5):
                y = 90 + j * 40
                cv2.line(img, (40, y), (560, y), (180, 180, 180), 1)
            cv2.imwrite(os.path.join(auth_dir, f"authentic_{i:04d}.jpg"), img)

        generate_synthetic_forgeries(auth_dir, forged_dir, n_samples=args.n_synth)
        validate_dataset(args.output)

    elif args.dataset == "custom":
        if not args.input_dir:
            print("[ERROR] --input_dir required for custom dataset")
        else:
            prepare_custom_dataset(args.input_dir, args.output)
            validate_dataset(args.output)
