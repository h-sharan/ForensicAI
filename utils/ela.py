"""
Error Level Analysis (ELA) Utility
====================================
ELA reveals regions saved at a different compression level —
a strong indicator of digital tampering or splicing.
"""

import os
import tempfile
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import cv2


def compute_ela(image_path: str, quality: int = 90, scale: int = 15) -> np.ndarray:
    """
    Compute ELA for a given image.

    Steps:
      1. Re-save the image at a known JPEG quality.
      2. Subtract the re-saved image from the original.
      3. Amplify the difference for visibility.

    Args:
        image_path : Path to the input image.
        quality    : JPEG re-save quality (default 90).
        scale      : Amplification factor (default 15).

    Returns:
        ela_array  : uint8 numpy array (H x W x 3).
    """
    original = Image.open(image_path).convert("RGB")

    # Use a proper temp file that works on Windows, Mac, and Linux
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(tmp_fd)

    try:
        original.save(tmp_path, "JPEG", quality=quality)
        recompressed = Image.open(tmp_path).copy()
    finally:
        os.remove(tmp_path)

    diff = ImageChops.difference(original, recompressed)
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    factor = 255.0 / max_diff * scale / 10
    enhancer = ImageEnhance.Brightness(diff)
    ela_image = enhancer.enhance(factor)

    ela_array = np.array(ela_image, dtype=np.uint8)
    return ela_array


def compute_ela_batch(image_paths: list, quality: int = 90, scale: int = 15,
                      output_dir: str = None) -> list:
    """Compute ELA for a list of images, optionally saving results."""
    results = []
    for path in image_paths:
        try:
            ela = compute_ela(path, quality, scale)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                base = os.path.splitext(os.path.basename(path))[0]
                save_path = os.path.join(output_dir, f"{base}_ela.png")
                Image.fromarray(ela).save(save_path)
            results.append((path, ela))
        except Exception as e:
            print(f"[ELA] Failed on {path}: {e}")
    return results


def ela_heatmap(image_path: str, quality: int = 90) -> np.ndarray:
    """Generate a grayscale ELA energy heatmap."""
    ela = compute_ela(image_path, quality)
    heatmap = np.sqrt(np.sum(ela.astype(np.float32) ** 2, axis=2))
    heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
    return heatmap


def overlay_ela_on_original(image_path: str, alpha: float = 0.5,
                             quality: int = 90, scale: int = 15) -> np.ndarray:
    """Blend the ELA result on top of the original image for visual inspection."""
    original = cv2.imread(image_path)
    ela = compute_ela(image_path, quality, scale)
    ela_bgr = cv2.cvtColor(ela, cv2.COLOR_RGB2BGR)
    ela_resized = cv2.resize(ela_bgr, (original.shape[1], original.shape[0]))
    blended = cv2.addWeighted(original, 1 - alpha, ela_resized, alpha, 0)
    return blended
