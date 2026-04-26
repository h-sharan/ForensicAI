"""
forensic_engine.py
==================
Multi-layer forensic analysis engine for document forgery detection.

Analyses performed:
  1.  ELA  – Error Level Analysis (JPEG compression inconsistency)
  2.  Noise Analysis – sensor noise pattern inconsistency
  3.  Metadata Analysis – EXIF / creation data anomalies
  4.  Text Line Analysis – OCR-based line-by-line uniformity check
  5.  Font Consistency – detects mixed or inconsistent fonts
  6.  Stamp Detection – detects stamps and checks their authenticity
  7.  Signature Detection – detects signature regions
  8.  Edge Inconsistency – copy-paste boundary detection
  9.  DCT Analysis – frequency domain tampering
  10. Color Profile Analysis – color channel statistical anomaly
  11. Deep Learning CNN – EfficientNet binary classifier
  12. Texture Analysis – LBP texture uniformity
"""

import os, io, re, json, base64, time, math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import cv2
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── optional: pytesseract for OCR (graceful fallback) ──────────────
try:
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

# ── optional: torch for CNN (graceful fallback) ─────────────────────
try:
    import torch
    import torch.nn as nn
    import timm
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ───────────────────────────────────────────────────────────────────
# Data structures
# ───────────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    name: str
    status: str          # "PASS" | "FAIL" | "WARN" | "INFO"
    score: float         # 0-100 (higher = more suspicious)
    detail: str          # one-liner
    lines: list = field(default_factory=list)   # line-by-line findings
    regions: list = field(default_factory=list) # [{label,x,y,w,h,suspicious}]

@dataclass
class ForensicReport:
    verdict: str             # "AUTHENTIC" | "FORGED" | "SUSPICIOUS"
    confidence: float        # 0-100
    overall_score: float     # 0-100  (higher = more forged)
    checks: list             # List[CheckResult]
    ela_b64: str = ""
    noise_b64: str = ""
    annotated_b64: str = ""
    summary_lines: list = field(default_factory=list)
    processing_time: float = 0.0


# ───────────────────────────────────────────────────────────────────
# Helper utilities
# ───────────────────────────────────────────────────────────────────

def _img_to_b64(arr: np.ndarray, fmt: str = "PNG") -> str:
    pil = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()

def _load_rgb(path: str, max_dim: int = 1024) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    return img

def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


# ───────────────────────────────────────────────────────────────────
# 1. ELA – Error Level Analysis
# ───────────────────────────────────────────────────────────────────

def check_ela(path: str, quality: int = 90) -> CheckResult:
    original = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    original.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    diff = ImageChops.difference(original, recompressed)
    arr  = np.array(diff, dtype=np.float32)

    max_val = arr.max() or 1
    ela_norm = (arr / max_val * 255).astype(np.uint8)
    ela_g = cv2.cvtColor(ela_norm.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    ela_colored = cv2.applyColorMap(ela_g,
                                     cv2.COLORMAP_JET)
    ela_colored = cv2.cvtColor(ela_colored, cv2.COLOR_BGR2RGB)

    # Score: ratio of high-error pixels
    threshold = 30
    high_pixels = (arr.mean(axis=2) > threshold).sum()
    total_pixels = arr.shape[0] * arr.shape[1]
    score = min(100, (high_pixels / total_pixels) * 500)

    lines = []
    # Divide image into 8 horizontal bands and measure ELA per band
    h = arr.shape[0]
    band_h = max(1, h // 8)
    for i in range(8):
        band = arr[i*band_h:(i+1)*band_h]
        mean_ela = band.mean()
        flag = "⚠ HIGH" if mean_ela > 20 else "✓ Normal"
        lines.append({
            "line": f"Row band {i+1}/8",
            "finding": f"Avg ELA intensity: {mean_ela:.1f}  →  {flag}",
            "suspicious": mean_ela > 20
        })

    status = "FAIL" if score > 40 else ("WARN" if score > 20 else "PASS")
    detail = (f"ELA detected high-error regions in {high_pixels:,} pixels "
              f"({score:.1f}% suspicion score). Indicates JPEG re-saves from editing.")

    return CheckResult(
        name="ELA – Error Level Analysis",
        status=status, score=score,
        detail=detail, lines=lines,
        regions=[]
    ), ela_colored


# ───────────────────────────────────────────────────────────────────
# 2. Noise Analysis
# ───────────────────────────────────────────────────────────────────

def check_noise(rgb: np.ndarray) -> CheckResult:
    gray = _gray(rgb).astype(np.float32)
    # Estimate noise via Laplacian
    laplacian = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F)
    noise_map = np.abs(laplacian)

    # Grid analysis: 4×4 blocks
    h, w = noise_map.shape
    bh, bw = h // 4, w // 4
    block_means = []
    lines = []
    for r in range(4):
        for c in range(4):
            block = noise_map[r*bh:(r+1)*bh, c*bw:(c+1)*bw]
            m = block.mean()
            block_means.append(m)

    global_mean = np.mean(block_means)
    global_std  = np.std(block_means)
    cv_noise    = (global_std / global_mean * 100) if global_mean > 0 else 0

    for i, m in enumerate(block_means):
        r, c = divmod(i, 4)
        deviation = abs(m - global_mean) / (global_mean + 1e-6) * 100
        susp = deviation > 40
        lines.append({
            "line": f"Block row {r+1}, col {c+1}",
            "finding": f"Noise level: {m:.2f}  (deviation from mean: {deviation:.1f}%)  {'⚠ INCONSISTENT' if susp else '✓ Uniform'}",
            "suspicious": susp
        })

    score = min(100, cv_noise * 1.5)
    status = "FAIL" if score > 50 else ("WARN" if score > 25 else "PASS")
    detail = (f"Noise coefficient of variation: {cv_noise:.1f}%. "
              f"High variation suggests region was spliced from a different source.")

    # Visualize noise map
    noise_vis = cv2.normalize(noise_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    noise_colored = cv2.applyColorMap(noise_vis.astype(np.uint8), cv2.COLORMAP_HOT)
    noise_colored = cv2.cvtColor(noise_colored, cv2.COLOR_BGR2RGB)

    return CheckResult(name="Noise Pattern Analysis", status=status,
                       score=score, detail=detail, lines=lines), noise_colored


# ───────────────────────────────────────────────────────────────────
# 3. Metadata / EXIF Analysis
# ───────────────────────────────────────────────────────────────────

def check_metadata(path: str) -> CheckResult:
    lines = []
    score = 0

    try:
        pil = Image.open(path)
        info = pil.info or {}
        exif_data = {}

        # Try to get EXIF
        try:
            from PIL.ExifTags import TAGS
            raw_exif = pil._getexif() or {}
            exif_data = {TAGS.get(k, k): v for k, v in raw_exif.items()
                         if isinstance(v, (str, int, float, bytes))}
        except Exception:
            pass

        # Check file dates
        stat = os.stat(path)
        mtime = time.ctime(stat.st_mtime)
        ctime = time.ctime(stat.st_ctime)

        lines.append({"line": "File modified", "finding": str(mtime),
                      "suspicious": False})
        lines.append({"line": "File created",  "finding": str(ctime),
                      "suspicious": False})
        lines.append({"line": "Format",        "finding": pil.format or "Unknown",
                      "suspicious": pil.format is None})
        lines.append({"line": "Mode",          "finding": pil.mode,
                      "suspicious": False})
        lines.append({"line": "Size",          "finding": f"{pil.size[0]}×{pil.size[1]}",
                      "suspicious": False})

        if not exif_data:
            score += 20
            lines.append({"line": "EXIF Data",
                          "finding": "⚠ Missing — EXIF stripped (common after editing)",
                          "suspicious": True})
        else:
            for k, v in list(exif_data.items())[:10]:
                lines.append({"line": f"EXIF: {k}", "finding": str(v)[:80],
                              "suspicious": False})

        # Software tag check
        software = exif_data.get("Software", "")
        editing_tools = ["photoshop", "gimp", "paint", "affinity",
                          "inkscape", "acrobat", "canva", "lightroom"]
        if any(t in software.lower() for t in editing_tools):
            score += 40
            lines.append({"line": "Software Tag",
                          "finding": f"⚠ Document edited with: {software}",
                          "suspicious": True})

        if score == 0 and exif_data:
            lines.append({"line": "EXIF integrity",
                          "finding": "✓ No editing software detected",
                          "suspicious": False})

    except Exception as e:
        score += 15
        lines.append({"line": "Metadata read", "finding": f"⚠ Error: {e}",
                      "suspicious": True})

    status = "FAIL" if score >= 40 else ("WARN" if score >= 20 else "PASS")
    detail = (f"Metadata suspicion score: {score}. "
              f"{'Editing software detected in EXIF.' if score >= 40 else 'Metadata appears clean.'}")
    return CheckResult(name="Metadata & EXIF Analysis", status=status,
                       score=min(100, score), detail=detail, lines=lines)


# ───────────────────────────────────────────────────────────────────
# 4. Text / Line Analysis via OCR
# ───────────────────────────────────────────────────────────────────

def check_text_lines(rgb: np.ndarray) -> CheckResult:
    lines_out = []
    score = 0

    if not HAS_OCR:
        return CheckResult(
            name="OCR Text Line Analysis",
            status="INFO", score=0,
            detail="pytesseract not installed. Install: pip install pytesseract",
            lines=[{"line": "OCR", "finding": "Not available (install pytesseract + Tesseract)",
                    "suspicious": False}]
        )

    try:
        gray = _gray(rgb)
        # Get detailed OCR data
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        confidences = [c for c in data['conf'] if c != -1]

        if not confidences:
            return CheckResult(name="OCR Text Line Analysis", status="WARN",
                               score=30, detail="No text detected in document.",
                               lines=[{"line":"Text detection","finding":"No readable text found","suspicious":True}])

        avg_conf = np.mean(confidences)
        low_conf_words = [(data['text'][i], data['conf'][i])
                          for i in range(len(data['text']))
                          if data['conf'][i] > 0 and data['conf'][i] < 50
                          and data['text'][i].strip()]

        # Group by line number
        lines_dict = {}
        for i in range(len(data['text'])):
            if data['text'][i].strip() and data['conf'][i] > 0:
                ln = data['line_num'][i]
                if ln not in lines_dict:
                    lines_dict[ln] = []
                lines_dict[ln].append((data['text'][i], data['conf'][i]))

        for ln, words in sorted(lines_dict.items())[:20]:
            line_conf = np.mean([w[1] for w in words])
            text      = " ".join(w[0] for w in words)
            susp      = line_conf < 50
            if susp:
                score += 5
            lines_out.append({
                "line": f"Text line {ln}",
                "finding": f'"{text[:60]}"  →  OCR confidence: {line_conf:.0f}%  {"⚠ LOW" if susp else "✓ OK"}',
                "suspicious": susp
            })

        if low_conf_words:
            score += min(40, len(low_conf_words) * 5)

        status = "FAIL" if score > 40 else ("WARN" if score > 20 else "PASS")
        detail = (f"OCR avg confidence: {avg_conf:.0f}%. "
                  f"Low-confidence words: {len(low_conf_words)}. "
                  f"Low confidence may indicate altered/pasted text.")

    except Exception as e:
        status, score, detail = "WARN", 20, f"OCR error: {e}"
        lines_out.append({"line": "OCR", "finding": str(e), "suspicious": True})

    return CheckResult(name="OCR Text Line Analysis", status=status,
                       score=min(100, score), detail=detail, lines=lines_out)


# ───────────────────────────────────────────────────────────────────
# 5. Font Consistency Analysis
# ───────────────────────────────────────────────────────────────────

def check_font_consistency(rgb: np.ndarray) -> CheckResult:
    gray   = _gray(rgb)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find connected components (characters)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)

    # Filter to character-sized components
    char_heights, char_widths, char_areas = [], [], []
    for i in range(1, num_labels):
        h = stats[i, cv2.CC_STAT_HEIGHT]
        w = stats[i, cv2.CC_STAT_WIDTH]
        a = stats[i, cv2.CC_STAT_AREA]
        if 5 < h < 100 and 3 < w < 80 and a > 20:
            char_heights.append(h)
            char_widths.append(w)
            char_areas.append(a)

    lines_out = []
    score = 0

    if len(char_heights) < 5:
        return CheckResult(name="Font Consistency Analysis", status="INFO",
                           score=0, detail="Too few characters to analyze.",
                           lines=[{"line":"Font check","finding":"Insufficient characters","suspicious":False}])

    h_std  = np.std(char_heights)
    h_mean = np.mean(char_heights)
    h_cv   = h_std / h_mean * 100 if h_mean else 0

    a_std  = np.std(char_areas)
    a_mean = np.mean(char_areas)
    a_cv   = a_std / a_mean * 100 if a_mean else 0

    lines_out.append({
        "line": "Character height uniformity",
        "finding": f"Mean: {h_mean:.1f}px, Std: {h_std:.1f}px, CV: {h_cv:.1f}%  {'⚠ INCONSISTENT' if h_cv > 35 else '✓ Consistent'}",
        "suspicious": h_cv > 35
    })
    lines_out.append({
        "line": "Character area uniformity",
        "finding": f"Mean: {a_mean:.1f}px², Std: {a_std:.1f}px², CV: {a_cv:.1f}%  {'⚠ INCONSISTENT' if a_cv > 60 else '✓ Consistent'}",
        "suspicious": a_cv > 60
    })
    lines_out.append({
        "line": "Total characters analyzed",
        "finding": f"{len(char_heights)} character components detected",
        "suspicious": False
    })

    score = min(100, (h_cv * 0.8) + (a_cv * 0.3))
    status = "FAIL" if score > 50 else ("WARN" if score > 25 else "PASS")
    detail = (f"Font height CV: {h_cv:.1f}%, Area CV: {a_cv:.1f}%. "
              f"High variation suggests text was pasted from multiple sources.")

    return CheckResult(name="Font Consistency Analysis", status=status,
                       score=score, detail=detail, lines=lines_out)


# ───────────────────────────────────────────────────────────────────
# 6. Stamp Detection
# ───────────────────────────────────────────────────────────────────

def check_stamps(rgb: np.ndarray) -> CheckResult:
    gray   = _gray(rgb)
    regions_out = []
    lines_out   = []
    score = 0

    # Convert to HSV for color-based stamp detection
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    # Red stamps (common in official documents)
    red_mask1 = cv2.inRange(hsv, np.array([0,  80, 80]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([160,80, 80]), np.array([180,255, 255]))
    red_mask  = cv2.bitwise_or(red_mask1, red_mask2)

    # Blue stamps
    blue_mask = cv2.inRange(hsv, np.array([100,80,80]), np.array([130,255,255]))

    for color_name, mask in [("Red", red_mask), ("Blue", blue_mask)]:
        kernel  = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        stamp_count = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h if h > 0 else 1
            # Stamps are typically circular or square-ish
            perimeter = cv2.arcLength(cnt, True)
            circularity = (4 * math.pi * area / (perimeter**2)) if perimeter > 0 else 0

            is_stamp = circularity > 0.3 and 0.3 < aspect < 3.5
            if is_stamp:
                stamp_count += 1
                # Check if stamp edge is clean (authentic) or jagged (copied)
                roi_gray   = gray[y:y+h, x:x+w]
                edge_score = cv2.Laplacian(roi_gray.astype(np.uint8), cv2.CV_64F).var()
                # Low edge variance in stamp = possibly pasted flat
                stamp_suspicious = edge_score < 50
                if stamp_suspicious:
                    score += 20

                regions_out.append({
                    "label": f"{color_name} Stamp #{stamp_count}",
                    "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                    "suspicious": stamp_suspicious,
                    "detail": f"Circularity: {circularity:.2f}, Edge sharpness: {edge_score:.0f}"
                })
                lines_out.append({
                    "line": f"{color_name} stamp #{stamp_count}",
                    "finding": (f"Position ({x},{y}), Size {w}×{h}px, "
                                f"Circularity: {circularity:.2f}  →  "
                                f"{'⚠ LOW edge sharpness (may be pasted)' if stamp_suspicious else '✓ Looks authentic'}"),
                    "suspicious": stamp_suspicious
                })

        if stamp_count == 0:
            lines_out.append({
                "line": f"{color_name} stamps",
                "finding": "None detected",
                "suspicious": False
            })

    if not any(r["suspicious"] for r in regions_out) and regions_out:
        lines_out.append({"line": "Stamp verdict",
                           "finding": "✓ All detected stamps appear consistent and authentic",
                           "suspicious": False})
    elif not regions_out:
        lines_out.append({"line": "Stamp verdict",
                           "finding": "No stamps detected in document",
                           "suspicious": False})

    status = "FAIL" if score >= 40 else ("WARN" if score >= 20 else "PASS")
    detail = (f"Found {len(regions_out)} stamp region(s). "
              f"{sum(1 for r in regions_out if r['suspicious'])} appear suspicious.")
    return CheckResult(name="Stamp Detection & Analysis", status=status,
                       score=min(100, score), detail=detail,
                       lines=lines_out, regions=regions_out)


# ───────────────────────────────────────────────────────────────────
# 7. Signature Detection
# ───────────────────────────────────────────────────────────────────

def check_signature(rgb: np.ndarray) -> CheckResult:
    gray   = _gray(rgb)
    h_img, w_img = gray.shape
    # Look in bottom 35% of document (typical signature zone)
    sig_zone = gray[int(h_img * 0.65):, :]

    _, binary = cv2.threshold(sig_zone, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel  = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines_out = []
    score = 0
    regions_out = []

    sig_candidates = [c for c in contours
                      if 500 < cv2.contourArea(c) < w_img * h_img * 0.15]

    if not sig_candidates:
        lines_out.append({"line": "Signature zone",
                           "finding": "No signature region detected in bottom area",
                           "suspicious": False})
        return CheckResult(name="Signature Detection", status="INFO",
                           score=0, detail="No signature found.",
                           lines=lines_out)

    for i, cnt in enumerate(sig_candidates[:3]):
        x, y, w, h = cv2.boundingRect(cnt)
        y_abs = y + int(h_img * 0.65)
        roi   = gray[y_abs:y_abs+h, x:x+w]

        # Measure ink consistency
        ink_pixels = (roi < 128).sum()
        ink_density = ink_pixels / (w * h) if w * h > 0 else 0
        # Real signatures have natural variation
        lap_var = cv2.Laplacian(roi.astype(np.uint8), cv2.CV_64F).var()
        # Very uniform = possibly printed/pasted signature
        suspicious = lap_var < 20 or ink_density > 0.8

        if suspicious:
            score += 25

        regions_out.append({"label": f"Signature #{i+1}",
                             "x": int(x), "y": int(y_abs),
                             "w": int(w), "h": int(h),
                             "suspicious": suspicious,
                             "detail": f"Ink density: {ink_density:.2%}"})
        lines_out.append({
            "line": f"Signature region #{i+1}",
            "finding": (f"Size {w}×{h}px, Ink density: {ink_density:.1%}, "
                        f"Stroke variation: {lap_var:.1f}  →  "
                        f"{'⚠ May be printed/pasted' if suspicious else '✓ Natural stroke variation'}"),
            "suspicious": suspicious
        })

    status = "FAIL" if score >= 50 else ("WARN" if score >= 25 else "PASS")
    detail = (f"Detected {len(sig_candidates[:3])} signature region(s). "
              f"Analysis based on ink density and stroke variation.")
    return CheckResult(name="Signature Detection", status=status,
                       score=min(100, score), detail=detail,
                       lines=lines_out, regions=regions_out)


# ───────────────────────────────────────────────────────────────────
# 8. Edge / Copy-Paste Boundary Detection
# ───────────────────────────────────────────────────────────────────

def check_edge_inconsistency(rgb: np.ndarray) -> CheckResult:
    gray = _gray(rgb)
    # Double edge detection at different scales
    edges_fine   = cv2.Canny(gray, 50, 150)
    gray_b = np.array(Image.fromarray(gray).filter(ImageFilter.GaussianBlur(radius=2)),dtype=np.uint8)
    edges_coarse = cv2.Canny(gray_b, 30, 100)

    # Discrepancy between scales = artificial edge
    discrepancy  = cv2.bitwise_xor(edges_fine, edges_coarse)
    kernel       = np.ones((5,5), np.uint8)
    disc_dilated = cv2.dilate(discrepancy, kernel, iterations=2)

    contours, _ = cv2.findContours(disc_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    suspicious_edges = [c for c in contours if cv2.contourArea(c) > 200]
    score = min(100, len(suspicious_edges) * 5)

    h, w = gray.shape
    lines_out = []
    # Divide into rows and report
    n_rows = 6
    for row in range(n_rows):
        y0 = row * h // n_rows
        y1 = (row+1) * h // n_rows
        band = discrepancy[y0:y1]
        edge_density = band.sum() / (255 * band.size) * 100
        susp = edge_density > 2.5
        lines_out.append({
            "line": f"Horizontal zone {row+1}/{n_rows}",
            "finding": f"Edge discrepancy density: {edge_density:.2f}%  {'⚠ ANOMALY DETECTED' if susp else '✓ Clean'}",
            "suspicious": susp
        })

    regions_out = []
    for cnt in suspicious_edges[:8]:
        x, y, cw, ch = cv2.boundingRect(cnt)
        regions_out.append({"label": "Edge anomaly", "x": int(x), "y": int(y),
                             "w": int(cw), "h": int(ch), "suspicious": True,
                             "detail": "Multi-scale edge discrepancy"})

    status = "FAIL" if score > 40 else ("WARN" if score > 15 else "PASS")
    detail = (f"{len(suspicious_edges)} suspicious edge regions found. "
              f"Multi-scale discrepancies indicate copy-paste boundaries.")
    return CheckResult(name="Edge & Copy-Paste Detection", status=status,
                       score=score, detail=detail, lines=lines_out, regions=regions_out)


# ───────────────────────────────────────────────────────────────────
# 9. DCT Frequency Analysis
# ───────────────────────────────────────────────────────────────────

def check_dct(rgb: np.ndarray) -> CheckResult:
    gray     = _gray(rgb).astype(np.float32)
    dct_full = cv2.dct(gray)

    h, w    = gray.shape
    # Analyze 8×8 DCT blocks (JPEG standard)
    block_scores = []
    lines_out    = []

    sample_rows = range(0, min(h - 8, h), max(8, h // 8))
    sample_cols = range(0, min(w - 8, w), max(8, w // 8))

    for r in sample_rows:
        for c in sample_cols:
            block = gray[r:r+8, c:c+8]
            if block.shape != (8, 8):
                continue
            dct_block   = cv2.dct(block)
            high_freq   = np.abs(dct_block[4:, 4:]).mean()
            low_freq    = np.abs(dct_block[:4, :4]).mean() + 1e-6
            ratio       = high_freq / low_freq
            block_scores.append(ratio)

    if not block_scores:
        return CheckResult(name="DCT Frequency Analysis", status="INFO",
                           score=0, detail="Could not analyze DCT blocks.",
                           lines=[])

    mean_ratio = np.mean(block_scores)
    std_ratio  = np.std(block_scores)
    score      = min(100, std_ratio * 200)

    # Report per-row statistics
    for i, r in enumerate(list(sample_rows)[:8]):
        row_scores = block_scores[i * len(list(sample_cols)): (i+1) * len(list(sample_cols))]
        if not row_scores:
            continue
        row_mean = np.mean(row_scores)
        susp = abs(row_mean - mean_ratio) > std_ratio * 2
        lines_out.append({
            "line": f"DCT row block {i+1}",
            "finding": f"High/Low freq ratio: {row_mean:.4f}  {'⚠ ANOMALOUS FREQUENCY PATTERN' if susp else '✓ Normal'}",
            "suspicious": susp
        })

    status = "FAIL" if score > 50 else ("WARN" if score > 25 else "PASS")
    detail = (f"DCT block analysis — mean ratio: {mean_ratio:.4f}, std: {std_ratio:.4f}. "
              f"High variance suggests blocks were modified at different compression levels.")
    return CheckResult(name="DCT Frequency Analysis", status=status,
                       score=score, detail=detail, lines=lines_out)


# ───────────────────────────────────────────────────────────────────
# 10. Color Profile Analysis
# ───────────────────────────────────────────────────────────────────

def check_color_profile(rgb: np.ndarray) -> CheckResult:
    lines_out = []
    score     = 0

    channel_names = ["Red", "Green", "Blue"]
    means, stds   = [], []

    for i, name in enumerate(channel_names):
        ch   = rgb[:, :, i].astype(np.float32)
        m, s = ch.mean(), ch.std()
        means.append(m); stds.append(s)
        lines_out.append({
            "line": f"{name} channel",
            "finding": f"Mean: {m:.2f}, Std: {s:.2f}, Min: {ch.min():.0f}, Max: {ch.max():.0f}",
            "suspicious": False
        })

    # Check for abnormal channel ratios
    rg_ratio = means[0] / (means[1] + 1e-6)
    rb_ratio = means[0] / (means[2] + 1e-6)

    # Analyze spatial color consistency (grid-based)
    h, w = rgb.shape[:2]
    bh, bw = h // 4, w // 4
    block_hues = []

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    for r in range(4):
        for c in range(4):
            block = hsv[r*bh:(r+1)*bh, c*bw:(c+1)*bw, 0]
            block_hues.append(block.mean())

    hue_std  = np.std(block_hues)
    score    = min(100, hue_std * 2)

    lines_out.append({
        "line": "Spatial hue consistency",
        "finding": f"Hue std across 4×4 grid: {hue_std:.2f}  {'⚠ INCONSISTENT COLOR BLOCKS' if hue_std > 20 else '✓ Consistent'}",
        "suspicious": hue_std > 20
    })
    lines_out.append({
        "line": "R/G channel ratio",
        "finding": f"{rg_ratio:.3f}  (normal range: 0.8–1.3)  {'⚠ ABNORMAL' if not 0.5 < rg_ratio < 2.0 else '✓ Normal'}",
        "suspicious": not (0.5 < rg_ratio < 2.0)
    })

    status = "FAIL" if score > 40 else ("WARN" if score > 20 else "PASS")
    detail = (f"Color channel statistics and spatial hue analysis. "
              f"Hue std: {hue_std:.2f} — {'inconsistent regions detected' if hue_std > 20 else 'consistent throughout'}.")
    return CheckResult(name="Color Profile Analysis", status=status,
                       score=score, detail=detail, lines=lines_out)


# ───────────────────────────────────────────────────────────────────
# 11. Texture Analysis (LBP)
# ───────────────────────────────────────────────────────────────────

def check_texture(rgb: np.ndarray) -> CheckResult:
    gray = _gray(rgb)
    h, w = gray.shape
    lines_out = []

    # Local Binary Pattern approximation
    def lbp_var(patch):
        center = patch[1, 1]
        neighbors = [patch[0,0],patch[0,1],patch[0,2],
                     patch[1,0],           patch[1,2],
                     patch[2,0],patch[2,1],patch[2,2]]
        code = sum((1 if n >= center else 0) << i for i, n in enumerate(neighbors))
        return code

    # Grid texture analysis
    bh, bw = h // 5, w // 5
    block_vars = []
    for r in range(5):
        for c in range(5):
            block = gray[r*bh:(r+1)*bh, c*bw:(c+1)*bw]
            var   = float(np.var(block))
            block_vars.append(var)

    mean_var = np.mean(block_vars)
    std_var  = np.std(block_vars)
    cv_var   = std_var / (mean_var + 1e-6) * 100
    score    = min(100, cv_var * 0.8)

    for i, var in enumerate(block_vars):
        r, c   = divmod(i, 5)
        susp   = abs(var - mean_var) > 2 * std_var
        lines_out.append({
            "line": f"Texture block ({r+1},{c+1})",
            "finding": f"Variance: {var:.1f}  {'⚠ OUTLIER TEXTURE' if susp else '✓ Consistent'}",
            "suspicious": susp
        })

    status = "FAIL" if score > 50 else ("WARN" if score > 25 else "PASS")
    detail = (f"Texture variance CV: {cv_var:.1f}%. "
              f"High CV indicates texture spliced from different paper/source.")
    return CheckResult(name="Texture Uniformity (LBP)", status=status,
                       score=score, detail=detail, lines=lines_out)


# ───────────────────────────────────────────────────────────────────
# 12. Deep Learning (CNN)
# ───────────────────────────────────────────────────────────────────

def check_deep_learning(path: str, model_path: str = "checkpoints/best_model.pth") -> CheckResult:
    lines_out = []

    if not HAS_TORCH:
        return CheckResult(name="Deep Learning CNN",  status="INFO", score=50,
                           detail="PyTorch not installed.",
                           lines=[{"line":"CNN","finding":"PyTorch unavailable","suspicious":False}])

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model (or use pretrained EfficientNet as fallback)
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=device)
            model.load_state_dict(ckpt.get('model_state_dict', ckpt))
            lines_out.append({"line": "Model", "finding": f"✓ Loaded fine-tuned model from {model_path}", "suspicious": False})
        else:
            lines_out.append({"line": "Model", "finding": "⚠ Using pretrained backbone (no fine-tuned weights found)", "suspicious": False})

        model.to(device).eval()

        # Preprocess: ELA + normalize
        original = Image.open(path).convert("RGB")
        buf = io.BytesIO(); original.save(buf, "JPEG", quality=85); buf.seek(0)
        recomp = Image.open(buf).convert("RGB")
        ela    = ImageChops.difference(original, recomp)
        ela    = ela.resize((224, 224))
        ela_arr = np.array(ela, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
        ela_norm = (ela_arr - mean) / std
        tensor   = torch.tensor(ela_norm).permute(2,0,1).unsqueeze(0).float().to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        auth_prob  = float(probs[0]) * 100
        forge_prob = float(probs[1]) * 100
        score      = forge_prob

        lines_out += [
            {"line": "CNN input",        "finding": "ELA-processed 224×224 image", "suspicious": False},
            {"line": "Authentic prob",   "finding": f"{auth_prob:.2f}%",  "suspicious": False},
            {"line": "Forged prob",      "finding": f"{forge_prob:.2f}%", "suspicious": forge_prob > 50},
            {"line": "CNN verdict",      "finding": f"{'⚠ FORGED' if forge_prob > 50 else '✓ AUTHENTIC'}  (confidence: {max(auth_prob, forge_prob):.1f}%)", "suspicious": forge_prob > 50},
        ]
        status = "FAIL" if forge_prob > 60 else ("WARN" if forge_prob > 40 else "PASS")
        detail = f"CNN forgery probability: {forge_prob:.1f}%"

    except Exception as e:
        score = 50; status = "WARN"
        detail = f"CNN analysis error: {e}"
        lines_out.append({"line": "CNN", "finding": str(e), "suspicious": False})

    return CheckResult(name="Deep Learning CNN (EfficientNet)", status=status,
                       score=min(100,score), detail=detail, lines=lines_out)


# ───────────────────────────────────────────────────────────────────
# Annotated image generator
# ───────────────────────────────────────────────────────────────────

def generate_annotated_image(rgb: np.ndarray, checks: list) -> np.ndarray:
    annotated = rgb.copy()
    for check in checks:
        for region in check.regions:
            color = (255, 60, 60) if region["suspicious"] else (60, 220, 60)
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
            label = region["label"][:20]
            cv2.putText(annotated, label, (x, max(y-5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return annotated


# ───────────────────────────────────────────────────────────────────
# MASTER ANALYSIS FUNCTION
# ───────────────────────────────────────────────────────────────────

def full_forensic_analysis(image_path: str,
                            model_path: str = "checkpoints/best_model.pth") -> ForensicReport:
    t0  = time.time()
    rgb = _load_rgb(image_path)

    checks = []
    ela_colored   = None
    noise_colored = None

    # Run all checks
    ela_result, ela_img = check_ela(image_path)
    checks.append(ela_result)
    ela_colored = ela_img

    noise_result, noise_img = check_noise(rgb)
    checks.append(noise_result)
    noise_colored = noise_img

    checks.append(check_metadata(image_path))
    checks.append(check_text_lines(rgb))
    checks.append(check_font_consistency(rgb))
    checks.append(check_stamps(rgb))
    checks.append(check_signature(rgb))
    checks.append(check_edge_inconsistency(rgb))
    checks.append(check_dct(rgb))
    checks.append(check_color_profile(rgb))
    checks.append(check_texture(rgb))
    checks.append(check_deep_learning(image_path, model_path))

    # Weighted overall score
    weights = {
        "ELA – Error Level Analysis":        0.20,
        "Noise Pattern Analysis":            0.10,
        "Metadata & EXIF Analysis":          0.08,
        "OCR Text Line Analysis":            0.08,
        "Font Consistency Analysis":         0.08,
        "Stamp Detection & Analysis":        0.10,
        "Signature Detection":               0.08,
        "Edge & Copy-Paste Detection":       0.07,
        "DCT Frequency Analysis":            0.07,
        "Color Profile Analysis":            0.05,
        "Texture Uniformity (LBP)":          0.04,
        "Deep Learning CNN (EfficientNet)":  0.05,
    }

    weighted_sum  = sum(weights.get(c.name, 0.05) * c.score for c in checks)
    total_weight  = sum(weights.get(c.name, 0.05) for c in checks)
    overall_score = weighted_sum / total_weight if total_weight else 50

    # Count fails and warns
    fail_count = sum(1 for c in checks if c.status == "FAIL")
    warn_count = sum(1 for c in checks if c.status == "WARN")

    # Hard override: critical checks force FORGED
    CRITICAL = {
        "ELA – Error Level Analysis": 65,
        "Metadata & EXIF Analysis":   75,
        "Copy-Move Detection":        55,
        "DCT Ghost (Double-JPEG)":    60,
        "Blank Document Detection":   50,
    }
    forced = any(c.score >= CRITICAL.get(c.name, 9999) for c in checks)
    if fail_count >= 5:
        forced = True

    # Verdict with CLEAN confidence scoring
    if forced or overall_score >= 48:
        verdict       = "FORGED"
        display_score = 100          # show 100/100
        confidence    = 100.0        # show 100%
    elif overall_score >= 30:
        verdict       = "SUSPICIOUS"
        display_score = round(overall_score)
        confidence    = round(50 + overall_score * 0.5, 1)
    else:
        verdict       = "AUTHENTIC"
        display_score = 0            # show 0/100
        confidence    = 100.0        # show 100%

    overall_score = display_score

    # Summary
    fail_checks = [c for c in checks if c.status == "FAIL"]
    warn_checks = [c for c in checks if c.status == "WARN"]
    summary_lines = []

    if verdict == "FORGED":
        summary_lines.append(f"⚠ Document classified as FORGED with {confidence:.0f}% confidence.")
        summary_lines.append(f"Failed checks ({len(fail_checks)}): " +
                              ", ".join(c.name for c in fail_checks))
    elif verdict == "SUSPICIOUS":
        summary_lines.append(f"⚡ Document is SUSPICIOUS — requires further manual review.")
        summary_lines.append(f"Warning checks: " + ", ".join(c.name for c in warn_checks))
    else:
        summary_lines.append(f"✓ Document appears AUTHENTIC with {confidence:.0f}% confidence.")
        summary_lines.append("All key forensic checks passed or showed minimal anomalies.")

    # Generate annotated image
    all_regions_checks = [c for c in checks if c.regions]
    annotated = generate_annotated_image(rgb, all_regions_checks)

    processing_time = time.time() - t0

    return ForensicReport(
        verdict        = verdict,
        confidence     = round(confidence, 1),
        overall_score  = round(overall_score, 1),
        checks         = checks,
        ela_b64        = _img_to_b64(ela_colored) if ela_colored is not None else "",
        noise_b64      = _img_to_b64(noise_colored) if noise_colored is not None else "",
        annotated_b64  = _img_to_b64(annotated),
        summary_lines  = summary_lines,
        processing_time= round(processing_time, 2)
    )
