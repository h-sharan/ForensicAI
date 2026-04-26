"""
generate_dataset.py
====================
Synthetic Document Dataset Generator for Forgery Detection.

Generates realistic-looking:
  - College Certificates
  - Mark Sheets
  - ID Cards
  - Bonafide Letters
  - Government Letters

Each document type is generated as:
  - AUTHENTIC version  → saved to data/raw/authentic/
  - FORGED  version    → saved to data/raw/forged/
      (with visible tampering: changed text, pasted stamps, altered numbers)

Usage:
    python generate_dataset.py --count 300
"""

import os
import random
import argparse
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np
import cv2

# ── Output folders ──────────────────────────────────────────────────
AUTHENTIC_DIR = "data/raw/authentic"
FORGED_DIR    = "data/raw/forged"
os.makedirs(AUTHENTIC_DIR, exist_ok=True)
os.makedirs(FORGED_DIR,    exist_ok=True)

# ── Fake data pools ─────────────────────────────────────────────────
FIRST_NAMES  = ["Arjun","Priya","Rahul","Sneha","Vikram","Ananya","Karthik","Divya",
                "Suresh","Meera","Rohan","Kavya","Aditya","Pooja","Nithya","Sanjay",
                "Lakshmi","Harish","Revathi","Deepak","Ishaan","Tanvi","Manish","Riya"]
LAST_NAMES   = ["Kumar","Sharma","Patel","Reddy","Nair","Iyer","Singh","Verma",
                "Krishnan","Mehta","Gupta","Rao","Pillai","Joshi","Bhat","Menon"]
UNIVERSITIES = ["Chennai University","Anna University","Bangalore University",
                "Pune University","Delhi University","Hyderabad University",
                "Madras Institute of Technology","VIT University"]
DEPARTMENTS  = ["Computer Science & Engineering","Electronics & Communication",
                "Mechanical Engineering","Civil Engineering","Information Technology",
                "Electrical Engineering","Biomedical Engineering"]
GRADES       = ["A+","A","A","B+","B+","B","A+"]
CGPA_VALS    = ["9.2","8.7","8.4","9.0","7.8","8.1","9.5","7.6","8.9"]
YEARS        = ["2020","2021","2022","2023","2024"]
REG_NUMS     = [f"20{random.randint(10,23)}{random.randint(1000,9999)}" for _ in range(50)]
SUBJECTS     = ["Mathematics","Physics","Chemistry","Programming","Data Structures",
                "Algorithms","Networks","Database","Operating Systems","AI & ML"]

# ── Color palette ────────────────────────────────────────────────────
NAVY   = (10,  36,  99)
BLUE   = (30,  80, 160)
GOLD   = (180,140,  20)
RED    = (180,  20,  20)
GREEN  = (20, 120,  40)
BLACK  = (10,  10,  10)
WHITE  = (255,255,255)
LGRAY  = (240,240,245)
DGRAY  = (80,  80,  80)
CREAM  = (255,252,240)

# ── Font helper ──────────────────────────────────────────────────────
def font(size, bold=False):
    """Return a PIL ImageFont — uses default if system fonts unavailable."""
    candidates = []
    if bold:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:/Windows/Fonts/arialbd.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:/Windows/Fonts/arial.ttf",
        ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()

def mono_font(size):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "C:/Windows/Fonts/cour.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return font(size)

# ── Drawing helpers ──────────────────────────────────────────────────
def draw_border(draw, img_w, img_h, color, thickness=6, margin=18):
    for t in range(thickness):
        m = margin + t
        draw.rectangle([m, m, img_w-m, img_h-m], outline=color)

def draw_header_band(draw, img_w, y0, y1, color):
    draw.rectangle([0, y0, img_w, y1], fill=color)

def draw_stamp(img, x, y, radius, color, text="VERIFIED", authentic=True):
    """Draw a circular stamp on the image."""
    draw = ImageDraw.Draw(img)
    # Outer ring
    draw.ellipse([x-radius, y-radius, x+radius, y+radius],
                 outline=color, width=3)
    draw.ellipse([x-radius+6, y-radius+6, x+radius-6, y+radius-6],
                 outline=color, width=2)
    # Inner text
    f = font(max(10, radius//3), bold=True)
    bbox = draw.textbbox((0,0), text, font=f)
    tw = bbox[2]-bbox[0]; th = bbox[3]-bbox[1]
    draw.text((x - tw//2, y - th//2), text, fill=color, font=f)
    # Star pattern
    for angle in range(0, 360, 45):
        rad = angle * 3.14159 / 180
        sx  = int(x + (radius-14) * np.cos(rad))
        sy  = int(y + (radius-14) * np.sin(rad))
        draw.ellipse([sx-2, sy-2, sx+2, sy+2], fill=color)

    if not authentic:
        # Forged stamps have jagged edges (simulated)
        for _ in range(8):
            ax = x + random.randint(-radius, radius)
            ay = y + random.randint(-radius, radius)
            draw.ellipse([ax-1, ay-1, ax+2, ay+2], fill=color)

def draw_signature(draw, x, y, w, h, color=BLACK, forged=False):
    """Draw a handwritten-style signature using bezier-like curves."""
    pts = []
    steps = 40 if not forged else 20
    for i in range(steps):
        px = x + int((w / steps) * i) + random.randint(-3, 3)
        py = y + h//2 + int(np.sin(i * 0.5) * h//3) + random.randint(-4, 4)
        pts.append((px, py))
    if len(pts) > 1:
        draw.line(pts, fill=color, width=2)
    # Forged signatures are too straight/uniform
    if forged:
        draw.line([(x, y+h//2), (x+w, y+h//2)], fill=color, width=2)

def add_paper_texture(img):
    """Add subtle paper texture and aging to the image."""
    arr  = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 3, arr.shape).astype(np.float32)
    arr  = np.clip(arr + noise, 0, 255).astype(np.uint8)
    result = Image.fromarray(arr)
    # Slight vignette
    vignette = Image.new("L", result.size, 255)
    vd = ImageDraw.Draw(vignette)
    w, h = result.size
    vd.ellipse([-w//3, -h//3, w+w//3, h+h//3], fill=255)
    result = Image.composite(result,
                              Image.new("RGB", result.size, CREAM),
                              vignette)
    return result

def add_forgery_artifacts(img):
    """Add visible forgery signs: blur patch, JPEG artifacts, clone marks."""
    arr = np.array(img)
    h, w = arr.shape[:2]

    # Randomly pick a forgery type
    ftype = random.randint(0, 2)

    if ftype == 0:
        # Copy-move: duplicate a region
        ry  = random.randint(h//4, h//2)
        rx  = random.randint(w//4, w//2)
        ph, pw = random.randint(30, 60), random.randint(60, 120)
        patch = arr[ry:ry+ph, rx:rx+pw].copy()
        ty  = random.randint(h//2, h-ph-10)
        tx  = random.randint(10, w-pw-10)
        arr[ty:ty+ph, tx:tx+pw] = patch

    elif ftype == 1:
        # Brightness splice on a text region
        ry  = random.randint(h//5, h//2)
        rh2 = random.randint(30, 70)
        rw2 = random.randint(80, 200)
        rx  = random.randint(10, max(11, w-rw2-10))
        region = arr[ry:ry+rh2, rx:rx+rw2].astype(np.float32)
        arr[ry:ry+rh2, rx:rx+rw2] = np.clip(region * 1.4, 0, 255).astype(np.uint8)

    else:
        # Gaussian blur on stamp/signature area (bottom 30%)
        ry  = int(h * 0.70)
        roi = arr[ry:, :]
        blurred = cv2.GaussianBlur(roi, (9, 9), 0)
        arr[ry:, :] = blurred

    return Image.fromarray(arr)


# ════════════════════════════════════════════════════════════════════
# DOCUMENT GENERATORS
# ════════════════════════════════════════════════════════════════════

# ── 1. College Certificate ───────────────────────────────────────────
def make_certificate(forged=False, idx=0):
    W, H = 900, 650
    img  = Image.new("RGB", (W, H), CREAM)
    draw = ImageDraw.Draw(img)

    name  = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    uni   = random.choice(UNIVERSITIES)
    dept  = random.choice(DEPARTMENTS)
    year  = random.choice(YEARS)
    grade = random.choice(GRADES)
    reg   = random.choice(REG_NUMS)

    if forged:
        # Change grade/year to suspicious values
        grade = random.choice(["A+","A+","A+"])
        year  = str(int(year) + random.choice([1, 2]))

    # Decorative border
    draw_border(draw, W, H, NAVY, thickness=4, margin=15)
    draw_border(draw, W, H, GOLD, thickness=2, margin=22)

    # Header band
    draw_header_band(draw, W, 0, 100, NAVY)
    draw.text((W//2, 30), uni.upper(), fill=GOLD,
              font=font(22, bold=True), anchor="mm")
    draw.text((W//2, 68), f"Department of {dept}", fill=WHITE,
              font=font(14), anchor="mm")

    # Certificate title
    draw.text((W//2, 135), "CERTIFICATE OF COMPLETION", fill=NAVY,
              font=font(26, bold=True), anchor="mm")
    draw.line([(W//4, 158), (3*W//4, 158)], fill=GOLD, width=2)

    # Body
    draw.text((W//2, 195), "This is to certify that", fill=DGRAY,
              font=font(16), anchor="mm")
    draw.text((W//2, 240), name, fill=NAVY, font=font(30, bold=True), anchor="mm")
    draw.line([(W//4, 265), (3*W//4, 265)], fill=NAVY, width=1)

    body_text = (f"has successfully completed the course requirements for the degree of\n"
                 f"Bachelor of Engineering in {dept}\n"
                 f"with Grade  {grade}  in the academic year {year}-{int(year)+1}")
    y = 285
    for line in body_text.split("\n"):
        draw.text((W//2, y), line, fill=BLACK, font=font(15), anchor="mm")
        y += 30

    draw.text((W//2, 390), f"Registration No: {reg}", fill=DGRAY,
              font=mono_font(13), anchor="mm")

    # Signature lines
    sig_y = 490
    for sx, label in [(200, "Principal"), (450, "HOD"), (700, "Examiner")]:
        draw.line([(sx-70, sig_y), (sx+70, sig_y)], fill=BLACK, width=1)
        draw_signature(draw, sx-60, sig_y-40, 120, 35, forged=forged)
        draw.text((sx, sig_y+10), label, fill=DGRAY, font=font(12), anchor="mm")

    # Stamps
    draw_stamp(img, 180, 560, 45, RED,  "OFFICIAL", authentic=not forged)
    draw_stamp(img, 720, 560, 45, BLUE, "VERIFIED", authentic=not forged)

    # Date
    draw.text((W//2, 620), f"Date: {random.randint(1,28):02d}/0{random.randint(1,9)}/{year}",
              fill=DGRAY, font=font(12), anchor="mm")

    img = add_paper_texture(img)
    if forged:
        img = add_forgery_artifacts(img)
    return img


# ── 2. Mark Sheet ────────────────────────────────────────────────────
def make_marksheet(forged=False, idx=0):
    W, H = 850, 720
    img  = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(img)

    name  = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    uni   = random.choice(UNIVERSITIES)
    dept  = random.choice(DEPARTMENTS)
    year  = random.choice(YEARS)
    sem   = random.choice(["I","II","III","IV","V","VI","VII","VIII"])
    reg   = random.choice(REG_NUMS)

    draw_border(draw, W, H, NAVY, thickness=5, margin=12)

    # Header
    draw_header_band(draw, W, 0, 90, NAVY)
    draw.text((W//2, 28), uni.upper(), fill=WHITE, font=font(20, bold=True), anchor="mm")
    draw.text((W//2, 62), "OFFICIAL MARK SHEET", fill=GOLD, font=font(16, bold=True), anchor="mm")

    # Student info
    draw.text((60, 110), f"Name         : {name}", fill=BLACK, font=font(14))
    draw.text((60, 132), f"Reg. No.     : {reg}", fill=BLACK, font=mono_font(14))
    draw.text((60, 154), f"Department   : {dept}", fill=BLACK, font=font(14))
    draw.text((500, 110), f"Semester : {sem}", fill=BLACK, font=font(14))
    draw.text((500, 132), f"Year     : {year}", fill=BLACK, font=font(14))
    draw.line([(40, 178), (W-40, 178)], fill=NAVY, width=2)

    # Table header
    row_y = 192
    cols  = [50, 380, 490, 590, 700]
    hdrs  = ["Subject", "Max Marks", "Marks Obtained", "Grade", "Result"]
    for col, hdr in zip(cols, hdrs):
        draw.rectangle([col-4, row_y, col+140, row_y+26], fill=NAVY)
        draw.text((col+4, row_y+4), hdr, fill=WHITE, font=font(12, bold=True))
    row_y += 30

    subjects = random.sample(SUBJECTS, 6)
    total_obtained = 0
    for i, subj in enumerate(subjects):
        max_m = 100
        if forged and i in [0, 2]:
            obtained = random.randint(90, 100)   # suspiciously high
        else:
            obtained = random.randint(55, 95)
        total_obtained += obtained
        grade = "O" if obtained>=90 else "A+" if obtained>=85 else "A" if obtained>=75 else "B+" if obtained>=65 else "B"
        result = "PASS"
        bg = LGRAY if i % 2 == 0 else WHITE
        draw.rectangle([40, row_y, W-40, row_y+26], fill=bg)
        draw.text((cols[0], row_y+4), subj[:30], fill=BLACK, font=font(12))
        draw.text((cols[1]+20, row_y+4), str(max_m), fill=BLACK, font=mono_font(12))
        draw.text((cols[2]+30, row_y+4), str(obtained), fill=BLACK, font=mono_font(12))
        draw.text((cols[3]+10, row_y+4), grade, fill=BLACK, font=font(12, bold=True))
        draw.text((cols[4]+10, row_y+4), result, fill=GREEN, font=font(12, bold=True))
        row_y += 28

    # Total
    draw.line([(40, row_y), (W-40, row_y)], fill=NAVY, width=2)
    cgpa = round(total_obtained / (len(subjects) * 10), 2)
    if forged:
        cgpa = round(cgpa + random.uniform(0.4, 0.8), 2)
        cgpa = min(cgpa, 10.0)
    draw.text((50, row_y+8), f"Total: {total_obtained}/{len(subjects)*100}", fill=BLACK, font=font(14, bold=True))
    draw.text((500, row_y+8), f"CGPA: {cgpa:.2f} / 10.00", fill=NAVY, font=font(14, bold=True))

    # Stamp + signature
    draw_stamp(img, 150, row_y+100, 45, RED, "RESULT", authentic=not forged)
    draw_stamp(img, W-150, row_y+100, 45, BLUE, "OFFICIAL", authentic=not forged)
    sig_x = W//2
    draw.line([(sig_x-70, row_y+135), (sig_x+70, row_y+135)], fill=BLACK, width=1)
    draw_signature(draw, sig_x-60, row_y+95, 120, 35, forged=forged)
    draw.text((sig_x, row_y+148), "Controller of Examinations", fill=DGRAY, font=font(11), anchor="mm")

    img = add_paper_texture(img)
    if forged:
        img = add_forgery_artifacts(img)
    return img


# ── 3. ID Card ───────────────────────────────────────────────────────
def make_id_card(forged=False, idx=0):
    W, H = 640, 400
    img  = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(img)

    name  = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    uni   = random.choice(UNIVERSITIES)
    dept  = random.choice(DEPARTMENTS)
    year  = random.choice(YEARS)
    reg   = random.choice(REG_NUMS)
    dob   = f"{random.randint(1,28):02d}/{random.randint(1,12):02d}/{random.randint(2000,2004)}"
    blood = random.choice(["A+","B+","O+","AB+","A-","B-"])

    # Sidebar
    draw.rectangle([0, 0, 180, H], fill=NAVY)
    draw.text((90, 30), uni[:12], fill=WHITE, font=font(11, bold=True), anchor="mm")

    # Photo placeholder
    draw.rectangle([30, 60, 150, 190], fill=LGRAY, outline=WHITE, width=2)
    draw.text((90, 125), "PHOTO", fill=DGRAY, font=font(14), anchor="mm")

    # QR placeholder
    draw.rectangle([50, 210, 130, 290], fill=LGRAY, outline=WHITE, width=1)
    draw.text((90, 250), "QR", fill=DGRAY, font=font(12), anchor="mm")
    draw.text((90, 310), blood, fill=RED, font=font(18, bold=True), anchor="mm")
    draw.text((90, 335), "Blood Group", fill=WHITE, font=font(10), anchor="mm")

    # Right panel
    draw.text((350, 30), "IDENTITY CARD", fill=NAVY, font=font(22, bold=True), anchor="mm")
    draw.line([(200, 55), (W-20, 55)], fill=NAVY, width=2)

    fields = [
        ("Name",        name),
        ("Reg. No.",    reg),
        ("Department",  dept[:28]),
        ("Year",        year),
        ("DOB",         dob),
        ("Valid Until", f"31/05/{int(year)+4}"),
    ]
    if forged:
        # Alter validity year
        fields[5] = ("Valid Until", f"31/05/{int(year)+8}")

    fy = 75
    for label, value in fields:
        draw.text((205, fy), f"{label}:", fill=DGRAY, font=font(12, bold=True))
        draw.text((330, fy), value, fill=BLACK, font=font(13))
        draw.line([(200, fy+20), (W-20, fy+20)], fill=LGRAY, width=1)
        fy += 36

    # Signature
    draw.text((500, 330), "Signature", fill=DGRAY, font=font(11), anchor="mm")
    draw_signature(draw, 440, 295, 120, 30, forged=forged)
    draw.line([(440, 322), (560, 322)], fill=DGRAY, width=1)

    # Stamp
    draw_stamp(img, 570, 340, 35, RED, "VALID", authentic=not forged)

    img = add_paper_texture(img)
    if forged:
        img = add_forgery_artifacts(img)
    return img


# ── 4. Bonafide Letter ───────────────────────────────────────────────
def make_bonafide(forged=False, idx=0):
    W, H = 800, 1000
    img  = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(img)

    name  = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    uni   = random.choice(UNIVERSITIES)
    dept  = random.choice(DEPARTMENTS)
    year  = random.choice(YEARS)
    reg   = random.choice(REG_NUMS)
    purpose = random.choice(["Bank Account Opening","Passport Application",
                              "Scholarship Application","Bus Pass","Visa Application"])

    draw_border(draw, W, H, NAVY, thickness=3, margin=20)

    # Logo area
    draw.ellipse([350, 28, 450, 108], outline=NAVY, width=3)
    draw.text((400, 68), "🎓", font=font(30), anchor="mm")

    draw.text((W//2, 125), uni.upper(), fill=NAVY, font=font(18, bold=True), anchor="mm")
    draw.text((W//2, 152), f"Department of {dept}", fill=DGRAY, font=font(13), anchor="mm")
    draw.line([(50, 170), (W-50, 170)], fill=NAVY, width=2)

    draw.text((W//2, 192), "BONAFIDE CERTIFICATE", fill=NAVY,
              font=font(20, bold=True), anchor="mm")
    draw.line([(200, 210), (W-200, 210)], fill=GOLD, width=2)

    ref_no = f"Ref: {uni[:3].upper()}/{dept[:3].upper()}/{random.randint(100,999)}/{year}"
    draw.text((60, 228), ref_no, fill=DGRAY, font=mono_font(11))
    date_str = f"Date: {random.randint(1,28):02d}/0{random.randint(1,9)}/{year}"
    draw.text((W-60, 228), date_str, fill=DGRAY, font=font(11), anchor="ra")

    para = (f"This is to certify that {name}, bearing Registration Number {reg}, "
            f"is a bonafide student of this institution, currently enrolled in "
            f"the {year} batch of {dept}.\n\n"
            f"This certificate is issued for the purpose of {purpose} and is valid "
            f"for the current academic year.\n\n"
            f"The student bears good conduct and is regular in attendance.")

    y = 280
    for paragraph in para.split("\n\n"):
        wrapped = textwrap.wrap(paragraph, width=75)
        for line in wrapped:
            draw.text((60, y), line, fill=BLACK, font=font(14))
            y += 24
        y += 14

    # Official fields box
    box_y = y + 20
    draw.rectangle([50, box_y, W-50, box_y+110], outline=NAVY, width=1)
    draw.text((80, box_y+12),  f"Student Name  : {name}", fill=BLACK, font=font(13))
    draw.text((80, box_y+34),  f"Reg. Number   : {reg}",  fill=BLACK, font=mono_font(13))
    draw.text((80, box_y+56),  f"Department    : {dept}", fill=BLACK, font=font(13))
    draw.text((80, box_y+78),  f"Academic Year : {year}-{int(year)+1}", fill=BLACK, font=font(13))
    draw.text((80, box_y+100), f"Purpose       : {purpose}", fill=BLACK, font=font(13))

    sig_y = box_y + 160
    for sx, label in [(200, "Class Advisor"), (600, "HOD / Principal")]:
        draw.line([(sx-80, sig_y), (sx+80, sig_y)], fill=BLACK, width=1)
        draw_signature(draw, sx-70, sig_y-45, 140, 40, forged=forged)
        draw.text((sx, sig_y+12), label, fill=DGRAY, font=font(12), anchor="mm")

    draw_stamp(img, 400, sig_y+20, 50, RED, "OFFICIAL", authentic=not forged)
    draw.text((W//2, H-35), f"{uni}  ·  {year}", fill=DGRAY, font=font(10), anchor="mm")

    img = add_paper_texture(img)
    if forged:
        img = add_forgery_artifacts(img)
    return img


# ── 5. Government Letter ─────────────────────────────────────────────
def make_govt_letter(forged=False, idx=0):
    W, H = 800, 1050
    img  = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(img)

    name     = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    dept_name = random.choice(["Ministry of Education","Department of Revenue",
                                "Municipal Corporation","State Government Office",
                                "District Collector Office"])
    year     = random.choice(YEARS)
    ref_no   = f"GOV/{random.randint(1000,9999)}/{year}"
    subject  = random.choice(["Income Certificate","Residence Certificate",
                               "Caste Certificate","Character Certificate"])

    draw_border(draw, W, H, DGRAY, thickness=2, margin=18)
    # Header stripe
    draw.rectangle([0, 0, W, 110], fill=(220, 230, 245))
    draw.text((W//2, 35), "GOVERNMENT OF INDIA", fill=NAVY, font=font(20, bold=True), anchor="mm")
    draw.text((W//2, 65), dept_name.upper(), fill=NAVY, font=font(15, bold=True), anchor="mm")
    draw.text((W//2, 90), "Official Document", fill=DGRAY, font=font(12), anchor="mm")
    draw.line([(30, 112), (W-30, 112)], fill=NAVY, width=2)

    draw.text((60, 128), f"Ref: {ref_no}", fill=DGRAY, font=mono_font(11))
    draw.text((W-60, 128), f"Date: {random.randint(1,28):02d}/0{random.randint(1,9)}/{year}",
              fill=DGRAY, font=font(11), anchor="ra")

    draw.text((W//2, 165), f"SUBJECT: {subject.upper()}", fill=NAVY,
              font=font(16, bold=True), anchor="mm")
    draw.line([(60, 182), (W-60, 182)], fill=NAVY, width=1)

    body = (f"This is to certify that {name}, resident of the jurisdiction under "
            f"this office, has been verified through official records.\n\n"
            f"The applicant has submitted the required documents and the same "
            f"have been verified by the concerned authorities. This certificate "
            f"is issued as per the standard norms.\n\n"
            f"This certificate is valid for a period of ONE YEAR from the date "
            f"of issue and is meant for official use only.")

    y = 205
    for paragraph in body.split("\n\n"):
        wrapped = textwrap.wrap(paragraph, width=72)
        for line in wrapped:
            draw.text((60, y), line, fill=BLACK, font=font(14))
            y += 26
        y += 12

    # Declaration box
    box_y = y + 15
    draw.rectangle([50, box_y, W-50, box_y+90], fill=(245,245,255), outline=NAVY, width=1)
    draw.text((80, box_y+10), f"Name    : {name}", fill=BLACK, font=font(13))
    draw.text((80, box_y+32), f"Ref No. : {ref_no}", fill=BLACK, font=mono_font(13))
    draw.text((80, box_y+54), f"Year    : {year}", fill=BLACK, font=font(13))
    draw.text((80, box_y+72), f"Subject : {subject}", fill=BLACK, font=font(13))

    sig_y = box_y + 155
    draw.text((W-180, sig_y-50), "Authorized Signatory", fill=DGRAY, font=font(12))
    draw_signature(draw, W-280, sig_y-90, 160, 40, forged=forged)
    draw.line([(W-290, sig_y-45), (W-80, sig_y-45)], fill=BLACK, width=1)

    draw_stamp(img, 170, sig_y, 55, RED,  "GOVT.", authentic=not forged)
    draw_stamp(img, W-170, sig_y, 55, BLUE, "SEALED", authentic=not forged)

    # Footer
    draw.rectangle([0, H-50, W, H], fill=(220,230,245))
    draw.text((W//2, H-25), f"{dept_name}  ·  Government of India  ·  {year}",
              fill=NAVY, font=font(11), anchor="mm")

    img = add_paper_texture(img)
    if forged:
        img = add_forgery_artifacts(img)
    return img


# ════════════════════════════════════════════════════════════════════
# MAIN GENERATOR
# ════════════════════════════════════════════════════════════════════

GENERATORS = [
    ("certificate", make_certificate),
    ("marksheet",   make_marksheet),
    ("idcard",      make_id_card),
    ("bonafide",    make_bonafide),
    ("govtletter",  make_govt_letter),
]

def generate_dataset(total_per_class=300):
    print(f"\n{'='*60}")
    print(f"  Synthetic Document Dataset Generator")
    print(f"  Generating {total_per_class} authentic + {total_per_class} forged")
    print(f"={'='*59}\n")

    per_type = max(1, total_per_class // len(GENERATORS))

    auth_count, forge_count = 0, 0

    for doc_name, gen_fn in GENERATORS:
        print(f"  Generating {doc_name}s ...")
        for i in range(per_type):
            try:
                # Authentic
                img = gen_fn(forged=False, idx=i)
                path = os.path.join(AUTHENTIC_DIR, f"{doc_name}_auth_{i:04d}.jpg")
                img.save(path, "JPEG", quality=92)
                auth_count += 1

                # Forged
                img = gen_fn(forged=True, idx=i)
                path = os.path.join(FORGED_DIR, f"{doc_name}_forged_{i:04d}.jpg")
                img.save(path, "JPEG", quality=88)
                forge_count += 1

                if (i+1) % 20 == 0:
                    print(f"    {doc_name}: {i+1}/{per_type} done")
            except Exception as e:
                print(f"    [WARN] {doc_name} #{i}: {e}")

    print(f"\n{'='*60}")
    print(f"  ✅ DONE!")
    print(f"  Authentic : {auth_count} images → data/raw/authentic/")
    print(f"  Forged    : {forge_count} images → data/raw/forged/")
    print(f"{'='*60}\n")
    print("  Next step:")
    print("  python train.py --data_dir data/raw --model efficientnet --epochs 30\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=300,
                        help="Total images per class (authentic/forged)")
    args = parser.parse_args()
    generate_dataset(total_per_class=args.count)
