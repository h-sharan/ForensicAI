# 🔍 ForensicAI — AI-Powered Document Forgery Detection
## Complete Setup & Execution Guide for VS Code

---

## 📁 Project Structure

```
ForensicAI_Complete/
│
├── app.py                  ← 🚀 MAIN FILE — Run this to start the web app
├── generate_dataset.py     ← Step 1: Generate synthetic document dataset
├── train.py                ← Step 2: Train the deep learning model
├── evaluate.py             ← Step 3: Evaluate model performance
├── predict.py              ← Predict on a single image from terminal
├── prepare_data.py         ← Organize custom datasets
├── requirements.txt        ← All Python dependencies
│
├── utils/
│   ├── forensic_engine.py  ← All 12 forensic analysis functions
│   ├── dataset.py          ← Dataset loader + augmentation
│   ├── ela.py              ← Error Level Analysis module
│   └── __init__.py
│
├── models/
│   ├── model.py            ← EfficientNet, ResNet50, UNet architectures
│   └── __init__.py
│
├── templates/
│   └── index.html          ← Web UI (served automatically by Flask)
│
├── static/
│   ├── uploads/            ← Uploaded documents (auto-created)
│   └── reports/            ← Analysis reports  (auto-created)
│
├── checkpoints/
│   └── best_model.pth      ← Saved model weights (created after training)
│
├── data/
│   ├── raw/
│   │   ├── authentic/      ← Genuine document images
│   │   └── forged/         ← Tampered document images
│   ├── processed/          ← Resized images (auto-created)
│   └── ela/                ← ELA images (auto-created)
│
├── evaluation/             ← Plots & metrics (created after evaluate.py)
└── logs/                   ← Training logs
```

---

## ⚙️ STEP 1 — Install Required Software

### 1.1 Install Python 3.10
- Download: https://www.python.org/downloads/
- ✅ During install → check "Add Python to PATH"
- Verify: open terminal → type `python --version`

### 1.2 Install VS Code
- Download: https://code.visualstudio.com/

### 1.3 Install Tesseract OCR
- Windows: https://github.com/UB-Mannheim/tesseract/wiki
  - Download the .exe installer → install it
  - Default path: C:\Program Files\Tesseract-OCR\tesseract.exe
- Ubuntu/Linux: sudo apt install tesseract-ocr
- Mac: brew install tesseract

---

## ⚙️ STEP 2 — Open Project in VS Code

```
1. Open VS Code
2. Click: File → Open Folder
3. Select the "ForensicAI_Complete" folder
4. Press Ctrl+` to open the Terminal panel
```

---

## ⚙️ STEP 3 — Install Python Extensions

```
1. Press Ctrl+Shift+X
2. Search "Python"  → Install (by Microsoft)
3. Search "Pylance" → Install (by Microsoft)
```

---

## ⚙️ STEP 4 — Create Virtual Environment

Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

Mac / Linux:
```bash
python -m venv venv
source venv/bin/activate
```

You will see (venv) appear in the terminal.

Then:
```
Press Ctrl+Shift+P
Type: Python: Select Interpreter
Choose: ./venv/Scripts/python.exe
```

---

## ⚙️ STEP 5 — Install All Dependencies

```bash
pip install -r requirements.txt
```

Takes 5-10 minutes.

---

## ⚙️ STEP 6 — Configure Tesseract (Windows Only)

Open utils/forensic_engine.py → find `import pytesseract` → add below it:

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

Save with Ctrl+S.

---

## 🗃️ STEP 7 — Generate Dataset

```bash
python generate_dataset.py --count 300
```

Generates 300 authentic + 300 forged documents:
- College Certificates
- Mark Sheets
- ID Cards
- Bonafide Letters
- Government Letters

Saved to: data/raw/authentic/ and data/raw/forged/

---

## 🏋️ STEP 8 — Train the Model

```bash
python train.py --data_dir data/raw --model efficientnet --epochs 30 --batch_size 16
```

Model saved to: checkpoints/best_model.pth

---

## 📊 STEP 9 — Evaluate the Model

```bash
python evaluate.py --model_path checkpoints/best_model.pth --data_dir data/raw
```

Saves plots to: evaluation/ folder

---

## 🌐 STEP 10 — Run the Web App

```bash
python app.py
```

Open browser → http://localhost:5000

---

## 🔧 Troubleshooting

| Error | Solution |
|---|---|
| activate fails | Use PowerShell → run: Set-ExecutionPolicy RemoteSigned -Scope CurrentUser |
| No module named flask | Ensure (venv) is active → re-run pip install -r requirements.txt |
| tesseract not installed | Complete Step 6 + reinstall Tesseract |
| Address already in use | Change port=5000 to port=5001 in app.py |
| No module named torch | pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu |
| CUDA out of memory | Add --batch_size 8 to train command |
| PDF not working | pip install pymupdf |
| Port 5000 blocked Mac | Change to port=5001 in app.py (Mac uses 5000 for AirPlay) |
