import os, json, uuid, traceback
import cv2
import numpy as np

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

ALLOWED = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB


def to_python(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [to_python(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    else:
        return obj


def allowed(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    if not file or not allowed(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    uid = str(uuid.uuid4())[:10]
    ext = file.filename.rsplit('.', 1)[1].lower()

    save_path = os.path.join(UPLOAD_FOLDER, f"{uid}.{ext}")
    file.save(save_path)

    try:
        if ext == 'pdf':
            try:
                import fitz
                doc = fitz.open(save_path)
                page = doc[0]
                pix = page.get_pixmap(dpi=150)
                img_path = os.path.join(UPLOAD_FOLDER, f"{uid}.png")
                pix.save(img_path)
                doc.close()
            except ImportError:
                return jsonify({"error": "PDF support requires PyMuPDF: pip install pymupdf"}), 500
        else:
            img_path = save_path

        img = cv2.imread(img_path)
        if img is None:
            return jsonify({"error": "Could not read image"}), 500
        img = img.astype(np.uint8)
        cv2.imwrite(img_path, img)

        from utils.forensic_engine import full_forensic_analysis
        report = full_forensic_analysis(img_path)

        def serialize_check(c):
            return to_python({
                "name": c.name,
                "status": c.status,
                "score": round(float(c.score), 1),
                "detail": c.detail,
                "lines": c.lines,
                "regions": c.regions
            })

        return jsonify(to_python({
            "verdict": report.verdict,
            "confidence": float(report.confidence),
            "overall_score": float(report.overall_score),
            "checks": [serialize_check(c) for c in report.checks],
            "ela_b64": report.ela_b64,
            "noise_b64": report.noise_b64,
            "annotated_b64": report.annotated_b64,
            "summary_lines": report.summary_lines,
            "processing_time": float(report.processing_time),
        }))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/static/<path:f>')
def static_files(f):
    return send_from_directory('static', f)


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
