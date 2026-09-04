"""
THERMAL IMAGE BASED FEVER & INFECTION DETECTION
------------------------------------------------
Flask Backend (Chapter 4 - Software Description / Chapter 5 - Modules)

Routes:
    GET  /            -> upload page
    POST /predict      -> runs preprocessing + CNN classification, shows result
    GET  /history       -> shows log of past screenings (Alert & Monitoring Module)

Run:
    python app.py
Then open:
    http://127.0.0.1:5000
"""

import os
import csv
import uuid
from datetime import datetime

import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

# ----------------------------------------------------------------------
# APP CONFIG
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "model", "thermal_fever_cnn.h5")
LOG_FILE = os.path.join(BASE_DIR, "alert_log.csv")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IMG_SIZE = (128, 128)
FEVER_THRESHOLD = 0.5  # sigmoid probability threshold for "fever" class

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB
app.secret_key = "thermal-fever-detection-secret-key"

# ----------------------------------------------------------------------
# LOAD CNN MODEL (Chapter 5.2 - CNN-Based Feature Extraction Module)
# ----------------------------------------------------------------------
MODEL = None
MODEL_LOADED = False

try:
    import tensorflow as tf
    if os.path.exists(MODEL_PATH):
        MODEL = tf.keras.models.load_model(MODEL_PATH)
        MODEL_LOADED = True
        print("[INFO] Trained CNN model loaded successfully.")
    else:
        print("[WARNING] No trained model found at:", MODEL_PATH)
        print("[WARNING] Run model/train_model.py after adding a dataset.")
        print("[WARNING] Falling back to a heuristic thermal-intensity demo mode.")
except Exception as e:  # pragma: no cover
    print("[WARNING] TensorFlow/model could not be loaded:", e)
    print("[WARNING] Falling back to a heuristic thermal-intensity demo mode.")


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    """
    Image Preprocessing Module (Chapter 5.2):
    - reads the thermal image
    - resizes to the CNN input size
    - normalizes pixel values
    - reduces noise with a light Gaussian blur
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read the uploaded image.")

    img = cv2.GaussianBlur(img, (3, 3), 0)          # noise reduction
    img_resized = cv2.resize(img, IMG_SIZE)           # standard input size
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype("float32") / 255.0       # normalization
    return img_norm, img_rgb


def heuristic_prediction(img_rgb):
    """
    Fallback classifier used ONLY when no trained CNN model is present.
    Estimates "heat" from warm color channels (thermal palettes typically
    render higher temperature as red/yellow/white). This lets the app run
    end-to-end for demo purposes before a real model has been trained.
    NOT a medical-grade measurement.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype("float32")
    r_channel = img_rgb[:, :, 0].astype("float32")

    # Combine overall brightness with red-channel dominance as a rough
    # proxy for "hot" regions in common thermal color palettes.
    hot_score = 0.5 * (gray.mean() / 255.0) + 0.5 * (r_channel.mean() / 255.0)
    probability = float(np.clip(hot_score, 0.0, 1.0))
    return probability


def predict_fever(image_path):
    img_norm, img_rgb = preprocess_image(image_path)

    if MODEL_LOADED:
        batch = np.expand_dims(img_norm, axis=0)
        probability = float(MODEL.predict(batch, verbose=0)[0][0])
        # Model trained with classes=["fever", "normal"] alphabetically ->
        # index 0 = fever, so class-1 output represents "normal" probability.
        # We convert so probability represents "fever" likelihood:
        fever_probability = 1.0 - probability
    else:
        fever_probability = heuristic_prediction(img_rgb)

    label = "Elevated Temperature (Possible Fever)" if fever_probability >= FEVER_THRESHOLD else "Normal Temperature"
    is_fever = fever_probability >= FEVER_THRESHOLD
    confidence = fever_probability if is_fever else (1.0 - fever_probability)

    return {
        "label": label,
        "is_fever": is_fever,
        "confidence": round(confidence * 100, 2),
        "fever_probability": round(fever_probability * 100, 2),
        "model_mode": "CNN Model" if MODEL_LOADED else "Demo Heuristic Mode (no trained model found)",
    }


def log_result(filename, result):
    """Alert & Monitoring Module (Chapter 5.2): keeps a record of screenings."""
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "filename", "result", "confidence_percent", "mode"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            filename,
            result["label"],
            result["confidence"],
            result["model_mode"],
        ])


def read_log(limit=25):
    rows = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    return list(reversed(rows))[:limit]


# ----------------------------------------------------------------------
# ROUTES
# ----------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", model_loaded=MODEL_LOADED)


@app.route("/predict", methods=["POST"])
def predict():
    if "thermal_image" not in request.files:
        flash("No file part in the request.")
        return redirect(url_for("index"))

    file = request.files["thermal_image"]

    if file.filename == "":
        flash("No file selected. Please choose a thermal image.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload a JPG or PNG thermal image.")
        return redirect(url_for("index"))

    unique_name = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)

    try:
        result = predict_fever(save_path)
    except Exception as e:
        flash(f"Error processing image: {e}")
        return redirect(url_for("index"))

    log_result(unique_name, result)

    return render_template(
        "result.html",
        result=result,
        image_url=url_for("static", filename=f"uploads/{unique_name}"),
    )


@app.route("/history", methods=["GET"])
def history():
    rows = read_log()
    return render_template("history.html", rows=rows)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
