# Thermal Image Based Fever & Infection Detection

Non-contact fever screening system using infrared thermal imaging and a
Convolutional Neural Network (CNN), served through a Flask web application
with an HTML/CSS frontend — matching the architecture described in the
project report (Chapters 4–7).

## 1. Project Structure

```
thermal_fever_detection/
├── app.py                     # Flask backend (routes, preprocessing, prediction)
├── requirements.txt           # Python dependencies
├── model/
│   ├── train_model.py         # CNN architecture + training script
│   └── thermal_fever_cnn.h5   # trained model (created after training)
├── dataset/
│   ├── train/normal/          # put normal-temperature thermal images here
│   ├── train/fever/           # put elevated-temperature thermal images here
│   ├── val/normal/
│   └── val/fever/
├── templates/                 # HTML pages (Jinja2)
│   ├── base.html
│   ├── index.html
│   ├── result.html
│   └── history.html
├── static/
│   ├── style.css
│   └── uploads/                # uploaded images are saved here
└── alert_log.csv               # auto-created screening log (Alert & Monitoring Module)
```

## 2. Setup

```bash
cd thermal_fever_detection
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. (Optional but recommended) Train the CNN model

The app will run out-of-the-box in a **demo heuristic mode** even without a
trained model (so you can test the full pipeline immediately), but for real
classification accuracy you should train the CNN on an actual thermal image
dataset such as:

- Public thermal face datasets (e.g., Terravic Facial IR / IRIS thermal
  face databases, or your own captured images from a FLIR / infrared
  camera).

Arrange images like this:

```
dataset/train/normal/*.jpg
dataset/train/fever/*.jpg
dataset/val/normal/*.jpg
dataset/val/fever/*.jpg
```

Then run:

```bash
python model/train_model.py
```

This trains the CNN (Chapter 5.2 CNN-Based Feature Extraction &
Classification Module), saves the best model to
`model/thermal_fever_cnn.h5`, and writes accuracy/loss curves to
`model/training_history.png`.

## 4. Run the web application

```bash
python app.py
```

Open your browser at: **http://127.0.0.1:5000**

- Upload a facial thermal image (JPG/PNG).
- The system preprocesses it, runs it through the CNN, and returns
  **Normal** or **Elevated Temperature (Possible Fever)** with a
  confidence score.
- Every screening is appended to `alert_log.csv` and viewable on the
  **History** page (Alert & Monitoring Module).

## 5. Notes

- `MAX_CONTENT_LENGTH` limits uploads to 8 MB; adjust in `app.py` if needed.
- The CNN input size is 128×128 RGB; change `IMG_SIZE` consistently in
  both `app.py` and `model/train_model.py` if you retrain with a
  different resolution.
- This is a **first-level, non-diagnostic screening tool** — it does not
  replace clinical thermometry or a medical diagnosis (see Chapter 9,
  Conclusion & Scope of the Project in the report).
