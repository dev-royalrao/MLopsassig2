# app/app.py
"""
Simple Flask app to upload an image, preprocess to 64x64 grayscale,
flatten and predict class using savedmodel.pth.
"""
import os
from flask import Flask, request, render_template, redirect, url_for, flash
from PIL import Image
import numpy as np
import joblib

APP_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(os.path.dirname(APP_DIR), 'saved', 'savedmodel.pth')

app = Flask(__name__)
app.secret_key = "change_this_secret_for_prod"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run training and place savedmodel.pth in saved/")
    return joblib.load(MODEL_PATH)

model = None

def preprocess_image_to_vector(pil_image):
    # convert to grayscale, resize to 64x64, flatten and scale to [0,1]
    im = pil_image.convert('L').resize((64,64))
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr.reshape(1, -1)

@app.before_first_request
def _load():
    global model
    model = load_model()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        flash("No image uploaded")
        return redirect(url_for('index'))
    file = request.files['image']
    if file.filename == '':
        flash("Empty filename")
        return redirect(url_for('index'))
    try:
        pil_image = Image.open(file.stream)
    except Exception as e:
        flash(f"Cannot open image: {e}")
        return redirect(url_for('index'))
    x = preprocess_image_to_vector(pil_image)
    pred = model.predict(x)[0]
    return render_template('index.html', prediction=int(pred))

if __name__ == "__main__":
    # For local testing only. In Docker use gunicorn or prod server
    app.run(host='0.0.0.0', port=5000, debug=False)
