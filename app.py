from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
MODEL_PATH = r'C:\chronic disease lungs\lungs_disease_app\model\inception_lung_disease_model.keras'

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = 299

# Ensure 'static/' folder exists
STATIC_DIR = 'static'
os.makedirs(STATIC_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(STATIC_DIR, filename)
        file.save(filepath)

        # Load and preprocess image
        img = Image.open(filepath).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)[0][0]
        result = "Lung Disease Detected" if prediction > 0.5 else "No Lung Disease Detected"

        return render_template('index.html', prediction=result, img_path=filepath)

    return "Prediction failed", 500

if __name__ == '__main__':
    app.run(debug=True)
