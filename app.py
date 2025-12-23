from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model('plant_disease_model.keras')

# THESE NAMES MUST MATCH YOUR JAVASCRIPT diseaseDB KEYS EXACTLY
CLASS_NAMES = [
    "Tomato - Healthy", "Tomato - Early Blight",
    "Potato - Healthy", "Potato - Late Blight",
    "Corn - Healthy", "Corn - Northern Leaf Blight",
    "Strawberry - Healthy", "Strawberry - Leaf Scorch",
    "Brinjal - Healthy", "Brinjal - Phomopsis Blight"
]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    img = Image.open(file.stream).convert('RGB')
    img = img.resize((128, 128))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    result_idx = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]) * 100)

    return jsonify({
        'prediction': CLASS_NAMES[result_idx],
        'confidence': f"{confidence:.2f}%"
    })


if __name__ == '__main__':
    app.run(debug=True)