import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from PIL import Image
import tensorflow as tf
import io

app = Flask(__name__)

# Load model on startup
MODEL = tf.keras.models.load_model('model.weights.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Read and preprocess the image
    img = Image.open(io.BytesIO(file.read()))
    img = img.resize((128, 128))  # Resize to match model input
    img_array = np.array(img) / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = MODEL.predict(img_array)[0][0]
    result = "dog" if prediction > 0.5 else "cat"
    confidence = float(prediction) if result == "dog" else float(1.0 - prediction)
    
    return jsonify({
        'prediction': result,
        'confidence': confidence,
        'raw_score': float(prediction)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))