from flask import Flask, request, jsonify, render_template, send_file
import os
import cv2
import numpy as np
from PIL import Image
import io
import base64
import torch

# Monkey patch torch.load to use weights_only=False
_original_torch_load = torch.load
def patched_torch_load(f, map_location=None, pickle_module=None, **kwargs):
    return _original_torch_load(f, map_location=map_location, pickle_module=pickle_module, weights_only=False, **kwargs)
torch.load = patched_torch_load

from ultralytics import YOLO

app = Flask(__name__)

# Configure Flask
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the trained model
print("Loading YOLO model...")
model = YOLO('runs/detect/train/weights/best.pt')
print("Model loaded successfully!")

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Received prediction request")
        print("Files in request:", request.files)
        
        if 'file' not in request.files:
            print("No file in request")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        print(f"File received: {file.filename}")
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Run YOLO prediction
        results = model.predict(source=filepath, conf=0.25, save=False)
        
        # Get the annotated image
        annotated_img = results[0].plot()
        
        # Convert to PIL Image and then to base64
        annotated_pil = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        
        # Convert original image to base64
        original_img = Image.open(filepath)
        
        # Convert images to base64
        original_b64 = image_to_base64(original_img)
        processed_b64 = image_to_base64(annotated_pil)
        
        # Get detection info
        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                detection = {
                    'confidence': float(box.conf[0]),
                    'class': 'pothole',
                    'bbox': box.xyxy[0].tolist()
                }
                detections.append(detection)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'original_image': original_b64,
            'processed_image': processed_b64,
            'detections': detections,
            'count': len(detections)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)