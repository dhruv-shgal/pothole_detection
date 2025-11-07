from ultralytics import YOLO
import os

# Load the trained YOLO model
model = YOLO('runs/detect/train/weights/best.pt')

# Path to image folder
image_folder = 'TEST/Pothole'

# Loop through all JPG files in the folder
for filename in os.listdir(image_folder):
    if filename.lower().endswith('.jpg'):
        image_path = os.path.join(image_folder, filename)
        print(f"Processing: {image_path}")
        results = model.predict(source=image_path, conf=0.10 , save=True)
        print(results)