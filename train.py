from ultralytics import YOLO
import os

# Initialize YOLO model
model = YOLO("yolov8n.yaml")  

# Train the model
results = model.train(data="data/data.yaml", epochs=100)

# Save the trained model
model.save("pothole_model.pt")
print("Model saved as 'pothole_model.pt'")

# Also save the best weights (automatically saved during training)
print(f"Best model weights saved at: {model.trainer.best}")
print("You can use 'pothole_model.pt' for inference without retraining")

