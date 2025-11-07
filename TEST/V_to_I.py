import cv2
import os

# ====== CONFIGURATION ======
video_path = "TEST/p.mp4"        # ✅ Your video file name (in same folder)
base_output_folder = "frames"      # Creates frames_1, frames_2, etc.
frame_interval = 30                # Extract every 30th frame
# ============================

# Get current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create unique output folder like frames_1, frames_2...
def get_unique_output_folder(base_name):
    i = 1
    while True:
        folder_name = f"{base_name}_{i}"
        full_path = os.path.join(script_dir, folder_name)
        if not os.path.exists(full_path):
            return full_path
        i += 1

# Create the output folder
output_path = get_unique_output_folder(base_output_folder)
os.makedirs(output_path)

# Open video using just the filename (relative path)
print("🔍 Opening:", video_path)
cap = cv2.VideoCapture(video_path)

frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        filename = f"frame_{saved_count:04d}.jpg"
        filepath = os.path.join(output_path, filename)
        cv2.imwrite(filepath, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"✅ Done! {saved_count} frames saved in '{output_path}'")
