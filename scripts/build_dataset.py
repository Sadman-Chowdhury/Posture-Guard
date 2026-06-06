import cv2
import mediapipe as mp
import os
import csv
from extract_features import extract_features  # Import your custom feature extractor

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5
)

DATASET_PATH = "dataset"
OUTPUT_FILE = "dataset/posture_landmarks.csv"

# Ensure dataset directory exists
os.makedirs(DATASET_PATH, exist_ok=True)

data = []
labels = ["good", "bad"]

print("Processing images and extracting structural features...")

for label in labels:
    folder = os.path.join(DATASET_PATH, label)

    if not os.path.exists(folder):
        print(f"Warning: Folder '{folder}' does not exist. Skipping.")
        continue

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            # Format the 33 landmarks into a list of [x, y, z] coordinates
            landmarks_formatted = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
            
            # Extract the 4 custom angles
            features = extract_features(landmarks_formatted)

            # Append the class label at the end
            features.append(label)
            data.append(features)

print("Total samples extracted:", len(data))

# Generate the header row for the 4 features + label
header = ['neck_angle', 'shoulder_slope', 'back_angle', 'head_forward', 'class']

# Save to CSV
with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)

print(f"Dataset successfully saved to {OUTPUT_FILE}")