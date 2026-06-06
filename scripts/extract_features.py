import numpy as np

def calculate_angle_2d(a, b, c):
    """
    Calculates the angle between three points using ONLY 2D (X, Y) coordinates.
    This makes the math highly stable for side-profile camera angles.
    """
    a = np.array([a[0], a[1]])
    b = np.array([b[0], b[1]])
    c = np.array([c[0], c[1]])

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = np.arccos(cosine)

    return np.degrees(angle)

def extract_features(landmarks):
    # 1. Dynamically determine which side of the body is facing the camera
    # MediaPipe assigns a smaller (more negative) Z value to points closer to the camera
    left_shoulder_z = landmarks[11][2]
    right_shoulder_z = landmarks[12][2]

    if left_shoulder_z < right_shoulder_z:
        # Left side is facing the camera
        ear = landmarks[7]
        shoulder = landmarks[11]
        hip = landmarks[23]
    else:
        # Right side is facing the camera
        ear = landmarks[8]
        shoulder = landmarks[12]
        hip = landmarks[24]

    # Feature 1: Neck/Spine Alignment Angle (Ear -> Shoulder -> Hip)
    # A straight back will have an angle close to 180 degrees. Slouching decreases it.
    spine_angle = calculate_angle_2d(ear, shoulder, hip)

    # Feature 2: Torso Lean Angle (Shoulder -> Hip -> Vertical Axis)
    # Calculates how far forward or backward the person is leaning in their chair.
    vertical_reference = [hip[0], hip[1] - 0.5] 
    torso_lean = calculate_angle_2d(shoulder, hip, vertical_reference)

    # Calculate Torso Length for dynamic scaling (makes it immune to camera distance)
    torso_length = np.linalg.norm(np.array([shoulder[0], shoulder[1]]) - np.array([hip[0], hip[1]]))
    if torso_length == 0: torso_length = 1.0

    # Feature 3: Forward Head Ratio (Horizontal Distance from Ear to Shoulder)
    # In good posture, the ear is directly above the shoulder. Slouching pushes the ear forward (X-axis).
    head_forward_distance = abs(ear[0] - shoulder[0])
    head_forward_ratio = head_forward_distance / torso_length

    # Feature 4: Neck Compression (Distance from Ear to Shoulder)
    # When people slouch severely, their shoulders often hunch up toward their ears.
    ear_shoulder_dist = np.linalg.norm(np.array([ear[0], ear[1]]) - np.array([shoulder[0], shoulder[1]]))
    neck_compression = ear_shoulder_dist / torso_length

    return [
        spine_angle,
        torso_lean,
        head_forward_ratio,
        neck_compression
    ]