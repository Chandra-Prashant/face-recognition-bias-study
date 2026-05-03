import cv2
import numpy as np
import os
from retinaface import RetinaFace

# 1. Setup Paths
# Point to the specific high-risk sample from your audit
IMG_PATH = 'results/sea_female_clean.png' 
OUTPUT_PATH = 'results/sea_female_occluded.png'

# 2. Load and Detect
image = cv2.imread(IMG_PATH)
if image is None:
    print(f"❌ Error: Could not find {IMG_PATH}")
    exit()

image = cv2.resize(image, (224, 224))
h, w, _ = image.shape

# 3. Scientific Occlusion Logic (Landmark-Based)
try:
    # Detect faces to get precise eye coordinates
    faces = RetinaFace.detect_faces(image)
    if faces and 'face_1' in faces:
        landmarks = faces['face_1']['landmarks']
        le = landmarks['left_eye']
        re = landmarks['right_eye']
        
        # Calculate a full periocular bar (eyes + eyebrows)
        # We use a 12% width and 15% height padding for a 'security mask' look
        x_start = int(min(le[0], re[0]) - w * 0.15)
        x_end = int(max(le[0], re[0]) + w * 0.15)
        y_start = int(min(le[1], re[1]) - h * 0.12) # Shifting up to cover brows
        y_end = int(max(le[1], re[1]) + h * 0.08)
    else:
        raise ValueError("No face detected")

except Exception as e:
    print(f"⚠️ Landmark detection failed: {e}. Using calibrated manual coordinates.")
    # Professional default coordinates for a centered 224x224 face
    # (x_start, y_start), (x_end, y_end)
    x_start, y_start, x_end, y_end = 30, 40, 195, 85

# 4. Apply the "Critical Stressor" Mask
occluded_img = image.copy()
cv2.rectangle(occluded_img, (x_start, y_start), (x_end, y_end), (0, 0, 0), -1)

# 5. Save for Paper
cv2.imwrite(OUTPUT_PATH, occluded_img)
print(f"✨ Corrected Research-Grade occlusion saved to: {OUTPUT_PATH}")