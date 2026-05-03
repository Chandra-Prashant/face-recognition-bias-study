import cv2
import numpy as np
import os
from perturbations import FacePerturber
from retinaface import RetinaFace

# 1. ROBUST PROJECT PATHING
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Define exact paths - Using the high-risk group as the subject
IMG_PATH = os.path.join(PROJECT_ROOT, 'data', 'FairFace', 'stratified_audit', 'Southeast Asian_Female', '1378.jpg')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'methodology')
os.makedirs(OUTPUT_DIR, exist_ok=True)

perturber = FacePerturber()

# 2. LOAD AND STANDARDIZE
raw_img = cv2.imread(IMG_PATH)
if raw_img is None:
    print(f"❌ Error: Image not found at {IMG_PATH}.")
    exit()
else:
    raw_img = cv2.resize(raw_img, (224, 224))

# 3. DETECT LANDMARKS & CORRECT OCCLUSION LOGIC
try:
    # We use RetinaFace to get precise coordinates for the bar
    faces = RetinaFace.detect_faces(IMG_PATH)
    if faces and 'face_1' in faces:
        landmarks = faces['face_1']['landmarks']
        
        # --- THE CORRECTION ---
        # Instead of two boxes, we create a single horizontal strip
        le = landmarks['left_eye']
        re = landmarks['right_eye']
        h, w, _ = raw_img.shape
        
        # Calculate bounds for a full periocular bar
        # Padding (0.12w and 0.08h) ensures eyebrows and corners are covered
        x_start = int(min(le[0], re[0]) - w * 0.12)
        x_end = int(max(le[0], re[0]) + w * 0.12)
        y_start = int(min(le[1], re[1]) - h * 0.08)
        y_end = int(max(le[1], re[1]) + h * 0.08)
        
        eye_bar_coords = (x_start, y_start, x_end, y_end)
    else:
        raise ValueError("No face detected")
except Exception as e:
    print(f"⚠️ Landmark detection issues: {e}. Using calibrated defaults.")
    # Fallback calibrated for a 224x224 centered face
    eye_bar_coords = (40, 85, 184, 115)

# 4. GENERATE SEPARATE PANELS
# Panel A: Baseline (Clean)
cv2.imwrite(os.path.join(OUTPUT_DIR, 'panel_a_baseline.png'), raw_img)

# Panel B: Gaussian Noise (Sigma=25)
noise_img = perturber.apply_gaussian_noise(raw_img, sigma=25)
cv2.imwrite(os.path.join(OUTPUT_DIR, 'panel_b_noise.png'), noise_img)

# Panel C: Gaussian Blur (K=15)
blur_img = perturber.apply_gaussian_blur(raw_img, kernel_size=15)
cv2.imwrite(os.path.join(OUTPUT_DIR, 'panel_c_blur.png'), blur_img)

# Panel D: Low Light (Gamma=0.5)
dark_img = perturber.apply_illumination_change(raw_img, gamma=0.5)
cv2.imwrite(os.path.join(OUTPUT_DIR, 'panel_d_gamma.png'), dark_img)

# Panel E: CORRECTED PERIOCULAR OCCLUSION
occluded_img = raw_img.copy()
cv2.rectangle(occluded_img, (eye_bar_coords[0], eye_bar_coords[1]), 
              (eye_bar_coords[2], eye_bar_coords[3]), (0, 0, 0), -1)
cv2.imwrite(os.path.join(OUTPUT_DIR, 'panel_e_occlusion.png'), occluded_img)

print(f"✨ Success! Corrected panels saved to: {OUTPUT_DIR}")