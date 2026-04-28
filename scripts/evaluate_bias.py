import os
import pandas as pd
import numpy as np
import cv2
from deepface import DeepFace
from perturbations import FacePerturber
from retinaface import RetinaFace
from typing import cast

# RQ1 & RQ2: Correct DeepFace identifiers
MODELS = ["Facenet", "ArcFace", "VGG-Face"]
print(f"DEBUG: Successfully loaded models: {MODELS}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "FairFace")
LABEL_CSV = os.path.join(DATA_DIR, "fairface_label_val.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "bias_metrics.csv")

# RQ2 Severity Levels from Methodology [cite: 46, 47, 48]
EXPERIMENTS = [
    ('eye_occlusion', 'high'),
    ('gaussian_noise', 15), 
    ('gaussian_blur', 7),
    ('gamma_correction', 0.5) 
]

os.makedirs(RESULTS_DIR, exist_ok=True)
perturber = FacePerturber()

def calculate_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    v1_f, v2_f = v1.flatten(), v2.flatten()
    denom = (np.linalg.norm(v1_f) * np.linalg.norm(v2_f))
    return float(np.dot(v1_f, v2_f) / denom) if denom != 0 else 0.0

try:
    labels_df = pd.read_csv(LABEL_CSV)
    print(f"✅ Loaded {len(labels_df)} labels for intersectional analysis.")
except Exception as e:
    print(f"❌ ERROR: Could not load {LABEL_CSV}: {e}")
    exit()

results_data = []
# Process a subset for baseline model evaluation [cite: 54]
sample_df = labels_df.head(1000) 

for idx, row in sample_df.iterrows():
    img_rel_path = str(row['file']) # 'val/1.jpg'
    race = str(row['race'])
    gender = str(row['gender'])
    
    # Correct path logic: Looking inside organized demographic folders
    filename = os.path.basename(img_rel_path)
    img_path = os.path.join(DATA_DIR, "val", race, filename)
    
    if not os.path.exists(img_path):
        # Fallback to original path if not organized
        img_path = os.path.join(DATA_DIR, img_rel_path)
        if not os.path.exists(img_path):
            continue

    img = cv2.imread(img_path)
    if img is None: continue

    # 1. Detection via RetinaFace [cite: 33]
    try:
        faces = RetinaFace.detect_faces(img_path)
        if not isinstance(faces, dict) or 'face_1' not in faces:
            continue
        landmarks = faces['face_1']['landmarks']
    except:
        continue

    print(f"🚀 [{idx}] Processing {race} | {gender} | {filename}")

    # 2. Embedding Extraction across Models
    for model_name in MODELS:
        current_model = "Facenet" if "Facenet" in model_name else model_name
        try:
            # Baseline (Clean)
            clean_res = DeepFace.represent(img, model_name=current_model, enforce_detection=False)
            
            if not clean_res or not isinstance(clean_res[0], dict):
                continue
            clean_vec = np.array(clean_res[0]["embedding"])

            # Experimental Perturbations
            for p_type, severity in EXPERIMENTS:
                if p_type == 'eye_occlusion':
                    mod_img = perturber.apply_occlusion(img, landmarks, area='eyes')
                elif p_type == 'gaussian_noise':
                    mod_img = perturber.apply_gaussian_noise(img, sigma=cast(int, severity))
                elif p_type == 'gaussian_blur':
                    mod_img = perturber.apply_gaussian_blur(img, kernel_size=cast(int, severity))
                elif p_type == 'gamma_correction':
                    mod_img = perturber.apply_illumination_change(img, gamma=cast(float, severity))
                else:
                    continue

                mod_res = DeepFace.represent(mod_img, model_name=current_model, enforce_detection=False)
                
                if not mod_res or not isinstance(mod_res[0], dict):
                    continue
                mod_vec = np.array(mod_res[0]["embedding"])

                # 3. Score & Record
                score = calculate_similarity(clean_vec, mod_vec)
                results_data.append({
                    "race": race,
                    "gender": gender,
                    "model": current_model,
                    "perturbation": p_type,
                    "severity": severity,
                    "similarity": score
                })
                print(f"  [✔] {current_model} | {p_type}: {score:.4f}")
        except Exception as e:
            print(f"  [!] DeepFace Error ({current_model}) on {filename}: {e}")

if results_data:
    final_df = pd.DataFrame(results_data)
    final_df.to_csv(RESULTS_FILE, index=False)
    print(f"\n✨ SUCCESS: {len(results_data)} intersectional records saved to {RESULTS_FILE}")