import os
import pandas as pd
import numpy as np
import cv2
from deepface import DeepFace
from perturbations import FacePerturber
from retinaface import RetinaFace
from typing import cast, Dict, List, Any

# RQ1 & RQ2 Constants
MODELS = ["Facenet", "ArcFace", "VGG-Face"]
THRESHOLD = 0.85 # Operational security threshold

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "FairFace")
AUDIT_DIR = os.path.join(DATA_DIR, "stratified_audit") 
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "bias_metrics_optimized.csv")

# Experiments to run per image
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

def get_match_status(score: float) -> str:
    return "MATCH" if score >= THRESHOLD else "FAILURE"

results_data = []

if not os.path.exists(AUDIT_DIR):
    print(f"❌ ERROR: Audit directory {AUDIT_DIR} not found.")
    exit()

group_folders = [f for f in os.listdir(AUDIT_DIR) if os.path.isdir(os.path.join(AUDIT_DIR, f))]

# Main Audit Loop
for group in group_folders:
    race, gender = group.split("_")
    group_path = os.path.join(AUDIT_DIR, group)
    images = os.listdir(group_path)

    for img_name in images:
        img_path = os.path.join(group_path, img_name)
        img = cv2.imread(img_path)
        if img is None: continue

        # 1. OPTIMIZATION: Landmark Detection (Run ONCE per image)
        try:
            faces = RetinaFace.detect_faces(img_path)
            if not isinstance(faces, dict) or 'face_1' not in faces:
                continue
            landmarks = faces['face_1']['landmarks']
        except:
            continue

        print(f"\n📸 FILE: {img_name} ({group})")

        # 2. OPTIMIZATION: Generate all perturbed versions upfront
        perturbed_set = []
        for p_type, severity in EXPERIMENTS:
            if p_type == 'eye_occlusion':
                mod = perturber.apply_occlusion(img, landmarks, area='eyes')
            elif p_type == 'gaussian_noise':
                mod = perturber.apply_gaussian_noise(img, sigma=cast(int, severity))
            elif p_type == 'gaussian_blur':
                mod = perturber.apply_gaussian_blur(img, kernel_size=cast(int, severity))
            elif p_type == 'gamma_correction':
                mod = perturber.apply_illumination_change(img, gamma=cast(float, severity))
            perturbed_set.append((p_type, severity, mod))

        # 3. Model Evaluation
        for model_name in MODELS:
            try:
                # Baseline embedding
                clean_res = cast(List[Dict[str, Any]], DeepFace.represent(img, model_name=model_name, enforce_detection=False))
                if not clean_res: continue
                clean_vec = np.array(clean_res[0]["embedding"])

                # Test each perturbation against baseline
                for p_type, severity, mod_img in perturbed_set:
                    mod_res = cast(List[Dict[str, Any]], DeepFace.represent(mod_img, model_name=model_name, enforce_detection=False))
                    if not mod_res: continue
                    
                    mod_vec = np.array(mod_res[0]["embedding"])
                    score = calculate_similarity(clean_vec, mod_vec)
                    status = get_match_status(score)

                    # LOGGING: See exactly what test is running
                    print(f"   ↳ [TEST] {model_name:8} | {p_type:16} | Score: {score:.3f} | {status}")

                    results_data.append({
                        "group": group,
                        "race": race,
                        "gender": gender,
                        "model": model_name,
                        "perturbation": p_type,
                        "severity": severity,
                        "similarity": score,
                        "status": status
                    })
            except Exception:
                continue

# Save and summarize
if results_data:
    final_df = pd.DataFrame(results_data)
    final_df.to_csv(RESULTS_FILE, index=False)
    fail_rate = (final_df['status'] == 'FAILURE').mean() * 100
    print(f"\n✨ AUDIT COMPLETE: {len(final_df)} tests performed.")
    print(f"📊 Final Intersectional Failure Rate: {fail_rate:.2f}%")
