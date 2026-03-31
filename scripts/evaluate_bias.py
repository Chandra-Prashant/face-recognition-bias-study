import os
import pandas as pd
import numpy as np
import cv2
from deepface import DeepFace
from perturbations import FacePerturber
from retinaface import RetinaFace

# Fix: Ensure these match DeepFace's exact internal strings
MODELS = ["FaceNet", "ArcFace", "VGG-Face"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "FairFace", "val")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "bias_metrics.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)
perturber = FacePerturber()

def calculate_similarity(v1, v2):
    """Calculates Cosine Similarity between two embedding vectors."""
    v1 = np.array(v1).flatten()
    v2 = np.array(v2).flatten()
    # Adding a small epsilon to avoid division by zero
    denominator = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denominator == 0: return 0
    return np.dot(v1, v2) / denominator

results_data = []

if not os.path.exists(DATA_DIR):
    print(f"❌ ERROR: DATA_DIR not found at {os.path.abspath(DATA_DIR)}")
    exit()

groups = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
print(f"✅ Found {len(groups)} demographic groups: {groups}")

for race in groups:
    race_path = os.path.join(DATA_DIR, race)
    print(f"\n--- Processing group: {race.upper()} ---")
    
    # Process 50 images per group for statistical significance
    image_list = [f for f in os.listdir(race_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:50]
    
    for img_name in image_list:
        img_path = os.path.join(race_path, img_name)
        img = cv2.imread(img_path)
        
        if img is None: continue
        
        # 1. Detection via RetinaFace
        try:
            faces = RetinaFace.detect_faces(img_path)
            if not isinstance(faces, dict) or 'face_1' not in faces:
                print(f"  [!] Skip: No face detected in {img_name}")
                continue
            
            landmarks = faces['face_1']['landmarks']
        except Exception as e:
            print(f"  [!] RetinaFace Error on {img_name}: {e}")
            continue

        # 2. Embedding Extraction across Models
        for model_name in MODELS:
            try:
                # Clean Embedding
                clean_res = DeepFace.represent(img, model_name=model_name, enforce_detection=False)
                if not clean_res or not isinstance(clean_res[0], dict):
                    continue
                clean_vec = clean_res[0]["embedding"]

                # Apply Perturbation (Robustness Experiment)
                occ_img = perturber.apply_occlusion(img, landmarks, area='eyes')
                occ_res = DeepFace.represent(occ_img, model_name=model_name, enforce_detection=False)
                
                if not occ_res or not isinstance(occ_res[0], dict):
                    continue
                occ_vec = occ_res[0]["embedding"]

                # 3. Record Results
                score = calculate_similarity(clean_vec, occ_vec)
                results_data.append({
                    "demographic": race,
                    "model": model_name,
                    "perturbation": "eye_occlusion",
                    "similarity": float(score)
                })
                print(f"  [✔] {img_name} | {model_name}: {score:.4f}")

            except Exception as e:
                print(f"  [!] DeepFace Error ({model_name}) on {img_name}: {e}")
                continue

# 4. Save results - this is your raw data for the ANOVA test
if results_data:
    df = pd.DataFrame(results_data)
    df.to_csv(RESULTS_FILE, index=False)
    print(f"\n🚀 SUCCESS: {len(results_data)} records saved to {os.path.abspath(RESULTS_FILE)}")
else:
    print("\n❌ FAILED: No data points collected.")