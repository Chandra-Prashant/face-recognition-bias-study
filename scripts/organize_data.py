import pandas as pd
import os
import shutil

# 1. Absolute Path Setup for Mac
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_PATH, "..", "data", "FairFace"))
VAL_DIR = os.path.join(DATA_DIR, "val")
CSV_PATH = os.path.join(DATA_DIR, "fairface_label_val.csv")

def organize_fairface():
    # 2. Verify CSV exists
    if not os.path.exists(CSV_PATH):
        print(f"❌ ERROR: CSV not found at {CSV_PATH}")
        print("Please ensure you ran 'curl' to download the labels first.")
        return

    # 3. Load Labels
    df = pd.read_csv(CSV_PATH)
    print(f"📈 Loaded {len(df)} labels from CSV.")

    # 4. Check if 'val' directory is empty or nested
    if not os.path.exists(VAL_DIR):
        print(f"❌ ERROR: 'val' directory not found at {VAL_DIR}")
        return

    items_in_val = os.listdir(VAL_DIR)
    print(f"📂 Found {len(items_in_val)} items in 'val' directory.")

    moved_count = 0
    skip_count = 0

    print("🚀 Starting Demographic Stratification...")

    # 5. The Loop: Using enumerate to avoid Pylance 'Hashable' errors
    for i, (_, row) in enumerate(df.iterrows()):
        # Extract filename (e.g., '1.jpg') and race
        img_name = os.path.basename(str(row['file']))
        race = str(row['race'])
        
        # Source and Destination paths
        src = os.path.join(VAL_DIR, img_name)
        dst_folder = os.path.join(VAL_DIR, race)
        
        # 6. Move logic
        if os.path.exists(src):
            os.makedirs(dst_folder, exist_ok=True)
            shutil.move(src, os.path.join(dst_folder, img_name))
            moved_count += 1
            
            # Progress reporting
            if moved_count % 500 == 0:
                print(f"✅ Moved {moved_count} images...")
        else:
            skip_count += 1
            # Show the first 5 skips to help debug if paths are wrong
            if i < 5:
                print(f"⚠️  Missing file: {src}")

    print(f"\n--- Process Complete ---")
    print(f"✨ Images Moved: {moved_count}")
    print(f"⏭️  Images Skipped: {skip_count}")
    print(f"📂 Folders created in: {VAL_DIR}")

if __name__ == "__main__":
    organize_fairface()