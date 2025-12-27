import os
import shutil
from pathlib import Path

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
TARGET_DIR = "/lab_data/leelab/VLA-Husain-Tianqin/magno_stream_encoder/data"
DATASET_NAME = "ambityga/imagenet100"

def download_and_extract():
    """Download dataset to the specific target directory."""
    print(f"Target Directory: {TARGET_DIR}")
    
    # Ensure directory exists
    Path(TARGET_DIR).mkdir(parents=True, exist_ok=True)
    
    print("Authenticating with Kaggle...")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    
    print(f"Downloading {DATASET_NAME}...")
    # unzip=True automatically extracts the contents
    api.dataset_download_files(
        DATASET_NAME, 
        path=TARGET_DIR, 
        unzip=True,
        quiet=False
    )
    print("Download and extraction complete.")

def post_process_merge():
    """
    Fixes the 'split folder' issue common with this specific dataset.
    Merges 'train.X' folders into a single 'train' folder.
    """
    root = Path(TARGET_DIR)
    train_final = root / "train"
    
    # Find all folders starting with 'train' (e.g., train, train.1, train.2)
    train_parts = sorted([p for p in root.glob("train*") if p.is_dir()])
    
    if not train_parts:
        return

    print("Checking for split training folders...")
    train_final.mkdir(exist_ok=True)
    
    # Move contents from parts to the main 'train' folder
    for part in train_parts:
        if part == train_final:
            continue
            
        print(f"Merging {part.name} into train/...")
        for item in part.iterdir():
            dest = train_final / item.name
            if not dest.exists():
                shutil.move(str(item), str(dest))
        
        # Remove the empty part folder
        try:
            part.rmdir() 
        except OSError:
            pass # Directory might not be empty if there were duplicates
            
    print(f"Unified training data located at: {train_final}")

if __name__ == "__main__":
    try:
        # Install kaggle if missing: pip install kaggle
        import kaggle
    except ImportError:
        print("Error: 'kaggle' library not found. Please run: pip install kaggle")
        exit(1)

    # download_and_extract()
    post_process_merge()