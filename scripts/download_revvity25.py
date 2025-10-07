#!/usr/bin/env python3
"""
Download Revvity-25 dataset from Kaggle using the correct API.
"""

import kagglehub
import os
import shutil
from pathlib import Path

def main():
    """Download the dataset using the correct API."""
    print("🚀 Downloading Revvity-25 dataset from Kaggle")
    print("="*60)
    
    try:
        # Use the correct API call
        print("📥 Downloading dataset...")
        path = kagglehub.dataset_download("slavkoprytula/revvity-25")
        print(f"✅ Path to dataset files: {path}")
        
        # List files in the dataset
        print(f"\n📁 Files in dataset:")
        all_files = []
        for root, dirs, files in os.walk(path):
            level = root.replace(str(path), '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:10]:  # Show first 10 files
                print(f"{subindent}{file}")
                all_files.append(os.path.join(root, file))
            if len(files) > 10:
                print(f"{subindent}... and {len(files) - 10} more files")
        
        # Count files by type
        image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        json_files = [f for f in all_files if f.lower().endswith('.json')]
        
        print(f"\n📊 File counts:")
        print(f"  Total files: {len(all_files)}")
        print(f"  Images: {len(image_files)}")
        print(f"  JSON files: {len(json_files)}")
        
        # Copy to our project structure
        print(f"\n📁 Copying to data/revvity25/...")
        target_dir = Path("data/revvity25")
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (target_dir / "images").mkdir(exist_ok=True)
        (target_dir / "annotations").mkdir(exist_ok=True)
        (target_dir / "train").mkdir(exist_ok=True)
        (target_dir / "valid").mkdir(exist_ok=True)
        
        # Copy images
        for img_file in image_files:
            dest_file = target_dir / "images" / os.path.basename(img_file)
            shutil.copy2(img_file, dest_file)
            print(f"  ✅ Copied: {os.path.basename(img_file)}")
        
        # Copy JSON files
        for json_file in json_files:
            dest_file = target_dir / "annotations" / os.path.basename(json_file)
            shutil.copy2(json_file, dest_file)
            print(f"  ✅ Copied: {os.path.basename(json_file)}")
        
        print(f"\n🎉 Dataset successfully downloaded and organized!")
        print(f"📁 Check data/revvity25/ directory")
        
        return dataset_path
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    main()
