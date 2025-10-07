#!/usr/bin/env python3
"""
Manual download of Revvity-25 dataset.
Since the Hugging Face dataset has issues, we'll create a placeholder structure
and provide instructions for manual download.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_dataset_structure():
    """Create the expected dataset directory structure."""
    print("📁 Creating Revvity-25 dataset structure...")
    
    # Create directories
    base_dir = Path("data/revvity25")
    dirs_to_create = [
        base_dir,
        base_dir / "images",
        base_dir / "annotations",
        base_dir / "processed",
        base_dir / "train",
        base_dir / "valid",
        base_dir / "test"
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {dir_path}")
    
    return base_dir

def create_dataset_info():
    """Create dataset information file."""
    print("\n📄 Creating dataset information...")
    
    dataset_info = {
        "name": "Revvity-25",
        "description": "Full Cell Segmentation Dataset for U2OS cells",
        "source": "https://huggingface.co/datasets/YaroslavPrytula/Revvity-25",
        "paper": "https://arxiv.org/abs/2508.01928",
        "github": "https://github.com/SlavkoPrytula/IAUNet",
        "project_page": "https://slavkoprytula.github.io/IAUNet/",
        "statistics": {
            "total_images": 110,
            "image_size": "1080x1080",
            "total_cells": 2937,
            "average_cells_per_image": 27,
            "annotation_type": "polygon",
            "average_points_per_cell": 60,
            "max_points_per_cell": 400
        },
        "splits": {
            "train": "Training split",
            "valid": "Validation split"
        },
        "format": {
            "images": "PNG format, 1080x1080 pixels",
            "annotations": "JSON format with polygon coordinates"
        }
    }
    
    info_path = Path("data/revvity25/dataset_info.json")
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"  ✅ Dataset info saved to: {info_path}")

def create_download_instructions():
    """Create instructions for manual dataset download."""
    print("\n📋 Creating download instructions...")
    
    instructions = """
# Revvity-25 Dataset Download Instructions

## Manual Download Steps:

1. **Visit the Hugging Face Dataset Page:**
   - Go to: https://huggingface.co/datasets/YaroslavPrytula/Revvity-25
   - Click on "Files and versions" tab

2. **Download Dataset Files:**
   - Download all files from the repository
   - Place images in: `data/revvity25/images/`
   - Place annotations in: `data/revvity25/annotations/`

3. **Expected File Structure:**
   ```
   data/revvity25/
   ├── images/
   │   ├── train_*.png
   │   └── valid_*.png
   ├── annotations/
   │   ├── train.json
   │   └── valid.json
   └── dataset_info.json
   ```

4. **Alternative Download Methods:**
   - Use `git clone` if the repository is public
   - Use Hugging Face CLI: `huggingface-cli download YaroslavPrytula/Revvity-25`
   - Download individual files from the web interface

## Dataset Statistics:
- **Total Images**: 110 (1080x1080 brightfield images)
- **Total Cells**: 2,937 manually annotated cells
- **Average Cells per Image**: 27
- **Annotation Type**: Detailed polygon annotations
- **Average Points per Cell**: 60 (up to 400 for complex structures)

## Usage for microSAM Fine-tuning:
- Images will be resized to 512x512 for training
- Annotations will be converted to segmentation masks
- Train/validation split: 70/20/10
- Focus on auto-segmentation (AIS) head training
"""
    
    instructions_path = Path("data/revvity25/DOWNLOAD_INSTRUCTIONS.md")
    with open(instructions_path, 'w') as f:
        f.write(instructions)
    
    print(f"  ✅ Download instructions saved to: {instructions_path}")

def create_sample_data_structure():
    """Create a sample data structure to test our pipeline."""
    print("\n🧪 Creating sample data structure for testing...")
    
    # Create sample image (placeholder)
    sample_image = np.random.randint(0, 255, (1080, 1080, 3), dtype=np.uint8)
    
    # Save sample image
    sample_img_path = Path("data/revvity25/sample_image.png")
    plt.imsave(sample_img_path, sample_image)
    print(f"  ✅ Sample image created: {sample_img_path}")
    
    # Create sample annotation
    sample_annotation = {
        "image_id": "sample_001",
        "image_size": [1080, 1080],
        "cells": [
            {
                "cell_id": 1,
                "category": "cell",
                "segmentation": {
                    "polygon": [
                        [100, 100], [200, 100], [200, 200], [100, 200]
                    ]
                },
                "area": 10000,
                "bbox": [100, 100, 100, 100]
            }
        ]
    }
    
    sample_ann_path = Path("data/revvity25/sample_annotation.json")
    with open(sample_ann_path, 'w') as f:
        json.dump(sample_annotation, f, indent=2)
    
    print(f"  ✅ Sample annotation created: {sample_ann_path}")

def create_data_exploration_script():
    """Create a script to explore the dataset once downloaded."""
    print("\n🔍 Creating data exploration script...")
    
    exploration_script = '''#!/usr/bin/env python3
"""
Explore Revvity-25 dataset structure and statistics.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

def explore_dataset():
    """Explore the downloaded dataset."""
    data_dir = Path("data/revvity25")
    
    if not data_dir.exists():
        print("❌ Dataset directory not found. Please download the dataset first.")
        return
    
    print("🔍 Exploring Revvity-25 dataset...")
    
    # Check directory structure
    print("\\n📁 Directory structure:")
    for root, dirs, files in os.walk(data_dir):
        level = root.replace(str(data_dir), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")
    
    # Count files
    image_files = list(data_dir.glob("**/*.png")) + list(data_dir.glob("**/*.jpg"))
    json_files = list(data_dir.glob("**/*.json"))
    
    print(f"\\n📊 File counts:")
    print(f"  Images: {len(image_files)}")
    print(f"  JSON files: {len(json_files)}")
    
    # Analyze images
    if image_files:
        print(f"\\n🖼️  Image analysis (first 3 images):")
        for img_path in image_files[:3]:
            try:
                with Image.open(img_path) as img:
                    print(f"  {img_path.name}: {img.size}, {img.mode}")
            except Exception as e:
                print(f"  ❌ Error reading {img_path.name}: {e}")
    
    # Analyze annotations
    if json_files:
        print(f"\\n📄 Annotation analysis:")
        for json_path in json_files[:3]:
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                print(f"  {json_path.name}: {type(data)} with {len(data) if isinstance(data, (list, dict)) else 'unknown'} items")
            except Exception as e:
                print(f"  ❌ Error reading {json_path.name}: {e}")

if __name__ == "__main__":
    explore_dataset()
'''
    
    script_path = Path("data/revvity25/explore_dataset.py")
    with open(script_path, 'w') as f:
        f.write(exploration_script)
    
    # Make it executable
    os.chmod(script_path, 0o755)
    print(f"  ✅ Exploration script created: {script_path}")

def main():
    """Main function to set up dataset structure."""
    print("🚀 Setting up Revvity-25 Dataset Structure")
    print("="*60)
    
    # Create directory structure
    base_dir = create_dataset_structure()
    
    # Create dataset information
    create_dataset_info()
    
    # Create download instructions
    create_download_instructions()
    
    # Create sample data for testing
    create_sample_data_structure()
    
    # Create exploration script
    create_data_exploration_script()
    
    print("\n🎉 Dataset structure setup completed!")
    print("\n📋 Next steps:")
    print("1. Follow the instructions in data/revvity25/DOWNLOAD_INSTRUCTIONS.md")
    print("2. Download the actual dataset files")
    print("3. Run: python data/revvity25/explore_dataset.py")
    print("4. Proceed with data preprocessing pipeline")

if __name__ == "__main__":
    main()
