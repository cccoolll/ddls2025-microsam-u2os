#!/usr/bin/env python3
"""
Data preprocessing for BBBC039 dataset following micro-sam tutorial pattern.
Converts BBBC039 format to simple image/mask pairs for microSAM training.
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from glob import glob


def preprocess_bbbc039(data_dir="data/BBBC039", target_size=512):
    """
    Preprocess BBBC039 dataset into simple image/mask format.
    
    BBBC039 dataset structure:
    - images/: TIF files with microscopy images
    - masks/: PNG files with segmentation masks
    - metadata/: Split information (training.txt, validation.txt, test.txt)
    
    Following micro-sam tutorial: just need images and instance masks.
    """
    print("🚀 Preprocessing BBBC039 Dataset")
    print("="*60)
    
    data_dir = Path(data_dir)
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(exist_ok=True)
    
    # Create simple structure: images and labels folders
    images_dir = processed_dir / "images"
    labels_dir = processed_dir / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    
    # Load split information
    train_files = load_split_files(data_dir / "metadata" / "training.txt")
    valid_files = load_split_files(data_dir / "metadata" / "validation.txt")
    test_files = load_split_files(data_dir / "metadata" / "test.txt")
    
    print(f"📊 Dataset splits:")
    print(f"   Training: {len(train_files)} images")
    print(f"   Validation: {len(valid_files)} images")
    print(f"   Test: {len(test_files)} images")
    
    # Process each split
    for split_name, split_files in [("train", train_files), ("valid", valid_files)]:
        print(f"\n📊 Processing {split_name} split...")
        
        processed_count = 0
        for idx, filename in enumerate(split_files):
            # Convert PNG mask filename to TIF image filename
            image_filename = filename.replace('.png', '.tif')
            
            # Check if both image and mask exist
            image_path = data_dir / "images" / image_filename
            mask_path = data_dir / "masks" / filename
            
            if not image_path.exists() or not mask_path.exists():
                print(f"  ⚠️  Skipping {filename}: missing image or mask")
                continue
            
            try:
                # Load image (TIF format)
                image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    print(f"  ❌ Failed to load image: {image_path}")
                    continue
                
                # Convert to RGB if needed
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif len(image.shape) == 2:
                    # Grayscale image, convert to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                
                # Load mask (PNG format)
                mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                if mask is None:
                    print(f"  ❌ Failed to load mask: {mask_path}")
                    continue
                
                # Convert mask to instance segmentation format
                # BBBC039 masks are typically binary or multi-class
                # Convert to instance IDs if needed
                if len(mask.shape) == 3:
                    # Convert to grayscale
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                
                # Ensure mask is uint16 for instance IDs
                mask = mask.astype(np.uint16)
                
                # Get original dimensions
                h, w = image.shape[:2]
                
                # Resize to target size
                image_resized = cv2.resize(image, (target_size, target_size))
                mask_resized = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
                
                # Convert to grayscale for microSAM (expects 2D images)
                if len(image_resized.shape) == 3:
                    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
                else:
                    image_gray = image_resized
                
                # Normalize image to [0, 255] range for microSAM
                # BBBC039 TIF images may have higher bit depth values
                if image_gray.max() > 255:
                    # Normalize to [0, 255] range
                    image_gray = ((image_gray - image_gray.min()) / (image_gray.max() - image_gray.min()) * 255).astype(np.uint8)
                else:
                    image_gray = image_gray.astype(np.uint8)
                
                # Save with simple naming: split_idx.tif
                out_name = f"{split_name}_{idx:03d}.tif"
                cv2.imwrite(str(images_dir / out_name), image_gray)
                cv2.imwrite(str(labels_dir / out_name), mask_resized)
                
                # Count cells in mask
                n_cells = len(np.unique(mask_resized)) - 1  # Subtract background (0)
                print(f"  ✅ {out_name}: {n_cells} cells")
                processed_count += 1
                
            except Exception as e:
                print(f"  ❌ Error processing {filename}: {e}")
                continue
        
        print(f"✅ Processed {processed_count} {split_name} images")
    
    # Save split info
    split_info = {
        'train_images': len([f for f in os.listdir(images_dir) if f.startswith('train_')]),
        'valid_images': len([f for f in os.listdir(images_dir) if f.startswith('valid_')]),
        'image_size': target_size,
        'format': 'tif',
        'dataset': 'BBBC039',
        'description': 'BBBC039 microscopy dataset with cell segmentation masks'
    }
    
    with open(processed_dir / "split_info.json", 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n✅ Preprocessing complete!")
    print(f"   Images: {processed_dir / 'images'}")
    print(f"   Labels: {processed_dir / 'labels'}")
    print(f"   Train: {split_info['train_images']} images")
    print(f"   Valid: {split_info['valid_images']} images")
    
    # Visualize one sample
    visualize_sample(images_dir, labels_dir, "train_000.tif", processed_dir)
    
    return processed_dir


def load_split_files(split_file):
    """Load list of filenames from split file."""
    if not os.path.exists(split_file):
        print(f"⚠️  Split file not found: {split_file}")
        return []
    
    with open(split_file, 'r') as f:
        files = [line.strip() for line in f.readlines() if line.strip()]
    
    return files


def visualize_sample(images_dir, labels_dir, filename, save_dir):
    """Visualize a sample image and mask."""
    img_path = images_dir / filename
    mask_path = labels_dir / filename
    
    if not img_path.exists() or not mask_path.exists():
        print(f"⚠️  Sample files not found for visualization")
        return
    
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    
    # Convert to RGB for display
    if len(img.shape) == 2:
        img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_display = img
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_display)
    axes[0].set_title('Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='tab20')
    n_cells = len(np.unique(mask)) - 1  # Subtract background
    axes[1].set_title(f'Mask ({n_cells} cells)')
    axes[1].axis('off')
    
    axes[2].imshow(img_display, alpha=0.7)
    axes[2].imshow(mask, cmap='tab20', alpha=0.3)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'sample_visualization.png', dpi=150, bbox_inches='tight')
    print(f"   Visualization: {save_dir / 'sample_visualization.png'}")
    plt.close()


def analyze_dataset(data_dir="data/BBBC039"):
    """Analyze the BBBC039 dataset structure and statistics."""
    print("🔍 Analyzing BBBC039 Dataset")
    print("="*60)
    
    data_dir = Path(data_dir)
    
    # Count files
    image_files = list((data_dir / "images").glob("*.tif"))
    mask_files = list((data_dir / "masks").glob("*.png"))
    
    print(f"📊 Dataset Statistics:")
    print(f"   Images: {len(image_files)}")
    print(f"   Masks: {len(mask_files)}")
    
    # Load splits
    train_files = load_split_files(data_dir / "metadata" / "training.txt")
    valid_files = load_split_files(data_dir / "metadata" / "validation.txt")
    test_files = load_split_files(data_dir / "metadata" / "test.txt")
    
    print(f"   Training: {len(train_files)}")
    print(f"   Validation: {len(valid_files)}")
    print(f"   Test: {len(test_files)}")
    
    # Sample a few images to check dimensions
    print(f"\n🔍 Sample Image Analysis:")
    for i, img_path in enumerate(image_files[:3]):
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is not None:
                print(f"   {img_path.name}: {img.shape}")
        except Exception as e:
            print(f"   {img_path.name}: Error - {e}")
    
    # Sample a few masks to check format
    print(f"\n🔍 Sample Mask Analysis:")
    for i, mask_path in enumerate(mask_files[:3]):
        try:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is not None:
                unique_values = np.unique(mask)
                print(f"   {mask_path.name}: {mask.shape}, unique values: {len(unique_values)}")
        except Exception as e:
            print(f"   {mask_path.name}: Error - {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess BBBC039 dataset")
    parser.add_argument("--data_dir", default="data/BBBC039", help="BBBC039 data directory")
    parser.add_argument("--target_size", type=int, default=512, help="Target image size")
    parser.add_argument("--analyze", action="store_true", help="Analyze dataset structure")
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_dataset(args.data_dir)
    else:
        preprocess_bbbc039(args.data_dir, args.target_size)
