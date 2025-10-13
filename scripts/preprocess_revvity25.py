#!/usr/bin/env python3
"""
Simple data preprocessing for Revvity-25 dataset following micro-sam tutorial pattern.
Converts COCO format to simple image/mask pairs.
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt


def preprocess_revvity25(data_dir="data/Revvity-25", target_size=512):
    """
    Preprocess Revvity-25 dataset into simple image/mask format.
    
    Following micro-sam tutorial: just need images and instance masks.
    """
    print("🚀 Preprocessing Revvity-25 Dataset")
    print("="*60)
    
    data_dir = Path(data_dir)
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(exist_ok=True)
    
    # Create simple structure: images and labels folders
    images_dir = processed_dir / "images"
    labels_dir = processed_dir / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    
    # Process train and valid splits
    for split in ["train", "valid"]:
        print(f"\n📊 Processing {split} split...")
        
        # Load COCO annotations
        ann_path = data_dir / "annotations" / f"{split}.json"
        with open(ann_path, 'r') as f:
            coco_data = json.load(f)
        
        images_info = coco_data['images']
        annotations = coco_data['annotations']
        
        # Group annotations by image_id
        anns_by_image = {}
        for ann in annotations:
            img_id = ann['image_id']
            if img_id not in anns_by_image:
                anns_by_image[img_id] = []
            anns_by_image[img_id].append(ann)
        
        # Process each image
        for idx, img_info in enumerate(images_info):
            img_id = img_info['id']
            img_name = img_info['file_name']
            
            # Load image
            img_path = data_dir / "images" / img_name
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # Create instance mask from COCO polygons
            mask = np.zeros((h, w), dtype=np.uint16)
            image_anns = anns_by_image.get(img_id, [])
            
            for obj_id, ann in enumerate(image_anns, start=1):
                if 'segmentation' in ann and ann['segmentation']:
                    for poly in ann['segmentation']:
                        if len(poly) >= 6:  # At least 3 points
                            polygon = np.array(poly).reshape(-1, 2).astype(np.int32)
                            cv2.fillPoly(mask, [polygon], obj_id)
            
            # Resize to target size
            image_resized = cv2.resize(image, (target_size, target_size))
            mask_resized = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
            
            # Save with simple naming: split_idx.tif
            out_name = f"{split}_{idx:03d}.tif"
            # Save as grayscale 2D images (micro-sam expects 2D)
            image_gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
            cv2.imwrite(str(images_dir / out_name), image_gray)
            cv2.imwrite(str(labels_dir / out_name), mask_resized)
            
            print(f"  ✅ {out_name}: {len(image_anns)} cells")
        
        print(f"✅ Processed {len(images_info)} {split} images")
    
    # Save split info
    split_info = {
        'train_images': len([f for f in os.listdir(images_dir) if f.startswith('train_')]),
        'valid_images': len([f for f in os.listdir(images_dir) if f.startswith('valid_')]),
        'image_size': target_size,
        'format': 'tif'
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


def visualize_sample(images_dir, labels_dir, filename, save_dir):
    """Visualize a sample image and mask."""
    img = cv2.imread(str(images_dir / filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(labels_dir / filename), cv2.IMREAD_UNCHANGED)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img)
    axes[0].set_title('Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='tab20')
    axes[1].set_title(f'Mask ({mask.max()} cells)')
    axes[1].axis('off')
    
    axes[2].imshow(img, alpha=0.7)
    axes[2].imshow(mask, cmap='tab20', alpha=0.3)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'sample_visualization.png', dpi=150, bbox_inches='tight')
    print(f"   Visualization: {save_dir / 'sample_visualization.png'}")
    plt.close()


if __name__ == "__main__":
    preprocess_revvity25()
