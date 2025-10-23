#!/usr/bin/env python3
"""
Create simple example data for testing microSAM service.
Just one image and one JSON annotation file.
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
import shutil


def create_simple_finetuning_example():
    """Create simple fine-tuning example with 4 images and annotations for proper train/val split."""
    print("🔧 Creating simple fine-tuning example with 4 images...")
    
    # Create output directory
    output_dir = Path("examples/simple_finetuning")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load train annotations
    train_ann_path = "data/Revvity-25/annotations/train.json"
    with open(train_ann_path, 'r') as f:
        train_data = json.load(f)
    
    # Select 4 images with good annotation coverage
    # Filter images that have annotations
    images_with_annotations = []
    for img in train_data['images']:
        img_annotations = [ann for ann in train_data['annotations'] if ann['image_id'] == img['id']]
        if len(img_annotations) > 0:  # Only include images with annotations
            images_with_annotations.append((img, img_annotations))
    
    # Sort by number of annotations (descending) and take top 4
    images_with_annotations.sort(key=lambda x: len(x[1]), reverse=True)
    selected_images_data = images_with_annotations[:4]
    
    print(f"  📊 Selected 4 images from {len(images_with_annotations)} available images")
    
    # Extract selected images and all their annotations
    selected_images = []
    selected_annotations = []
    all_annotations = []
    
    for i, (img, img_annotations) in enumerate(selected_images_data):
        selected_images.append(img)
        selected_annotations.extend(img_annotations)
        all_annotations.extend(img_annotations)
        print(f"    Image {i+1}: {img['file_name']} ({len(img_annotations)} annotations)")
    
    # Create COCO format with 4 images
    simple_coco = {
        "images": selected_images,
        "annotations": all_annotations,
        "categories": train_data['categories']
    }
    
    # Save COCO annotations
    with open(output_dir / "annotations.json", 'w') as f:
        json.dump(simple_coco, f, indent=2)
    
    # Copy all selected images
    copied_images = []
    for i, (img, _) in enumerate(selected_images_data):
        src_path = Path("data/Revvity-25/images") / img['file_name']
        dst_path = output_dir / img['file_name']
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            copied_images.append(img['file_name'])
            print(f"  ✅ Copied {img['file_name']}")
    
    # Create numpy arrays for bioengine-app (all 4 images)
    image_arrays = []
    for i, (img, _) in enumerate(selected_images_data):
        src_path = Path("data/Revvity-25/images") / img['file_name']
        if src_path.exists():
            # Load image and convert to numpy array
            img_cv = cv2.imread(str(src_path))
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            # Convert to (C, H, W) format
            img_array = np.transpose(img_cv, (2, 0, 1))
            image_arrays.append(img_array)
    
    # Save all images as numpy arrays
    for i, img_array in enumerate(image_arrays):
        np.save(output_dir / f"image_{i}.npy", img_array)
        print(f"  ✅ Created image_{i}.npy")
    
    # Also create a combined array for backward compatibility
    if image_arrays:
        # Use the first image as the main one for backward compatibility
        np.save(output_dir / "image.npy", image_arrays[0])
        print(f"  ✅ Created image.npy (main image)")
    
    print(f"✅ Simple fine-tuning example created in {output_dir}")
    print(f"   Images: {len(selected_images)} images")
    print(f"   Annotations: {len(all_annotations)} total cells")
    print(f"   Files: annotations.json, image.npy, image_0.npy, image_1.npy, image_2.npy, image_3.npy")
    print(f"   Copied images: {', '.join(copied_images)}")
    
    return output_dir, simple_coco, image_arrays


def create_simple_segmentation_example():
    """Create simple segmentation example with one test image."""
    print("🔍 Creating simple segmentation example...")
    
    # Create output directory
    output_dir = Path("examples/simple_segmentation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load validation annotations
    valid_ann_path = "data/Revvity-25/annotations/valid.json"
    with open(valid_ann_path, 'r') as f:
        valid_data = json.load(f)
    
    # Take just the first validation image
    sample_image = valid_data['images'][0]
    
    # Copy the image and resize to 512x512
    src_path = Path("data/Revvity-25/images") / sample_image['file_name']
    dst_path = output_dir / "test_image.png"
    
    if src_path.exists():
        # Load and resize
        img = cv2.imread(str(src_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (512, 512))
        
        # Save as PNG
        cv2.imwrite(str(dst_path), cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
        print(f"  ✅ Created {dst_path.name}")
        
        # Create ground truth mask
        sample_image_id = sample_image['id']
        sample_annotations = [ann for ann in valid_data['annotations'] if ann['image_id'] == sample_image_id]
        
        # Create ground truth mask
        mask = np.zeros((512, 512), dtype=np.uint16)
        
        for obj_id, ann in enumerate(sample_annotations, start=1):
            if 'segmentation' in ann and ann['segmentation']:
                for poly in ann['segmentation']:
                    if len(poly) >= 6:  # At least 3 points
                        # Scale polygon to 512x512
                        scaled_poly = []
                        for i in range(0, len(poly), 2):
                            x = int(poly[i] * 512 / sample_image['width'])
                            y = int(poly[i+1] * 512 / sample_image['height'])
                            scaled_poly.extend([x, y])
                        
                        # Draw polygon
                        if len(scaled_poly) >= 6:
                            polygon = np.array(scaled_poly).reshape(-1, 2).astype(np.int32)
                            cv2.fillPoly(mask, [polygon], obj_id)
        
        # Save ground truth mask
        mask_path = output_dir / "ground_truth.png"
        cv2.imwrite(str(mask_path), mask)
        print(f"  ✅ Created ground truth mask {mask_path.name}")
    
    print(f"✅ Simple segmentation example created in {output_dir}")
    print(f"   Test image: test_image.png")
    print(f"   Ground truth: ground_truth.png")
    
    return output_dir

def main():
    """Create simple example data."""
    print("🚀 Creating Simple microSAM Example Data")
    print("="*50)
    
    # Create examples directory
    Path("examples").mkdir(exist_ok=True)
    
    # Create simple fine-tuning example
    finetuning_dir, coco_data, image_arrays = create_simple_finetuning_example()
    
    # Create simple segmentation example
    segmentation_dir = create_simple_segmentation_example()
    
    print("\n✅ Simple example data created!")
    print(f"📁 Fine-tuning data: {finetuning_dir}")
    print(f"📁 Segmentation data: {segmentation_dir}")
    
    # Print summary
    print(f"\n📊 Summary:")
    print(f"   Fine-tuning: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
    print(f"   Segmentation: 1 test image + ground truth")
    print(f"   Train/Val split: {len(coco_data['images'])} images → 80/20 split")
    
    print(f"\n🎯 Usage:")
    print(f"   1. Fine-tuning: Use examples/simple_finetuning/ with start_fit()")
    print(f"   2. Segmentation: Use examples/simple_segmentation/ with segment_all()")
    print(f"   3. See examples/simple_*.py for code examples")
    print(f"   4. Training now supports proper train/validation split!")


if __name__ == "__main__":
    main()
