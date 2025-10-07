#!/usr/bin/env python3
"""
Data preprocessing pipeline for Revvity-25 dataset.
Prepares the dataset for microSAM fine-tuning.
"""

import os
import json
import numpy as np
import cv2
from PIL import Image
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

class Revvity25Preprocessor:
    """Preprocessor for Revvity-25 dataset."""
    
    def __init__(self, data_dir: str = "data/revvity25", target_size: int = 512):
        """
        Initialize the preprocessor.
        
        Args:
            data_dir: Path to the dataset directory
            target_size: Target image size for training (default: 512x512)
        """
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.processed_dir / "images").mkdir(exist_ok=True)
        (self.processed_dir / "masks").mkdir(exist_ok=True)
        (self.processed_dir / "train").mkdir(exist_ok=True)
        (self.processed_dir / "valid").mkdir(exist_ok=True)
        
    def load_annotations(self, split: str) -> Dict:
        """Load annotations for a specific split."""
        ann_path = self.data_dir / "annotations" / f"{split}.json"
        
        if not ann_path.exists():
            print(f"❌ Annotation file not found: {ann_path}")
            return {}
        
        with open(ann_path, 'r') as f:
            annotations = json.load(f)
        
        print(f"✅ Loaded COCO format annotations for {split}")
        print(f"  Images: {len(annotations.get('images', []))}")
        print(f"  Annotations: {len(annotations.get('annotations', []))}")
        return annotations
    
    def create_segmentation_mask(self, image_shape: Tuple[int, int], 
                                cells: List[Dict]) -> np.ndarray:
        """
        Create segmentation mask from cell annotations.
        
        Args:
            image_shape: (height, width) of the image
            cells: List of cell annotations with polygon coordinates
            
        Returns:
            Segmentation mask with cell IDs
        """
        mask = np.zeros(image_shape, dtype=np.uint8)
        
        for cell_id, cell in enumerate(cells, 1):
            if 'segmentation' in cell and 'polygon' in cell['segmentation']:
                polygon = np.array(cell['segmentation']['polygon'], dtype=np.int32)
                cv2.fillPoly(mask, [polygon], cell_id)
        
        return mask
    
    def create_coco_segmentation_mask(self, image_shape: Tuple[int, int], 
                                    annotations: List[Dict]) -> np.ndarray:
        """
        Create segmentation mask from COCO format annotations.
        
        Args:
            image_shape: (height, width) of the image
            annotations: List of COCO annotations
            
        Returns:
            Segmentation mask with cell IDs
        """
        mask = np.zeros(image_shape, dtype=np.uint8)
        
        for ann_id, ann in enumerate(annotations, 1):
            if 'segmentation' in ann:
                segmentation = ann['segmentation']
                if segmentation:  # Check if not empty
                    # COCO segmentation can be a list of polygons
                    if isinstance(segmentation[0], list):
                        # Multiple polygons for one object
                        for poly in segmentation:
                            if len(poly) >= 6:  # At least 3 points (x,y pairs)
                                polygon = np.array(poly, dtype=np.int32).reshape(-1, 2)
                                cv2.fillPoly(mask, [polygon], ann_id)
                    else:
                        # Single polygon
                        if len(segmentation) >= 6:  # At least 3 points
                            polygon = np.array(segmentation, dtype=np.int32).reshape(-1, 2)
                            cv2.fillPoly(mask, [polygon], ann_id)
        
        return mask
    
    def resize_image_and_mask(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Resize image and mask to target size."""
        # Resize image
        resized_image = cv2.resize(image, (self.target_size, self.target_size))
        
        # Resize mask (nearest neighbor to preserve cell IDs)
        resized_mask = cv2.resize(mask, (self.target_size, self.target_size), 
                                interpolation=cv2.INTER_NEAREST)
        
        return resized_image, resized_mask
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        return image.astype(np.float32) / 255.0
    
    def process_split(self, split: str) -> Dict[str, Any]:
        """Process a specific split (train/valid)."""
        print(f"\n🔄 Processing {split} split...")
        
        # Load annotations
        coco_data = self.load_annotations(split)
        if not coco_data:
            return {}
        
        processed_data = {
            'images': [],
            'masks': [],
            'metadata': []
        }
        
        # Get images and annotations
        images = coco_data.get('images', [])
        annotations = coco_data.get('annotations', [])
        
        # Group annotations by image_id
        annotations_by_image = {}
        for ann in annotations:
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)
        
        for i, image_info in enumerate(images):
            image_id = image_info['id']
            image_filename = image_info['file_name']
            
            # Get image path
            image_path = self.data_dir / "images" / image_filename
            
            if not image_path.exists():
                print(f"⚠️  Image not found: {image_path}")
                continue
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"❌ Failed to load image: {image_path}")
                continue
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_shape = image.shape[:2]
            
            # Get annotations for this image
            image_annotations = annotations_by_image.get(image_id, [])
            
            # Create segmentation mask from COCO annotations
            mask = self.create_coco_segmentation_mask(original_shape, image_annotations)
            
            # Resize image and mask
            resized_image, resized_mask = self.resize_image_and_mask(image, mask)
            
            # Normalize image
            normalized_image = self.normalize_image(resized_image)
            
            # Save processed data
            processed_image_path = self.processed_dir / "images" / f"{split}_{i:03d}.png"
            processed_mask_path = self.processed_dir / "masks" / f"{split}_{i:03d}.png"
            
            # Save as PNG
            cv2.imwrite(str(processed_image_path), (resized_image * 255).astype(np.uint8))
            cv2.imwrite(str(processed_mask_path), resized_mask)
            
            # Store metadata
            metadata = {
                'image_id': image_id,
                'filename': image_filename,
                'original_shape': original_shape,
                'target_shape': (self.target_size, self.target_size),
                'num_cells': len(image_annotations),
                'image_path': str(processed_image_path),
                'mask_path': str(processed_mask_path)
            }
            
            processed_data['images'].append(normalized_image)
            processed_data['masks'].append(resized_mask)
            processed_data['metadata'].append(metadata)
            
            print(f"  ✅ Processed {image_filename}: {len(image_annotations)} cells")
        
        print(f"✅ Processed {len(processed_data['images'])} images for {split}")
        return processed_data
    
    def create_train_valid_split(self):
        """Create train/validation split from processed data."""
        print("\n📊 Creating train/validation split...")
        
        # Process both splits
        train_data = self.process_split('train')
        valid_data = self.process_split('valid')
        
        # Save split information
        split_info = {
            'train': {
                'num_images': len(train_data.get('images', [])),
                'num_cells': sum(meta['num_cells'] for meta in train_data.get('metadata', []))
            },
            'valid': {
                'num_images': len(valid_data.get('images', [])),
                'num_cells': sum(meta['num_cells'] for meta in valid_data.get('metadata', []))
            }
        }
        
        # Save split info
        split_path = self.processed_dir / "split_info.json"
        with open(split_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"✅ Split info saved to: {split_path}")
        return split_info
    
    def create_dataset_summary(self):
        """Create a summary of the processed dataset."""
        print("\n📋 Creating dataset summary...")
        
        # Count files
        image_files = list(self.processed_dir.glob("images/*.png"))
        mask_files = list(self.processed_dir.glob("masks/*.png"))
        
        # Load split info
        split_path = self.processed_dir / "split_info.json"
        split_info = {}
        if split_path.exists():
            with open(split_path, 'r') as f:
                split_info = json.load(f)
        
        summary = {
            'dataset_name': 'Revvity-25',
            'preprocessing': {
                'target_size': self.target_size,
                'normalization': '0-1 range',
                'format': 'PNG'
            },
            'statistics': {
                'total_images': len(image_files),
                'total_masks': len(mask_files),
                'splits': split_info
            },
            'file_structure': {
                'images': str(self.processed_dir / "images"),
                'masks': str(self.processed_dir / "masks"),
                'split_info': str(split_path)
            }
        }
        
        # Save summary
        summary_path = self.processed_dir / "dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Dataset summary saved to: {summary_path}")
        return summary
    
    def visualize_sample(self, split: str = 'train', sample_idx: int = 0):
        """Visualize a sample from the processed dataset."""
        print(f"\n🖼️  Visualizing sample {sample_idx} from {split}...")
        
        # Find sample files
        image_path = self.processed_dir / "images" / f"{split}_{sample_idx:03d}.png"
        mask_path = self.processed_dir / "masks" / f"{split}_{sample_idx:03d}.png"
        
        if not image_path.exists() or not mask_path.exists():
            print(f"❌ Sample files not found")
            return
        
        # Load image and mask
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title(f'{split} Image {sample_idx}')
        axes[0].axis('off')
        
        # Segmentation mask
        axes[1].imshow(mask, cmap='tab20')
        axes[1].set_title(f'Segmentation Mask ({mask.max()} cells)')
        axes[1].axis('off')
        
        # Overlay
        overlay = image.copy()
        overlay[mask > 0] = [255, 0, 0]  # Red overlay for cells
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.processed_dir / f"sample_visualization_{split}_{sample_idx}.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to: {viz_path}")
        
        plt.show()

def main():
    """Main preprocessing function."""
    print("🚀 Revvity-25 Dataset Preprocessing Pipeline")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = Revvity25Preprocessor(target_size=512)
    
    # Process the dataset
    split_info = preprocessor.create_train_valid_split()
    
    # Create dataset summary
    summary = preprocessor.create_dataset_summary()
    
    # Visualize a sample
    preprocessor.visualize_sample('train', 0)
    
    print("\n🎉 Dataset preprocessing completed!")
    print("\n📋 What we have:")
    print("✅ Processed images (512x512)")
    print("✅ Segmentation masks")
    print("✅ Train/validation split")
    print("✅ Dataset summary and statistics")
    print("✅ Sample visualizations")
    print("\n📁 Check data/revvity25/processed/ directory")

if __name__ == "__main__":
    main()
