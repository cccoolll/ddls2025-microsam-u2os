#!/usr/bin/env python3
"""
Simple microSAM fine-tuning script for U2OS cell segmentation.
This script loads data from the simple_finetuning folder and runs training.
"""

import asyncio
import json
import numpy as np
import os
import tempfile
import shutil
import torch
import imageio.v3 as imageio
from PIL import Image, ImageDraw

async def main():
    # Load the data from simple_finetuning folder
    data_dir = "/home/scheng/workspace/ddls2025-microsam-u2os/examples/simple_finetuning"
    
    # Load all 4 images
    images = []
    for i in range(4):
        image_path = os.path.join(data_dir, f"image_{i}.npy")
        if os.path.exists(image_path):
            image = np.load(image_path)
            images.append(image)
            print(f"Loaded image_{i} with shape: {image.shape}")
        else:
            print(f"Warning: image_{i}.npy not found")
    
    print(f"Loaded {len(images)} images total")
    
    # Load annotations
    annotations_path = os.path.join(data_dir, "annotations.json")
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    print(f"Loaded annotations with {len(annotations['images'])} images and {len(annotations['annotations'])} annotations")
    
    # Create a simple trainer class
    class SimpleMicroSamTrainer:
        def __init__(self):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model_type = "vit_b_lm"
            self.checkpoint_dir = "models/checkpoints"
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            print(f"Using device: {self.device}")
        
        def _prepare_training_data(self, images, annotations, temp_dir):
            """Convert COCO format data to .tif files for training."""
            
            # Create subdirectories
            images_dir = os.path.join(temp_dir, "images")
            labels_dir = os.path.join(temp_dir, "labels")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
            
            # Create image ID to image mapping
            image_id_to_image = {img['id']: img for img in annotations['images']}
            
            # Group annotations by image
            image_annotations = {}
            for ann in annotations['annotations']:
                image_id = ann['image_id']
                if image_id not in image_annotations:
                    image_annotations[image_id] = []
                image_annotations[image_id].append(ann)
            
            # Process each image
            for i, image in enumerate(images):
                if i >= len(annotations['images']):
                    break
                    
                img_meta = annotations['images'][i]
                image_id = img_meta['id']
                
                # Save image as .tif
                if len(image.shape) == 3 and image.shape[0] in [1, 3]:
                    if image.shape[0] == 1:
                        image_2d = image[0]  # (H, W)
                    else:
                        image_2d = np.transpose(image, (1, 2, 0))  # (H, W, C)
                else:
                    image_2d = image  # Assume already (H, W)
                
                # Normalize to 0-255 if needed
                if image_2d.max() <= 1.0:
                    image_2d = (image_2d * 255).astype(np.uint8)
                else:
                    image_2d = image_2d.astype(np.uint8)
                
                # Save image
                img_filename = f"train_{i:03d}.tif"
                imageio.imwrite(os.path.join(images_dir, img_filename), image_2d)
                
                # Create segmentation mask
                if image_id in image_annotations:
                    h, w = image_2d.shape[:2]
                    mask = np.zeros((h, w), dtype=np.uint16)
                    
                    for ann_idx, ann in enumerate(image_annotations[image_id]):
                        if 'segmentation' in ann and ann['segmentation']:
                            for seg in ann['segmentation']:
                                if len(seg) >= 6:  # At least 3 points
                                    pil_mask = Image.new('L', (w, h), 0)
                                    draw = ImageDraw.Draw(pil_mask)
                                    
                                    points = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
                                    draw.polygon(points, fill=ann_idx + 1)
                                    
                                    seg_mask = np.array(pil_mask)
                                    mask[seg_mask > 0] = ann_idx + 1
                
                # Save label
                label_filename = f"train_{i:03d}.tif"
                imageio.imwrite(os.path.join(labels_dir, label_filename), mask)
            
            return temp_dir
        
        async def train(self, images, annotations, n_epochs=10):
            """Run training without Ray Serve."""
            try:
                # Enforce minimum 2 images requirement
                if len(images) < 2:
                    raise ValueError(
                        f"Minimum 2 images required for proper train/validation split. "
                        f"Received {len(images)} image(s). Please upload at least 2 images."
                    )
                
                print(f"✅ Training with {len(images)} images - proper train/val split enabled")
                
                # Create temporary directory for training data
                temp_dir = tempfile.mkdtemp(prefix="microsam_training_")
                print(f"Created temp directory: {temp_dir}")
                
                # Prepare training data
                self._prepare_training_data(images, annotations, temp_dir)
                print("Prepared training data")
                
                # Create proper train/val split (80/20) - requires minimum 2 images
                images_dir = os.path.join(temp_dir, "images")
                labels_dir = os.path.join(temp_dir, "labels")
                
                import glob
                all_images = sorted(glob.glob(os.path.join(images_dir, "*.tif")))
                all_labels = sorted(glob.glob(os.path.join(labels_dir, "*.tif")))
                
                # Split into train/val (80/20)
                split_idx = int(0.8 * len(all_images))
                train_images = all_images[:split_idx]
                train_labels = all_labels[:split_idx]
                val_images = all_images[split_idx:]
                val_labels = all_labels[split_idx:]
                
                print(f"📊 Data split: {len(train_images)} train, {len(val_images)} validation")
                
                # Rename files for train/val split
                for i, (img_path, label_path) in enumerate(zip(train_images, train_labels)):
                    new_img_path = os.path.join(images_dir, f"train_{i:03d}.tif")
                    new_label_path = os.path.join(labels_dir, f"train_{i:03d}.tif")
                    os.rename(img_path, new_img_path)
                    os.rename(label_path, new_label_path)
                
                for i, (img_path, label_path) in enumerate(zip(val_images, val_labels)):
                    new_img_path = os.path.join(images_dir, f"valid_{i:03d}.tif")
                    new_label_path = os.path.join(labels_dir, f"valid_{i:03d}.tif")
                    os.rename(img_path, new_img_path)
                    os.rename(label_path, new_label_path)
                
                print("✅ Created proper train/validation split")
                
                # Import training modules - following the working pattern from train_revvity25_ais.py
                import micro_sam.training as sam_training
                from torch_em.data import MinInstanceSampler
                
                # Training configuration - optimized for memory
                batch_size = 1  # Reduced for memory efficiency
                n_objects_per_batch = 4  # Reduced for memory efficiency
                patch_shape = (512, 512)
                learning_rate = 1e-4
                freeze_encoder = True
                early_stopping = 15
                scheduler_patience = 5
                
                # Use MinInstanceSampler
                sampler = MinInstanceSampler(min_size=20)
                
                # Create dataloaders - following the working pattern
                train_loader = sam_training.default_sam_loader(
                    raw_paths=images_dir,
                    raw_key="train_*.tif",
                    label_paths=labels_dir,
                    label_key="train_*.tif",
                    with_segmentation_decoder=True,
                    patch_shape=(1, *patch_shape),
                    batch_size=batch_size,
                    shuffle=True,
                    sampler=sampler,
                    num_workers=4,
                    pin_memory=True,
                )
                
                val_loader = sam_training.default_sam_loader(
                    raw_paths=images_dir,
                    raw_key="valid_*.tif",
                    label_paths=labels_dir,
                    label_key="valid_*.tif",
                    with_segmentation_decoder=True,
                    patch_shape=(1, *patch_shape),
                    batch_size=batch_size,
                    shuffle=False,
                    sampler=sampler,
                    num_workers=4,
                    pin_memory=True,
                )
                
                print("Created data loaders")
                
                # Configure training parameters - following the working pattern
                freeze_parts = ["image_encoder"] if freeze_encoder else None
                scheduler_kwargs = {
                    "mode": "min",
                    "factor": 0.8,
                    "patience": scheduler_patience,
                    "min_lr": 1e-7,
                }
                
                # Train the model - following the working pattern
                checkpoint_name = "bioengine_shared"
                print(f"Starting training for {n_epochs} epochs...")
                
                sam_training.train_sam(
                    name=checkpoint_name,
                    save_root="models",
                    model_type=self.model_type,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    n_epochs=n_epochs,
                    n_objects_per_batch=n_objects_per_batch,
                    with_segmentation_decoder=True,
                    freeze=freeze_parts,
                    device=self.device,
                    lr=learning_rate,
                    early_stopping=early_stopping,
                    scheduler_kwargs=scheduler_kwargs,
                    save_every_kth_epoch=10,
                    n_sub_iteration=6,
                    mask_prob=0.6,
                    checkpoint_path=None,
                    overwrite_training=False,
                )
                
                print("Training completed successfully!")
                
            except Exception as e:
                print(f"Training failed: {str(e)}")
                import traceback
                traceback.print_exc()
                raise
            finally:
                # Clean up temporary files
                if 'temp_dir' in locals():
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    print(f"Cleaned up temp directory: {temp_dir}")
    
    # Create trainer instance
    trainer = SimpleMicroSamTrainer()
    
    # Start fine-tuning
    print("Starting fine-tuning...")
    print(f"✅ Using {len(images)} images for proper train/validation split")
    print(f"   Train/Val split: {int(0.8 * len(images))} train, {len(images) - int(0.8 * len(images))} validation")
    
    await trainer.train(
        images=images,  # List of 4 images
        annotations=annotations,  # COCO format annotations
        n_epochs=10  # Number of epochs for training
    )
    print("Training finished!")

if __name__ == "__main__":
    # Set CUDA memory optimization
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    asyncio.run(main())
