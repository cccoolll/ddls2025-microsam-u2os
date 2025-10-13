#!/usr/bin/env python3
"""
Simple training script for microSAM AIS following the tutorial pattern.
"""

import os
import numpy as np
import torch
from torch_em.data import MinInstanceSampler
import micro_sam.training as sam_training


def train_revvity25_ais(
    data_dir="data/Revvity-25/processed",
    save_root="models",
    model_type="vit_b_lm",
    n_epochs=10,
    n_objects_per_batch=5,
    batch_size=1,
    patch_shape=(512, 512),
):
    """
    Train microSAM AIS on Revvity-25 dataset.
    
    Following micro-sam tutorial pattern - simple and clean!
    """
    print("🚀 Training microSAM AIS on Revvity-25")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Setup paths (following tutorial pattern)
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")
    
    # Define ROIs for train/val split (following tutorial)
    # Count files to determine split
    import glob
    train_files = sorted(glob.glob(os.path.join(images_dir, "train_*.tif")))
    valid_files = sorted(glob.glob(os.path.join(images_dir, "valid_*.tif")))
    
    print(f"Train images: {len(train_files)}")
    print(f"Valid images: {len(valid_files)}")
    
    # Use MinInstanceSampler to ensure we sample patches with objects
    sampler = MinInstanceSampler(min_size=25)
    
    # Create dataloaders using micro-sam's default_sam_loader (from tutorial)
    train_loader = sam_training.default_sam_loader(
        raw_paths=images_dir,
        raw_key="train_*.tif",  # Pattern to match train images
        label_paths=labels_dir,
        label_key="train_*.tif",  # Pattern to match train labels
        with_segmentation_decoder=True,  # Enable AIS training
        patch_shape=(1, *patch_shape),  # (channels, height, width)
        batch_size=batch_size,
        shuffle=True,
        sampler=sampler,
    )
    
    val_loader = sam_training.default_sam_loader(
        raw_paths=images_dir,
        raw_key="valid_*.tif",
        label_paths=labels_dir,
        label_key="valid_*.tif",
        with_segmentation_decoder=True,
        patch_shape=(1, *patch_shape),
        batch_size=batch_size,
        shuffle=True,
        sampler=sampler,
    )
    
    print(f"\nTraining configuration:")
    print(f"  Model: {model_type}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Objects per batch: {n_objects_per_batch}")
    print(f"  Patch shape: {patch_shape}")
    print(f"  AIS training: True")
    
    # Train the model (following tutorial - simple!)
    checkpoint_name = "revvity25_ais"
    
    print(f"\n🏋️  Starting training...")
    sam_training.train_sam(
        name=checkpoint_name,
        save_root=save_root,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        n_objects_per_batch=n_objects_per_batch,
        with_segmentation_decoder=True,  # Enable AIS
        device=device,
    )
    
    best_checkpoint = os.path.join(save_root, "checkpoints", checkpoint_name, "best.pt")
    print(f"\n🎉 Training complete!")
    print(f"   Best checkpoint: {best_checkpoint}")
    
    return best_checkpoint


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train microSAM AIS on Revvity-25")
    parser.add_argument("--data_dir", default="data/Revvity-25/processed", help="Processed data directory")
    parser.add_argument("--save_root", default="models", help="Save directory for checkpoints")
    parser.add_argument("--model_type", default="vit_b_lm", help="Model type")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--n_objects", type=int, default=5, help="Objects per batch")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--patch_shape", type=int, default=512, help="Patch shape (square)")
    
    args = parser.parse_args()
    
    train_revvity25_ais(
        data_dir=args.data_dir,
        save_root=args.save_root,
        model_type=args.model_type,
        n_epochs=args.n_epochs,
        n_objects_per_batch=args.n_objects,
        batch_size=args.batch_size,
        patch_shape=(args.patch_shape, args.patch_shape),
    )
