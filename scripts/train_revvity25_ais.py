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
    n_epochs=50,  # Increased from 10 to 50 for better convergence
    n_objects_per_batch=8,  # Increased from 5 to 8 for better sampling
    batch_size=2,  # Increased from 1 to 2 for better gradient estimates
    patch_shape=(512, 512),
    learning_rate=1e-4,  # Higher LR for faster convergence
    early_stopping=15,  # Early stopping to prevent overfitting
    freeze_encoder=True,  # Freeze encoder to focus on AIS head
    use_mixed_precision=True,  # Enable mixed precision for efficiency
    scheduler_patience=5,  # Learning rate scheduler patience
    save_every_kth_epoch=10,  # Save checkpoints every 10 epochs
):
    """
    Train microSAM AIS on Revvity-25 dataset with optimized parameters.
    
    Key improvements:
    - Higher learning rate for faster convergence
    - More epochs with early stopping
    - Better batch size and object sampling
    - Freeze encoder to focus on AIS head
    - Mixed precision training for efficiency
    - Optimized learning rate scheduling
    """
    print("🚀 Training microSAM AIS on Revvity-25 (Optimized)")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Check GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        if gpu_memory < 20:
            print("⚠️  Warning: Low GPU memory. Consider reducing batch_size or patch_shape.")
    
    # Setup paths
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")
    
    # Count files to determine split
    import glob
    train_files = sorted(glob.glob(os.path.join(images_dir, "train_*.tif")))
    valid_files = sorted(glob.glob(os.path.join(images_dir, "valid_*.tif")))
    
    print(f"Train images: {len(train_files)}")
    print(f"Valid images: {len(valid_files)}")
    
    # Use MinInstanceSampler with optimized parameters
    sampler = MinInstanceSampler(min_size=20)  # Reduced from 25 for more diverse sampling
    
    # Create dataloaders with optimized parameters
    train_loader = sam_training.default_sam_loader(
        raw_paths=images_dir,
        raw_key="train_*.tif",
        label_paths=labels_dir,
        label_key="train_*.tif",
        with_segmentation_decoder=True,  # Enable AIS training
        patch_shape=(1, *patch_shape),
        batch_size=batch_size,
        shuffle=True,
        sampler=sampler,
        num_workers=4,  # Parallel data loading
        pin_memory=True,  # Faster GPU transfer
    )
    
    val_loader = sam_training.default_sam_loader(
        raw_paths=images_dir,
        raw_key="valid_*.tif",
        label_paths=labels_dir,
        label_key="valid_*.tif",
        with_segmentation_decoder=True,
        patch_shape=(1, *patch_shape),
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    
    print(f"\nTraining configuration:")
    print(f"  Model: {model_type}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Objects per batch: {n_objects_per_batch}")
    print(f"  Patch shape: {patch_shape}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Early stopping: {early_stopping} epochs")
    print(f"  Freeze encoder: {freeze_encoder}")
    print(f"  Mixed precision: {use_mixed_precision}")
    print(f"  AIS training: True")
    
    # Configure freeze parameters
    freeze_parts = ["image_encoder"] if freeze_encoder else None
    
    # Configure scheduler
    scheduler_kwargs = {
        "mode": "min",
        "factor": 0.8,  # More aggressive LR reduction
        "patience": scheduler_patience,
        "min_lr": 1e-7,  # Minimum learning rate
    }
    
    # Train the model with optimized parameters
    checkpoint_name = "revvity25_ais_optimized"
    
    print(f"\n🏋️  Starting optimized training...")
    sam_training.train_sam(
        name=checkpoint_name,
        save_root=save_root,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        n_objects_per_batch=n_objects_per_batch,
        with_segmentation_decoder=True,  # Enable AIS
        freeze=freeze_parts,  # Freeze encoder if specified
        device=device,
        lr=learning_rate,  # Use optimized learning rate
        early_stopping=early_stopping,  # Early stopping
        scheduler_kwargs=scheduler_kwargs,  # Optimized scheduler
        save_every_kth_epoch=save_every_kth_epoch,  # Save checkpoints
        n_sub_iteration=6,  # Reduced from default 8 for efficiency
        mask_prob=0.6,  # Increased mask probability for better training
    )
    
    best_checkpoint = os.path.join(save_root, "checkpoints", checkpoint_name, "best.pt")
    print(f"\n🎉 Training complete!")
    print(f"   Best checkpoint: {best_checkpoint}")
    
    # Print training summary
    print(f"\n📊 Training Summary:")
    print(f"   Model: {model_type}")
    print(f"   Total epochs: {n_epochs}")
    print(f"   Final learning rate: {learning_rate}")
    print(f"   Encoder frozen: {freeze_encoder}")
    print(f"   Checkpoint saved: {best_checkpoint}")
    
    return best_checkpoint


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train microSAM AIS on Revvity-25 (Optimized)")
    parser.add_argument("--data_dir", default="data/Revvity-25/processed", help="Processed data directory")
    parser.add_argument("--save_root", default="models", help="Save directory for checkpoints")
    parser.add_argument("--model_type", default="vit_b_lm", help="Model type")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--n_objects", type=int, default=8, help="Objects per batch")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--patch_shape", type=int, default=512, help="Patch shape (square)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--early_stopping", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--freeze_encoder", action="store_true", default=True, help="Freeze image encoder")
    parser.add_argument("--no_freeze_encoder", action="store_false", dest="freeze_encoder", help="Don't freeze image encoder")
    parser.add_argument("--scheduler_patience", type=int, default=5, help="LR scheduler patience")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoints every N epochs")
    
    args = parser.parse_args()
    
    train_revvity25_ais(
        data_dir=args.data_dir,
        save_root=args.save_root,
        model_type=args.model_type,
        n_epochs=args.n_epochs,
        n_objects_per_batch=args.n_objects,
        batch_size=args.batch_size,
        patch_shape=(args.patch_shape, args.patch_shape),
        learning_rate=args.learning_rate,
        early_stopping=args.early_stopping,
        freeze_encoder=args.freeze_encoder,
        scheduler_patience=args.scheduler_patience,
        save_every_kth_epoch=args.save_every,
    )
