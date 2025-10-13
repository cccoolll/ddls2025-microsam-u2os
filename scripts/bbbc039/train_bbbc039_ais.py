#!/usr/bin/env python3
"""
Training script for microSAM AIS on BBBC039 dataset.
Continues training from existing checkpoint if available.
"""

import os
import numpy as np
import torch
from torch_em.data import MinInstanceSampler
import micro_sam.training as sam_training


def train_bbbc039_ais(
    data_dir="data/BBBC039/processed",
    save_root="models",
    model_type="vit_b_lm",
    n_epochs=50,
    n_objects_per_batch=8,
    batch_size=2,
    patch_shape=(512, 512),
    learning_rate=1e-4,
    early_stopping=15,
    freeze_encoder=True,
    use_mixed_precision=True,
    scheduler_patience=5,
    save_every_kth_epoch=10,
    continue_from_checkpoint="models/checkpoints/latest.pt",  # Continue from existing checkpoint
):
    """
    Train microSAM AIS on BBBC039 dataset with optimized parameters.
    
    Key features:
    - Continues training from existing checkpoint (latest.pt)
    - Creates BBBC039-specific checkpoint folder
    - Optimized for BBBC039 dataset characteristics
    - Freeze encoder to focus on AIS head
    - Mixed precision training for efficiency
    """
    print("🚀 Training microSAM AIS on BBBC039 (Continuing from checkpoint)")
    print("="*70)
    
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
    
    if len(train_files) == 0 or len(valid_files) == 0:
        print("❌ No training or validation files found. Please run preprocessing first.")
        return None
    
    # Use MinInstanceSampler with optimized parameters
    sampler = MinInstanceSampler(min_size=20)
    
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
    print(f"  Continue from: {continue_from_checkpoint}")
    
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
    checkpoint_name = "bbbc039_ais"
    
    # Check if we should continue from existing checkpoint
    resume_from_checkpoint = None
    if os.path.exists(continue_from_checkpoint):
        print(f"🔄 Found existing checkpoint: {continue_from_checkpoint}")
        print("   Continuing training from existing checkpoint...")
        resume_from_checkpoint = continue_from_checkpoint
    else:
        print(f"🆕 No existing checkpoint found at {continue_from_checkpoint}")
        print("   Starting fresh training...")
    
    # Create BBBC039-specific checkpoint directory
    bbbc039_checkpoint_dir = os.path.join(save_root, "checkpoints", checkpoint_name)
    os.makedirs(bbbc039_checkpoint_dir, exist_ok=True)
    print(f"📁 Checkpoint directory: {bbbc039_checkpoint_dir}")
    
    print(f"\n🏋️  Starting BBBC039 training...")
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
        checkpoint_path=resume_from_checkpoint,  # Continue from checkpoint if available
        overwrite_training=False,  # Don't overwrite if resuming
    )
    
    best_checkpoint = os.path.join(save_root, "checkpoints", checkpoint_name, "best.pt")
    latest_checkpoint = os.path.join(save_root, "checkpoints", checkpoint_name, "latest.pt")
    
    print(f"\n🎉 BBBC039 training complete!")
    print(f"   Best checkpoint: {best_checkpoint}")
    print(f"   Latest checkpoint: {latest_checkpoint}")
    
    # Print training summary
    print(f"\n📊 Training Summary:")
    print(f"   Dataset: BBBC039")
    print(f"   Model: {model_type}")
    print(f"   Total epochs: {n_epochs}")
    print(f"   Final learning rate: {learning_rate}")
    print(f"   Encoder frozen: {freeze_encoder}")
    print(f"   Continued from: {continue_from_checkpoint}")
    print(f"   Best checkpoint: {best_checkpoint}")
    print(f"   Latest checkpoint: {latest_checkpoint}")
    
    # Copy latest checkpoint to main checkpoint directory for next dataset
    if os.path.exists(latest_checkpoint):
        import shutil
        main_latest = os.path.join(save_root, "checkpoints", "latest.pt")
        shutil.copy2(latest_checkpoint, main_latest)
        print(f"   Updated main latest checkpoint: {main_latest}")
    
    return best_checkpoint


def evaluate_bbbc039_model(
    checkpoint_path="models/checkpoints/bbbc039_ais/best.pt",
    data_dir="data/BBBC039/processed",
    n_samples=5,
):
    """
    Quick evaluation of BBBC039 model performance.
    """
    print("🔍 Evaluating BBBC039 Model")
    print("="*50)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Load a few validation samples
    import glob
    valid_images = sorted(glob.glob(os.path.join(data_dir, "images", "valid_*.tif")))[:n_samples]
    valid_labels = sorted(glob.glob(os.path.join(data_dir, "labels", "valid_*.tif")))[:n_samples]
    
    print(f"📊 Evaluating on {len(valid_images)} validation samples...")
    
    # This is a placeholder for actual evaluation
    # In practice, you would load the model and run inference
    print("✅ Evaluation complete (placeholder)")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train microSAM AIS on BBBC039 (Continuing from checkpoint)")
    parser.add_argument("--data_dir", default="data/BBBC039/processed", help="Processed data directory")
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
    parser.add_argument("--continue_from", default="models/checkpoints/latest.pt", help="Checkpoint to continue from")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation after training")
    
    args = parser.parse_args()
    
    # Train the model
    best_checkpoint = train_bbbc039_ais(
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
        continue_from_checkpoint=args.continue_from,
    )
    
    # Run evaluation if requested
    if args.evaluate and best_checkpoint:
        evaluate_bbbc039_model(
            checkpoint_path=best_checkpoint,
            data_dir=args.data_dir,
        )
