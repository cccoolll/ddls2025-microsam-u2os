#!/usr/bin/env python3
"""
Training script for microSAM AIS (Auto Instance Segmentation) head on Revvity-25 dataset.
This script focuses on fine-tuning the Segmentation Decoder for zero-prompt cell segmentation.
"""

import os
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch_em.transform.label import PerObjectDistanceTransform
import imageio.v3 as imageio

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model
from micro_sam.training.util import identity


class Revvity25Dataset(Dataset):
    """Dataset class for Revvity-25 brightfield microscopy images."""
    
    def __init__(self, data_dir, split="train", transform=None, label_transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Path to the processed dataset directory
            split: 'train' or 'valid'
            transform: Transform to apply to images
            label_transform: Transform to apply to labels
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.label_transform = label_transform
        
        # Get list of processed image files
        images_dir = self.data_dir / "processed" / "images"
        if split == "train":
            self.image_files = [f for f in os.listdir(images_dir) if f.startswith("train_") and f.endswith(".png")]
        else:
            self.image_files = [f for f in os.listdir(images_dir) if f.startswith("valid_") and f.endswith(".png")]
        
        print(f"📊 Loaded {len(self.image_files)} {split} images")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.data_dir / "processed" / "images" / self.image_files[idx]
        image = imageio.imread(img_path)
        
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        
        # Load corresponding mask
        mask_path = self.data_dir / "processed" / "masks" / self.image_files[idx]
        mask = imageio.imread(mask_path)
        
        # Convert to torch tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # HWC -> CHW
        mask = torch.from_numpy(mask).long()
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.label_transform:
            mask = self.label_transform(mask)
        
        return image, mask


def get_dataloaders(data_dir, patch_shape=(512, 512), batch_size=2, num_workers=4):
    """
    Create data loaders for Revvity-25 dataset.
    
    Args:
        data_dir: Path to the dataset directory
        patch_shape: Shape of patches for training
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
    
    Returns:
        train_loader, val_loader: Data loaders for training and validation
    """
    # Label transform for distance-based segmentation
    label_transform = PerObjectDistanceTransform(
        distances=True, 
        boundary_distances=True, 
        directed_distances=False, 
        foreground=True, 
        instances=True, 
        min_size=25
    )
    
    # Raw transform (identity for now, can be customized)
    raw_transform = identity
    
    # Create datasets
    train_dataset = Revvity25Dataset(
        data_dir=data_dir,
        split="train",
        transform=raw_transform,
        label_transform=label_transform
    )
    
    val_dataset = Revvity25Dataset(
        data_dir=data_dir,
        split="valid", 
        transform=raw_transform,
        label_transform=label_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    train_loader.shuffle = True  # Add shuffle attribute
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Use batch size 1 for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader.shuffle = False  # Add shuffle attribute
    
    return train_loader, val_loader


def finetune_revvity25_ais(args):
    """
    Fine-tune microSAM AIS head on Revvity-25 dataset.
    
    This function focuses on training the Auto Instance Segmentation (AIS) head
    for zero-prompt cell segmentation without user interaction.
    """
    print("🚀 Starting microSAM AIS fine-tuning on Revvity-25 dataset")
    print(f"📁 Data directory: {args.data_dir}")
    print(f"🤖 Model type: {args.model_type}")
    print(f"💾 Save directory: {args.save_root}")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    
    if device == "cpu":
        print("⚠️  Warning: CUDA not available, training will be very slow!")
    
    # Training configuration
    model_type = args.model_type
    checkpoint_path = None  # Start from pre-trained weights
    patch_shape = (512, 512)  # Fixed patch size for U2OS cells
    n_objects_per_batch = args.n_objects
    freeze_parts = args.freeze  # Parts to freeze during training
    checkpoint_name = f"{args.model_type}/revvity25_ais"
    
    # Create data loaders
    print("📊 Creating data loaders...")
    train_loader, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        patch_shape=patch_shape,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Learning rate scheduler configuration
    scheduler_kwargs = {
        "mode": "min", 
        "factor": 0.9, 
        "patience": 10
    }
    
    print("🏋️  Starting training...")
    print(f"   - Training samples: {len(train_loader.dataset)}")
    print(f"   - Validation samples: {len(val_loader.dataset)}")
    print(f"   - Batch size: {args.batch_size}")
    print(f"   - Objects per batch: {n_objects_per_batch}")
    print(f"   - Learning rate: {args.learning_rate}")
    print(f"   - Max iterations: {args.iterations}")
    
    # Run AIS training (Auto Instance Segmentation with UNETR decoder)
    print("🤖 Starting AIS training with UNETR decoder...")
    
    training_kwargs = {
        "name": checkpoint_name,
        "model_type": model_type,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "early_stopping": 10,
        "n_objects_per_batch": n_objects_per_batch,
        "checkpoint_path": checkpoint_path,
        "freeze": freeze_parts,
        "device": device,
        "lr": args.learning_rate,
        "n_iterations": args.iterations,
        "save_root": args.save_root,
        "scheduler_kwargs": scheduler_kwargs,
    }
    
    # Only add save_every_kth_epoch if it's not None
    if args.save_every_kth_epoch is not None:
        training_kwargs["save_every_kth_epoch"] = args.save_every_kth_epoch
    
    # Use train_instance_segmentation for AIS training (includes UNETR decoder)
    sam_training.train_instance_segmentation(**training_kwargs)
    
    # Export the trained model
    if args.export_path is not None:
        print("📤 Exporting trained model...")
        checkpoint_path = os.path.join(
            "" if args.save_root is None else args.save_root, 
            "checkpoints", 
            checkpoint_name, 
            "best.pt"
        )
        export_custom_sam_model(
            checkpoint_path=checkpoint_path, 
            model_type=model_type, 
            save_path=args.export_path,
        )
        print(f"✅ Model exported to: {args.export_path}")
    
    print("🎉 Training completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune microSAM AIS head for Revvity-25 U2OS cell segmentation."
    )
    
    # Data arguments
    parser.add_argument(
        "--data_dir", "-d", 
        default="/home/scheng/workspace/ddls2025-microsam-u2os/data",
        help="Path to the Revvity-25 dataset directory"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_type", "-m", 
        default="vit_b_lm",
        choices=["vit_t", "vit_b", "vit_l", "vit_h", "vit_b_lm"],
        help="microSAM model type to use for fine-tuning"
    )
    
    # Training arguments
    parser.add_argument(
        "--save_root", "-s",
        default="/home/scheng/workspace/ddls2025-microsam-u2os/models",
        help="Directory to save checkpoints and logs"
    )
    
    parser.add_argument(
        "--iterations", "-i", 
        type=int, 
        default=10000,
        help="Number of training iterations (default: 10000)"
    )
    
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=2,
        help="Batch size for training (default: 2)"
    )
    
    parser.add_argument(
        "--learning_rate", "-lr",
        type=float,
        default=1e-5,
        help="Learning rate for training (default: 1e-5)"
    )
    
    parser.add_argument(
        "--n_objects", "-n",
        type=int,
        default=25,
        help="Number of objects per batch for training (default: 25)"
    )
    
    parser.add_argument(
        "--num_workers", "-w",
        type=int,
        default=4,
        help="Number of worker processes for data loading (default: 4)"
    )
    
    # Fine-tuning strategy arguments
    parser.add_argument(
        "--freeze", 
        type=str, 
        nargs="+", 
        default=None,
        help="Model parts to freeze during training (e.g., 'image_encoder' 'mask_decoder')"
    )
    
    parser.add_argument(
        "--lora_rank", 
        type=int, 
        default=None,
        help="LoRA rank for parameter-efficient fine-tuning (default: None, use full fine-tuning)"
    )
    
    # Export arguments
    parser.add_argument(
        "--export_path", "-e",
        help="Path to export the final trained model"
    )
    
    parser.add_argument(
        "--save_every_kth_epoch", 
        type=int, 
        default=None,
        help="Save checkpoint every k epochs (default: None, only save best and latest)"
    )
    
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_root, exist_ok=True)
    
    # Start training
    finetune_revvity25_ais(args)


if __name__ == "__main__":
    main()
