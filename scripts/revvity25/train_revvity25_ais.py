#!/usr/bin/env python3
"""
Simple training script for microSAM AIS following the tutorial pattern.
"""

import os
import numpy as np
import torch
from torch_em.data import MinInstanceSampler
import micro_sam.training as sam_training
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from micro_sam.training.joint_sam_trainer import JointSamLogger


class MetricsTrackingLogger(JointSamLogger):
    """
    Custom logger that tracks training metrics and saves plots after each epoch.
    
    This logger extends JointSamLogger to:
    1. Accumulate batch-level metrics during training
    2. Compute epoch-level averages
    3. Generate and save visualization plots after each validation
    
    For beginners: This class "wraps" the existing logger to add plotting functionality.
    It intercepts the metric logging calls and stores them for visualization.
    """
    
    def __init__(self, trainer, save_root, plot_dir=None, **kwargs):
        """
        Initialize the metrics tracking logger.
        
        Args:
            trainer: The trainer object
            save_root: Root directory for saving logs
            plot_dir: Directory to save plots (if None, creates 'results/{trainer.name}/')
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(trainer, save_root, **kwargs)
        
        # Create plot directory
        if plot_dir is None:
            plot_dir = os.path.join("results", trainer.name)
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # Storage for epoch-level metrics
        # These lists will grow by 1 element per epoch
        self.train_metrics = {
            'epoch': [],           # Epoch number
            'loss': [],            # Total loss
            'mask_loss': [],       # Mask segmentation loss
            'iou_loss': [],        # IoU regression loss
            'model_iou': [],       # Predicted IoU score
            'lr': [],              # Learning rate
            'instance_loss': []    # AIS decoder loss
        }
        self.val_metrics = {
            'epoch': [],
            'loss': [],
            'mask_loss': [],
            'iou_loss': [],
            'model_iou': [],
            'instance_loss': []
        }
        
        self.current_epoch = 0
        # Temporary storage for batch-level metrics within an epoch
        self.train_batch_metrics = []
        self.val_batch_metrics = []
        
        print(f"📊 Metrics tracking enabled. Plots will be saved to: {self.plot_dir}")
    
    def log_train(self, step, loss, lr, x, y, samples, mask_loss, iou_loss, model_iou, instance_loss):
        """
        Called after each training batch to log metrics.
        
        For beginners: This method is automatically called by the trainer during training.
        We intercept it to store the metrics for later plotting.
        """
        # Call parent to maintain TensorBoard logging
        super().log_train(step, loss, lr, x, y, samples, mask_loss, iou_loss, model_iou, instance_loss)
        
        # Accumulate batch metrics (convert tensors to Python floats)
        self.train_batch_metrics.append({
            'loss': float(loss) if torch.is_tensor(loss) else loss,
            'mask_loss': float(mask_loss) if torch.is_tensor(mask_loss) else mask_loss,
            'iou_loss': float(iou_loss) if torch.is_tensor(iou_loss) else iou_loss,
            'model_iou': float(model_iou) if torch.is_tensor(model_iou) else model_iou,
            'lr': float(lr) if torch.is_tensor(lr) else lr,
            'instance_loss': float(instance_loss) if torch.is_tensor(instance_loss) else instance_loss
        })
    
    def log_validation(self, step, metric, loss, x, y, samples, mask_loss, iou_loss, model_iou, instance_loss):
        """
        Called after validation completes for an epoch.
        
        For beginners: This is called once per epoch after all validation batches.
        This is where we compute epoch averages and generate plots.
        """
        # Call parent
        super().log_validation(step, metric, loss, x, y, samples, mask_loss, iou_loss, model_iou, instance_loss)
        
        # Store validation metrics (these are already averaged by the trainer)
        self.val_metrics['epoch'].append(self.current_epoch)
        self.val_metrics['loss'].append(float(loss) if torch.is_tensor(loss) else loss)
        self.val_metrics['mask_loss'].append(float(mask_loss) if torch.is_tensor(mask_loss) else mask_loss)
        self.val_metrics['iou_loss'].append(float(iou_loss) if torch.is_tensor(iou_loss) else iou_loss)
        self.val_metrics['model_iou'].append(float(model_iou) if torch.is_tensor(model_iou) else model_iou)
        self.val_metrics['instance_loss'].append(float(instance_loss) if torch.is_tensor(instance_loss) else instance_loss)
        
        # Compute epoch averages from training batches and save
        self._save_epoch_metrics()
        
        # Generate and save plots
        self._plot_training_curves()
        
        # Reset batch accumulators for next epoch
        self.train_batch_metrics = []
        self.val_batch_metrics = []
        self.current_epoch += 1
    
    def _save_epoch_metrics(self):
        """
        Compute epoch-level averages from batch metrics.
        
        For beginners: During training, we collect metrics from many batches.
        This function averages them to get one number per epoch.
        """
        if not self.train_batch_metrics:
            return
        
        # Average all metrics across batches in this epoch
        train_avg = {
            k: np.mean([batch[k] for batch in self.train_batch_metrics])
            for k in ['loss', 'mask_loss', 'iou_loss', 'model_iou', 'instance_loss']
        }
        # Learning rate: use the last value from the epoch
        train_avg['lr'] = self.train_batch_metrics[-1]['lr']
        
        # Store epoch-level metrics
        self.train_metrics['epoch'].append(self.current_epoch)
        for k, v in train_avg.items():
            self.train_metrics[k].append(v)
    
    def _plot_training_curves(self):
        """
        Generate and save training visualization plots.
        
        For beginners: This creates a 2x2 grid of plots showing:
        - How loss decreases over time (we want this to go down)
        - How IoU increases over time (we want this to go up)
        - How learning rate changes (shows scheduler behavior)
        """
        if not self.train_metrics['epoch']:
            return
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Training Progress - Epoch {self.current_epoch}', fontsize=16, fontweight='bold')
        
        epochs = self.train_metrics['epoch']
        
        # Plot 1: Total Loss (Top-Left)
        # Lower is better - shows if model is learning
        ax = axes[0, 0]
        ax.plot(epochs, self.train_metrics['loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
        if self.val_metrics['loss']:
            ax.plot(self.val_metrics['epoch'], self.val_metrics['loss'], 'r-s', 
                   label='Val Loss', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Model IoU (Top-Right)
        # Higher is better - shows segmentation quality
        ax = axes[0, 1]
        ax.plot(epochs, self.train_metrics['model_iou'], 'b-o', label='Train IoU', linewidth=2, markersize=4)
        if self.val_metrics['model_iou']:
            ax.plot(self.val_metrics['epoch'], self.val_metrics['model_iou'], 'r-s', 
                   label='Val IoU', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('IoU Score', fontsize=11)
        ax.set_title('Model IoU Score', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Mask Loss (Bottom-Left)
        # Shows how well the model segments masks
        ax = axes[1, 0]
        ax.plot(epochs, self.train_metrics['mask_loss'], 'b-o', label='Train Mask Loss', linewidth=2, markersize=4)
        if self.val_metrics['mask_loss']:
            ax.plot(self.val_metrics['epoch'], self.val_metrics['mask_loss'], 'r-s', 
                   label='Val Mask Loss', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Mask Loss', fontsize=11)
        ax.set_title('Mask Segmentation Loss', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Learning Rate (Bottom-Right)
        # Shows how learning rate changes (scheduler behavior)
        ax = axes[1, 1]
        ax.plot(epochs, self.train_metrics['lr'], 'g-o', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Learning Rate', fontsize=11)
        ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        ax.set_yscale('log')  # Log scale to see small changes
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save with epoch number in filename
        plot_path = os.path.join(self.plot_dir, f'training_curves_epoch_{self.current_epoch:03d}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        
        # Also save as "latest" for easy viewing during training
        latest_path = os.path.join(self.plot_dir, 'training_curves_latest.png')
        plt.savefig(latest_path, dpi=150, bbox_inches='tight')
        
        plt.close(fig)
        
        print(f"📈 Saved training plots: {plot_path}")


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
    
    # Create results directory for plots
    plot_dir = os.path.join("results", checkpoint_name)
    os.makedirs(plot_dir, exist_ok=True)
    print(f"📊 Training plots will be saved to: {plot_dir}")
    
    # Check if we should resume from existing checkpoint
    latest_checkpoint = os.path.join(save_root, "checkpoints", checkpoint_name, "latest.pt")
    resume_from_checkpoint = None
    if os.path.exists(latest_checkpoint):
        print(f"🔄 Found existing checkpoint: {latest_checkpoint}")
        print("   Resuming training from latest checkpoint...")
        resume_from_checkpoint = latest_checkpoint
    else:
        print(f"🆕 No existing checkpoint found, starting fresh training...")
    
    # Monkey-patch the JointSamLogger to use our custom logger
    # For beginners: This temporarily replaces the default logger with our custom one
    # that adds plotting functionality
    from micro_sam.training import joint_sam_trainer
    original_logger = joint_sam_trainer.JointSamLogger
    
    # Create a wrapper that passes plot_dir to our custom logger
    def custom_logger_factory(trainer, save_root, **kwargs):
        return MetricsTrackingLogger(trainer, save_root, plot_dir=plot_dir, **kwargs)
    
    joint_sam_trainer.JointSamLogger = custom_logger_factory
    
    try:
        print(f"\n🏋️  Starting optimized training with metrics tracking...")
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
            checkpoint_path=resume_from_checkpoint,  # Resume from checkpoint if available
            overwrite_training=False,  # Don't overwrite if resuming
        )
    finally:
        # Restore original logger
        # For beginners: This ensures we don't affect other code that might use the logger
        joint_sam_trainer.JointSamLogger = original_logger
    
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
    print(f"   Training plots saved to: {plot_dir}")
    print(f"   Latest plot: {os.path.join(plot_dir, 'training_curves_latest.png')}")
    
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
