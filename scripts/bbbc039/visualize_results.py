#!/usr/bin/env python3
"""
Enhanced evaluation script comparing pre-trained vs fine-tuned microSAM AIS on BBBC039.
Shows before/after fine-tuning results for clear comparison.
"""

import os
import numpy as np
import imageio.v3 as imageio
import matplotlib.pyplot as plt
from glob import glob
import torch

from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation


def run_automatic_instance_segmentation(image, checkpoint_path=None, model_type="vit_b_lm", device="cuda"):
    """
    Run AIS following the tutorial pattern.
    
    Args:
        image: Input image array
        checkpoint_path: Path to checkpoint (None for pre-trained model)
        model_type: Model type (default: "vit_b_lm")
        device: Device to run on
    """
    # Step 1: Get predictor and segmenter (from tutorial)
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type,
        checkpoint=checkpoint_path,  # None will use pre-trained model
        device=device,
        is_tiled=False,
    )
    
    # Step 2: Run automatic instance segmentation (from tutorial)
    prediction = automatic_instance_segmentation(
        predictor=predictor,
        segmenter=segmenter,
        input_path=image,
        ndim=2,
    )
    
    return prediction


def evaluate_bbbc039_models_comparison(
    finetuned_checkpoint_path="models/checkpoints/bbbc039_ais/best.pt",
    data_dir="data/BBBC039/processed",
    output_dir="results/bbbc039_visualizations",
    n_samples=6,
    pretrained_model_type="vit_b_lm",
):
    """
    Compare pre-trained vs fine-tuned model predictions on BBBC039.
    
    This shows the improvement achieved through fine-tuning on BBBC039!
    """
    print("🔍 Comparing Pre-trained vs Fine-tuned microSAM AIS on BBBC039")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    
    # Check fine-tuned checkpoint
    if not os.path.exists(finetuned_checkpoint_path):
        print(f"❌ Fine-tuned checkpoint not found: {finetuned_checkpoint_path}")
        print("   Please train the model first using train_bbbc039_ais.py")
        return
    
    print(f"✅ Fine-tuned checkpoint found: {finetuned_checkpoint_path}")
    print(f"   Size: {os.path.getsize(finetuned_checkpoint_path) / (1024**2):.1f} MB")
    print(f"✅ Pre-trained model: {pretrained_model_type} (will be downloaded if needed)")
    
    # Get validation images
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")
    
    valid_images = sorted(glob(os.path.join(images_dir, "valid_*.tif")))[:n_samples]
    valid_labels = sorted(glob(os.path.join(labels_dir, "valid_*.tif")))[:n_samples]
    
    if len(valid_images) == 0:
        print(f"❌ No validation images found in {images_dir}")
        print("   Please run preprocessing first using preprocess_bbbc039.py")
        return
    
    print(f"\n📊 Processing {len(valid_images)} BBBC039 validation images...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    results = []
    for i, (img_path, label_path) in enumerate(zip(valid_images, valid_labels)):
        print(f"\n[{i+1}/{len(valid_images)}] {os.path.basename(img_path)}")
        
        # Load image and ground truth
        image = imageio.imread(img_path)
        gt_mask = imageio.imread(label_path)
        
        # Run pre-trained model prediction
        try:
            print("  🔄 Running pre-trained model...")
            pretrained_pred = run_automatic_instance_segmentation(
                image=image,
                checkpoint_path=None,  # None = use pre-trained model
                model_type=pretrained_model_type,
                device=device,
            )
            n_pretrained = pretrained_pred.max()
            print(f"    Pre-trained: {n_pretrained} cells")
        except Exception as e:
            print(f"    ❌ Pre-trained model error: {e}")
            continue
        
        # Run fine-tuned model prediction
        try:
            print("  🔄 Running fine-tuned model...")
            finetuned_pred = run_automatic_instance_segmentation(
                image=image,
                checkpoint_path=finetuned_checkpoint_path,
                model_type=pretrained_model_type,  # Same architecture
                device=device,
            )
            n_finetuned = finetuned_pred.max()
            print(f"    Fine-tuned: {n_finetuned} cells")
        except Exception as e:
            print(f"    ❌ Fine-tuned model error: {e}")
            continue
        
        # Ground truth count
        n_gt = gt_mask.max()
        print(f"    Ground truth: {n_gt} cells")
        
        # Calculate improvement
        improvement = n_finetuned - n_pretrained
        improvement_pct = (improvement / max(n_pretrained, 1)) * 100 if n_pretrained > 0 else 0
        
        print(f"    📈 Improvement: {improvement:+d} cells ({improvement_pct:+.1f}%)")
        
        results.append({
            'image': image,
            'gt': gt_mask,
            'pretrained_pred': pretrained_pred,
            'finetuned_pred': finetuned_pred,
            'name': os.path.basename(img_path),
            'n_gt': n_gt,
            'n_pretrained': n_pretrained,
            'n_finetuned': n_finetuned,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
        })
    
    # Create visualization
    if results:
        print(f"\n📊 Creating BBBC039 comparison visualization...")
        create_bbbc039_comparison_visualization(results, output_dir)
        print(f"✅ Results saved to: {output_dir}")
        
        # Print summary statistics
        print_bbbc039_summary_stats(results)
    
    return results


def create_bbbc039_comparison_visualization(results, output_dir):
    """Create side-by-side comparison visualization for BBBC039."""
    n_samples = len(results)
    fig, axes = plt.subplots(n_samples, 4, figsize=(20, 5*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        # Original image
        axes[i, 0].imshow(result['image'], cmap='gray')
        axes[i, 0].set_title(f"BBBC039 Image: {result['name']}", fontsize=12)
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(result['gt'], cmap='tab20')
        axes[i, 1].set_title(f"Ground Truth\n({result['n_gt']} cells)", fontsize=12)
        axes[i, 1].axis('off')
        
        # Pre-trained prediction
        axes[i, 2].imshow(result['pretrained_pred'], cmap='tab20')
        axes[i, 2].set_title(f"Pre-trained\n({result['n_pretrained']} cells)", 
                            fontsize=12, color='red')
        axes[i, 2].axis('off')
        
        # Fine-tuned prediction
        axes[i, 3].imshow(result['finetuned_pred'], cmap='tab20')
        improvement_text = f"({result['improvement']:+d})" if result['improvement'] != 0 else ""
        axes[i, 3].set_title(f"Fine-tuned\n({result['n_finetuned']} cells {improvement_text})", 
                            fontsize=12, color='green')
        axes[i, 3].axis('off')
    
    plt.suptitle('BBBC039 Dataset: Pre-trained vs Fine-tuned microSAM AIS Comparison', 
                 fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bbbc039_pretrained_vs_finetuned_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create a summary plot showing improvement statistics
    create_bbbc039_improvement_summary_plot(results, output_dir)


def create_bbbc039_improvement_summary_plot(results, output_dir):
    """Create a summary plot showing improvement statistics for BBBC039."""
    if not results:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Cell count comparison
    sample_names = [f"BBBC039 Sample {i+1}" for i in range(len(results))]
    gt_counts = [r['n_gt'] for r in results]
    pretrained_counts = [r['n_pretrained'] for r in results]
    finetuned_counts = [r['n_finetuned'] for r in results]
    
    x = np.arange(len(sample_names))
    width = 0.25
    
    ax1.bar(x - width, gt_counts, width, label='Ground Truth', alpha=0.8, color='blue')
    ax1.bar(x, pretrained_counts, width, label='Pre-trained', alpha=0.8, color='red')
    ax1.bar(x + width, finetuned_counts, width, label='Fine-tuned', alpha=0.8, color='green')
    
    ax1.set_xlabel('BBBC039 Samples')
    ax1.set_ylabel('Number of Cells')
    ax1.set_title('BBBC039 Cell Count Comparison: Pre-trained vs Fine-tuned')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sample_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Improvement percentage
    improvements = [r['improvement_pct'] for r in results]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    
    bars = ax2.bar(sample_names, improvements, color=colors, alpha=0.7)
    ax2.set_xlabel('BBBC039 Samples')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('BBBC039 Fine-tuning Improvement by Sample')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                f'{imp:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
    
    plt.suptitle('BBBC039 Dataset: Fine-tuning Performance Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bbbc039_improvement_summary.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()


def print_bbbc039_summary_stats(results):
    """Print summary statistics for BBBC039 evaluation."""
    if not results:
        return
    
    print(f"\n📈 BBBC039 Summary Statistics:")
    print(f"   Total samples: {len(results)}")
    
    # Average counts
    avg_gt = np.mean([r['n_gt'] for r in results])
    avg_pretrained = np.mean([r['n_pretrained'] for r in results])
    avg_finetuned = np.mean([r['n_finetuned'] for r in results])
    
    print(f"   Average Ground Truth: {avg_gt:.1f} cells")
    print(f"   Average Pre-trained: {avg_pretrained:.1f} cells")
    print(f"   Average Fine-tuned: {avg_finetuned:.1f} cells")
    
    # Improvement statistics
    improvements = [r['improvement'] for r in results]
    improvement_pcts = [r['improvement_pct'] for r in results]
    
    print(f"\n🎯 BBBC039 Fine-tuning Improvements:")
    print(f"   Average improvement: {np.mean(improvements):+.1f} cells")
    print(f"   Average improvement %: {np.mean(improvement_pcts):+.1f}%")
    print(f"   Best improvement: {max(improvements):+d} cells ({max(improvement_pcts):+.1f}%)")
    print(f"   Worst case: {min(improvements):+d} cells ({min(improvement_pcts):+.1f}%)")
    
    # Accuracy metrics
    pretrained_errors = [abs(r['n_pretrained'] - r['n_gt']) for r in results]
    finetuned_errors = [abs(r['n_finetuned'] - r['n_gt']) for r in results]
    
    print(f"\n🎯 BBBC039 Accuracy Metrics:")
    print(f"   Pre-trained MAE: {np.mean(pretrained_errors):.1f} cells")
    print(f"   Fine-tuned MAE: {np.mean(finetuned_errors):.1f} cells")
    print(f"   Accuracy improvement: {np.mean(pretrained_errors) - np.mean(finetuned_errors):.1f} cells")
    
    # Dataset-specific insights
    print(f"\n🔬 BBBC039 Dataset Insights:")
    print(f"   Dataset: BBBC039 (Microscopy cell segmentation)")
    print(f"   Model: microSAM AIS (Auto Instance Segmentation)")
    print(f"   Training: Continued from existing checkpoint")
    print(f"   Focus: Zero-prompt cell segmentation")


def evaluate_bbbc039_model_only(
    checkpoint_path="models/checkpoints/bbbc039_ais/best.pt",
    data_dir="data/BBBC039/processed",
    output_dir="results/bbbc039_visualizations",
    n_samples=6,
):
    """
    Evaluate only the fine-tuned BBBC039 model (without comparison).
    """
    print("🔍 Evaluating BBBC039 Fine-tuned Model")
    print("="*60)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Get validation images
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")
    
    valid_images = sorted(glob(os.path.join(images_dir, "valid_*.tif")))[:n_samples]
    valid_labels = sorted(glob(os.path.join(labels_dir, "valid_*.tif")))[:n_samples]
    
    if len(valid_images) == 0:
        print(f"❌ No validation images found in {images_dir}")
        return
    
    print(f"📊 Processing {len(valid_images)} BBBC039 validation images...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for i, (img_path, label_path) in enumerate(zip(valid_images, valid_labels)):
        print(f"\n[{i+1}/{len(valid_images)}] {os.path.basename(img_path)}")
        
        # Load image and ground truth
        image = imageio.imread(img_path)
        gt_mask = imageio.imread(label_path)
        
        # Run fine-tuned model prediction
        try:
            print("  🔄 Running fine-tuned model...")
            finetuned_pred = run_automatic_instance_segmentation(
                image=image,
                checkpoint_path=checkpoint_path,
                model_type="vit_b_lm",
                device=device,
            )
            n_finetuned = finetuned_pred.max()
            n_gt = gt_mask.max()
            print(f"    Fine-tuned: {n_finetuned} cells")
            print(f"    Ground truth: {n_gt} cells")
            print(f"    Error: {abs(n_finetuned - n_gt)} cells")
            
            results.append({
                'image': image,
                'gt': gt_mask,
                'finetuned_pred': finetuned_pred,
                'name': os.path.basename(img_path),
                'n_gt': n_gt,
                'n_finetuned': n_finetuned,
                'error': abs(n_finetuned - n_gt),
            })
            
        except Exception as e:
            print(f"    ❌ Fine-tuned model error: {e}")
            continue
    
    if results:
        print(f"\n📊 BBBC039 Fine-tuned Model Results:")
        avg_error = np.mean([r['error'] for r in results])
        print(f"   Average error: {avg_error:.1f} cells")
        print(f"   Total samples: {len(results)}")
    
    return results


# Keep the original function for backward compatibility
def evaluate_finetuned_model(
    checkpoint_path="models/checkpoints/bbbc039_ais/best.pt",
    data_dir="data/BBBC039/processed",
    output_dir="results/bbbc039_visualizations",
    n_samples=6,
):
    """
    Original function - now calls the comparison function.
    """
    return evaluate_bbbc039_models_comparison(
        finetuned_checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        output_dir=output_dir,
        n_samples=n_samples,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate BBBC039 microSAM AIS model")
    parser.add_argument("--checkpoint", default="models/checkpoints/bbbc039_ais/best.pt", 
                       help="Fine-tuned checkpoint path")
    parser.add_argument("--data_dir", default="data/BBBC039/processed", help="Data directory")
    parser.add_argument("--output_dir", default="results/bbbc039_visualizations", help="Output directory")
    parser.add_argument("--n_samples", type=int, default=6, help="Number of samples to evaluate")
    parser.add_argument("--pretrained_model", default="vit_b_lm", 
                       help="Pre-trained model type (default: vit_b_lm)")
    parser.add_argument("--compare", action="store_true", default=True,
                       help="Compare pre-trained vs fine-tuned (default: True)")
    parser.add_argument("--no_compare", action="store_false", dest="compare",
                       help="Evaluate only fine-tuned model")
    
    args = parser.parse_args()
    
    if args.compare:
        evaluate_bbbc039_models_comparison(
            finetuned_checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            n_samples=args.n_samples,
            pretrained_model_type=args.pretrained_model,
        )
    else:
        evaluate_bbbc039_model_only(
            checkpoint_path=args.checkpoint,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            n_samples=args.n_samples,
        )
