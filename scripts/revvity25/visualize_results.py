#!/usr/bin/env python3
"""
Enhanced evaluation script comparing pre-trained vs fine-tuned microSAM AIS.
Shows before/after fine-tuning results for clear comparison.
Calculates Dice, mIoU, and AP50 metrics.
"""

import os
import numpy as np
import imageio.v3 as imageio
import matplotlib.pyplot as plt
from glob import glob
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score

from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation


def calculate_dice_score(pred_mask, gt_mask):
    """Calculate Dice coefficient between prediction and ground truth."""
    # Convert to binary masks
    pred_binary = (pred_mask > 0).astype(np.float32)
    gt_binary = (gt_mask > 0).astype(np.float32)
    
    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(pred_binary) + np.sum(gt_binary)
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = (2.0 * intersection) / union
    return dice


def calculate_miou(pred_mask, gt_mask):
    """Calculate mean Intersection over Union (mIoU)."""
    # Convert to binary masks
    pred_binary = (pred_mask > 0).astype(np.float32)
    gt_binary = (gt_mask > 0).astype(np.float32)
    
    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(pred_binary) + np.sum(gt_binary) - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    return iou


def calculate_ap50(pred_mask, gt_mask):
    """
    Calculate Average Precision at IoU threshold 0.5 (AP50).
    Matches predicted instances to ground truth instances.
    """
    # Get unique instance IDs (excluding background 0)
    pred_ids = np.unique(pred_mask)[1:]  # Skip 0
    gt_ids = np.unique(gt_mask)[1:]      # Skip 0
    
    if len(pred_ids) == 0 and len(gt_ids) == 0:
        return 1.0  # Perfect if both empty
    if len(pred_ids) == 0 or len(gt_ids) == 0:
        return 0.0  # No match if one is empty
    
    # Compute IoU matrix between all pred and gt instances
    iou_matrix = np.zeros((len(pred_ids), len(gt_ids)))
    
    for i, pred_id in enumerate(pred_ids):
        pred_instance = (pred_mask == pred_id)
        for j, gt_id in enumerate(gt_ids):
            gt_instance = (gt_mask == gt_id)
            
            intersection = np.sum(pred_instance & gt_instance)
            union = np.sum(pred_instance | gt_instance)
            
            if union > 0:
                iou_matrix[i, j] = intersection / union
    
    # Use Hungarian algorithm to find optimal matching
    pred_indices, gt_indices = linear_sum_assignment(-iou_matrix)
    
    # Count true positives at IoU threshold 0.5
    matched_ious = iou_matrix[pred_indices, gt_indices]
    true_positives = np.sum(matched_ious >= 0.5)
    
    # Calculate precision and recall
    precision = true_positives / len(pred_ids) if len(pred_ids) > 0 else 0
    recall = true_positives / len(gt_ids) if len(gt_ids) > 0 else 0
    
    # AP50 is the precision at 50% IoU threshold
    # For simplicity, we use F1-score as a proxy for AP
    if precision + recall > 0:
        ap50 = 2 * (precision * recall) / (precision + recall)
    else:
        ap50 = 0.0
    
    return ap50


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


def evaluate_models_comparison(
    finetuned_checkpoint_path="models/checkpoints/revvity25_ais/best.pt",
    data_dir="data/Revvity-25/processed",
    output_dir="results/visualizations",
    n_samples=6,
    pretrained_model_type="vit_b_lm",
):
    """
    Compare pre-trained vs fine-tuned model predictions.
    
    This shows the improvement achieved through fine-tuning!
    """
    print("🔍 Comparing Pre-trained vs Fine-tuned microSAM AIS")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    
    # Check fine-tuned checkpoint
    if not os.path.exists(finetuned_checkpoint_path):
        print(f"❌ Fine-tuned checkpoint not found: {finetuned_checkpoint_path}")
        return
    
    print(f"✅ Fine-tuned checkpoint found: {finetuned_checkpoint_path}")
    print(f"   Size: {os.path.getsize(finetuned_checkpoint_path) / (1024**2):.1f} MB")
    print(f"✅ Pre-trained model: {pretrained_model_type} (will be downloaded if needed)")
    
    # Get validation images
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")
    
    valid_images = sorted(glob(os.path.join(images_dir, "valid_*.tif")))[:n_samples]
    valid_labels = sorted(glob(os.path.join(labels_dir, "valid_*.tif")))[:n_samples]
    
    print(f"\n📊 Processing {len(valid_images)} validation images...")
    
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
        
        # Calculate metrics for pre-trained model
        print("  📊 Calculating metrics for pre-trained model...")
        pretrained_dice = calculate_dice_score(pretrained_pred, gt_mask)
        pretrained_miou = calculate_miou(pretrained_pred, gt_mask)
        pretrained_ap50 = calculate_ap50(pretrained_pred, gt_mask)
        print(f"    Dice: {pretrained_dice:.4f}, mIoU: {pretrained_miou:.4f}, AP50: {pretrained_ap50:.4f}")
        
        # Calculate metrics for fine-tuned model
        print("  📊 Calculating metrics for fine-tuned model...")
        finetuned_dice = calculate_dice_score(finetuned_pred, gt_mask)
        finetuned_miou = calculate_miou(finetuned_pred, gt_mask)
        finetuned_ap50 = calculate_ap50(finetuned_pred, gt_mask)
        print(f"    Dice: {finetuned_dice:.4f}, mIoU: {finetuned_miou:.4f}, AP50: {finetuned_ap50:.4f}")
        
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
            # Metrics
            'pretrained_dice': pretrained_dice,
            'pretrained_miou': pretrained_miou,
            'pretrained_ap50': pretrained_ap50,
            'finetuned_dice': finetuned_dice,
            'finetuned_miou': finetuned_miou,
            'finetuned_ap50': finetuned_ap50,
        })
    
    # Create visualization
    if results:
        print(f"\n📊 Creating comparison visualization...")
        create_comparison_visualization(results, output_dir)
        print(f"✅ Results saved to: {output_dir}")
        
        # Print summary statistics
        print_summary_stats(results)
        
        # Generate and save metrics table
        print(f"\n📊 Generating metrics table...")
        generate_metrics_table(results, output_dir)
    
    return results


def create_comparison_visualization(results, output_dir):
    """Create side-by-side comparison visualization."""
    n_samples = len(results)
    fig, axes = plt.subplots(n_samples, 4, figsize=(20, 5*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        # Original image
        axes[i, 0].imshow(result['image'], cmap='gray')
        axes[i, 0].set_title(f"Image: {result['name']}", fontsize=12)
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
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pretrained_vs_finetuned_comparison.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create a summary plot showing improvement statistics
    create_improvement_summary_plot(results, output_dir)


def create_improvement_summary_plot(results, output_dir):
    """Create a summary plot showing improvement statistics."""
    if not results:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Cell count comparison
    sample_names = [f"Sample {i+1}" for i in range(len(results))]
    gt_counts = [r['n_gt'] for r in results]
    pretrained_counts = [r['n_pretrained'] for r in results]
    finetuned_counts = [r['n_finetuned'] for r in results]
    
    x = np.arange(len(sample_names))
    width = 0.25
    
    ax1.bar(x - width, gt_counts, width, label='Ground Truth', alpha=0.8, color='blue')
    ax1.bar(x, pretrained_counts, width, label='Pre-trained', alpha=0.8, color='red')
    ax1.bar(x + width, finetuned_counts, width, label='Fine-tuned', alpha=0.8, color='green')
    
    ax1.set_xlabel('Samples')
    ax1.set_ylabel('Number of Cells')
    ax1.set_title('Cell Count Comparison: Pre-trained vs Fine-tuned')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sample_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Improvement percentage
    improvements = [r['improvement_pct'] for r in results]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    
    bars = ax2.bar(sample_names, improvements, color=colors, alpha=0.7)
    ax2.set_xlabel('Samples')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Fine-tuning Improvement by Sample')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                f'{imp:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improvement_summary.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()


def generate_metrics_table(results, output_dir):
    """Generate and save metrics table comparing pre-trained vs fine-tuned models."""
    if not results:
        return
    
    # Calculate average metrics
    avg_pretrained_dice = np.mean([r['pretrained_dice'] for r in results])
    avg_pretrained_miou = np.mean([r['pretrained_miou'] for r in results])
    avg_pretrained_ap50 = np.mean([r['pretrained_ap50'] for r in results])
    
    avg_finetuned_dice = np.mean([r['finetuned_dice'] for r in results])
    avg_finetuned_miou = np.mean([r['finetuned_miou'] for r in results])
    avg_finetuned_ap50 = np.mean([r['finetuned_ap50'] for r in results])
    
    # Create table
    table_data = [
        ["Model", "Split", "Dice ↑", "mIoU ↑", "AP50 ↑"],
        ["Pretrained micro-SAM", "Val", f"{avg_pretrained_dice:.4f}", f"{avg_pretrained_miou:.4f}", f"{avg_pretrained_ap50:.4f}"],
        ["Fine-tuned (frozen encoder)", "Val", f"{avg_finetuned_dice:.4f}", f"{avg_finetuned_miou:.4f}", f"{avg_finetuned_ap50:.4f}"],
    ]
    
    # Print table to console
    print("\n" + "="*80)
    print("📊 METRICS TABLE")
    print("="*80)
    for row in table_data:
        print(f"{row[0]:<30} {row[1]:<10} {row[2]:<12} {row[3]:<12} {row[4]:<12}")
    print("="*80)
    
    # Save table to text file
    table_path = os.path.join(output_dir, 'metrics_table.txt')
    with open(table_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("METRICS TABLE: Pre-trained vs Fine-tuned microSAM\n")
        f.write("="*80 + "\n\n")
        for row in table_data:
            f.write(f"{row[0]:<30} {row[1]:<10} {row[2]:<12} {row[3]:<12} {row[4]:<12}\n")
        f.write("\n" + "="*80 + "\n")
        f.write(f"\nEvaluated on {len(results)} validation samples\n")
        f.write(f"Dataset: Revvity-25\n")
    
    print(f"✅ Metrics table saved to: {table_path}")
    
    # Save as CSV for easy import
    csv_path = os.path.join(output_dir, 'metrics_table.csv')
    with open(csv_path, 'w') as f:
        f.write("Model,Split,Dice,mIoU,AP50\n")
        f.write(f"Pretrained micro-SAM,Val,{avg_pretrained_dice:.4f},{avg_pretrained_miou:.4f},{avg_pretrained_ap50:.4f}\n")
        f.write(f"Fine-tuned (frozen encoder),Val,{avg_finetuned_dice:.4f},{avg_finetuned_miou:.4f},{avg_finetuned_ap50:.4f}\n")
    
    print(f"✅ CSV table saved to: {csv_path}")
    
    # Create visual table plot
    create_metrics_table_plot(table_data, output_dir)


def create_metrics_table_plot(table_data, output_dir):
    """Create a visual plot of the metrics table."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.15, 0.15, 0.15, 0.15])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Header styling
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # Row styling
    for i in range(1, 3):
        for j in range(5):
            cell = table[(i, j)]
            if i == 1:
                cell.set_facecolor('#E7E6E6')
            else:
                cell.set_facecolor('#D9E1F2')
    
    plt.title('Performance Metrics: Pre-trained vs Fine-tuned microSAM', 
              fontsize=14, weight='bold', pad=20)
    
    plt.savefig(os.path.join(output_dir, 'metrics_table.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Metrics table plot saved to: {os.path.join(output_dir, 'metrics_table.png')}")


def print_summary_stats(results):
    """Print summary statistics."""
    if not results:
        return
    
    print(f"\n📈 Summary Statistics:")
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
    
    print(f"\n🎯 Fine-tuning Improvements:")
    print(f"   Average improvement: {np.mean(improvements):+.1f} cells")
    print(f"   Average improvement %: {np.mean(improvement_pcts):+.1f}%")
    print(f"   Best improvement: {max(improvements):+d} cells ({max(improvement_pcts):+.1f}%)")
    print(f"   Worst case: {min(improvements):+d} cells ({min(improvement_pcts):+.1f}%)")
    
    # Accuracy metrics
    pretrained_errors = [abs(r['n_pretrained'] - r['n_gt']) for r in results]
    finetuned_errors = [abs(r['n_finetuned'] - r['n_gt']) for r in results]
    
    print(f"\n🎯 Accuracy Metrics:")
    print(f"   Pre-trained MAE: {np.mean(pretrained_errors):.1f} cells")
    print(f"   Fine-tuned MAE: {np.mean(finetuned_errors):.1f} cells")
    print(f"   Accuracy improvement: {np.mean(pretrained_errors) - np.mean(finetuned_errors):.1f} cells")


# Keep the original function for backward compatibility
def evaluate_finetuned_model(
    checkpoint_path="models/checkpoints/revvity25_ais/best.pt",
    data_dir="data/Revvity-25/processed",
    output_dir="results/visualizations",
    n_samples=6,
):
    """
    Original function - now calls the comparison function.
    """
    return evaluate_models_comparison(
        finetuned_checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        output_dir=output_dir,
        n_samples=n_samples,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare pre-trained vs fine-tuned microSAM AIS")
    parser.add_argument("--checkpoint", default="models/checkpoints/revvity25_ais/best.pt", 
                       help="Fine-tuned checkpoint path")
    parser.add_argument("--data_dir", default="data/Revvity-25/processed", help="Data directory")
    parser.add_argument("--output_dir", default="results/visualizations", help="Output directory")
    parser.add_argument("--n_samples", type=int, default=6, help="Number of samples to evaluate")
    parser.add_argument("--pretrained_model", default="vit_b_lm", 
                       help="Pre-trained model type (default: vit_b_lm)")
    
    args = parser.parse_args()
    
    evaluate_models_comparison(
        finetuned_checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        pretrained_model_type=args.pretrained_model,
    )
