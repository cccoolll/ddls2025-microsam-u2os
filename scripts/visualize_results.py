#!/usr/bin/env python3
"""
Simple evaluation script following micro-sam tutorial pattern.
"""

import os
import numpy as np
import imageio.v3 as imageio
import matplotlib.pyplot as plt
from glob import glob
import torch

from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation


def run_automatic_instance_segmentation(image, checkpoint_path, model_type="vit_b_lm", device="cuda"):
    """
    Run AIS following the tutorial pattern.
    
    This is the exact pattern from sam_finetuning.ipynb!
    """
    # Step 1: Get predictor and segmenter (from tutorial)
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type,
        checkpoint=checkpoint_path,
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


def evaluate_finetuned_model(
    checkpoint_path="models/checkpoints/revvity25_ais/best.pt",
    data_dir="data/Revvity-25/processed",
    output_dir="results/visualizations",
    n_samples=6,
):
    """
    Evaluate the fine-tuned model on validation images.
    
    Following tutorial pattern - simple and clean!
    """
    print("🔍 Evaluating Fine-tuned microSAM AIS")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"✅ Checkpoint found: {checkpoint_path}")
    print(f"   Size: {os.path.getsize(checkpoint_path) / (1024**2):.1f} MB")
    
    # Get validation images
    images_dir = os.path.join(data_dir, "images")
    labels_dir = os.path.join(data_dir, "labels")
    
    valid_images = sorted(glob(os.path.join(images_dir, "valid_*.tif")))[:n_samples]
    valid_labels = sorted(glob(os.path.join(labels_dir, "valid_*.tif")))[:n_samples]
    
    print(f"\nProcessing {len(valid_images)} validation images...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    results = []
    for i, (img_path, label_path) in enumerate(zip(valid_images, valid_labels)):
        print(f"\n[{i+1}/{len(valid_images)}] {os.path.basename(img_path)}")
        
        # Load image and ground truth
        image = imageio.imread(img_path)
        gt_mask = imageio.imread(label_path)
        
        # Run AIS prediction (following tutorial)
        try:
            prediction = run_automatic_instance_segmentation(
                image=image,
                checkpoint_path=checkpoint_path,
                model_type="vit_b_lm",
                device=device,
            )
            
            n_pred = prediction.max()
            n_gt = gt_mask.max()
            print(f"  Predicted: {n_pred} cells | Ground truth: {n_gt} cells")
            
            results.append({
                'image': image,
                'gt': gt_mask,
                'pred': prediction,
                'name': os.path.basename(img_path),
            })
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    # Create visualization
    if results:
        print(f"\n📊 Creating visualization...")
        create_visualization(results, output_dir)
        print(f"✅ Results saved to: {output_dir}")
    
    return results


def create_visualization(results, output_dir):
    """Create side-by-side visualization."""
    n_samples = len(results)
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5*n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        # Original image
        axes[i, 0].imshow(result['image'], cmap='gray')
        axes[i, 0].set_title(f"Image: {result['name']}")
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(result['gt'], cmap='tab20')
        axes[i, 1].set_title(f"Ground Truth ({result['gt'].max()} cells)")
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(result['pred'], cmap='tab20')
        axes[i, 2].set_title(f"Prediction ({result['pred'].max()} cells)")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ais_evaluation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print summary
    print(f"\n📈 Summary:")
    print(f"   Total images: {len(results)}")
    print(f"   Avg predicted cells: {np.mean([r['pred'].max() for r in results]):.1f}")
    print(f"   Avg ground truth cells: {np.mean([r['gt'].max() for r in results]):.1f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned microSAM AIS")
    parser.add_argument("--checkpoint", default="models/checkpoints/revvity25_ais/best.pt", help="Checkpoint path")
    parser.add_argument("--data_dir", default="data/Revvity-25/processed", help="Data directory")
    parser.add_argument("--output_dir", default="results/visualizations", help="Output directory")
    parser.add_argument("--n_samples", type=int, default=6, help="Number of samples to evaluate")
    
    args = parser.parse_args()
    
    evaluate_finetuned_model(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
    )
