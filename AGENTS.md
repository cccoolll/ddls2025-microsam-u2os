# AGENTS.md - AI Coding Agent Guide

This file contains essential information about the project structure, technology stack, and development conventions. AI coding agents should read this before making any changes.

## Project Overview

**ddls2025-microsam-u2os** is a DDLS 2025 course project for fine-tuning microSAM (Segment Anything Model for Microscopy) for automatic U2OS cell segmentation in brightfield microscopy images.

The project fine-tunes the Auto Instance Segmentation (AIS) component of microSAM using the Revvity-25 dataset (110 images, 2,937 cell annotations) and deploys the model via BioEngine for cloud-based access.

### Key Features
- Fine-tuned microSAM for zero-prompt auto-segmentation
- Dual-model service supporting both microSAM and Cellpose
- Web application for browser-based segmentation
- BioEngine deployment for scalable cloud inference

## Technology Stack

### Core Dependencies
- **Python**: 3.11 (managed via conda)
- **Deep Learning**: PyTorch >=2.5, torchvision, segment-anything
- **Microscopy**: micro_sam (cloned submodule), torch_em >=0.8, trackastra
- **Deployment**: Ray Serve, hypha-rpc, BioEngine worker
- **Image Processing**: OpenCV, PIL, imageio, scikit-image
- **Data**: COCO format annotations, xarray, zarr

### Environment Setup
```bash
conda create -n microsam python=3.11 -y
conda activate microsam
pip install git+https://github.com/aicell-lab/bioengine-worker.git
conda install -c conda-forge micro_sam -y
```

**IMPORTANT**: Always run commands under the `microsam` conda environment.

## Project Structure

```
ddls2025-microsam-u2os/
├── bioengine-app/              # BioEngine service deployment
│   ├── cell-segmenter.py      # Main Ray Serve deployment class
│   └── manifest.yaml          # BioEngine application manifest
├── data/                       # Dataset storage
│   ├── Revvity-25/            # Primary training dataset
│   ├── BBBC039/               # Additional cell segmentation dataset
│   ├── chunk_cache/           # Cached embeddings for faster training
│   └── annotations/           # COCO format annotations
├── micro-sam/                  # Git submodule - microSAM library
│   ├── micro_sam/             # Main Python package
│   ├── finetuning/            # Fine-tuning scripts and evaluation
│   ├── test/                  # Unit tests
│   └── pyproject.toml         # Build configuration
├── scripts/                    # Project-specific scripts
│   ├── revvity25/
│   │   ├── preprocess_revvity25.py   # Data preprocessing
│   │   ├── train_revvity25_ais.py    # AIS training script
│   │   └── visualize_results.py      # Results visualization
│   └── deploy_cellsegmenter.py       # Deployment script
├── models/                     # Model checkpoints
│   └── checkpoints/           # Saved model weights
├── results/                    # Training curves and evaluation plots
├── examples/                   # Example data and scripts
│   ├── simple_finetuning/     # Example training data
│   └── simple_segmentation/   # Example segmentation outputs
├── microsam-segmenter.html    # Web application (single-file HTML)
├── .cursorrules               # Development plan and guidelines
└── .env                       # Environment variables (gitignored)
```

## Configuration Files

### Build and Linting (`micro-sam/pyproject.toml`)
- **Build**: setuptools >=42.0.0
- **Testing**: pytest with coverage reporting
- **Formatting**: Black (line-length: 79)
- **Linting**: Ruff with specific rule selections (E, F, W, UP, I, BLE, B, A, C4, ISC, G, PIE, SIM)

### Conda Environment (`micro-sam/environment.yaml`)
Key packages: pytorch >=2.5, segment-anything, torch_em >=0.8, napari, magicgui, trackastra

### BioEngine Manifest (`bioengine-app/manifest.yaml`)
- Service ID: `cell-segmenter`
- Type: `ray-serve`
- Workspace: `reef-imaging`
- Service endpoint: `reef-imaging/cell-segmenter`

## microSAM Architecture

The project focuses on fine-tuning the **Auto Instance Segmentation (AIS)** head:

1. **Image Encoder** (ViT-B): Frozen during AIS training
2. **Mask Decoder**: Handles prompt-based segmentation
3. **AIS Head**: Auto Instance Segmentation for zero-prompt segmentation (our focus)

### Training Strategies
1. **AIS Head Only**: Freeze encoder, train only AIS (~1-2 hours)
2. **LoRA + AIS**: Add LoRA adapters to encoder + train AIS (~3-4 hours)

Default configuration:
- Model: `vit_b_lm` (light microscopy pre-trained)
- Epochs: 50 with early stopping (patience=15)
- Batch size: 2
- Objects per batch: 8
- Learning rate: 1e-4 with ReduceLROnPlateau scheduler
- Patch shape: (512, 512)

## Key Commands

### Data Preprocessing
```bash
conda activate microsam
python scripts/revvity25/preprocess_revvity25.py
```

### Training
```bash
conda activate microsam
cd scripts/revvity25
python train_revvity25_ais.py --n_epochs 50 --batch_size 2
```

### Deployment
```bash
conda activate microsam
python scripts/deploy_cellsegmenter.py
```

### Web Application
Open `microsam-segmenter.html` in a browser - no installation needed.

## Code Style Guidelines

### Python Standards
- **Style**: PEP 8 compliant
- **Line length**: 79 characters (enforced by Black and Ruff)
- **Type hints**: Required for all function signatures
- **Docstrings**: Comprehensive docstrings for all public functions
- **Comments**: Inline comments explaining complex logic (beginner-friendly)

### Import Organization
```python
# 1. Standard library
import os
import json
from pathlib import Path

# 2. Third-party
import numpy as np
import torch
import cv2

# 3. microSAM specific
import micro_sam.training as sam_training
from micro_sam.training.joint_sam_trainer import JointSamLogger
```

### Function Documentation Pattern
```python
def function_name(param1: type1, param2: type2) -> return_type:
    """
    Brief description.
    
    For beginners: Explain what this function does in simple terms.
    
    Args:
        param1: Description
        param2: Description
        
    Returns:
        Description of return value
    """
    pass
```

## Data Format Requirements

### Input Format
- **Images**: COCO format or simple TIFF files
- **Preprocessing**: Resized to 512x512, converted to grayscale
- **Minimum dataset size**: 2 images (enforced for train/val split)

### COCO Annotations
```python
{
    "images": [{"id": int, "file_name": str, "height": int, "width": int}],
    "annotations": [{"id": int, "image_id": int, "segmentation": [...], "category_id": int}],
    "categories": [{"id": int, "name": str}]
}
```

## Testing Strategy

### Running Tests
```bash
# From micro-sam directory
cd micro-sam
pytest -v --durations=10

# Skip slow tests
pytest -v -m "not slow"

# Skip GUI tests
pytest -v -m "not gui"
```

### Test Categories
- `slow`: Long-running tests (deselect with `-m "not slow"`)
- `gui`: GUI-related tests requiring display (deselect with `-m "not gui"`)

## Deployment Architecture

### BioEngine Service (`bioengine-app/cell-segmenter.py`)
- **Framework**: Ray Serve
- **Resources**: 4 CPUs, 2 GPUs, 16GB memory
- **Methods**:
  - `segment_all()`: Main segmentation endpoint (supports microSAM and Cellpose)
  - `start_fit()`: Online training endpoint
  - `get_fit_status()`: Training status check
  - `encode_image()`: Extract embeddings

### Web Application (`microsam-segmenter.html`)
- Pure HTML/JavaScript (no build step)
- Connects to BioEngine via hypha-rpc
- Supports image upload, segmentation, and visualization

## Security Considerations

### Environment Variables
- `.env` file contains `HYPHA_TOKEN` - **NEVER commit to git**
- Token used for BioEngine authentication
- `.env` is already in `.gitignore`

### Service Authorization
Authorized users listed in `bioengine-app/manifest.yaml`:
- nils.mech@gmail.com
- songtao.cheng@scilifelab.se

## Common Issues and Solutions

### GPU Memory
- Reduce `batch_size` to 1 if OOM
- Enable `use_mixed_precision=True`
- Use `freeze_encoder=True` to reduce memory

### Data Loading
- Minimum 2 images required for train/val split
- Use `MinInstanceSampler` for proper instance sampling
- Ensure COCO annotations have valid polygons (>= 6 points)

### Version Compatibility
- numpy/xarray compatibility issues resolved with specific versions
- micro-sam 1.6.2 and trackastra 0.4.1 are tested versions

## External Resources

- **micro-sam**: https://github.com/computational-cell-analytics/micro-sam
- **BioEngine**: https://github.com/bioimage-io/bioengine
- **Revvity-25 Dataset**: https://huggingface.co/datasets/YaroslavPrytula/Revvity-25/
- **BioEngine Service**: https://hypha.aicell.io (workspace: reef-imaging)

## Contact and Attribution

- **Authors**: Songtao Cheng (cccoolll), Nils Mechtel (nilsmecthel)
- **Affiliation**: KTH / SciLifeLab
- **Course**: DDLS 2025
- **License**: MIT
