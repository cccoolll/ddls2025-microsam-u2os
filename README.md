# 🔬 ddls2025-microsam-u2os

Fine-tuning microSAM for U2OS cell segmentation with BioEngine deployment — a DDLS 2025 course project.

## Project Overview

Fine-tuning the **Auto Instance Segmentation (AIS)** component of microSAM for automatic U2OS cell segmentation in brightfield microscopy. The model is deployed via BioEngine and accessible through a web application.

## Project Structure

```
ddls2025-microsam-u2os/
├── bioengine-app/              # BioEngine service (microSAM + Cellpose)
├── data/
│   ├── Revvity-25/            # Training dataset (110 images, 2,937 cells)
│   └── chunk_cache/           # Cached embeddings
├── micro-sam/                  # Cloned micro-sam repository
├── models/checkpoints/         # Fine-tuned model weights
├── results/                    # Training curves and evaluations
├── scripts/                    # Training and deployment scripts
├── microsam-segmenter.html    # 🌟 Web application
└── README.md
```

## Quick Start

### 1. Environment Setup

```bash
conda create -n microsam python=3.11 -y
conda activate microsam
pip install git+https://github.com/aicell-lab/bioengine-worker.git
conda install -c conda-forge micro_sam -y
```

### 2. Web Application

**Easiest way to use the model:**

1. Open `microsam-segmenter.html` in your browser
2. Login when prompted
3. Upload a microscopy image (PNG/JPEG/TIFF)
4. Click "Segment Cells"
5. View instance mask and polygon overlay results

No installation needed - runs in browser and connects to cloud service!

### 3. Training

```bash
conda activate microsam
cd scripts/revvity25
python train_ais_optimized.py
```


## BioEngine Service

- **Service**: `reef-imaging/cell-segmenter` at `https://hypha.aicell.io`
- **Methods**: microSAM (fine-tuned) or Cellpose

```python
from hypha_rpc import connect_to_server

server = await connect_to_server({"server_url": "https://hypha.aicell.io"})
service = await server.get_service("reef-imaging/cell-segmenter")
result = await service.segment_all(image_base64, method="microsam")
```

## References

- [micro-sam](https://github.com/computational-cell-analytics/micro-sam)
- [BioEngine](https://github.com/bioimage-io/bioengine)
- [Revvity-25 Dataset](https://huggingface.co/datasets/YaroslavPrytula/Revvity-25/)