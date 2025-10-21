# ddls2025-microsam-u2os
Fine-tuning the microSAM model for U2OS brightfield cell segmentation and integrating real-time confluence analysis with the Squid+ microscope via Agent-Lens — a DDLS 2025 course project.

Conda environment setup:
Create conda environment with Python version: 3.11
Install micro-sam with command: `conda install -c conda-forge micro_sam`

Note: You might still get `module not found` error when trying to run scripts. Then you will need to install the missing modules.


## Installation

> The instructions below use `conda` for environment management. Alternatively, you can use `mamba` if preferred.

Create a new conda environment and activate it:
```bash
conda create -n microsam python=3.11 -y
conda activate microsam
```

Install BioEngine Worker:
```bash
pip install git+https://github.com/aicell-lab/bioengine-worker.git
```

Validate installation by checking the version:
```bash
python -c "import bioengine.worker; print(bioengine.__version__)"
```

Install micro-sam:
```bash
conda install -c conda-forge micro_sam -y
```