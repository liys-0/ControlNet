# ControlNet Virtual Environment Setup Guide

This guide provides step-by-step instructions to set up a Python virtual environment for ControlNet on a new server.

## System Requirements

- **Python**: 3.8.5 or higher (3.10+ recommended for modern systems)
- **CUDA Toolkit**: 11.8 (for cu118 PyTorch builds)
- **GPU**: NVIDIA GPU with CUDA support
- **OS**: Linux/MacOS/Windows with bash support
- **Disk Space**: At least 20GB for dependencies and models

## Quick Setup

### 1. Create Virtual Environment

```bash
# Using Python's venv
python3 -m venv controlnet_env

# Activate the environment
source controlnet_env/bin/activate  # On Linux/MacOS
# or
controlnet_env\Scripts\activate  # On Windows
```

### 2. Upgrade pip and Install Core Dependencies

```bash
# Upgrade pip to latest version
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8 support (REQUIRED - match your GPU driver)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install ControlNet Requirements

```bash
# Install all project dependencies
pip install -r requirements.txt
```

## Detailed Requirements

### Core ML/DL Libraries
- **torch==2.0.1+cu118** - PyTorch deep learning framework with CUDA support
- **torchvision==0.15.2+cu118** - Computer vision utilities
- **pytorch-lightning==2.0.9** - Training framework
- **diffusers==0.20.2** - Diffusion model utilities
- **transformers==4.30.2** - Transformer models (CLIP, etc.)

### Image Processing
- **opencv-contrib-python==4.8.1.78** - Computer vision library
- **albumentations==1.3.1** - Image augmentation
- **imageio==2.31.1** - Image I/O
- **imageio-ffmpeg==0.4.9** - Video support
- **kornia==0.6.12** - Geometric vision operations

### UI/Interface
- **gradio==3.16.2** - Web interface for models
- **streamlit==1.28.0** - Alternative web framework
- **streamlit-drawable-canvas==0.9.3** - Interactive drawing interface

### Model/Training Utilities
- **basicsr==1.4.2** - Image restoration components
- **timm==0.9.2** - PyTorch image models
- **open_clip_torch==2.20.0** - OpenAI CLIP implementation
- **einops==0.6.1** - Tensor operations
- **addict==2.4.0** - Dictionary utilities
- **safetensors==0.3.3** - Model serialization
- **omegaconf==2.3.0** - Configuration management
- **webdataset==0.2.48** - Large-scale dataset handling
- **invisible-watermark==0.2.0** - Watermarking
- **torchmetrics==0.11.4** - Training metrics
- **prettytable==3.9.0** - Formatted output
- **yapf==0.40.1** - Code formatter
- **test-tube==0.7.5** - Experiment tracking

## Optional: Conda Environment Setup

If you prefer using Conda instead of venv:

```bash
# Create conda environment from environment.yaml
conda env create -f environment.yaml

# Activate environment
conda activate control
```

**Note**: The `environment.yaml` contains older versions. Recommended to use the pip + requirements.txt approach above for more recent versions.

## Downloading Models and Weights

Several scripts are provided to download necessary models:

```bash
# Download Hugging Face models
python download_hf.py

# Download Stable Diffusion 1.5
python download_sd15.py
```

## Verification

Test the installation:

```bash
# Test basic imports
python -c "import torch; import diffusers; import transformers; print('✓ All imports successful')"

# Check CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device:', torch.cuda.get_device_name(0))"

# Try running a test script
python test_gradio.py
```

## Common Issues & Solutions

### CUDA Version Mismatch
If you get CUDA errors, check your GPU driver CUDA version:
```bash
nvidia-smi
```
Ensure the PyTorch version matches your CUDA version (cu118 for CUDA 11.8, cu121 for CUDA 12.1, etc.)

### Out of Memory
If you encounter OOM errors during generation, reduce batch size or image resolution in config.py

### Missing Models
Run the download scripts before attempting inference:
```bash
python download_hf.py
python download_sd15.py
```

### GPU Not Detected
Verify NVIDIA drivers are installed and PyTorch can access GPU:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Running the Application

After setup, you can run the various Gradio interfaces:

```bash
# Image-to-Image with ControlNet
python gradio_canny2image.py

# Interactive Scribble-to-Image
python gradio_scribble2image_interactive.py

# Pose estimation-guided generation
python gradio_pose2image.py

# Other available interfaces
python gradio_depth2image.py
python gradio_seg2image.py
python gradio_normal2image.py
python gradio_hough2image.py
python gradio_hed2image.py
python gradio_annotator.py
```

## Troubleshooting Setup

### Update pip if installation fails
```bash
pip install --upgrade pip
```

### Use --no-cache-dir if disk space is limited
```bash
pip install --no-cache-dir -r requirements.txt
```

### Install packages with specific versions only if needed
```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118 --no-deps
```

## Environment Variables (Optional)

```bash
# Set to reduce GPU memory usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# Set to disable TensorFlow for ControlNet-only usage
export TF_CPP_MIN_LOG_LEVEL=3
```

## Notes

- PyTorch 2.0.1 is significantly faster than 1.12.1 due to optimizations
- CUDA 11.8 (cu118) supports most modern GPUs (RTX 20xx, 30xx, 40xx series)
- Total installation size: ~15-20GB including dependencies and models
- First run may take longer as models are downloaded automatically

---

**Created**: April 29, 2026  
**Last Updated**: April 29, 2026
