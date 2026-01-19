# audio-deepfake-detection

## Installation

### macOS (Apple Silicon / Intel)

NVIDIA CUDA libraries are not available for macOS. Use the macOS-specific requirements file:

```bash
pip install -r requirements-macos.txt
```

### Linux / Windows

For systems with NVIDIA GPU support:

```bash
pip install -r requirements.txt
```

**Note**: The macOS requirements file excludes CUDA dependencies (`nvidia-*` packages) and `triton`, which are only available on Linux/Windows systems with NVIDIA GPUs. PyTorch on macOS will use CPU or Metal (MPS) acceleration instead.