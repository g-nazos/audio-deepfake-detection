# audio-deepfake-detection

## Quick Start (Docker) -- Recommended for evaluation

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Clone this repository:

   ```bash
   git clone <repo-url>
   cd audio-deepfake-detection
   ```

3. Download the datasets zip and extract it into the project root (see [Data Setup](#data-setup) below)
4. Start the environment:

   ```bash
   docker compose up --build
   ```

5. Open http://localhost:8888 in your browser
6. Navigate to the `notebooks/` folder and open any notebook

To stop the environment, press `Ctrl+C` in the terminal or run `docker compose down`.

## Data Setup

The datasets are not included in the repository due to their size. Download and extract them into the project root so the folder structure matches:

```
audio-deepfake-detection/
├── FoR_dataset/
│   ├── for-norm/for-norm/
│   │   ├── training/
│   │   │   ├── real/*.wav
│   │   │   └── fake/*.wav
│   │   ├── testing/
│   │   │   ├── real/*.wav
│   │   │   └── fake/*.wav
│   │   └── validation/
│   │       ├── real/*.wav
│   │       └── fake/*.wav
│   └── features/
│       └── *.parquet
├── in-the-wild-audio-deepfake/
│   ├── release_in_the_wild/
│   │   ├── real/*.wav
│   │   └── fake/*.wav
│   ├── features/
│   │   └── *.parquet
│   └── meta.csv
└── elevenlabs-dataset/
    ├── fake/*.wav
    ├── real/*.wav
    └── features/
        └── *.parquet
```

The ML notebooks (in `notebooks/`) only require the `.parquet` feature files. The raw `.wav` audio files are only needed for feature extraction and audio exploration.

## Manual Setup (without Docker)

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

For Windows with Python 3.13+, also install:

```bash
pip install audioop-lts
```

**Note**: The macOS requirements file excludes CUDA dependencies (`nvidia-*` packages) and `triton`, which are only available on Linux/Windows systems with NVIDIA GPUs. PyTorch on macOS will use CPU or Metal (MPS) acceleration instead.
