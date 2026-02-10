# audio-deepfake-detection

## Quick Start (Docker) -- Recommended for evaluation using JupyterLab in Localhost

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Clone this repository:

   ```bash
   git clone <repo-url>
   cd audio-deepfake-detection
   ```

3. Download the datasets zip and extract it into the project root (see [Data Setup](#data-setup) below)
4. Build and start the environment (using the respective command for the docker version):

   ```bash
   docker compose up --build
   # or
   docker-compose up --build
   ```

5. Open http://localhost:8888 in your browser
6. Navigate to the `notebooks/` folder and open any notebook

To stop the environment, press `Ctrl+C` in the terminal or run `docker compose down`.

## Data Setup

The datasets are not included in the repository due to their size. Download and extract them from [Google Drive](https://drive.google.com/file/d/1IRYglNlMVYHwYFksumPMJEnrOjJYWkIm/view?usp=sharing) into the project root so the folder structure matches:

```
audio-deepfake-detection/
├── FoR_dataset/
│   ...
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
│   ...
│
│   ├── release_in_the_wild_trimmed_normalized/
│   │   ├── real/*.wav
│   │   └── fake/*.wav
│   ├── normalized_features/
│   │   └── *.parquet
│   └── meta.csv
└── elevenlabs-dataset/
    ├── ...   
    ├── fake/*.wav
    ├── real/*.wav
    └── features/
        └── *.parquet
```

The ML notebooks (in `notebooks/`) only require the `.parquet` feature files. The raw `.wav` audio files are only needed for feature extraction and audio exploration.

**Note**: The Docker image does not include PyTorch, HuggingFace Transformers, or Gradio to keep it lightweight. The following notebooks wont run because of this:

- `notebooks/gender_recognition.ipynb`
- `notebooks/itw_data_exploration.ipynb`

In case they are needed, they can be installed by running the following commands:

```bash
pip install torch transformers gradio
```

## Manual Setup (without Docker)

1. Install Python 3.12+
2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   # or
   .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. For Python 3.13+, also install:

   ```bash
   pip install audioop-lts
   ```

5. For notebooks that use PyTorch (gender_recognition, itw_data_exploration):

   ```bash
   pip install torch transformers gradio
   ```