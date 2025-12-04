import os
import librosa
import numpy as np
import pandas as pd
from typing import Dict, Callable, Any
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


# Global feature registry (must be module-level for multiprocessing)
FEATURE_FUNCTIONS = {
    "mfcc": lambda y, sr, n_mfcc=20: librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc),
    "mfcc_delta": lambda y, sr, n_mfcc=20: librosa.feature.delta(
        librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    ),
    "mfcc_delta2": lambda y, sr, n_mfcc=20: librosa.feature.delta(
        librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc), order=2
    ),
    "spectral_centroid": lambda y, sr, **p: librosa.feature.spectral_centroid(
        y=y, sr=sr, **p
    ),
    "spectral_bandwidth": lambda y, sr, **p: librosa.feature.spectral_bandwidth(
        y=y, sr=sr, **p
    ),
    "spectral_contrast": lambda y, sr, **p: librosa.feature.spectral_contrast(
        y=y, sr=sr, **p
    ),
    "zero_crossing_rate": lambda y, sr, **p: librosa.feature.zero_crossing_rate(y, **p),
    "rmse": lambda y, sr, **p: librosa.feature.rms(y=y, **p),
    "spectral_flatness": lambda y, sr, **p: librosa.feature.spectral_flatness(y=y, **p),
    "spectral_rolloff": lambda y, sr, **p: librosa.feature.spectral_rolloff(
        y=y, sr=sr, **p
    ),
    "poly_features": lambda y, sr, **p: librosa.feature.poly_features(y=y, sr=sr, **p),
    "tonnetz": lambda y, sr, **p: librosa.feature.tonnetz(
        y=librosa.effects.harmonic(y), sr=sr, **p
    ),
    "chroma_stft": lambda y, sr, **p: librosa.feature.chroma_stft(y=y, sr=sr, **p),
    "chroma_cqt": lambda y, sr, **p: librosa.feature.chroma_cqt(y=y, sr=sr, **p),
    "chroma_cens": lambda y, sr, **p: librosa.feature.chroma_cens(y=y, sr=sr, **p),
    "melspectrogram": lambda y, sr, **p: librosa.feature.melspectrogram(
        y=y, sr=sr, **p
    ),
    "tempogram": lambda y, sr, **p: librosa.feature.tempogram(y=y, sr=sr, **p),
    "fourier_tempogram": lambda y, sr, **p: librosa.feature.fourier_tempogram(
        y=y, sr=sr, **p
    ),
    "onset_strength": lambda y, sr, **p: librosa.onset.onset_strength(y=y, sr=sr, **p),
    "pitch_yin": lambda y, sr, fmin=50, fmax=300: librosa.yin(y, fmin=fmin, fmax=fmax),
}


def _process_single_file(args):
    """
    Worker function (runs inside a process).
    Extracts features for one audio file.

    Parameters
    ----------
    args: tuple
        (file_path, label, feature_config, sample_rate)

    Returns
    -------
    dict
        Flattened feature dictionary for this file.
    """
    file_path, label, feature_config, sample_rate = args

    filename = os.path.basename(file_path)

    # Load audio once
    y, sr = librosa.load(file_path, sr=sample_rate)

    row_data = {"label": label, "filename": filename}

    # Compute selected features
    for feat_name, params in feature_config.items():
        matrix = FEATURE_FUNCTIONS[feat_name](y, sr, **params)
        flat = matrix.flatten()
        for i, value in enumerate(flat):
            row_data[f"{feat_name}_{i}"] = value

    return row_data


def extract_features_from_folder(
    folder_path: str,
    feature_config: Dict[str, Dict[str, Any]],
    sample_rate: int = 16000,
    num_workers: int = None,
) -> pd.DataFrame:
    """
    Extract features from audio files in 'real' and 'fake' subfolders using multiprocessing.

    Parameters
    ----------
    folder_path : str
        Path to dataset split (e.g., 'training', 'testing', 'validation').

    feature_config : dict
        Dictionary defining which features to compute and their librosa parameters.
        Example:
            {
                "mfcc": {"n_mfcc": 20},
                "spectral_centroid": {}
            }

    sample_rate : int
        Audio sampling rate used during librosa.load().

    num_workers : int, optional
        Number of CPU processes. If None, uses all available cores.

    Returns
    -------
    DataFrame
        One row per file containing flattened feature values + metadata.
    """

    # Validate feature names
    for name in feature_config.keys():
        if name not in FEATURE_FUNCTIONS:
            raise ValueError(f"Unsupported feature: {name}")

    # Collect all files to process
    tasks = []
    for label in ["real", "fake"]:
        class_dir = os.path.join(folder_path, label)
        if not os.path.isdir(class_dir):
            continue
        for f in os.listdir(class_dir):
            if f.lower().endswith((".mp3", ".wav")):
                full_path = os.path.join(class_dir, f)
                tasks.append((full_path, label, feature_config, sample_rate))

    # Run multiprocessing with progress bar
    if num_workers is None:
        cpu_count = os.cpu_count()
        num_workers = max(1, cpu_count - 1)

    print(f"Using {num_workers} workers...")
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for row in tqdm(
            executor.map(_process_single_file, tasks),
            total=len(tasks),
            desc="Extracting features",
        ):
            results.append(row)

    return pd.DataFrame(results)


if __name__ == "__main__":
    feature_config = {
        # Time-domain
        "zero_crossing_rate": {},  # Zero Crossing Rate
        "rmse": {},  # Root-mean-square energy
        # Spectral features
        "spectral_centroid": {},  # Spectral centroid
        "spectral_bandwidth": {},  # Spectral bandwidth
        "spectral_contrast": {},  # Spectral contrast
        "spectral_flatness": {},  # Spectral flatness
        "spectral_rolloff": {},  # Spectral rolloff
        "poly_features": {},  # Polynomial coefficients of the spectrum
        # Harmonic / tonal
        "tonnetz": {},  # Tonal centroid features (requires harmonic component)
        # Chroma features
        # "chroma_stft": {},              # Chroma from STFT
        "chroma_cqt": {},  # Chroma from CQT
        "chroma_cens": {},  # Chroma energy normalized
        # Mel-scale features
        "melspectrogram": {},  # Mel-scaled spectrogram
        "mfcc": {"n_mfcc": 20},  # MFCCs
        "mfcc_delta": {"n_mfcc": 20},  # Delta MFCCs
        "mfcc_delta2": {"n_mfcc": 20},  # Delta-delta MFCCs
        # Temporal / rhythm
        "tempogram": {},  # Tempogram
        "fourier_tempogram": {},  # Fourier tempogram
        "onset_strength": {},  # Onset strength envelope
        # Pitch
        "pitch_yin": {
            "fmin": 50,
            "fmax": 300,
        },  # Fundamental frequency using YIN algorithm
    }

    df = extract_features_from_folder(
        r"C:\Users\konst\Documents\for-2seconds\testing",
        feature_config,
        sample_rate=16000,
        num_workers=4,
    )

    print(df.head())
