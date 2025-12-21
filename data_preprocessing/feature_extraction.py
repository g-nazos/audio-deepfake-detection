import os
import librosa
import numpy as np
import pandas as pd
from typing import Dict, Callable, Any
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def mfcc(y, sr, n_mfcc=40, n_fft=2048, hop_length=512):
    return librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )


def mfcc_delta(y, sr, **params):
    return librosa.feature.delta(mfcc(y, sr, **params), order=1)


def mfcc_delta2(y, sr, **params):
    return librosa.feature.delta(mfcc(y, sr, **params), order=2)


def pitch_yin(y, sr, fmin=50, fmax=300):
    pitch = librosa.yin(y, fmin=fmin, fmax=fmax)

    pitch = pitch[~np.isnan(pitch)]
    if len(pitch) == 0:
        pitch = np.array([0.0])

    return pitch[np.newaxis, :]  # shape (1, time)


FEATURE_FUNCTIONS = {
    # Energy / time
    "rmse": lambda y, sr, **p: librosa.feature.rms(y=y, **p),
    "zero_crossing_rate": lambda y, sr, **p: librosa.feature.zero_crossing_rate(y, **p),
    # Spectral statistics
    "spectral_centroid": lambda y, sr, **p: librosa.feature.spectral_centroid(
        y=y, sr=sr, **p
    ),
    "spectral_bandwidth": lambda y, sr, **p: librosa.feature.spectral_bandwidth(
        y=y, sr=sr, **p
    ),
    "spectral_flatness": lambda y, sr, **p: librosa.feature.spectral_flatness(y=y, **p),
    "spectral_rolloff": lambda y, sr, **p: librosa.feature.spectral_rolloff(
        y=y, sr=sr, **p
    ),
    # Core forensic features
    "mfcc": mfcc,
    "mfcc_delta": mfcc_delta,
    "mfcc_delta2": mfcc_delta2,
    # Pitch dynamics
    "pitch_yin": pitch_yin,
}


def aggregate_feature(matrix: np.ndarray) -> np.ndarray:
    """
    Aggregate time-varying features into fixed-length statistics.
    Assumes shape (n_features, time)
    """
    return np.concatenate(
        [
            np.mean(matrix, axis=1),
            np.std(matrix, axis=1),
            # np.min(matrix, axis=1),
            # np.max(matrix, axis=1),
        ]
    )


def _process_single_file(args):
    """
    Extract features for a single file with error handling.
    Logs filenames that cause problems instead of crashing.
    """
    file_path, label, feature_config, sample_rate = args
    filename = os.path.basename(file_path)

    row_data = {"label": label, "filename": filename}

    try:
        y, sr = librosa.load(file_path, sr=sample_rate)

        # -----------------------------
        # Handle MFCC and delta features
        # -----------------------------
        mfcc_params = feature_config["mfcc"]
        mfcc_matrix = mfcc(y, sr, **mfcc_params)

        # Only compute delta if there is more than 1 frame
        if mfcc_matrix.shape[1] > 1:
            mfcc_features = {
                "mfcc": mfcc_matrix,
                "mfcc_delta": librosa.feature.delta(mfcc_matrix, order=1),
                "mfcc_delta2": librosa.feature.delta(mfcc_matrix, order=2),
            }
        else:
            print(f"Warning: {filename} has too few frames for delta computation.")
            mfcc_features = {"mfcc": mfcc_matrix}

        for feat_name, matrix in mfcc_features.items():
            agg = aggregate_feature(matrix)
            n = matrix.shape[0]
            for i in range(n):
                row_data[f"{feat_name}_mean_{i}"] = agg[i]
                row_data[f"{feat_name}_std_{i}"] = agg[i + n]

        # -----------------------------
        # Handle other features
        # -----------------------------
        for feat_name, params in feature_config.items():
            if feat_name in ["mfcc", "mfcc_delta", "mfcc_delta2"]:
                continue
            matrix = FEATURE_FUNCTIONS[feat_name](y, sr, **params)
            agg = aggregate_feature(matrix)
            n = matrix.shape[0]
            for i in range(n):
                row_data[f"{feat_name}_mean_{i}"] = agg[i]
                row_data[f"{feat_name}_std_{i}"] = agg[i + n]

    except Exception as e:
        # Catch errors per file and log
        print(f"Error processing {filename}: {e}")
        # fill row_data with NaNs to keep DataFrame shape consistent
        for feat_name in feature_config.keys():
            row_data[f"{feat_name}_mean_0"] = np.nan
            row_data[f"{feat_name}_std_0"] = np.nan

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
    N_MFCC = 40
    N_FTT = 2048
    HOP_LENGTH = 512
    feature_config = {
        # Energy / time
        "rmse": {},
        "zero_crossing_rate": {},
        # Spectral statistics
        "spectral_centroid": {},
        "spectral_bandwidth": {},
        "spectral_flatness": {},
        "spectral_rolloff": {},
        # MFCC-based (core forensic features)
        "mfcc": {"n_mfcc": N_MFCC, "n_fft": N_FTT, "hop_length": HOP_LENGTH},
        "mfcc_delta": {"n_mfcc": N_MFCC, "n_fft": N_FTT, "hop_length": HOP_LENGTH},
        "mfcc_delta2": {"n_mfcc": N_MFCC, "n_fft": N_FTT, "hop_length": HOP_LENGTH},
        # Pitch
        "pitch_yin": {"fmin": 50, "fmax": 300},
    }

    df = extract_features_from_folder(
        r"c:\Users\konst\Documents\FoR_dataset\for-norm\for-norm\testing",
        feature_config,
        sample_rate=22050,
        num_workers=4,
    )

    print(df.head())
    print(df.shape)
    print(df.isna().sum().max())
    print(df.columns)
