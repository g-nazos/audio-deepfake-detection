import os
import tqdm
import subprocess
import numpy as np
import librosa
from utils.utils import get_file_path
from scipy.io.wavfile import read, write


def get_wav_duration(filepath):
    """
    This is a helper function that checks if a given file
    exists and is a .wav and if so returns the file duration.

    Input:
    filepath = audio file path

    Returns:
    float - audio length duration (in seconds) of the input time series or spectrogram.

    FileNotFoundError
    """
    if os.path.exists(filepath):
        duration = librosa.get_duration(path=filepath)
        return duration
    else:
        raise FileNotFoundError(f"File: {filepath} is not found!")


def get_sample_rate(filepath):
    """
    This function uses soxi to get the sample rate of an audio file.

    Input:
    filepath = audio file path

    Returns:
    float - sample rate in kHz (e.g., 44.1 for 44100 Hz)

    Raises:
    FileNotFoundError - if the file does not exist
    RuntimeError - if soxi command fails
    """
    # file_path = get_file_path(filepath)
    # if file_path is None:
    #     raise FileNotFoundError(f"File: {filepath} is not found!")

    file_path = filepath

    soxi_command = "soxi -r " + file_path
    try:
        result = subprocess.run(
            soxi_command, shell=True, capture_output=True, text=True, check=True
        )
        sample_rate_hz = int(result.stdout.strip())
        return sample_rate_hz
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"soxi failed for {filepath}: {e.stderr}")
