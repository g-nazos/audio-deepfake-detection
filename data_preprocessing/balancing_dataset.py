import pandas as pd
import os
import shutil
from pathlib import Path


def find_imabalnaced_speakers(
    metadata_path="../../in-the-wild-audio-deepfake/meta.csv", threshold=0.2
):
    """
    Identify speakers who have a significant imbalance between spoof and bona-fide samples.

    This function loads a metadata CSV containing audio filenames, speaker identities,
    and labels ("spoof" or "bona-fide"). It computes the number of spoof and bona-fide
    samples per speaker, determines the minority-class ratio for each speaker, and
    returns both the complete per-speaker summary and the subset of speakers whose
    minority class ratio falls below a specified threshold.

    Args:
        metadata_path (str):
            Path to the metadata CSV file containing columns:
            "filename", "speaker", and "label".
        threshold (float):
            Minimum acceptable ratio of minority samples (min(spoof, bona-fide) / total).
            Speakers falling below this ratio are considered significantly imbalanced.
            Default is 0.2 (i.e., minority class < 20% of total samples).

    Returns:
        tuple:
            result (pd.DataFrame):
                DataFrame containing all speakers with columns:
                ['speaker', 'num_spoof', 'num_bona_fide'].
            significantly_imbalanced (pd.DataFrame):
                Subset of `result` containing only speakers whose minority ratio < threshold.

    Example:
        result, imbalanced = find_imabalnaced_speakers("meta.csv", threshold=0.15)
    """
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv("../../in-the-wild-audio-deepfake/meta.csv")

    # Pivot table to count the number of occurrences of each label per speaker
    result = df.pivot_table(
        index="speaker",  # Rows will be unique speakers
        columns="label",  # Columns will be unique labels ('spoof', 'bona-fide')
        aggfunc="size",  # Count the number of rows for each speaker-label combination
        fill_value=0,  # Fill missing combinations with 0 instead of NaN
    ).reset_index()  # Reset index to turn 'speaker' back into a normal column

    # Clean up column names for clarity
    result.columns.name = None
    result = result.rename(
        columns={
            "spoof": "num_spoof",  # Rename 'spoof' column to 'num_spoof'
            "bona-fide": "num_bona_fide",  # Rename 'bona-fide' column to 'num_bona_fide'
        }
    )

    # Calculate total samples per speaker
    result["total"] = result["num_spoof"] + result["num_bona_fide"]

    # Calculate the ratio of the minority class
    result["min_ratio"] = (
        result[["num_spoof", "num_bona_fide"]].min(axis=1) / result["total"]
    )

    # Filter speakers where the minority class is less than 20% of total
    significantly_imbalanced = result[result["min_ratio"] < threshold]

    # Drop helper columns if you want clean output
    significantly_imbalanced = significantly_imbalanced.drop(
        columns=["total", "min_ratio"]
    )

    return df, significantly_imbalanced


def get_files_to_remove_for_balancing(df, significantly_imbalanced):
    """
    Determine which audio files should be removed to balance spoof and bona-fide
    samples for each significantly imbalanced speaker.

    This function examines each speaker listed in `significantly_imbalanced` and
    calculates how many spoof and bona-fide samples must be removed so that both
    classes contain an equal number of samples for that speaker. It keeps a
    random subset of size equal to the minority class and returns the extra
    spoof and bona-fide samples separately.

    Args:
        df (pd.DataFrame):
            The complete dataset containing the columns:
            "filename", "speaker", and "label".
        significantly_imbalanced (pd.DataFrame):
            A DataFrame containing only significantly imbalanced speakers, with:
                - 'speaker'
                - 'num_spoof'
                - 'num_bona_fide'

    Returns:
        tuple:
            spoof_remove_df (pd.DataFrame):
                DataFrame containing filenames of spoof samples to remove.
            bona_remove_df (pd.DataFrame):
                DataFrame containing filenames of bona-fide samples to remove.

    Example:
        spoof_df, bona_df = get_files_to_remove_for_balancing(
            df=metadata_df,
            significantly_imbalanced=imbalanced_speakers
        )
    """

    spoof_to_remove = []
    bona_to_remove = []

    for _, row in significantly_imbalanced.iterrows():
        speaker = row["speaker"]
        n_spoof = row["num_spoof"]
        n_bona = row["num_bona_fide"]

        # Target samples per class
        target = min(n_spoof, n_bona)

        # All samples of this speaker
        spk_df = df[df["speaker"] == speaker]

        # --- Spoof files ---
        spoof_files = spk_df[spk_df["label"] == "spoof"]["file"]
        spoof_keep = spoof_files.sample(target, random_state=42)
        spoof_extra = spoof_files[~spoof_files.isin(spoof_keep)]
        spoof_to_remove.extend(list(spoof_extra))

        # --- Bona-fide files ---
        bona_files = spk_df[spk_df["label"] == "bona-fide"]["file"]
        bona_keep = bona_files.sample(target, random_state=42)
        bona_extra = bona_files[~bona_files.isin(bona_keep)]
        bona_to_remove.extend(list(bona_extra))

    # Convert to DataFrames
    spoof_remove_df = pd.DataFrame(spoof_to_remove, columns=["file"])
    bona_remove_df = pd.DataFrame(bona_to_remove, columns=["file"])

    return spoof_remove_df, bona_remove_df


import os
import shutil
from pathlib import Path


def move_balancing_files(
    real_path,
    fake_path,
    spoof_remove_df,
    bona_remove_df,
    output_root=None,
    work_on_copy=False,
):
    """
    Move (or optionally copy and then move) spoof and bona-fide WAV files based on
    balancing logic.

    Args:
        real_path (str):
            Path to directory containing bona-fide (real) .wav files.
        fake_path (str):
            Path to directory containing spoofed (fake) .wav files.
        spoof_remove_df (pd.DataFrame):
            DataFrame containing 'filename' column with spoof files to move.
        bona_remove_df (pd.DataFrame):
            DataFrame containing 'filename' column with bona-fide files to move.
        output_root (str, optional):
            Where to store removed files AND copies (if enabled).
            If None, defaults to creating folders inside the original directories.
        work_on_copy (bool):
            If True:
                - Creates a full copy of both real and fake directories under output_root
                - Performs all operations on the copied structure
            If False:
                - Modifies the original dataset in-place.

    Returns:
        dict:
            Paths of final working directories:
                {
                    "real_dir": <path used for real files>,
                    "fake_dir": <path used for fake files>,
                    "removed_bona_dir": <dir for removed real>,
                    "removed_spoof_dir": <dir for removed fake>
                }
    """

    # Convert to Paths
    real_path = Path(real_path)
    fake_path = Path(fake_path)

    # If output_root is not given, place removed dirs next to original dataset
    if output_root is None:
        output_root = real_path.parent
    output_root = Path(output_root)

    # Determine working directories
    if work_on_copy:
        working_real = output_root / "real_copy"
        working_fake = output_root / "fake_copy"

        # Copy dirs only if they don't exist
        if not working_real.exists():
            shutil.copytree(real_path, working_real)
        if not working_fake.exists():
            shutil.copytree(fake_path, working_fake)
    else:
        working_real = real_path
        working_fake = fake_path

    # Create "removed" folders
    removed_bona_dir = output_root / "removed_bona"
    removed_spoof_dir = output_root / "removed_spoof"
    removed_bona_dir.mkdir(parents=True, exist_ok=True)
    removed_spoof_dir.mkdir(parents=True, exist_ok=True)

    # --- Move bona-fide files ---
    for fname in bona_remove_df["filename"]:
        src = working_real / fname
        if src.exists():
            shutil.move(str(src), removed_bona_dir / fname)

    # --- Move spoof files ---
    for fname in spoof_remove_df["filename"]:
        src = working_fake / fname
        if src.exists():
            shutil.move(str(src), removed_spoof_dir / fname)

    return {
        "real_dir": working_real,
        "fake_dir": working_fake,
        "removed_bona_dir": removed_bona_dir,
        "removed_spoof_dir": removed_spoof_dir,
    }
