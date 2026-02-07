import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(PROJECT_ROOT, "FoR_dataset", "for-norm", "for-norm")
ITW_DATASET_PATH = os.path.join(PROJECT_ROOT, "in-the-wild-audio-deepfake")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "FoR_dataset", "features")
DATASET_PATH_FEATURES = FEATURES_DIR
ITW_FEATURES_DIR = os.path.join(PROJECT_ROOT, "in-the-wild-audio-deepfake", "features")
ELEVEN_LABS_FEATURES_PATH = os.path.join(PROJECT_ROOT, "elevenlabs-dataset", "features")
ELEVEN_LABS_DATASET_PATH = os.path.join(PROJECT_ROOT, "elevenlabs-dataset")
MODELS_PATH = os.path.join(PROJECT_ROOT, "notebooks", "experiments")
FINAL_MODELS_PATH = os.path.join(PROJECT_ROOT, "notebooks", "experiments", "final")