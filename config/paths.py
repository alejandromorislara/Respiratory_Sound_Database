"""
Path configurations for the Quantum Respiratory Classification project.
"""
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Raw data paths
RAW_DATA_DIR = DATA_DIR / "raw"
AUDIO_DIR = PROJECT_ROOT / "audio_and_txt_files"  # Original location
PATIENT_DIAGNOSIS_FILE = PROJECT_ROOT / "patient_diagnosis.csv"

# Processed data paths
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_FILE = PROCESSED_DATA_DIR / "features.csv"
FEATURES_PCA_FILE = PROCESSED_DATA_DIR / "features_pca.csv"

# Split data paths
SPLITS_DIR = DATA_DIR / "splits"
TRAIN_FILE = SPLITS_DIR / "train.csv"
TEST_FILE = SPLITS_DIR / "test.csv"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
QSVM_MODEL_FILE = MODELS_DIR / "qsvm_model.pkl"
QNN_MODEL_FILE = MODELS_DIR / "qnn_model.pkl"
HYBRID_MODEL_FILE = MODELS_DIR / "hybrid_model.pt"

# Results paths
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def ensure_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SPLITS_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        FIGURES_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Create directories on import
ensure_directories()

