from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
AUDIO_DIR = PROJECT_ROOT / "audio_and_txt_files"
PATIENT_DIAGNOSIS_FILE = PROJECT_ROOT / "patient_diagnosis.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"


def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, RESULTS_DIR / "figures"]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


ensure_directories()
