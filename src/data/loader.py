import pandas as pd
import numpy as np
import librosa
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import SAMPLE_RATE, TARGET_CLASSES
from config.paths import AUDIO_DIR, PATIENT_DIAGNOSIS_FILE


class DataLoader:
    """
    Loads and processes respiratory sound data from the ICBHI database.
    
    This class handles:
    - Loading patient diagnosis information
    - Reading audio files and their annotations
    - Segmenting audio into respiratory cycles
    """
    
    def __init__(self, audio_dir: Optional[Path] = None, 
                 diagnosis_file: Optional[Path] = None):
        """
        Initialize the DataLoader.
        
        Args:
            audio_dir: Path to directory containing audio and annotation files
            diagnosis_file: Path to patient diagnosis CSV file
        """
        self.audio_dir = audio_dir or AUDIO_DIR
        self.diagnosis_file = diagnosis_file or PATIENT_DIAGNOSIS_FILE
        self.sample_rate = SAMPLE_RATE
        self._diagnosis_df = None
        
    @property
    def diagnosis_df(self) -> pd.DataFrame:
        """Lazy load the diagnosis dataframe."""
        if self._diagnosis_df is None:
            self._diagnosis_df = self._load_diagnosis()
        return self._diagnosis_df
    
    def _load_diagnosis(self) -> pd.DataFrame:
        """Load patient diagnosis data from CSV."""
        df = pd.read_csv(self.diagnosis_file, header=None, 
                         names=["patient_id", "diagnosis"])
        return df
    
    def get_patient_diagnosis(self, patient_id: int) -> str:
        """Get diagnosis for a specific patient."""
        row = self.diagnosis_df[self.diagnosis_df["patient_id"] == patient_id]
        if len(row) == 0:
            raise ValueError(f"Patient {patient_id} not found")
        return row["diagnosis"].values[0]
    
    def get_audio_files(self, filter_classes: Optional[List[str]] = None) -> List[Path]:
        """
        Get list of audio files, optionally filtered by diagnosis class.
        
        Args:
            filter_classes: List of diagnosis classes to include (e.g., ["Healthy", "COPD"])
            
        Returns:
            List of paths to audio files
        """
        audio_files = list(self.audio_dir.glob("*.wav"))
        
        if filter_classes is not None:
            # Filter by diagnosis
            valid_patients = self.diagnosis_df[
                self.diagnosis_df["diagnosis"].isin(filter_classes)
            ]["patient_id"].values
            
            audio_files = [
                f for f in audio_files 
                if self._extract_patient_id(f) in valid_patients
            ]
        
        return sorted(audio_files)
    
    def _extract_patient_id(self, audio_path: Path) -> int:
        """Extract patient ID from audio filename."""
        # Filename format: PatientID_RecordingIndex_Location_Mode_Equipment.wav
        return int(audio_path.stem.split("_")[0])
    
    def _load_annotations(self, audio_path: Path) -> List[Dict]:
        """
        Load respiratory cycle annotations for an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            List of dictionaries with cycle information:
            - start: Start time in seconds
            - end: End time in seconds
            - crackles: 1 if crackles present, 0 otherwise
            - wheezes: 1 if wheezes present, 0 otherwise
        """
        annotation_path = audio_path.with_suffix(".txt")
        
        if not annotation_path.exists():
            return []
        
        cycles = []
        with open(annotation_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    cycles.append({
                        "start": float(parts[0]),
                        "end": float(parts[1]),
                        "crackles": int(parts[2]),
                        "wheezes": int(parts[3])
                    })
        
        return cycles
    
    def load_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """
        Load an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple of (audio signal, sample rate)
        """
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        return audio, sr
    
    def segment_audio(self, audio: np.ndarray, sr: int, 
                      cycles: List[Dict]) -> List[Tuple[np.ndarray, Dict]]:
        """
        Segment audio into respiratory cycles.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            cycles: List of cycle annotations
            
        Returns:
            List of tuples (segment_audio, cycle_info)
        """
        segments = []
        
        for cycle in cycles:
            start_sample = int(cycle["start"] * sr)
            end_sample = int(cycle["end"] * sr)
            
            # Ensure valid indices
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            
            if end_sample > start_sample:
                segment = audio[start_sample:end_sample]
                segments.append((segment, cycle))
        
        return segments
    
    def load_all_segments(self, filter_classes: Optional[List[str]] = None,
                          show_progress: bool = True) -> pd.DataFrame:
        """
        Load all audio segments with their metadata.
        
        Args:
            filter_classes: List of diagnosis classes to include
            show_progress: Whether to show progress bar
            
        Returns:
            DataFrame with columns: patient_id, filename, diagnosis, 
                                   start, end, crackles, wheezes, audio
        """
        audio_files = self.get_audio_files(filter_classes)
        
        all_segments = []
        iterator = tqdm(audio_files, desc="Loading audio") if show_progress else audio_files
        
        for audio_path in iterator:
            patient_id = self._extract_patient_id(audio_path)
            diagnosis = self.get_patient_diagnosis(patient_id)
            
            # Skip if not in target classes
            if filter_classes and diagnosis not in filter_classes:
                continue
            
            audio, sr = self.load_audio(audio_path)
            cycles = self._load_annotations(audio_path)
            segments = self.segment_audio(audio, sr, cycles)
            
            for segment_audio, cycle_info in segments:
                all_segments.append({
                    "patient_id": patient_id,
                    "filename": audio_path.name,
                    "diagnosis": diagnosis,
                    "start": cycle_info["start"],
                    "end": cycle_info["end"],
                    "crackles": cycle_info["crackles"],
                    "wheezes": cycle_info["wheezes"],
                    "audio": segment_audio,
                    "label": TARGET_CLASSES.get(diagnosis, -1)
                })
        
        return pd.DataFrame(all_segments)


def get_class_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """Get the distribution of classes in the dataset."""
    return df["diagnosis"].value_counts().to_dict()

