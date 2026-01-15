"""
Audio feature extraction and augmentation for respiratory sounds.
"""
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import SAMPLE_RATE, N_MFCC, HOP_LENGTH, N_FFT, TARGET_CLASSES
from config.paths import AUDIO_DIR, PATIENT_DIAGNOSIS_FILE


class AudioFeatureExtractor:
    """
    Extracts audio features (MFCCs and deltas) from respiratory sound segments.
    Also provides audio augmentation methods for data balancing.
    """
    
    def __init__(self, n_mfcc: int = N_MFCC, 
                 sr: int = SAMPLE_RATE,
                 hop_length: int = HOP_LENGTH,
                 n_fft: int = N_FFT):
        self.n_mfcc = n_mfcc
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.feature_names = self._generate_feature_names()
        
    def _generate_feature_names(self) -> List[str]:
        """Generate feature names based on n_mfcc."""
        names = []
        for stat in ["mean", "std"]:
            for i in range(self.n_mfcc):
                names.append(f"mfcc_{i}_{stat}")
        for stat in ["mean", "std"]:
            for i in range(self.n_mfcc):
                names.append(f"delta_mfcc_{i}_{stat}")
        return names
    
    def time_stretch(self, audio: np.ndarray, rate: Optional[float] = None) -> np.ndarray:
        """
        Apply time stretching to audio without changing pitch.
        Rate > 1.0 speeds up, rate < 1.0 slows down.
        """
        if rate is None:
            rate = np.random.uniform(0.8, 1.2)
        rate = np.clip(rate, 0.5, 2.0)
        try:
            return librosa.effects.time_stretch(audio, rate=rate)
        except Exception:
            return audio
    
    def pitch_shift(self, audio: np.ndarray, n_steps: Optional[float] = None) -> np.ndarray:
        """
        Shift pitch by n_steps semitones without changing duration.
        """
        if n_steps is None:
            n_steps = np.random.uniform(-2, 2)
        try:
            return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)
        except Exception:
            return audio
    
    def add_noise(self, audio: np.ndarray, noise_factor: Optional[float] = None) -> np.ndarray:
        """
        Add white noise to audio for robustness.
        """
        if noise_factor is None:
            noise_factor = np.random.uniform(0.001, 0.01)
        noise = np.random.randn(len(audio))
        signal_power = np.sqrt(np.mean(audio ** 2))
        noise_power = noise_factor * signal_power
        return audio + noise_power * noise
    
    def augment_audio(self, audio: np.ndarray, 
                      n_augmentations: int = 3,
                      use_time_stretch: bool = True,
                      use_pitch_shift: bool = True,
                      use_noise: bool = True) -> List[np.ndarray]:
        """
        Generate multiple augmented versions of an audio sample.
        Combines time stretching, pitch shifting, and noise addition.
        """
        augmented = [audio]
        
        for _ in range(n_augmentations):
            aug = audio.copy()
            if use_time_stretch and np.random.random() > 0.5:
                aug = self.time_stretch(aug)
            if use_pitch_shift and np.random.random() > 0.5:
                aug = self.pitch_shift(aug)
            if use_noise and np.random.random() > 0.5:
                aug = self.add_noise(aug)
            augmented.append(aug)
        
        return augmented
    
    def extract_from_segment(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from a single audio segment.
        Returns feature vector of 52 dimensions.
        """
        if len(audio) < self.n_fft:
            audio = np.pad(audio, (0, self.n_fft - len(audio)), mode='constant')
        
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=self.sr, 
            n_mfcc=self.n_mfcc,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        delta_mfccs = librosa.feature.delta(mfccs)
        
        features = np.concatenate([
            mfccs.mean(axis=1),
            mfccs.std(axis=1),
            delta_mfccs.mean(axis=1),
            delta_mfccs.std(axis=1)
        ])
        return features
    
    def extract_from_file(self, audio_path: Path) -> List[Tuple[np.ndarray, dict]]:
        """Extract features from all respiratory cycles in an audio file."""
        audio, sr = librosa.load(audio_path, sr=self.sr)
        
        annotation_path = audio_path.with_suffix(".txt")
        if not annotation_path.exists():
            return []
        
        cycles = self._load_annotations(annotation_path)
        results = []
        
        for cycle in cycles:
            start_sample = int(cycle["start"] * sr)
            end_sample = int(cycle["end"] * sr)
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            
            if end_sample > start_sample:
                segment = audio[start_sample:end_sample]
                features = self.extract_from_segment(segment)
                results.append((features, cycle))
        
        return results
    
    def _load_annotations(self, annotation_path: Path) -> List[dict]:
        """Load respiratory cycle annotations from file."""
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
    
    def extract_dataset(self, audio_dir: Optional[Path] = None,
                        diagnosis_file: Optional[Path] = None,
                        filter_classes: Optional[List[str]] = None,
                        save_path: Optional[Path] = None,
                        show_progress: bool = True) -> pd.DataFrame:
        """
        Extract features from all audio files in a directory.
        """
        audio_dir = audio_dir or AUDIO_DIR
        diagnosis_file = diagnosis_file or PATIENT_DIAGNOSIS_FILE
        
        diagnosis_df = pd.read_csv(diagnosis_file, header=None,
                                   names=["patient_id", "diagnosis"])
        
        audio_files = sorted(audio_dir.glob("*.wav"))
        
        if filter_classes:
            valid_patients = diagnosis_df[
                diagnosis_df["diagnosis"].isin(filter_classes)
            ]["patient_id"].values
            audio_files = [
                f for f in audio_files
                if int(f.stem.split("_")[0]) in valid_patients
            ]
        
        all_data = []
        iterator = tqdm(audio_files, desc="Processing files") if show_progress else audio_files
        
        for audio_path in iterator:
            patient_id = int(audio_path.stem.split("_")[0])
            diagnosis = diagnosis_df[
                diagnosis_df["patient_id"] == patient_id
            ]["diagnosis"].values[0]
            
            results = self.extract_from_file(audio_path)
            
            for features, cycle_info in results:
                row = {
                    "patient_id": patient_id,
                    "filename": audio_path.name,
                    "diagnosis": diagnosis,
                    "start": cycle_info["start"],
                    "end": cycle_info["end"],
                    "crackles": cycle_info["crackles"],
                    "wheezes": cycle_info["wheezes"],
                    "label": TARGET_CLASSES.get(diagnosis, -1)
                }
                for i, feat_name in enumerate(self.feature_names):
                    row[feat_name] = features[i]
                all_data.append(row)
        
        df = pd.DataFrame(all_data)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"Features saved to {save_path}")
        
        return df
    
    def get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Extract feature matrix from a DataFrame with feature columns."""
        return df[self.feature_names].values
