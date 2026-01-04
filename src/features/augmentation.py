"""
Audio Data Augmentation for Respiratory Sounds.

Techniques implemented:
1. Time Stretching: Accelerate/decelerate audio without pitch change
2. Pitch Shifting: Shift pitch without duration change
3. Noise Injection: Add controlled white noise for robustness
"""
import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Union
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import SAMPLE_RATE, RANDOM_STATE


class AudioAugmenter:
    """
    Audio augmentation for respiratory sound classification.
    
    Applies augmentation techniques to balance class distribution
    by generating synthetic samples from the minority class.
    """
    
    def __init__(self, 
                 sr: int = SAMPLE_RATE,
                 random_state: int = RANDOM_STATE):
        """
        Initialize the augmenter.
        
        Args:
            sr: Sample rate for audio processing
            random_state: Random seed for reproducibility
        """
        self.sr = sr
        self.random_state = random_state
        np.random.seed(random_state)
    
    def time_stretch(self, audio: np.ndarray, 
                     rate: Optional[float] = None) -> np.ndarray:
        """
        Apply time stretching to audio.
        
        Stretches or compresses the audio in time without changing the pitch.
        Rate > 1.0 speeds up (shorter), rate < 1.0 slows down (longer).
        
        Args:
            audio: Input audio signal
            rate: Stretch factor. If None, randomly chosen from [0.8, 1.2]
            
        Returns:
            Time-stretched audio
        """
        if rate is None:
            rate = np.random.uniform(0.8, 1.2)
        
        # Ensure rate is within reasonable bounds
        rate = np.clip(rate, 0.5, 2.0)
        
        try:
            stretched = librosa.effects.time_stretch(audio, rate=rate)
            return stretched
        except Exception:
            # Return original if stretching fails
            return audio
    
    def pitch_shift(self, audio: np.ndarray,
                    n_steps: Optional[float] = None) -> np.ndarray:
        """
        Apply pitch shifting to audio.
        
        Shifts the pitch up or down by a number of semitones
        without changing the duration.
        
        Args:
            audio: Input audio signal
            n_steps: Number of semitones to shift. If None, randomly chosen from [-2, 2]
            
        Returns:
            Pitch-shifted audio
        """
        if n_steps is None:
            n_steps = np.random.uniform(-2, 2)
        
        try:
            shifted = librosa.effects.pitch_shift(
                audio, 
                sr=self.sr, 
                n_steps=n_steps
            )
            return shifted
        except Exception:
            # Return original if shifting fails
            return audio
    
    def add_noise(self, audio: np.ndarray,
                  noise_factor: Optional[float] = None) -> np.ndarray:
        """
        Add white noise to audio.
        
        Adds Gaussian white noise at a controlled SNR level to make
        the model robust to recording quality variations.
        
        Args:
            audio: Input audio signal
            noise_factor: Noise amplitude relative to signal. If None, randomly chosen from [0.001, 0.01]
            
        Returns:
            Noisy audio
        """
        if noise_factor is None:
            noise_factor = np.random.uniform(0.001, 0.01)
        
        # Generate white noise
        noise = np.random.randn(len(audio))
        
        # Scale noise relative to signal amplitude
        signal_power = np.sqrt(np.mean(audio ** 2))
        noise_power = noise_factor * signal_power
        
        noisy_audio = audio + noise_power * noise
        
        return noisy_audio
    
    def augment_audio(self, audio: np.ndarray,
                      techniques: List[str] = None) -> List[Tuple[np.ndarray, str]]:
        """
        Apply multiple augmentation techniques to a single audio.
        
        Args:
            audio: Input audio signal
            techniques: List of techniques to apply. Options: 'time_stretch', 'pitch_shift', 'noise'
                       If None, applies all techniques.
            
        Returns:
            List of (augmented_audio, technique_name) tuples
        """
        if techniques is None:
            techniques = ['time_stretch', 'pitch_shift', 'noise']
        
        augmented = []
        
        for technique in techniques:
            if technique == 'time_stretch':
                aug_audio = self.time_stretch(audio)
                augmented.append((aug_audio, 'time_stretch'))
            elif technique == 'pitch_shift':
                aug_audio = self.pitch_shift(audio)
                augmented.append((aug_audio, 'pitch_shift'))
            elif technique == 'noise':
                aug_audio = self.add_noise(audio)
                augmented.append((aug_audio, 'noise'))
            elif technique == 'combined':
                # Apply all transformations sequentially
                aug_audio = self.time_stretch(audio)
                aug_audio = self.pitch_shift(aug_audio)
                aug_audio = self.add_noise(aug_audio)
                augmented.append((aug_audio, 'combined'))
        
        return augmented
    
    def augment_segment_from_file(self, 
                                   audio_path: Path,
                                   start: float,
                                   end: float,
                                   techniques: List[str] = None) -> List[Tuple[np.ndarray, str]]:
        """
        Load an audio segment and apply augmentation.
        
        Args:
            audio_path: Path to the audio file
            start: Start time in seconds
            end: End time in seconds
            techniques: Augmentation techniques to apply
            
        Returns:
            List of (augmented_audio, technique_name) tuples
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sr)
        
        # Extract segment
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)
        
        if end_sample <= start_sample:
            return []
        
        segment = audio[start_sample:end_sample]
        
        return self.augment_audio(segment, techniques)
    
    def augment_minority_class(self,
                               df: pd.DataFrame,
                               audio_dir: Path,
                               feature_extractor,
                               target_ratio: float = 1.0,
                               minority_label: int = 0,
                               techniques: List[str] = None,
                               verbose: bool = True) -> pd.DataFrame:
        """
        Augment minority class samples to achieve target balance ratio.
        
        This is the main pipeline function that:
        1. Identifies minority class samples
        2. Applies audio augmentation
        3. Extracts features from augmented audio
        4. Returns balanced dataframe
        
        Args:
            df: DataFrame with columns: filename, start, end, label, and feature columns
            audio_dir: Directory containing audio files
            feature_extractor: AudioFeatureExtractor instance
            target_ratio: Target ratio minority/majority (1.0 = balanced)
            minority_label: Label of minority class (0 = Healthy)
            techniques: Augmentation techniques to apply
            verbose: Whether to show progress
            
        Returns:
            DataFrame with original + augmented samples
        """
        if techniques is None:
            techniques = ['time_stretch', 'pitch_shift', 'noise']
        
        # Count samples per class
        class_counts = df['label'].value_counts()
        n_minority = class_counts.get(minority_label, 0)
        n_majority = class_counts.get(1 - minority_label, 0)
        
        # Calculate how many samples we need to generate
        target_minority = int(n_majority * target_ratio)
        n_to_generate = target_minority - n_minority
        
        if n_to_generate <= 0:
            if verbose:
                print("Classes already balanced or minority is larger.")
            return df
        
        if verbose:
            print(f"Current balance: {n_minority} minority vs {n_majority} majority")
            print(f"Target: {target_minority} minority samples")
            print(f"Generating {n_to_generate} augmented samples...")
        
        # Get minority class samples
        minority_df = df[df['label'] == minority_label].copy()
        
        # Calculate augmentations per sample
        n_techniques = len(techniques)
        samples_per_original = n_techniques
        n_iterations = int(np.ceil(n_to_generate / (len(minority_df) * samples_per_original)))
        
        augmented_rows = []
        generated_count = 0
        
        iterator = minority_df.iterrows()
        if verbose:
            iterator = tqdm(list(iterator), desc="Augmenting minority class")
        
        for _ in range(n_iterations):
            if generated_count >= n_to_generate:
                break
                
            for idx, row in iterator:
                if generated_count >= n_to_generate:
                    break
                
                audio_path = audio_dir / row['filename']
                if not audio_path.exists():
                    continue
                
                try:
                    # Get augmented audio segments
                    augmented_list = self.augment_segment_from_file(
                        audio_path,
                        row['start'],
                        row['end'],
                        techniques
                    )
                    
                    for aug_audio, technique in augmented_list:
                        if generated_count >= n_to_generate:
                            break
                        
                        # Extract features from augmented audio
                        features = feature_extractor.extract_from_segment(aug_audio)
                        
                        # Create new row
                        new_row = row.copy()
                        new_row['augmented'] = True
                        new_row['aug_technique'] = technique
                        
                        # Update features
                        for i, feat_name in enumerate(feature_extractor.feature_names):
                            new_row[feat_name] = features[i]
                        
                        augmented_rows.append(new_row)
                        generated_count += 1
                        
                except Exception as e:
                    if verbose:
                        print(f"Error augmenting {row['filename']}: {e}")
                    continue
        
        # Create augmented dataframe
        if augmented_rows:
            augmented_df = pd.DataFrame(augmented_rows)
            
            # Add augmented column to original if not present
            if 'augmented' not in df.columns:
                df = df.copy()
                df['augmented'] = False
                df['aug_technique'] = 'original'
            
            # Combine original and augmented
            result_df = pd.concat([df, augmented_df], ignore_index=True)
            
            if verbose:
                new_counts = result_df['label'].value_counts()
                print(f"\nAfter augmentation:")
                print(f"  Minority (Healthy): {new_counts.get(minority_label, 0)}")
                print(f"  Majority (COPD): {new_counts.get(1-minority_label, 0)}")
                print(f"  Total samples: {len(result_df)}")
            
            return result_df
        
        return df


def augment_features_smote_style(X: np.ndarray, 
                                  y: np.ndarray,
                                  target_ratio: float = 1.0,
                                  minority_label: int = 0,
                                  random_state: int = RANDOM_STATE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple SMOTE-like augmentation directly on feature vectors.
    
    Creates synthetic samples by interpolating between minority class samples.
    This is faster than audio augmentation but less realistic.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels
        target_ratio: Target ratio minority/majority
        minority_label: Label of minority class
        random_state: Random seed
        
    Returns:
        Tuple of (augmented_X, augmented_y)
    """
    np.random.seed(random_state)
    
    # Separate classes
    X_minority = X[y == minority_label]
    X_majority = X[y != minority_label]
    
    n_minority = len(X_minority)
    n_majority = len(X_majority)
    
    # Calculate samples to generate
    target_minority = int(n_majority * target_ratio)
    n_to_generate = target_minority - n_minority
    
    if n_to_generate <= 0:
        return X, y
    
    # Generate synthetic samples by interpolation (SMOTE-like)
    synthetic_samples = []
    
    for _ in range(n_to_generate):
        # Randomly select two minority samples
        idx1, idx2 = np.random.choice(n_minority, 2, replace=False)
        
        # Interpolate
        alpha = np.random.uniform(0, 1)
        synthetic = X_minority[idx1] + alpha * (X_minority[idx2] - X_minority[idx1])
        synthetic_samples.append(synthetic)
    
    # Combine
    X_synthetic = np.array(synthetic_samples)
    y_synthetic = np.full(n_to_generate, minority_label)
    
    X_augmented = np.vstack([X, X_synthetic])
    y_augmented = np.concatenate([y, y_synthetic])
    
    return X_augmented, y_augmented

