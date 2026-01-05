import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit, StratifiedGroupKFold, GroupKFold
)
from typing import Tuple, Optional, Generator, List, Dict
from itertools import product

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import TEST_SIZE, RANDOM_STATE, N_OUTER_FOLDS, N_INNER_FOLDS


class PatientWiseSplitter:
    """
    Splits data ensuring all samples from the same patient are in the same split.
    
    This is crucial to avoid data leakage since respiratory cycles from the same
    patient are highly correlated.
    """
    
    def __init__(self, test_size: float = TEST_SIZE, 
                 random_state: int = RANDOM_STATE):
        """
        Initialize the splitter.
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        
    def split(self, X: np.ndarray, y: np.ndarray, 
              patient_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data into train and test sets by patient.
        
        Args:
            X: Feature matrix
            y: Target labels
            patient_ids: Array of patient IDs for each sample
            
        Returns:
            Tuple of (train_indices, test_indices)
        """
        gss = GroupShuffleSplit(
            n_splits=1, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        train_idx, test_idx = next(gss.split(X, y, groups=patient_ids))
        
        return train_idx, test_idx
    
    def split_dataframe(self, df: pd.DataFrame, 
                        patient_col: str = "patient_id",
                        target_col: str = "label") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split a DataFrame into train and test sets by patient.
        
        Args:
            df: Input DataFrame
            patient_col: Name of the patient ID column
            target_col: Name of the target column
            
        Returns:
            Tuple of (train_df, test_df)
        """
        X = df.index.values
        y = df[target_col].values
        patient_ids = df[patient_col].values
        
        train_idx, test_idx = self.split(X, y, patient_ids)
        
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        
        return train_df, test_df
    
    def get_patient_split_info(self, df: pd.DataFrame,
                               patient_col: str = "patient_id",
                               target_col: str = "label") -> dict:
        """
        Get information about the patient-wise split.
        
        Returns:
            Dictionary with split statistics
        """
        train_df, test_df = self.split_dataframe(df, patient_col, target_col)
        
        train_patients = train_df[patient_col].nunique()
        test_patients = test_df[patient_col].nunique()
        
        train_samples = len(train_df)
        test_samples = len(test_df)
        
        train_class_dist = train_df[target_col].value_counts().to_dict()
        test_class_dist = test_df[target_col].value_counts().to_dict()
        
        return {
            "train_patients": train_patients,
            "test_patients": test_patients,
            "train_samples": train_samples,
            "test_samples": test_samples,
            "train_class_distribution": train_class_dist,
            "test_class_distribution": test_class_dist,
            "test_ratio": test_samples / (train_samples + test_samples)
        }


class StratifiedPatientSplitter(PatientWiseSplitter):
    """
    Extension of PatientWiseSplitter that attempts to maintain class balance.
    
    Note: Perfect stratification is not always possible with patient-wise splits
    since patients have different numbers of samples.
    """
    
    def split(self, X: np.ndarray, y: np.ndarray,
              patient_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data attempting to maintain class balance.
        
        This uses a greedy approach to assign patients to train/test sets
        while trying to maintain the target class ratio.
        """
        # Get unique patients and their primary class
        unique_patients = np.unique(patient_ids)
        
        # Determine primary class for each patient (majority vote)
        patient_classes = {}
        patient_counts = {}
        
        for patient in unique_patients:
            patient_mask = patient_ids == patient
            patient_labels = y[patient_mask]
            patient_classes[patient] = np.bincount(patient_labels).argmax()
            patient_counts[patient] = len(patient_labels)
        
        # Separate patients by class
        class_0_patients = [p for p, c in patient_classes.items() if c == 0]
        class_1_patients = [p for p, c in patient_classes.items() if c == 1]
        
        # Shuffle patients
        np.random.seed(self.random_state)
        np.random.shuffle(class_0_patients)
        np.random.shuffle(class_1_patients)
        
        # Split each class separately
        n_test_class_0 = max(1, int(len(class_0_patients) * self.test_size))
        n_test_class_1 = max(1, int(len(class_1_patients) * self.test_size))
        
        test_patients = set(class_0_patients[:n_test_class_0] + 
                           class_1_patients[:n_test_class_1])
        train_patients = set(unique_patients) - test_patients
        
        # Convert to indices
        train_idx = np.where(np.isin(patient_ids, list(train_patients)))[0]
        test_idx = np.where(np.isin(patient_ids, list(test_patients)))[0]
        
        return train_idx, test_idx
