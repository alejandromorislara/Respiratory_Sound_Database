"""
Data splitting utilities with patient-wise splits to avoid data leakage.

This module provides:
1. PatientWiseSplitter: Basic patient-wise train/test split
2. StratifiedPatientSplitter: Patient-wise split with class balance
3. NestedCrossValidator: Nested CV for rigorous hyperparameter tuning
4. RepeatedStratifiedGroupKFold: 5x2cv scheme for statistical testing
"""
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


class NestedCrossValidator:
    """
    Nested Cross-Validation with patient-wise grouping.
    
    This class implements a rigorous evaluation protocol that separates
    model selection (hyperparameter tuning) from performance estimation,
    avoiding optimistic bias in reported metrics.
    
    Structure:
    - Outer loop (k folds): Estimates true generalization performance
    - Inner loop (m folds): Selects optimal hyperparameters
    
    The outer loop uses StratifiedGroupKFold to maintain:
    1. Patient independence: No patient appears in both train and test
    2. Class balance: Approximate stratification by diagnosis
    """
    
    def __init__(self, outer_splits: int = N_OUTER_FOLDS,
                 inner_splits: int = N_INNER_FOLDS,
                 random_state: int = RANDOM_STATE):
        """
        Initialize the nested cross-validator.
        
        Args:
            outer_splits: Number of folds for outer loop (performance estimation)
            inner_splits: Number of folds for inner loop (hyperparameter tuning)
            random_state: Random seed for reproducibility
        """
        self.outer_splits = outer_splits
        self.inner_splits = inner_splits
        self.random_state = random_state
        
        # Results storage
        self.outer_results_ = []
        self.inner_results_ = []
        self.best_params_ = []
        
    def split(self, X: np.ndarray, y: np.ndarray,
              groups: np.ndarray) -> Generator:
        """
        Generate nested cross-validation splits.
        
        Args:
            X: Feature matrix
            y: Target labels
            groups: Patient IDs for grouping
            
        Yields:
            Tuple of (outer_fold_idx, train_idx, test_idx, inner_cv)
            where inner_cv is a generator of (inner_train_idx, inner_val_idx)
        """
        # Outer CV splitter
        outer_cv = StratifiedGroupKFold(
            n_splits=self.outer_splits,
            shuffle=True,
            random_state=self.random_state
        )
        
        for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
            # Extract training data for inner CV
            X_train = X[train_idx]
            y_train = y[train_idx]
            groups_train = groups[train_idx]
            
            # Inner CV splitter
            inner_cv = StratifiedGroupKFold(
                n_splits=self.inner_splits,
                shuffle=True,
                random_state=self.random_state + outer_fold
            )
            
            # Generate inner splits (relative to training indices)
            def inner_generator():
                for inner_train_rel, inner_val_rel in inner_cv.split(
                    X_train, y_train, groups_train
                ):
                    # Convert relative indices to absolute
                    inner_train_abs = train_idx[inner_train_rel]
                    inner_val_abs = train_idx[inner_val_rel]
                    yield inner_train_abs, inner_val_abs
            
            yield outer_fold, train_idx, test_idx, inner_generator()
    
    def get_outer_splits(self, X: np.ndarray, y: np.ndarray,
                         groups: np.ndarray) -> Generator:
        """
        Get only outer splits (for simpler iteration when inner CV is manual).
        
        Args:
            X: Feature matrix
            y: Target labels
            groups: Patient IDs
            
        Yields:
            Tuple of (fold_idx, train_idx, test_idx)
        """
        outer_cv = StratifiedGroupKFold(
            n_splits=self.outer_splits,
            shuffle=True,
            random_state=self.random_state
        )
        
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
            yield fold_idx, train_idx, test_idx
    
    def get_split_info(self, X: np.ndarray, y: np.ndarray,
                       groups: np.ndarray) -> Dict:
        """
        Get information about the nested CV splits.
        
        Returns:
            Dictionary with split statistics
        """
        outer_cv = StratifiedGroupKFold(
            n_splits=self.outer_splits,
            shuffle=True,
            random_state=self.random_state
        )
        
        info = {
            'outer_splits': self.outer_splits,
            'inner_splits': self.inner_splits,
            'folds': []
        }
        
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
            fold_info = {
                'fold': fold_idx,
                'train_samples': len(train_idx),
                'test_samples': len(test_idx),
                'train_patients': len(np.unique(groups[train_idx])),
                'test_patients': len(np.unique(groups[test_idx])),
                'train_class_dist': dict(zip(*np.unique(y[train_idx], return_counts=True))),
                'test_class_dist': dict(zip(*np.unique(y[test_idx], return_counts=True)))
            }
            info['folds'].append(fold_info)
        
        return info
    
    def print_summary(self, X: np.ndarray, y: np.ndarray,
                      groups: np.ndarray) -> None:
        """Print a summary of the nested CV configuration."""
        info = self.get_split_info(X, y, groups)
        
        print("\n" + "=" * 60)
        print("Nested Cross-Validation Configuration")
        print("=" * 60)
        print(f"Outer folds: {info['outer_splits']}")
        print(f"Inner folds: {info['inner_splits']}")
        print(f"Total evaluations: {info['outer_splits']} × {info['inner_splits']}")
        
        print("\nOuter fold distribution:")
        for fold in info['folds']:
            print(f"  Fold {fold['fold']+1}:")
            print(f"    Train: {fold['train_samples']} samples, {fold['train_patients']} patients")
            print(f"    Test:  {fold['test_samples']} samples, {fold['test_patients']} patients")
        print("=" * 60 + "\n")


class Repeated5x2CVSplitter:
    """
    5x2 Cross-Validation splitter for statistical comparison of algorithms.
    
    This implements the 5×2cv scheme recommended by Dietterich (1998) for
    comparing machine learning algorithms when computational cost is high.
    
    The scheme performs 5 replications of 2-fold cross-validation:
    - Each replication uses different random splits
    - Provides data for the paired 5×2cv t-test
    
    Advantages:
    - More reliable than single train/test split
    - Lower computational cost than 10-fold CV
    - Suitable for expensive algorithms (quantum simulation)
    - Provides variance estimates across repetitions
    """
    
    def __init__(self, n_repetitions: int = 5,
                 random_state: int = RANDOM_STATE):
        """
        Initialize the 5x2cv splitter.
        
        Args:
            n_repetitions: Number of 2-fold CV repetitions (default 5)
            random_state: Base random seed
        """
        self.n_repetitions = n_repetitions
        self.random_state = random_state
        
    def split(self, X: np.ndarray, y: np.ndarray,
              groups: np.ndarray) -> Generator:
        """
        Generate 5x2cv splits respecting patient groups.
        
        Args:
            X: Feature matrix
            y: Target labels
            groups: Patient IDs
            
        Yields:
            Tuple of (repetition_idx, fold_idx, train_idx, test_idx)
        """
        for rep in range(self.n_repetitions):
            # Create 2-fold CV with different seed for each repetition
            cv = StratifiedGroupKFold(
                n_splits=2,
                shuffle=True,
                random_state=self.random_state + rep * 100
            )
            
            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
                yield rep, fold_idx, train_idx, test_idx
    
    def get_all_splits(self, X: np.ndarray, y: np.ndarray,
                       groups: np.ndarray) -> List[Tuple]:
        """
        Get all splits as a list for easy iteration.
        
        Returns:
            List of (rep, fold, train_idx, test_idx) tuples
        """
        return list(self.split(X, y, groups))
    
    def compute_5x2cv_statistics(self, 
                                  scores_1: np.ndarray,
                                  scores_2: np.ndarray) -> Dict:
        """
        Compute 5x2cv t-test statistics for algorithm comparison.
        
        The 5x2cv t-test (Dietterich, 1998) is computed as:
        t = p̄₁ / sqrt((1/5) * Σᵢ sᵢ²)
        
        where:
        - p̄₁ is the mean difference in the first fold of each repetition
        - sᵢ² is the variance of the differences in repetition i
        
        Args:
            scores_1: Scores from algorithm 1, shape (n_repetitions, 2)
            scores_2: Scores from algorithm 2, shape (n_repetitions, 2)
            
        Returns:
            Dictionary with t-statistic and p-value
        """
        from scipy.stats import t as t_dist
        
        # Compute differences
        diffs = scores_1 - scores_2  # Shape: (5, 2)
        
        # Mean of first fold differences
        p_bar_1 = diffs[:, 0].mean()
        
        # Variance for each repetition
        variances = np.zeros(self.n_repetitions)
        for i in range(self.n_repetitions):
            p_i1 = diffs[i, 0]
            p_i2 = diffs[i, 1]
            p_mean = (p_i1 + p_i2) / 2
            variances[i] = (p_i1 - p_mean)**2 + (p_i2 - p_mean)**2
        
        # Compute t-statistic
        denominator = np.sqrt((1/self.n_repetitions) * variances.sum())
        
        if denominator == 0:
            return {'t_statistic': 0, 'p_value': 1.0, 'significant': False}
        
        t_stat = p_bar_1 / denominator
        
        # Degrees of freedom = 5 for 5x2cv
        df = self.n_repetitions
        
        # Two-tailed p-value
        p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df))
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'mean_difference': diffs.mean(),
            'std_difference': diffs.std()
        }

