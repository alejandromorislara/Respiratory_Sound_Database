"""
Feature-level balancing utilities for imbalanced datasets.

This module provides SMOTE and other balancing techniques that operate
on extracted feature vectors (post-MFCC extraction).
"""
import numpy as np
from typing import Tuple, Dict, Optional, Union
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import RANDOM_STATE

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN


class FeatureBalancer:
    """
    Balances feature vectors using various oversampling techniques.
    
    Supports:
    - SMOTE (Synthetic Minority Over-sampling Technique)
    - ADASYN (Adaptive Synthetic Sampling)
    - BorderlineSMOTE (focuses on borderline samples)
    - Custom interpolation-based augmentation
    """
    
    def __init__(self, random_state: int = RANDOM_STATE):
        """
        Initialize the balancer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
    
    def apply_smote(self, 
                    X: np.ndarray, 
                    y: np.ndarray,
                    sampling_strategy: Union[str, float, Dict] = 'minority',
                    k_neighbors: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE to balance classes.

        SMOTE creates synthetic samples by interpolating between
        existing minority class samples and their k nearest neighbors.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels
            sampling_strategy: 'minority', 'not majority', 'all', or float ratio
            k_neighbors: Number of neighbors for SMOTE

        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        # Adjust k_neighbors if minority class is too small
        n_minority = min(np.bincount(y.astype(int)))
        k_neighbors = min(k_neighbors, n_minority - 1)
        k_neighbors = max(1, k_neighbors)

        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=self.random_state
        )
        return smote.fit_resample(X, y)
    
    def apply_adasyn(self,
                     X: np.ndarray,
                     y: np.ndarray,
                     sampling_strategy: Union[str, float, Dict] = 'minority',
                     n_neighbors: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply ADASYN (Adaptive Synthetic Sampling).

        ADASYN focuses on generating samples near the decision boundary
        where the minority class is harder to learn.

        Args:
            X: Feature matrix
            y: Target labels
            sampling_strategy: Resampling strategy
            n_neighbors: Number of neighbors

        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        # Adjust n_neighbors if minority class is too small
        n_minority = min(np.bincount(y.astype(int)))
        n_neighbors = min(n_neighbors, n_minority - 1)
        n_neighbors = max(1, n_neighbors)

        adasyn = ADASYN(
            sampling_strategy=sampling_strategy,
            n_neighbors=n_neighbors,
            random_state=self.random_state
        )
        return adasyn.fit_resample(X, y)
    
    def apply_borderline_smote(self,
                               X: np.ndarray,
                               y: np.ndarray,
                               sampling_strategy: Union[str, float, Dict] = 'minority',
                               k_neighbors: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Borderline-SMOTE.

        Focuses on generating samples only for minority instances
        that are near the decision boundary (borderline cases).

        Args:
            X: Feature matrix
            y: Target labels
            sampling_strategy: Resampling strategy
            k_neighbors: Number of neighbors

        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        # Adjust k_neighbors if minority class is too small
        n_minority = min(np.bincount(y.astype(int)))
        k_neighbors = min(k_neighbors, n_minority - 1)
        k_neighbors = max(1, k_neighbors)

        bsmote = BorderlineSMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=self.random_state
        )
        return bsmote.fit_resample(X, y)
    
    @staticmethod
    def get_balanced_weights(y: np.ndarray) -> Dict[int, float]:
        """
        Compute balanced class weights for training.
        
        Weights are inversely proportional to class frequency:
        w_i = n_samples / (n_classes * n_samples_i)
        
        Args:
            y: Target labels
            
        Returns:
            Dictionary mapping class labels to weights
        """
        classes, counts = np.unique(y, return_counts=True)
        n_samples = len(y)
        n_classes = len(classes)
        
        weights = {}
        for cls, count in zip(classes, counts):
            weights[int(cls)] = n_samples / (n_classes * count)
        
        return weights
    
    @staticmethod
    def get_sample_weights(y: np.ndarray) -> np.ndarray:
        """
        Compute per-sample weights based on class frequency.
        
        Args:
            y: Target labels
            
        Returns:
            Array of sample weights
        """
        class_weights = FeatureBalancer.get_balanced_weights(y)
        return np.array([class_weights[int(label)] for label in y])
    
    def balance_to_ratio(self,
                         X: np.ndarray,
                         y: np.ndarray,
                         target_ratio: float = 1.0,
                         method: str = 'smote') -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance classes to a specific ratio.
        
        Args:
            X: Feature matrix
            y: Target labels
            target_ratio: Desired ratio of minority to majority (1.0 = equal)
            method: 'smote', 'adasyn', or 'borderline'
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        if method == 'smote':
            return self.apply_smote(X, y, sampling_strategy=target_ratio)
        elif method == 'adasyn':
            return self.apply_adasyn(X, y, sampling_strategy=target_ratio)
        elif method == 'borderline':
            return self.apply_borderline_smote(X, y, sampling_strategy=target_ratio)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def get_balance_report(self, y: np.ndarray) -> Dict:
        """
        Generate a report on class balance.
        
        Args:
            y: Target labels
            
        Returns:
            Dictionary with balance statistics
        """
        classes, counts = np.unique(y, return_counts=True)
        
        report = {
            'total_samples': len(y),
            'n_classes': len(classes),
            'class_counts': dict(zip(classes.tolist(), counts.tolist())),
            'imbalance_ratio': counts.max() / counts.min() if counts.min() > 0 else float('inf'),
            'minority_class': int(classes[np.argmin(counts)]),
            'majority_class': int(classes[np.argmax(counts)]),
            'minority_percentage': 100 * counts.min() / len(y)
        }
        
        return report
    
    def print_balance_report(self, y: np.ndarray, title: str = "Class Balance Report"):
        """
        Print a formatted balance report.
        
        Args:
            y: Target labels
            title: Report title
        """
        report = self.get_balance_report(y)
        
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
        print(f"Total samples: {report['total_samples']}")
        print(f"Number of classes: {report['n_classes']}")
        print(f"\nClass distribution:")
        for cls, count in report['class_counts'].items():
            pct = 100 * count / report['total_samples']
            print(f"  Class {cls}: {count} samples ({pct:.1f}%)")
        print(f"\nImbalance ratio: {report['imbalance_ratio']:.2f}:1")
        print(f"Minority class: {report['minority_class']} ({report['minority_percentage']:.1f}%)")
        print(f"{'='*60}\n")

