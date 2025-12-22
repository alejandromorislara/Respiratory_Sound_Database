"""
Balanced Ensemble methods for quantum classifiers.

This module implements ensemble strategies that address the severe class
imbalance (18:1 ratio) while respecting computational constraints of 
quantum simulation.
"""
import numpy as np
from typing import List, Optional, Union, Callable
from pathlib import Path
import pickle
import time
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import (
    N_ENSEMBLE_ESTIMATORS, ENSEMBLE_VOTING, RANDOM_STATE
)


class BalancedQuantumEnsemble:
    """
    Ensemble of quantum classifiers trained on balanced subsets.
    
    This approach addresses the fundamental tension between:
    1. Severe class imbalance (5746 COPD vs 322 Healthy)
    2. Computational constraints of quantum simulation (O(N²) for kernels)
    3. Need to utilize all available data
    
    Strategy:
    - Divide majority class into M disjoint subsets
    - Train M quantum models, each seeing all minority + subset of majority
    - Aggregate predictions via voting or probability averaging
    
    Advantages:
    - Uses ALL majority class samples (no data waste)
    - Each model sees balanced training data
    - Ensemble reduces prediction variance
    - Respects quantum simulation constraints
    """
    
    def __init__(self, 
                 base_estimator_class: type,
                 base_estimator_params: dict = None,
                 n_estimators: Union[int, str] = N_ENSEMBLE_ESTIMATORS,
                 voting: str = ENSEMBLE_VOTING,
                 random_state: int = RANDOM_STATE):
        """
        Initialize the balanced ensemble.
        
        Args:
            base_estimator_class: Class of the base quantum estimator
            base_estimator_params: Parameters for the base estimator
            n_estimators: Number of estimators ('auto' = ceil(n_maj/n_min))
            voting: 'soft' (probability averaging) or 'hard' (majority vote)
            random_state: Random seed
        """
        self.base_estimator_class = base_estimator_class
        self.base_estimator_params = base_estimator_params or {}
        self.n_estimators = n_estimators
        self.voting = voting
        self.random_state = random_state
        
        # Will be populated during fit
        self.estimators_: List = []
        self.n_estimators_: int = 0
        self.classes_: np.ndarray = None
        self.training_time: float = 0
        self._is_fitted = False
        
    def _compute_n_estimators(self, n_majority: int, n_minority: int) -> int:
        """Compute number of estimators based on class ratio."""
        if isinstance(self.n_estimators, str) and self.n_estimators == 'auto':
            # At least 3 estimators for stability
            return max(3, int(np.ceil(n_majority / n_minority)))
        return self.n_estimators
    
    def _create_balanced_subsets(self, X: np.ndarray, y: np.ndarray) -> List[tuple]:
        """
        Create balanced training subsets.
        
        Each subset contains:
        - ALL samples from minority class
        - A disjoint portion of the majority class
        
        Returns:
            List of (X_subset, y_subset) tuples
        """
        np.random.seed(self.random_state)
        
        # Identify classes
        self.classes_ = np.unique(y)
        class_counts = {c: (y == c).sum() for c in self.classes_}
        
        # Determine minority and majority
        minority_class = min(class_counts, key=class_counts.get)
        majority_class = max(class_counts, key=class_counts.get)
        
        n_minority = class_counts[minority_class]
        n_majority = class_counts[majority_class]
        
        # Compute number of estimators
        self.n_estimators_ = self._compute_n_estimators(n_majority, n_minority)
        
        # Get indices
        minority_idx = np.where(y == minority_class)[0]
        majority_idx = np.where(y == majority_class)[0]
        
        # Shuffle majority indices
        np.random.shuffle(majority_idx)
        
        # Split majority into n_estimators subsets
        subset_size = n_majority // self.n_estimators_
        
        subsets = []
        for i in range(self.n_estimators_):
            # Get majority subset
            start_idx = i * subset_size
            if i == self.n_estimators_ - 1:
                # Last subset takes remaining samples
                end_idx = n_majority
            else:
                end_idx = start_idx + subset_size
            
            maj_subset = majority_idx[start_idx:end_idx]
            
            # Combine with all minority samples
            subset_idx = np.concatenate([minority_idx, maj_subset])
            np.random.shuffle(subset_idx)
            
            X_subset = X[subset_idx]
            y_subset = y[subset_idx]
            
            subsets.append((X_subset, y_subset))
        
        return subsets
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            verbose: bool = True,
            **fit_params) -> 'BalancedQuantumEnsemble':
        """
        Fit the ensemble on imbalanced data.
        
        Args:
            X: Feature matrix
            y: Target labels
            verbose: Whether to show progress
            **fit_params: Additional parameters passed to base estimator fit
            
        Returns:
            Self
        """
        start_time = time.time()
        
        # Create balanced subsets
        subsets = self._create_balanced_subsets(X, y)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training Balanced Quantum Ensemble")
            print(f"{'='*60}")
            print(f"Base estimator: {self.base_estimator_class.__name__}")
            print(f"Number of estimators: {self.n_estimators_}")
            print(f"Samples per estimator: ~{len(subsets[0][0])}")
            print(f"Voting strategy: {self.voting}")
        
        # Train each estimator
        self.estimators_ = []
        iterator = enumerate(subsets)
        if verbose:
            iterator = tqdm(list(iterator), desc="Training estimators")
        
        for i, (X_subset, y_subset) in iterator:
            # Create fresh estimator with modified random state
            params = self.base_estimator_params.copy()
            if 'random_state' in params:
                params['random_state'] = self.random_state + i
            
            estimator = self.base_estimator_class(**params)
            
            # Disable individual verbose for cleaner output
            fit_params_i = fit_params.copy()
            if 'verbose' in fit_params_i:
                fit_params_i['verbose'] = False
            if 'show_progress' in fit_params_i:
                fit_params_i['show_progress'] = False
            
            estimator.fit(X_subset, y_subset, **fit_params_i)
            self.estimators_.append(estimator)
        
        self.training_time = time.time() - start_time
        self._is_fitted = True
        
        if verbose:
            print(f"\nEnsemble trained in {self.training_time:.2f} seconds")
            print(f"Average time per estimator: {self.training_time/self.n_estimators_:.2f}s")
        
        return self
    
    def predict(self, X: np.ndarray, **predict_params) -> np.ndarray:
        """
        Predict class labels using ensemble voting.
        
        Args:
            X: Features to predict
            **predict_params: Additional parameters for prediction
            
        Returns:
            Predicted labels
        """
        if not self._is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        if self.voting == 'soft':
            # Use probability averaging
            proba = self.predict_proba(X, **predict_params)
            return (proba[:, 1] >= 0.5).astype(int)
        else:
            # Hard voting (majority)
            predictions = np.zeros((len(X), self.n_estimators_))
            
            for i, estimator in enumerate(self.estimators_):
                predictions[:, i] = estimator.predict(X, **predict_params)
            
            # Majority vote
            return (predictions.mean(axis=1) >= 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray, **predict_params) -> np.ndarray:
        """
        Predict class probabilities by averaging across estimators.
        
        Args:
            X: Features to predict
            **predict_params: Additional parameters for prediction
            
        Returns:
            Predicted probabilities (n_samples, n_classes)
        """
        if not self._is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Collect probabilities from all estimators
        all_proba = np.zeros((len(X), len(self.classes_), self.n_estimators_))
        
        for i, estimator in enumerate(self.estimators_):
            proba = estimator.predict_proba(X, **predict_params)
            all_proba[:, :, i] = proba
        
        # Average across estimators
        return all_proba.mean(axis=2)
    
    def get_estimator_predictions(self, X: np.ndarray,
                                   **predict_params) -> np.ndarray:
        """
        Get predictions from each individual estimator.
        
        Useful for analyzing ensemble diversity.
        
        Returns:
            Array of shape (n_samples, n_estimators)
        """
        if not self._is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        predictions = np.zeros((len(X), self.n_estimators_))
        
        for i, estimator in enumerate(self.estimators_):
            predictions[:, i] = estimator.predict(X, **predict_params)
        
        return predictions
    
    def get_agreement_score(self, X: np.ndarray, **predict_params) -> np.ndarray:
        """
        Compute agreement score (proportion of estimators agreeing).
        
        High agreement = high confidence
        Low agreement = uncertain prediction
        
        Returns:
            Array of agreement scores (0.5 to 1.0) for each sample
        """
        predictions = self.get_estimator_predictions(X, **predict_params)
        
        # Compute proportion voting for majority class
        agreement = np.maximum(
            predictions.mean(axis=1),
            1 - predictions.mean(axis=1)
        )
        
        return agreement
    
    def get_params(self) -> dict:
        """Get ensemble parameters."""
        return {
            'base_estimator_class': self.base_estimator_class.__name__,
            'base_estimator_params': self.base_estimator_params,
            'n_estimators': self.n_estimators_,
            'voting': self.voting,
            'random_state': self.random_state
        }
    
    def save(self, path: Path) -> None:
        """Save ensemble to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as f:
            pickle.dump({
                'estimators': self.estimators_,
                'n_estimators': self.n_estimators_,
                'classes': self.classes_,
                'voting': self.voting,
                'training_time': self.training_time,
                'base_estimator_class': self.base_estimator_class,
                'base_estimator_params': self.base_estimator_params
            }, f)
        
        print(f"Ensemble saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'BalancedQuantumEnsemble':
        """Load ensemble from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        ensemble = cls(
            base_estimator_class=data['base_estimator_class'],
            base_estimator_params=data['base_estimator_params'],
            voting=data['voting']
        )
        ensemble.estimators_ = data['estimators']
        ensemble.n_estimators_ = data['n_estimators']
        ensemble.classes_ = data['classes']
        ensemble.training_time = data['training_time']
        ensemble._is_fitted = True
        
        return ensemble


class BootstrapQuantumEnsemble:
    """
    Bootstrap ensemble for uncertainty quantification.
    
    Unlike BalancedQuantumEnsemble which divides majority class,
    this creates bootstrap samples (with replacement) to estimate
    prediction uncertainty and confidence intervals.
    """
    
    def __init__(self,
                 base_estimator_class: type,
                 base_estimator_params: dict = None,
                 n_bootstrap: int = 10,
                 sample_fraction: float = 0.8,
                 random_state: int = RANDOM_STATE):
        """
        Initialize bootstrap ensemble.
        
        Args:
            base_estimator_class: Class of base estimator
            base_estimator_params: Parameters for base estimator
            n_bootstrap: Number of bootstrap samples
            sample_fraction: Fraction of data per bootstrap
            random_state: Random seed
        """
        self.base_estimator_class = base_estimator_class
        self.base_estimator_params = base_estimator_params or {}
        self.n_bootstrap = n_bootstrap
        self.sample_fraction = sample_fraction
        self.random_state = random_state
        
        self.estimators_: List = []
        self._is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            verbose: bool = True, **fit_params) -> 'BootstrapQuantumEnsemble':
        """Fit ensemble on bootstrap samples."""
        np.random.seed(self.random_state)
        
        n_samples = int(len(X) * self.sample_fraction)
        
        self.estimators_ = []
        iterator = range(self.n_bootstrap)
        if verbose:
            iterator = tqdm(iterator, desc="Training bootstrap estimators")
        
        for i in iterator:
            # Create bootstrap sample (with replacement)
            indices = np.random.choice(len(X), n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Create and train estimator
            params = self.base_estimator_params.copy()
            if 'random_state' in params:
                params['random_state'] = self.random_state + i
            
            estimator = self.base_estimator_class(**params)
            
            fit_params_i = fit_params.copy()
            fit_params_i['verbose'] = False
            if 'show_progress' in fit_params_i:
                fit_params_i['show_progress'] = False
            
            estimator.fit(X_boot, y_boot, **fit_params_i)
            self.estimators_.append(estimator)
        
        self._is_fitted = True
        return self
    
    def predict_with_uncertainty(self, X: np.ndarray,
                                  **predict_params) -> tuple:
        """
        Predict with uncertainty estimates.
        
        Returns:
            Tuple of (predictions, std, confidence_intervals)
        """
        if not self._is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Collect probabilities from all estimators
        all_proba = np.zeros((len(X), self.n_bootstrap))
        
        for i, estimator in enumerate(self.estimators_):
            proba = estimator.predict_proba(X, **predict_params)
            all_proba[:, i] = proba[:, 1]
        
        # Compute statistics
        mean_proba = all_proba.mean(axis=1)
        std_proba = all_proba.std(axis=1)
        
        # Confidence intervals (2.5th and 97.5th percentiles)
        ci_low = np.percentile(all_proba, 2.5, axis=1)
        ci_high = np.percentile(all_proba, 97.5, axis=1)
        
        predictions = (mean_proba >= 0.5).astype(int)
        
        return predictions, std_proba, (ci_low, ci_high)
    
    def predict(self, X: np.ndarray, **predict_params) -> np.ndarray:
        """Predict class labels."""
        predictions, _, _ = self.predict_with_uncertainty(X, **predict_params)
        return predictions
    
    def predict_proba(self, X: np.ndarray, **predict_params) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        all_proba = np.zeros((len(X), 2, self.n_bootstrap))
        
        for i, estimator in enumerate(self.estimators_):
            all_proba[:, :, i] = estimator.predict_proba(X, **predict_params)
        
        return all_proba.mean(axis=2)

