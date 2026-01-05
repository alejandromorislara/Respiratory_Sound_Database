import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple, Optional, Union
from pathlib import Path
import pickle

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import N_COMPONENTS_PCA, PCA_VARIANCE_THRESHOLD
from config.paths import PROCESSED_DATA_DIR


class DimensionalityReducer:
    """
    Reduces feature dimensionality for quantum circuit compatibility.
    
    Pipeline:
    1. StandardScaler - Normalize features to zero mean, unit variance
    2. PCA - Reduce to target number of components (qubits)
    """
    
    def __init__(self, n_components: int = N_COMPONENTS_PCA,
                 variance_threshold: Optional[float] = None):
        """
        Initialize the dimensionality reducer.
        
        Args:
            n_components: Target number of components (= number of qubits)
            variance_threshold: If set, select components to explain this variance
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.scaler = StandardScaler()
        self.pca = None
        self._is_fitted = False
        
    def fit(self, X: np.ndarray) -> 'DimensionalityReducer':
        """
        Fit the reducer on training data.
        
        Args:
            X: Feature matrix (n_samples x n_features)
            
        Returns:
            Self
        """
        # Fit scaler
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine number of components
        if self.variance_threshold is not None:
            # Fit PCA with all components first
            pca_full = PCA()
            pca_full.fit(X_scaled)
            
            # Find number of components for variance threshold
            cumsum = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= self.variance_threshold) + 1
            self.n_components = min(n_components, self.n_components)
        
        # Fit final PCA
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X_scaled)
        
        self._is_fitted = True
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features to reduced dimensionality.
        
        Args:
            X: Feature matrix (n_samples x n_features)
            
        Returns:
            Reduced feature matrix (n_samples x n_components)
        """
        if not self._is_fitted:
            raise ValueError("Reducer must be fitted before transform")
        
        X_scaled = self.scaler.transform(X)
        X_reduced = self.pca.transform(X_scaled)
        
        return X_reduced
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            X: Feature matrix
            
        Returns:
            Reduced feature matrix
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        """
        Inverse transform to original feature space.
        
        Args:
            X_reduced: Reduced feature matrix
            
        Returns:
            Reconstructed feature matrix (approximate)
        """
        if not self._is_fitted:
            raise ValueError("Reducer must be fitted before inverse_transform")
        
        X_scaled = self.pca.inverse_transform(X_reduced)
        X_original = self.scaler.inverse_transform(X_scaled)
        
        return X_original
    
    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio for each component."""
        if self.pca is None:
            raise ValueError("Reducer must be fitted first")
        return self.pca.explained_variance_ratio_
    
    @property
    def total_explained_variance(self) -> float:
        """Get total explained variance."""
        return np.sum(self.explained_variance_ratio)
    
    def get_component_names(self) -> list:
        """Get names for the reduced components."""
        return [f"PC{i+1}" for i in range(self.n_components)]
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the fitted reducer to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "pca": self.pca,
                "n_components": self.n_components,
                "variance_threshold": self.variance_threshold
            }, f)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'DimensionalityReducer':
        """Load a fitted reducer from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        reducer = cls(
            n_components=data["n_components"],
            variance_threshold=data["variance_threshold"]
        )
        reducer.scaler = data["scaler"]
        reducer.pca = data["pca"]
        reducer._is_fitted = True
        
        return reducer
    
    def print_summary(self) -> None:
        """Print a summary of the reduction."""
        if not self._is_fitted:
            print("Reducer not fitted yet")
            return
        
        print("\n" + "=" * 50)
        print("Dimensionality Reduction Summary")
        print("=" * 50)
        print(f"Input dimensions: {self.scaler.n_features_in_}")
        print(f"Output dimensions: {self.n_components}")
        print(f"Total explained variance: {self.total_explained_variance:.4f}")
        print("\nVariance by component:")
        for i, var in enumerate(self.explained_variance_ratio):
            cumvar = np.sum(self.explained_variance_ratio[:i+1])
            print(f"  PC{i+1}: {var:.4f} (cumulative: {cumvar:.4f})")
        print("=" * 50 + "\n")


def reduce_features_pipeline(df: pd.DataFrame,
                             feature_cols: list,
                             n_components: int = N_COMPONENTS_PCA,
                             save_reducer: bool = True) -> Tuple[np.ndarray, DimensionalityReducer]:
    """
    Complete dimensionality reduction pipeline.
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        n_components: Target number of components
        save_reducer: Whether to save the reducer
        
    Returns:
        Tuple of (reduced features, fitted reducer)
    """
    X = df[feature_cols].values
    
    reducer = DimensionalityReducer(n_components=n_components)
    X_reduced = reducer.fit_transform(X)
    
    reducer.print_summary()
    
    if save_reducer:
        reducer.save(PROCESSED_DATA_DIR / "reducer.pkl")
    
    return X_reduced, reducer


def normalize_for_quantum(X: np.ndarray, 
                          method: str = "minmax_pi") -> np.ndarray:
    """
    Normalize features for quantum encoding.
    
    Args:
        X: Feature matrix
        method: Normalization method
            - "minmax_pi": Scale to [-pi, pi]
            - "minmax_2pi": Scale to [0, 2*pi]
            - "tanh_pi": Apply tanh and scale to [-pi, pi]
            
    Returns:
        Normalized feature matrix
    """
    if method == "minmax_pi":
        # Scale to [-pi, pi]
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1  # Avoid division by zero
        X_norm = (X - X_min) / X_range * 2 * np.pi - np.pi
        
    elif method == "minmax_2pi":
        # Scale to [0, 2*pi]
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1
        X_norm = (X - X_min) / X_range * 2 * np.pi
        
    elif method == "tanh_pi":
        # Apply tanh (maps to [-1, 1]) then scale to [-pi, pi]
        X_norm = np.tanh(X) * np.pi
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return X_norm

