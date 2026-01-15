import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Optional, Union
from pathlib import Path
import pickle

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import N_COMPONENTS_PCA


class DimensionalityReducer:
    """Reduces feature dimensionality using StandardScaler + PCA."""
    
    def __init__(self, n_components: int = N_COMPONENTS_PCA,
                 variance_threshold: Optional[float] = None):
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.scaler = StandardScaler()
        self.pca = None
        self._is_fitted = False
        
    def fit(self, X: np.ndarray) -> 'DimensionalityReducer':
        X_scaled = self.scaler.fit_transform(X)
        
        if self.variance_threshold is not None:
            pca_full = PCA()
            pca_full.fit(X_scaled)
            cumsum = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= self.variance_threshold) + 1
            self.n_components = min(n_components, self.n_components)
        
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X_scaled)
        self._is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Reducer must be fitted before transform")
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Reducer must be fitted before inverse_transform")
        X_scaled = self.pca.inverse_transform(X_reduced)
        return self.scaler.inverse_transform(X_scaled)
    
    @property
    def explained_variance_ratio(self) -> np.ndarray:
        if self.pca is None:
            raise ValueError("Reducer must be fitted first")
        return self.pca.explained_variance_ratio_
    
    @property
    def total_explained_variance(self) -> float:
        return np.sum(self.explained_variance_ratio)
    
    def get_component_names(self) -> list:
        return [f"PC{i+1}" for i in range(self.n_components)]
    
    def save(self, path: Union[str, Path]) -> None:
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
