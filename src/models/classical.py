import numpy as np
import pickle
import time
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from typing import Optional

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import RANDOM_STATE
from .base import BaseQuantumClassifier


class ClassicalSVM(BaseQuantumClassifier):
    """
    Classical Support Vector Machine classifier (baseline).
    
    Uses an RBF kernel for comparison with quantum kernels.
    """
    
    def __init__(self, kernel: str = "rbf", C: float = 1.0,
                 gamma: str = "scale", class_weight=None, 
                 random_state: int = RANDOM_STATE):
        """
        Initialize the classical SVM.
        
        Args:
            kernel: Kernel type ("rbf", "linear", "poly")
            C: Regularization parameter
            gamma: Kernel coefficient
            class_weight: Class weights - 'balanced' or dict {class: weight}
            random_state: Random seed
        """
        super().__init__(name="Classical SVM")
        
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.class_weight = class_weight
        self.random_state = random_state
        
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            class_weight=class_weight,  # Handle imbalanced data
            random_state=random_state,
            probability=True  # Enable probability estimates
        )
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ClassicalSVM':
        """
        Train the SVM classifier.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Self
        """
        start_time = time.time()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        self.training_time = time.time() - start_time
        self._is_fitted = True
        
        print(f"Classical SVM trained in {self.training_time:.2f} seconds")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted labels
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted probabilities (n_samples, 2)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            "name": self.name,
            "kernel": self.kernel,
            "C": self.C,
            "gamma": self.gamma,
            "class_weight": self.class_weight,
            "random_state": self.random_state
        }
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "params": self.get_params(),
                "training_time": self.training_time
            }, f)
        
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'ClassicalSVM':
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        classifier = cls(
            kernel=data["params"]["kernel"],
            C=data["params"]["C"],
            gamma=data["params"]["gamma"],
            class_weight=data["params"].get("class_weight"),
            random_state=data["params"]["random_state"]
        )
        classifier.model = data["model"]
        classifier.scaler = data["scaler"]
        classifier.training_time = data["training_time"]
        classifier._is_fitted = True
        
        return classifier

