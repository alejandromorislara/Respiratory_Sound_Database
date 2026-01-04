from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path


class BaseQuantumClassifier(ABC):
    """
    Abstract base class for all quantum and hybrid classifiers.
    
    Interface for training and predicting.
    """
    
    def __init__(self, name: str = "BaseClassifier"):
        """
        Initialize the classifier.
        
        Args:
            name: Name of the classifier
        """
        self.name = name
        self._is_fitted = False
        self.training_history = []
        self.training_time = 0.0
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseQuantumClassifier':
        """
        Train the classifier.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Self
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted labels
        """
        pass
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted probabilities
        """
        # Default implementation for classifiers without probabilities
        predictions = self.predict(X)
        # Convert to one-hot-like probabilities
        proba = np.zeros((len(predictions), 2))
        proba[np.arange(len(predictions)), predictions.astype(int)] = 1.0
        return proba
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score.
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def get_params(self) -> Dict[str, Any]:
        """Get classifier parameters."""
        return {"name": self.name}
    
    def set_params(self, **params) -> 'BaseQuantumClassifier':
        """Set classifier parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model to disk."""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> 'BaseQuantumClassifier':
        """Load model from disk."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

