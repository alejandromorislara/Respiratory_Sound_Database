import numpy as np
import pennylane as qml
import pickle
import time
from pathlib import Path
from sklearn.svm import SVC
from typing import Optional, Callable
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import N_QUBITS, RANDOM_STATE
from .base import BaseQuantumClassifier
from ..circuits import create_kernel_circuit, compute_kernel_value


class QuantumKernelSVM(BaseQuantumClassifier):
    """
    Quantum Kernel Support Vector Machine.

    Uses a quantum circuit to compute kernel values between data points,
    then uses a classical SVM with the precomputed kernel.
    """

    @staticmethod
    def _get_optimal_device(n_qubits: int):
        try:
            dev = qml.device("lightning.gpu", wires=n_qubits)
            print(f"Using Lightning GPU device")
            return dev
        except Exception as e:
            print(f"Lightning GPU not available: {e}")

        try:
            dev = qml.device("lightning.qubit", wires=n_qubits)
            print(f"Using Lightning CPU device")
            return dev
        except Exception as e:
            print(f"Lightning CPU not available: {e}")

        print(f"Using default.qubit device (classical simulation)")
        return qml.device("default.qubit", wires=n_qubits)

    def __init__(self, n_qubits: int = N_QUBITS,
                 feature_map: str = "angle",
                 C: float = 1.0,
                 class_weight=None,
                 random_state: int = RANDOM_STATE):
        """
        Initialize the QSVM.
        
        Args:
            n_qubits: Number of qubits (must equal feature dimension)
            feature_map: Type of feature map ("angle", "custom", "zz")
                        - "angle": Uses AngleEmbedding 
                        - "custom": H-RZ-CNOT-RY pattern
                        - "zz": ZZ feature map with entanglement
            C: SVM regularization parameter
            class_weight: Class weights - 'balanced' or dict {class: weight}
            random_state: Random seed
        """
        super().__init__(name="Quantum SVM")
        
        self.n_qubits = n_qubits
        self.feature_map = feature_map
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state
        
        self.dev = self._get_optimal_device(n_qubits)
        print(f"QSVM using device: {self.dev}")

        self._kernel_circuit = create_kernel_circuit(
            self.dev, 
            n_qubits=self.n_qubits, 
            feature_map=self.feature_map
        )
        
        self.X_train = None
        self.svm = None
    
    def quantum_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute quantum kernel between two data points.
        
        K(x1, x2) = |<φ(x1)|φ(x2)>|^2 = probability of |00...0⟩
        
        kernel(a, b)[0]
        
        Args:
            x1: First data point
            x2: Second data point
            
        Returns:
            Kernel value (float)
        """
        return compute_kernel_value(self._kernel_circuit, x1, x2)
    
    def kernel_matrix(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Compute kernel matrix between two sets of data points.
        
        Args:
            A: First set of data points (n_a, n_features)
            B: Second set of data points (n_b, n_features)
            
        Returns:
            Kernel matrix (n_a, n_b)
        """
        return np.array([[self.quantum_kernel(a, b) for b in B] for a in A])
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            show_progress: bool = True) -> 'QuantumKernelSVM':
        """
        Train the QSVM.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels
            show_progress: Whether to show progress bar

        Returns:
            Self
        """
        start_time = time.time()

        if X.shape[1] != self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} features, got {X.shape[1]}")

        self.X_train = X.copy()

        print(f"Training QSVM with {len(y)} samples")
        self.svm = SVC(
            kernel=self.kernel_matrix,  
            C=self.C,
            class_weight=self.class_weight,
            random_state=self.random_state,
            probability=True
        )

        self.svm.fit(X, y)

        self.training_time = time.time() - start_time
        self._is_fitted = True

        print(f"QSVM trained in {self.training_time:.2f} seconds")

        return self
    
    def predict(self, X: np.ndarray, show_progress: bool = True) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features to predict
            show_progress: Whether to show progress bar (ignored - scikit-learn handles kernels)

        Returns:
            Predicted labels
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")

        return self.svm.predict(X)
    
    def predict_proba(self, X: np.ndarray, show_progress: bool = True) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features to predict
            show_progress: Whether to show progress bar (ignored - scikit-learn handles kernels)

        Returns:
            Predicted probabilities
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")

        return self.svm.predict_proba(X)
    
    def get_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix between two sets of data points."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")
        return self.kernel_matrix(X1, X2)
    
    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            "name": self.name,
            "n_qubits": self.n_qubits,
            "feature_map": self.feature_map,
            "C": self.C,
            "class_weight": self.class_weight,
            "random_state": self.random_state
        }
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as f:
            pickle.dump({
                "svm": self.svm,
                "X_train": self.X_train,
                "params": self.get_params(),
                "training_time": self.training_time
            }, f)
        
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'QuantumKernelSVM':
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        classifier = cls(
            n_qubits=data["params"]["n_qubits"],
            feature_map=data["params"]["feature_map"],
            C=data["params"]["C"],
            class_weight=data["params"].get("class_weight"),
            random_state=data["params"]["random_state"]
        )
        classifier.svm = data["svm"]
        classifier.X_train = data["X_train"]
        classifier.training_time = data["training_time"]
        classifier._is_fitted = True
        
        return classifier
    
    def draw_circuit(self):
        """Draw the quantum kernel circuit."""
        x1 = np.zeros(self.n_qubits)
        x2 = np.zeros(self.n_qubits)
        print(qml.draw(self._kernel_circuit)(x1, x2))
