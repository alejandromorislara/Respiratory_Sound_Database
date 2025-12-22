"""
Quantum Kernel Support Vector Machine (QSVM).
"""
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


class QuantumKernelSVM(BaseQuantumClassifier):
    """
    Quantum Kernel Support Vector Machine.

    Uses a quantum circuit to compute kernel values between data points,
    then uses a classical SVM with the precomputed kernel.
    """

    @staticmethod
    def _get_optimal_device(n_qubits: int):
        """Get the best available quantum device (GPU if possible)."""
        try:
            # Try Lightning GPU first
            dev = qml.device("lightning.gpu", wires=n_qubits)
            print(f"Using Lightning GPU device")
            return dev
        except Exception as e:
            print(f"Lightning GPU not available: {e}")

        try:
            # Try Lightning CPU
            dev = qml.device("lightning.qubit", wires=n_qubits)
            print(f"Using Lightning CPU device")
            return dev
        except Exception as e:
            print(f"Lightning CPU not available: {e}")

        # Fallback to default qubit
        print(f"Using default.qubit device (classical simulation)")
        return qml.device("default.qubit", wires=n_qubits)

    def __init__(self, n_qubits: int = N_QUBITS,
                 feature_map: str = "custom",
                 C: float = 1.0,
                 class_weight=None,
                 random_state: int = RANDOM_STATE):
        """
        Initialize the QSVM.
        
        Args:
            n_qubits: Number of qubits (must equal feature dimension)
            feature_map: Type of feature map ("angle", "custom", "zz")
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
        
        # Create quantum device and kernel
        self.dev = self._get_optimal_device(n_qubits)
        print(f"QSVM using device: {self.dev}")
        self._create_kernel()
        
        # SVM with precomputed kernel (supports class_weight for imbalanced data)
        self.svm = SVC(
            kernel="precomputed",
            C=C,
            class_weight=class_weight,
            random_state=random_state,
            probability=True
        )
        
        self.X_train = None
        self.K_train = None
        
    def _create_kernel(self):
        """Create the quantum kernel circuit."""
        wires = list(range(self.n_qubits))
        
        def feature_map_circuit(x):
            """Apply the feature map."""
            if self.feature_map == "angle":
                qml.AngleEmbedding(x, wires=wires)
            elif self.feature_map == "custom":
                for i in range(self.n_qubits):
                    qml.Hadamard(wires=i)
                    qml.RZ(x[i], wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(self.n_qubits):
                    qml.RY(x[i], wires=i)
            elif self.feature_map == "zz":
                for i in range(self.n_qubits):
                    qml.Hadamard(wires=i)
                    qml.RZ(2 * x[i], wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                    qml.RZ(2 * x[i] * x[i + 1], wires=i + 1)
                    qml.CNOT(wires=[i, i + 1])
        
        @qml.qnode(self.dev)
        def kernel_circuit(x1, x2):
            feature_map_circuit(x1)
            qml.adjoint(feature_map_circuit)(x2)
            return qml.probs(wires=wires)
        
        self._kernel_circuit = kernel_circuit
    
    def quantum_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute quantum kernel between two data points.
        
        K(x1, x2) = |<φ(x1)|φ(x2)>|^2
        
        Args:
            x1: First data point
            x2: Second data point
            
        Returns:
            Kernel value
        """
        probs = self._kernel_circuit(x1, x2)
        return float(probs[0])  # Probability of |00...0⟩
    
    def _compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray,
                               symmetric: bool = False,
                               show_progress: bool = True) -> np.ndarray:
        """
        Compute kernel matrix between two sets of data points.
        
        Args:
            X1: First set of data points
            X2: Second set of data points
            symmetric: Whether X1 == X2 (for efficiency)
            show_progress: Whether to show progress bar
            
        Returns:
            Kernel matrix
        """
        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))
        
        if symmetric:
            # Only compute upper triangle
            pairs = [(i, j) for i in range(n1) for j in range(i, n2)]
            desc = "Computing kernel (symmetric)"
        else:
            pairs = [(i, j) for i in range(n1) for j in range(n2)]
            desc = "Computing kernel"
        
        iterator = tqdm(pairs, desc=desc) if show_progress else pairs
        
        for i, j in iterator:
            k_val = self.quantum_kernel(X1[i], X2[j])
            K[i, j] = k_val
            if symmetric and i != j:
                K[j, i] = k_val
        
        return K
    
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
        
        # Check dimensions
        if X.shape[1] != self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} features, got {X.shape[1]}")
        
        self.X_train = X.copy()
        
        # Compute training kernel matrix
        print(f"Computing quantum kernel matrix ({len(X)}x{len(X)})...")
        self.K_train = self._compute_kernel_matrix(X, X, symmetric=True, 
                                                    show_progress=show_progress)
        
        # Train SVM with precomputed kernel
        self.svm.fit(self.K_train, y)
        
        self.training_time = time.time() - start_time
        self._is_fitted = True
        
        print(f"QSVM trained in {self.training_time:.2f} seconds")
        
        return self
    
    def predict(self, X: np.ndarray, show_progress: bool = True) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features to predict
            show_progress: Whether to show progress bar
            
        Returns:
            Predicted labels
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Compute kernel between test and training points
        K_test = self._compute_kernel_matrix(X, self.X_train, symmetric=False,
                                              show_progress=show_progress)
        
        return self.svm.predict(K_test)
    
    def predict_proba(self, X: np.ndarray, show_progress: bool = True) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict
            show_progress: Whether to show progress bar
            
        Returns:
            Predicted probabilities
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        K_test = self._compute_kernel_matrix(X, self.X_train, symmetric=False,
                                              show_progress=show_progress)
        
        return self.svm.predict_proba(K_test)
    
    def get_kernel_matrix(self) -> np.ndarray:
        """Get the training kernel matrix."""
        if self.K_train is None:
            raise ValueError("Model must be fitted first")
        return self.K_train
    
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
                "K_train": self.K_train,
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
        classifier.K_train = data["K_train"]
        classifier.training_time = data["training_time"]
        classifier._is_fitted = True
        
        return classifier
    
    def draw_circuit(self):
        """Draw the quantum kernel circuit."""
        # Create sample inputs
        x1 = np.zeros(self.n_qubits)
        x2 = np.zeros(self.n_qubits)
        
        # Draw circuit
        print(qml.draw(self._kernel_circuit)(x1, x2))

