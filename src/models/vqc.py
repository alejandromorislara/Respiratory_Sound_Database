"""
Variational Quantum Classifier (VQC).
"""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.optimize import AdamOptimizer, GradientDescentOptimizer
import pickle
import time
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import N_QUBITS, N_LAYERS_VQC, LEARNING_RATE, VQC_EPOCHS, RANDOM_STATE
from .base import BaseQuantumClassifier


class VariationalQuantumClassifier(BaseQuantumClassifier):
    """
    Variational Quantum Classifier.

    Uses parameterized quantum circuits for classification.
    Architecture:
    1. Angle Embedding for data encoding
    2. Strongly Entangling Layers (trainable)
    3. Pauli-Z measurement on first qubit
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
                 n_layers: int = N_LAYERS_VQC,
                 learning_rate: float = LEARNING_RATE,
                 epochs: int = VQC_EPOCHS,
                 class_weight: dict = None,
                 random_state: int = RANDOM_STATE):
        """
        Initialize the VQC.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of variational layers
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs
            class_weight: Dict {0: weight_0, 1: weight_1} for imbalanced data
            random_state: Random seed
        """
        super().__init__(name="Variational Quantum Classifier")
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.class_weight = class_weight
        self.random_state = random_state
        
        # Initialize weights
        np.random.seed(random_state)
        weight_shape = qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)
        self.weights = pnp.array(np.random.randn(*weight_shape) * 0.1, 
                                  requires_grad=True)
        
        # Create quantum circuit
        self.dev = self._get_optimal_device(n_qubits)
        print(f"VQC using device: {self.dev}")
        self._create_circuit()
        
        # Optimizer
        self.optimizer = AdamOptimizer(stepsize=learning_rate)
        
    def _create_circuit(self):
        """Create the VQC circuit."""
        @qml.qnode(self.dev, interface="autograd")
        def circuit(weights, x):
            # Data encoding
            qml.AngleEmbedding(x, wires=range(self.n_qubits))
            
            # Variational layers
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            
            # Measurement
            return qml.expval(qml.PauliZ(0))
        
        self._circuit = circuit
    
    def _predict_single(self, x: np.ndarray) -> float:
        """Get raw circuit output for a single sample."""
        return float(self._circuit(self.weights, x))
    
    def _predictions_batch(self, X: np.ndarray) -> np.ndarray:
        """Get predictions for a batch of samples."""
        return np.array([self._predict_single(x) for x in X])
    
    def _cost(self, weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the cost function (weighted mean squared error).
        
        Args:
            weights: Current weights
            X: Training features
            y: Training labels (0 or 1)
            
        Returns:
            Cost value
        """
        # Get predictions (in range [-1, 1])
        predictions = pnp.array([self._circuit(weights, x) for x in X])
        
        # Map labels from {0, 1} to {-1, 1}
        y_mapped = 2 * y - 1
        
        # Compute sample weights if class_weight is provided
        if self.class_weight is not None:
            sample_weights = pnp.array([self.class_weight.get(int(yi), 1.0) for yi in y])
            # Weighted mean squared error
            cost = pnp.sum(sample_weights * (predictions - y_mapped) ** 2) / pnp.sum(sample_weights)
        else:
            # Standard mean squared error
            cost = pnp.mean((predictions - y_mapped) ** 2)
        
        return cost
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            batch_size: Optional[int] = None,
            verbose: bool = True) -> 'VariationalQuantumClassifier':
        """
        Train the VQC.
        
        Args:
            X: Training features
            y: Training labels
            batch_size: If set, use mini-batch training
            verbose: Whether to print progress
            
        Returns:
            Self
        """
        start_time = time.time()
        
        # Check dimensions
        if X.shape[1] != self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} features, got {X.shape[1]}")
        
        self.training_history = []
        
        iterator = range(self.epochs)
        if verbose:
            iterator = tqdm(iterator, desc="Training VQC")
        
        for epoch in iterator:
            if batch_size is not None and batch_size < len(X):
                # Mini-batch training
                indices = np.random.choice(len(X), batch_size, replace=False)
                X_batch, y_batch = X[indices], y[indices]
            else:
                X_batch, y_batch = X, y
            
            # Update weights
            self.weights = self.optimizer.step(
                lambda w: self._cost(w, X_batch, y_batch),
                self.weights
            )
            
            # Compute full cost for logging
            cost = float(self._cost(self.weights, X, y))
            self.training_history.append(cost)
            
            if verbose and (epoch + 1) % 10 == 0:
                iterator.set_postfix({"loss": f"{cost:.4f}"})
        
        self.training_time = time.time() - start_time
        self._is_fitted = True
        
        if verbose:
            print(f"VQC trained in {self.training_time:.2f} seconds")
            print(f"Final loss: {self.training_history[-1]:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted labels (0 or 1)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get raw predictions
        raw_predictions = self._predictions_batch(X)
        
        # Convert from [-1, 1] to {0, 1}
        predictions = (raw_predictions > 0).astype(int)
        
        return predictions
    
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
        
        # Get raw predictions in [-1, 1]
        raw_predictions = self._predictions_batch(X)
        
        # Convert to probabilities [0, 1]
        prob_class_1 = (raw_predictions + 1) / 2
        prob_class_0 = 1 - prob_class_1
        
        return np.column_stack([prob_class_0, prob_class_1])
    
    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            "name": self.name,
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "class_weight": self.class_weight,
            "random_state": self.random_state
        }
    
    def get_training_history(self) -> List[float]:
        """Get training loss history."""
        return self.training_history
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as f:
            pickle.dump({
                "weights": np.array(self.weights),
                "params": self.get_params(),
                "training_history": self.training_history,
                "training_time": self.training_time
            }, f)
        
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'VariationalQuantumClassifier':
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        classifier = cls(
            n_qubits=data["params"]["n_qubits"],
            n_layers=data["params"]["n_layers"],
            learning_rate=data["params"]["learning_rate"],
            epochs=data["params"]["epochs"],
            class_weight=data["params"].get("class_weight"),
            random_state=data["params"]["random_state"]
        )
        classifier.weights = pnp.array(data["weights"], requires_grad=True)
        classifier.training_history = data["training_history"]
        classifier.training_time = data["training_time"]
        classifier._is_fitted = True
        
        return classifier
    
    def draw_circuit(self, x: Optional[np.ndarray] = None):
        """Draw the quantum circuit."""
        if x is None:
            x = np.zeros(self.n_qubits)
        
        print(qml.draw(self._circuit)(self.weights, x))
    
    def get_circuit_specs(self) -> dict:
        """Get circuit specifications."""
        x = np.zeros(self.n_qubits)
        specs = qml.specs(self._circuit)(self.weights, x)
        return specs

