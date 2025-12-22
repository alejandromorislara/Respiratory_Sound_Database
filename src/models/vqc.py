"""
Variational Quantum Classifier (VQC) with gradient monitoring.

This module implements a VQC with barren plateau detection capabilities.
Barren plateaus are exponentially flat regions in the optimization landscape
where gradients vanish, making training impossible.
"""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.optimize import AdamOptimizer, GradientDescentOptimizer
import pickle
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from tqdm import tqdm
import warnings

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
                 monitor_gradients: bool = True,
                 gradient_threshold: float = 1e-5,
                 random_state: int = RANDOM_STATE):
        """
        Initialize the VQC.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of variational layers
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs
            class_weight: Dict {0: weight_0, 1: weight_1} for imbalanced data
            monitor_gradients: Whether to monitor gradient norms for barren plateaus
            gradient_threshold: Threshold below which gradients are considered vanishing
            random_state: Random seed
        """
        super().__init__(name="Variational Quantum Classifier")
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.class_weight = class_weight
        self.monitor_gradients = monitor_gradients
        self.gradient_threshold = gradient_threshold
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
        
        # Gradient monitoring
        self.gradient_history: List[float] = []
        self.barren_plateau_detected: bool = False
        self.barren_plateau_epoch: Optional[int] = None
        
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
    
    def _compute_gradient_norm(self, weights: np.ndarray, 
                                X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the L2 norm of the gradient of the cost function.
        
        This is used to detect barren plateaus, where gradients become
        exponentially small, making training impossible.
        
        Args:
            weights: Current weights
            X: Training features
            y: Training labels
            
        Returns:
            L2 norm of the gradient
        """
        # Use finite differences for gradient approximation
        epsilon = 1e-4
        grad = np.zeros_like(weights.flatten())
        weights_flat = weights.flatten()
        
        # Sample a subset of parameters for efficiency
        n_params = len(weights_flat)
        sample_size = min(n_params, 20)  # Sample at most 20 parameters
        param_indices = np.random.choice(n_params, sample_size, replace=False)
        
        cost_current = float(self._cost(weights, X, y))
        
        for idx in param_indices:
            weights_plus = weights_flat.copy()
            weights_plus[idx] += epsilon
            weights_plus = pnp.array(weights_plus.reshape(weights.shape), requires_grad=True)
            
            cost_plus = float(self._cost(weights_plus, X, y))
            grad[idx] = (cost_plus - cost_current) / epsilon
        
        # Scale by sampling factor
        grad_norm = np.linalg.norm(grad) * np.sqrt(n_params / sample_size)
        
        return grad_norm
    
    def _check_barren_plateau(self, epoch: int, grad_norm: float) -> bool:
        """
        Check if training is stuck in a barren plateau.
        
        Criteria:
        1. Gradient norm below threshold for multiple consecutive epochs
        2. Loss not decreasing significantly
        
        Args:
            epoch: Current epoch
            grad_norm: Current gradient norm
            
        Returns:
            True if barren plateau detected
        """
        if grad_norm < self.gradient_threshold:
            # Check if this is persistent
            n_recent = min(5, len(self.gradient_history))
            if n_recent >= 3:
                recent_grads = self.gradient_history[-n_recent:]
                if all(g < self.gradient_threshold for g in recent_grads):
                    return True
        return False
    
    def get_gradient_statistics(self) -> Dict:
        """
        Get statistics about gradient evolution during training.
        
        Returns:
            Dictionary with gradient statistics
        """
        if not self.gradient_history:
            return {'error': 'No gradient history available'}
        
        grads = np.array(self.gradient_history)
        
        return {
            'mean_gradient_norm': grads.mean(),
            'std_gradient_norm': grads.std(),
            'min_gradient_norm': grads.min(),
            'max_gradient_norm': grads.max(),
            'final_gradient_norm': grads[-1],
            'gradient_trend': 'decreasing' if grads[-1] < grads[0] else 'stable_or_increasing',
            'barren_plateau_detected': self.barren_plateau_detected,
            'barren_plateau_epoch': self.barren_plateau_epoch,
            'n_epochs_below_threshold': (grads < self.gradient_threshold).sum()
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            batch_size: Optional[int] = None,
            verbose: bool = True) -> 'VariationalQuantumClassifier':
        """
        Train the VQC with optional gradient monitoring.
        
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
        self.gradient_history = []
        self.barren_plateau_detected = False
        self.barren_plateau_epoch = None
        
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
            
            # Monitor gradients if enabled
            if self.monitor_gradients and (epoch % 5 == 0 or epoch == self.epochs - 1):
                grad_norm = self._compute_gradient_norm(self.weights, X_batch, y_batch)
                self.gradient_history.append(grad_norm)
                
                # Check for barren plateau
                if not self.barren_plateau_detected:
                    if self._check_barren_plateau(epoch, grad_norm):
                        self.barren_plateau_detected = True
                        self.barren_plateau_epoch = epoch
                        if verbose:
                            warnings.warn(
                                f"\n⚠️ Barren plateau detected at epoch {epoch}! "
                                f"Gradient norm: {grad_norm:.2e}. "
                                "Consider reducing circuit depth or using different ansatz.",
                                UserWarning
                            )
            
            if verbose and (epoch + 1) % 10 == 0:
                postfix = {"loss": f"{cost:.4f}"}
                if self.monitor_gradients and self.gradient_history:
                    postfix["grad"] = f"{self.gradient_history[-1]:.2e}"
                iterator.set_postfix(postfix)
        
        self.training_time = time.time() - start_time
        self._is_fitted = True
        
        if verbose:
            print(f"VQC trained in {self.training_time:.2f} seconds")
            print(f"Final loss: {self.training_history[-1]:.4f}")
            
            if self.monitor_gradients:
                stats = self.get_gradient_statistics()
                print(f"Final gradient norm: {stats['final_gradient_norm']:.2e}")
                if self.barren_plateau_detected:
                    print(f"⚠️ Warning: Barren plateau detected at epoch {self.barren_plateau_epoch}")
        
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
            "monitor_gradients": self.monitor_gradients,
            "gradient_threshold": self.gradient_threshold,
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
                "gradient_history": self.gradient_history,
                "barren_plateau_detected": self.barren_plateau_detected,
                "barren_plateau_epoch": self.barren_plateau_epoch,
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
            monitor_gradients=data["params"].get("monitor_gradients", True),
            gradient_threshold=data["params"].get("gradient_threshold", 1e-5),
            random_state=data["params"]["random_state"]
        )
        classifier.weights = pnp.array(data["weights"], requires_grad=True)
        classifier.training_history = data["training_history"]
        classifier.gradient_history = data.get("gradient_history", [])
        classifier.barren_plateau_detected = data.get("barren_plateau_detected", False)
        classifier.barren_plateau_epoch = data.get("barren_plateau_epoch", None)
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

