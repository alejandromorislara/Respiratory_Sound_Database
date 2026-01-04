"""
Quantum Neural Network (QNN) with gradient monitoring.

This module implements a QNN following the style of PL11 notebook,
with barren plateau detection capabilities. Barren plateaus are 
exponentially flat regions in the optimization landscape where 
gradients vanish, making training impossible.
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

from config.settings import N_QUBITS, N_LAYERS_QNN, LEARNING_RATE, QNN_EPOCHS, RANDOM_STATE
from .base import BaseQuantumClassifier
from ..circuits import angle_embedding, strongly_entangling_layers, hermitian_projector


class QuantumNeuralNetwork(BaseQuantumClassifier):
    """
    Quantum Neural Network (QNN).

    Uses parameterized quantum circuits for classification.
    Architecture:
    1. Angle Embedding for data encoding
    2. Strongly Entangling Layers (trainable)
    3. Hermitian measurement projecting to |0> state on first qubit
    """

    @staticmethod
    def _get_optimal_device(n_qubits: int):
        """Get the best available quantum device (GPU if possible)."""
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
                 n_layers: int = N_LAYERS_QNN,
                 learning_rate: float = LEARNING_RATE,
                 epochs: int = QNN_EPOCHS,
                 class_weight: dict = None,
                 monitor_gradients: bool = True,
                 gradient_threshold: float = 1e-5,
                 early_stopping_patience: int = None,
                 validation_split: float = 0.0,
                 random_state: int = RANDOM_STATE):
        """
        Initialize the QNN.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of variational layers
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs
            class_weight: Dict {0: weight_0, 1: weight_1} for imbalanced data
            monitor_gradients: Whether to monitor gradient norms for barren plateaus
            gradient_threshold: Threshold below which gradients are considered vanishing
            early_stopping_patience: Number of epochs without improvement before stopping (None = disabled)
            validation_split: Fraction of training data to use for validation (0.0 = disabled)
            random_state: Random seed
        """
        super().__init__(name="Quantum Neural Network")
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.class_weight = class_weight
        self.monitor_gradients = monitor_gradients
        self.gradient_threshold = gradient_threshold
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        self.random_state = random_state
        
        # Initialize weights
        np.random.seed(random_state)
        weight_shape = qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)
        self.weights = pnp.array(np.random.randn(*weight_shape) * 0.1, 
                                  requires_grad=True)
        
        # Create quantum circuit
        self.dev = self._get_optimal_device(n_qubits)
        print(f"QNN using device: {self.dev}")
        self._create_circuit()
        
        # Optimizer
        self.optimizer = AdamOptimizer(stepsize=learning_rate)
        
        # Gradient monitoring
        self.gradient_history: List[float] = []
        self.barren_plateau_detected: bool = False
        self.barren_plateau_epoch: Optional[int] = None
        
        # Early stopping tracking
        self.best_val_loss: float = float('inf')
        self.best_weights: Optional[np.ndarray] = None
        self.stopped_epoch: Optional[int] = None
        self.validation_history: List[float] = []
        
    def _create_circuit(self):
        """Create the QNN circuit using circuits module."""
        n_qubits = self.n_qubits
        
        @qml.qnode(self.dev, interface="autograd")
        def circuit(weights, x):
            # Data encoding using circuits module
            angle_embedding(x, wires=range(n_qubits))
            
            # Variational layers using circuits module
            strongly_entangling_layers(weights, wires=range(n_qubits))
            
            return qml.expval(hermitian_projector(wire=0))
        
        self._circuit = circuit
    
    def _predict_single(self, x: np.ndarray) -> float:
        """Get raw circuit output for a single sample."""
        return float(self._circuit(self.weights, x))
    
    def _predictions_batch(self, X: np.ndarray) -> np.ndarray:
        """Get predictions for a batch of samples."""
        return np.array([self._predict_single(x) for x in X])
    
    def _cost(self, weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the cost function using weighted MSE.
        
        Args:
            weights: Current weights
            X: Training features
            y: Training labels (0 or 1)
            
        Returns:
            Cost value
        """
        predictions = pnp.array([self._circuit(weights, x) for x in X])
        
        if self.class_weight is not None:
            sample_weights = pnp.array([self.class_weight.get(int(yi), 1.0) for yi in y])
            cost = pnp.sum(sample_weights * (predictions - y) ** 2) / pnp.sum(sample_weights)
        else:
            cost = pnp.mean((predictions - y) ** 2)
        
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
        epsilon = 1e-4
        grad = np.zeros_like(weights.flatten())
        weights_flat = weights.flatten()
        
        n_params = len(weights_flat)
        sample_size = min(n_params, 20)
        param_indices = np.random.choice(n_params, sample_size, replace=False)
        
        cost_current = float(self._cost(weights, X, y))
        
        for idx in param_indices:
            weights_plus = weights_flat.copy()
            weights_plus[idx] += epsilon
            weights_plus = pnp.array(weights_plus.reshape(weights.shape), requires_grad=True)
            
            cost_plus = float(self._cost(weights_plus, X, y))
            grad[idx] = (cost_plus - cost_current) / epsilon
        
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
            verbose: bool = True) -> 'QuantumNeuralNetwork':
        """
        Train the QNN with optional gradient monitoring and early stopping.
        
        Args:
            X: Training features
            y: Training labels
            batch_size: If set, use mini-batch training
            verbose: Whether to print progress
            
        Returns:
            Self
        """
        start_time = time.time()
        
        if X.shape[1] != self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} features, got {X.shape[1]}")
        
        # Split data for validation if early stopping is enabled
        X_train, y_train = X, y
        X_val, y_val = None, None
        
        if self.validation_split > 0 and self.early_stopping_patience is not None:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, 
                test_size=self.validation_split, 
                stratify=y, 
                random_state=self.random_state
            )
            if verbose:
                print(f"Using {len(X_train)} samples for training, {len(X_val)} for validation")
        
        self.training_history = []
        self.validation_history = []
        self.gradient_history = []
        self.barren_plateau_detected = False
        self.barren_plateau_epoch = None
        self.stopped_epoch = None
        
        # Early stopping state
        self.best_val_loss = float('inf')
        self.best_weights = np.array(self.weights).copy()
        patience_counter = 0
        
        iterator = range(self.epochs)
        if verbose:
            iterator = tqdm(iterator, desc="Training QNN")
        
        for epoch in iterator:
            if batch_size is not None and batch_size < len(X_train):
                indices = np.random.choice(len(X_train), batch_size, replace=False)
                X_batch, y_batch = X_train[indices], y_train[indices]
            else:
                X_batch, y_batch = X_train, y_train
            
            # Update weights
            self.weights = self.optimizer.step(
                lambda w: self._cost(w, X_batch, y_batch),
                self.weights
            )
            
            # Compute training cost for logging
            cost = float(self._cost(self.weights, X_train, y_train))
            self.training_history.append(cost)
            
            # Compute validation cost if validation data is available
            val_loss = None
            if X_val is not None:
                val_loss = float(self._cost(self.weights, X_val, y_val))
                self.validation_history.append(val_loss)
                
                # Early stopping check
                if self.early_stopping_patience is not None:
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.best_weights = np.array(self.weights).copy()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                        if patience_counter >= self.early_stopping_patience:
                            self.stopped_epoch = epoch
                            if verbose:
                                print(f"\n🛑 Early stopping triggered at epoch {epoch}. "
                                      f"Best val_loss: {self.best_val_loss:.4f}")
                            break
            
            # Monitor gradients if enabled
            if self.monitor_gradients and (epoch % 5 == 0 or epoch == self.epochs - 1):
                grad_norm = self._compute_gradient_norm(self.weights, X_batch, y_batch)
                self.gradient_history.append(grad_norm)
                
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
                if val_loss is not None:
                    postfix["val_loss"] = f"{val_loss:.4f}"
                if self.monitor_gradients and self.gradient_history:
                    postfix["grad"] = f"{self.gradient_history[-1]:.2e}"
                iterator.set_postfix(postfix)
        
        # Restore best weights if early stopping was used
        if self.early_stopping_patience is not None and X_val is not None:
            self.weights = pnp.array(self.best_weights, requires_grad=True)
            if verbose:
                print(f"Restored best weights from epoch with val_loss: {self.best_val_loss:.4f}")
        
        self.training_time = time.time() - start_time
        self._is_fitted = True
        
        if verbose:
            print(f"QNN trained in {self.training_time:.2f} seconds")
            print(f"Final training loss: {self.training_history[-1]:.4f}")
            if self.validation_history:
                print(f"Best validation loss: {self.best_val_loss:.4f}")
            
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
        
        raw_predictions = self._predictions_batch(X)
        predictions = (raw_predictions > 0.5).astype(int)
        
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
        
        raw_predictions = self._predictions_batch(X)
        
        prob_class_1 = raw_predictions
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
            "early_stopping_patience": self.early_stopping_patience,
            "validation_split": self.validation_split,
            "random_state": self.random_state
        }
    
    def get_training_history(self) -> List[float]:
        """Get training loss history."""
        return self.training_history
    
    def get_validation_history(self) -> List[float]:
        """Get validation loss history."""
        return self.validation_history
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as f:
            pickle.dump({
                "weights": np.array(self.weights),
                "params": self.get_params(),
                "training_history": self.training_history,
                "validation_history": self.validation_history,
                "gradient_history": self.gradient_history,
                "barren_plateau_detected": self.barren_plateau_detected,
                "barren_plateau_epoch": self.barren_plateau_epoch,
                "best_val_loss": self.best_val_loss,
                "stopped_epoch": self.stopped_epoch,
                "training_time": self.training_time
            }, f)
        
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'QuantumNeuralNetwork':
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
            early_stopping_patience=data["params"].get("early_stopping_patience"),
            validation_split=data["params"].get("validation_split", 0.0),
            random_state=data["params"]["random_state"]
        )
        classifier.weights = pnp.array(data["weights"], requires_grad=True)
        classifier.training_history = data["training_history"]
        classifier.validation_history = data.get("validation_history", [])
        classifier.gradient_history = data.get("gradient_history", [])
        classifier.barren_plateau_detected = data.get("barren_plateau_detected", False)
        classifier.barren_plateau_epoch = data.get("barren_plateau_epoch", None)
        classifier.best_val_loss = data.get("best_val_loss", float('inf'))
        classifier.stopped_epoch = data.get("stopped_epoch", None)
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
