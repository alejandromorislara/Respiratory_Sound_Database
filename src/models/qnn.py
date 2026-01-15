import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.optimize import AdamOptimizer
import pickle
import time
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm

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
                 n_layers: int = N_LAYERS_QNN,
                 learning_rate: float = LEARNING_RATE,
                 epochs: int = QNN_EPOCHS,
                 class_weight: dict = None,
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
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        self.random_state = random_state
        
        np.random.seed(random_state)
        weight_shape = qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)
        self.weights = pnp.array(np.random.randn(*weight_shape) * 0.1, 
                                  requires_grad=True)
        
        self.dev = self._get_optimal_device(n_qubits)
        print(f"QNN using device: {self.dev}")
        self._create_circuit()
        
        self.optimizer = AdamOptimizer(stepsize=learning_rate)
        
        self.best_val_loss: float = float('inf')
        self.best_weights: Optional[np.ndarray] = None
        self.stopped_epoch: Optional[int] = None
        self.validation_history: List[float] = []
        
    def _create_circuit(self):
        """Create the QNN circuit using circuits module."""
        n_qubits = self.n_qubits
        
        @qml.qnode(self.dev, interface="autograd")
        def circuit(weights, x):
            angle_embedding(x, wires=range(n_qubits))
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
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            batch_size: Optional[int] = None,
            verbose: bool = True) -> 'QuantumNeuralNetwork':
        """
        Train the QNN with optional early stopping.
        
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
        self.stopped_epoch = None
        
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
            
            self.weights = self.optimizer.step(
                lambda w: self._cost(w, X_batch, y_batch),
                self.weights
            )
            
            cost = float(self._cost(self.weights, X_train, y_train))
            self.training_history.append(cost)
            
            val_loss = None
            if X_val is not None:
                val_loss = float(self._cost(self.weights, X_val, y_val))
                self.validation_history.append(val_loss)
                
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
                                print(f"\nEarly stopping triggered at epoch {epoch}. "
                                      f"Best val_loss: {self.best_val_loss:.4f}")
                            break
            
            if verbose and (epoch + 1) % 10 == 0:
                postfix = {"loss": f"{cost:.4f}"}
                if val_loss is not None:
                    postfix["val_loss"] = f"{val_loss:.4f}"
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
            early_stopping_patience=data["params"].get("early_stopping_patience"),
            validation_split=data["params"].get("validation_split", 0.0),
            random_state=data["params"]["random_state"]
        )
        classifier.weights = pnp.array(data["weights"], requires_grad=True)
        classifier.training_history = data["training_history"]
        classifier.validation_history = data.get("validation_history", [])
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
