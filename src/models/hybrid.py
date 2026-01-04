"""
Hybrid Quantum-Classical Neural Network Classifier.

This module provides:
HybridQuantumClassifier: Neural network with embedded quantum layer
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml
import time
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import (
    N_QUBITS_HYBRID, N_LAYERS_HYBRID, 
    LEARNING_RATE, HYBRID_EPOCHS, BATCH_SIZE, RANDOM_STATE
)
from .base import BaseQuantumClassifier
from ..circuits import angle_embedding, strongly_entangling_layers


class HybridQuantumClassifier(BaseQuantumClassifier):
    """
    Hybrid Quantum-Classical Neural Network.
    
    Architecture:
    1. Classical preprocessing: Linear layers to reduce dimension
    2. Quantum layer: Parameterized quantum circuit
    3. Classical postprocessing: Linear layer to output
    
    Input -> [Linear -> ReLU -> Linear] -> Quantum -> [Linear -> Sigmoid] -> Output
    """
    
    def __init__(self, input_dim: int = 52,
                 n_qubits: int = N_QUBITS_HYBRID,
                 n_layers: int = N_LAYERS_HYBRID,
                 learning_rate: float = LEARNING_RATE,
                 epochs: int = HYBRID_EPOCHS,
                 batch_size: int = BATCH_SIZE,
                 pos_weight: float = None,
                 dropout_rate: float = 0.2,
                 early_stopping_patience: int = None,
                 random_state: int = RANDOM_STATE):
        """
        Initialize the hybrid classifier following PL12 style.
        
        Args:
            input_dim: Number of input features
            n_qubits: Number of qubits in quantum layer
            n_layers: Number of variational layers
            learning_rate: Learning rate
            epochs: Number of training epochs
            batch_size: Batch size for training
            pos_weight: Weight for positive class (for imbalanced data: n_neg/n_pos)
            dropout_rate: Dropout rate for regularization (default 0.2)
            early_stopping_patience: Number of epochs without improvement before stopping (None = disabled)
            random_state: Random seed
        """
        super().__init__(name="Hybrid Quantum Classifier")
        
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.pos_weight = pos_weight
        self.dropout_rate = dropout_rate
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        
        # Early stopping tracking
        self.best_val_loss: float = float('inf')
        self.best_model_state: Optional[Dict] = None
        self.stopped_epoch: Optional[int] = None
        
        # Set seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Detect and use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ PyTorch using: {self.device}")
        if self.device.type == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
        
        # Create quantum device (try GPU first)
        self.dev = self._get_optimal_quantum_device(n_qubits)
        print(f"⚛️ Quantum device: {self.dev}")
        
        # Initialize the model and move to GPU (no sigmoid at output)
        self.model = self._build_model().to(self.device)
        
        # Loss function: BCE with optional class weighting
        if pos_weight is not None:
            pw_tensor = torch.tensor([pos_weight], device=self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    
    @staticmethod
    def _get_optimal_quantum_device(n_qubits: int):
        """Get the best available quantum device."""
        try:
            dev = qml.device("lightning.gpu", wires=n_qubits)
            print(f"   Using Lightning GPU for quantum layer")
            return dev
        except Exception:
            pass
        
        print(f"   Using default.qubit for quantum layer")
        return qml.device("default.qubit", wires=n_qubits)
        
    def _build_model(self) -> nn.Module:
        """Build the hybrid model using circuits module."""
        
        n_qubits = self.n_qubits
        n_layers = self.n_layers
        dev = self.dev
        dropout_rate = self.dropout_rate
        
        # Quantum circuit using circuits module
        @qml.qnode(dev, interface="torch")
        def quantum_circuit(inputs, weights):
            # Angle embedding from circuits module
            angle_embedding(inputs, wires=range(n_qubits))
            
            # Variational layers from circuits module
            strongly_entangling_layers(weights, wires=range(n_qubits))
            
            # Return expectations for each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        class QuantumLayer(nn.Module):
            """Quantum layer that wraps a PennyLane circuit."""
            
            def __init__(self, n_qubits, n_layers):
                super().__init__()
                weight_shape = qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)
                self.weights = nn.Parameter(torch.randn(weight_shape) * 0.1)
                self.n_qubits = n_qubits
            
            def forward(self, x):
                batch_size = x.shape[0]
                outputs = []
                
                for i in range(batch_size):
                    # Scale inputs to [-pi, pi]
                    inputs = torch.tanh(x[i]) * np.pi
                    
                    # Run quantum circuit
                    q_out = quantum_circuit(inputs, self.weights)
                    outputs.append(torch.stack(q_out))
                
                return torch.stack(outputs)
        
        class HybridNet(nn.Module):
            """
            Architecture: Linear -> ReLU -> QuantumLayer -> Linear
            Output is raw logits (no sigmoid - use BCEWithLogitsLoss).
            """
            
            def __init__(self, input_dim, n_qubits, n_layers, dropout_rate):
                super().__init__()
                
                # Classical preprocessing with configurable dropout
                self.classical_pre = nn.Sequential(
                    nn.Linear(input_dim, 32),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(16, n_qubits)
                )
                
                # Quantum layer
                self.quantum = QuantumLayer(n_qubits, n_layers)
                
                # Classical postprocessing with dropout (no sigmoid - use BCEWithLogitsLoss)
                self.classical_post = nn.Sequential(
                    nn.Linear(n_qubits, 8),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(8, 1)
                )
            
            def forward(self, x):
                x = self.classical_pre(x)
                x = self.quantum(x)
                x = self.classical_post(x)
                return x.squeeze(-1)
        
        return HybridNet(self.input_dim, self.n_qubits, self.n_layers, dropout_rate)
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            validation_split: float = 0.0,
            verbose: bool = True) -> 'HybridQuantumClassifier':
        """
        Train the hybrid classifier with optional early stopping.
        
        Args:
            X: Training features
            y: Training labels
            validation_data: Optional (X_val, y_val) tuple
            validation_split: Fraction of training data to use for validation (if validation_data not provided)
            verbose: Whether to print progress
            
        Returns:
            Self
        """
        start_time = time.time()
        
        # Create validation split if early stopping is enabled and no validation_data provided
        X_train, y_train = X, y
        X_val, y_val = None, None
        
        if validation_data is not None:
            X_val, y_val = validation_data
        elif validation_split > 0 and self.early_stopping_patience is not None:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, 
                test_size=validation_split, 
                stratify=y, 
                random_state=self.random_state
            )
            if verbose:
                print(f"Using {len(X_train)} samples for training, {len(X_val)} for validation")
        
        # Convert to tensors and move to GPU
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.training_history = {"loss": [], "val_loss": []}
        
        # Early stopping state
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.stopped_epoch = None
        patience_counter = 0
        
        iterator = range(self.epochs)
        if verbose:
            iterator = tqdm(iterator, desc=f"Training Hybrid ({self.device})")
        
        for epoch in iterator:
            self.model.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            epoch_loss /= len(dataloader)
            self.training_history["loss"].append(epoch_loss)
            
            # Validation and early stopping
            val_loss = None
            if X_val is not None:
                val_loss = self._compute_loss(X_val, y_val)
                self.training_history["val_loss"].append(val_loss)
                
                # Early stopping check
                if self.early_stopping_patience is not None:
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                        if patience_counter >= self.early_stopping_patience:
                            self.stopped_epoch = epoch
                            if verbose:
                                print(f"\n🛑 Early stopping triggered at epoch {epoch}. "
                                      f"Best val_loss: {self.best_val_loss:.4f}")
                            break
            
            if verbose and (epoch + 1) % 5 == 0:
                msg = f"loss: {epoch_loss:.4f}"
                if val_loss is not None:
                    msg += f", val_loss: {val_loss:.4f}"
                iterator.set_postfix_str(msg)
        
        # Restore best model if early stopping was used
        if self.early_stopping_patience is not None and self.best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_model_state.items()})
            if verbose:
                print(f"Restored best weights from epoch with val_loss: {self.best_val_loss:.4f}")
        
        self.training_time = time.time() - start_time
        self._is_fitted = True
        
        if verbose:
            print(f"Hybrid model trained in {self.training_time:.2f} seconds")
            print(f"Final training loss: {self.training_history['loss'][-1]:.4f}")
            if self.training_history["val_loss"]:
                print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return self
    
    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute loss on a dataset."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
        return float(loss)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features to predict
            threshold: Decision threshold (default 0.5)
            
        Returns:
            Predicted labels
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        proba = self.predict_proba(X)
        return (proba[:, 1] > threshold).astype(int)
    
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
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            logits = self.model(X_tensor)
            prob_class_1 = torch.sigmoid(logits).cpu().numpy()
        
        prob_class_0 = 1 - prob_class_1
        return np.column_stack([prob_class_0, prob_class_1])
    
    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            "name": self.name,
            "input_dim": self.input_dim,
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "pos_weight": self.pos_weight,
            "dropout_rate": self.dropout_rate,
            "early_stopping_patience": self.early_stopping_patience,
            "random_state": self.random_state
        }
    
    def get_training_history(self) -> dict:
        """Get training history."""
        return self.training_history
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "params": self.get_params(),
            "training_history": self.training_history,
            "best_val_loss": self.best_val_loss,
            "stopped_epoch": self.stopped_epoch,
            "training_time": self.training_time
        }, path)
        
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'HybridQuantumClassifier':
        """Load model from disk."""
        checkpoint = torch.load(path)
        
        classifier = cls(
            input_dim=checkpoint["params"]["input_dim"],
            n_qubits=checkpoint["params"]["n_qubits"],
            n_layers=checkpoint["params"]["n_layers"],
            learning_rate=checkpoint["params"]["learning_rate"],
            epochs=checkpoint["params"]["epochs"],
            batch_size=checkpoint["params"]["batch_size"],
            pos_weight=checkpoint["params"].get("pos_weight"),
            dropout_rate=checkpoint["params"].get("dropout_rate", 0.2),
            early_stopping_patience=checkpoint["params"].get("early_stopping_patience"),
            random_state=checkpoint["params"]["random_state"]
        )
        
        classifier.model.load_state_dict(checkpoint["model_state_dict"])
        classifier.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        classifier.training_history = checkpoint["training_history"]
        classifier.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        classifier.stopped_epoch = checkpoint.get("stopped_epoch")
        classifier.training_time = checkpoint["training_time"]
        classifier._is_fitted = True
        
        return classifier
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


