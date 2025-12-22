"""
Hybrid Quantum-Classical Neural Network Classifier.

This module provides:
1. HybridQuantumClassifier: Neural network with embedded quantum layer
2. ClassicalControlClassifier: Ablation study control (pure classical)

The ablation study compares both to isolate quantum contribution.
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
                 random_state: int = RANDOM_STATE):
        """
        Initialize the hybrid classifier.
        
        Args:
            input_dim: Number of input features
            n_qubits: Number of qubits in quantum layer
            n_layers: Number of variational layers
            learning_rate: Learning rate
            epochs: Number of training epochs
            batch_size: Batch size for training
            pos_weight: Weight for positive class (for imbalanced data: n_neg/n_pos)
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
        self.random_state = random_state
        
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
        
        # Loss with pos_weight for imbalanced data (BCEWithLogitsLoss is numerically stable)
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
        
        try:
            dev = qml.device("lightning.qubit", wires=n_qubits)
            print(f"   Using Lightning CPU for quantum layer")
            return dev
        except Exception:
            pass
        
        print(f"   Using default.qubit for quantum layer")
        return qml.device("default.qubit", wires=n_qubits)
        
    def _build_model(self) -> nn.Module:
        """Build the hybrid model."""
        
        # Quantum layer definition
        n_qubits = self.n_qubits
        n_layers = self.n_layers
        dev = self.dev
        
        @qml.qnode(dev, interface="torch")
        def quantum_circuit(inputs, weights):
            # Angle embedding
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            
            # Variational layers
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            
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
                # Process each sample
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
            """Complete hybrid network. Output is raw logits (no sigmoid)."""
            
            def __init__(self, input_dim, n_qubits, n_layers):
                super().__init__()
                
                # Classical preprocessing
                self.classical_pre = nn.Sequential(
                    nn.Linear(input_dim, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, n_qubits)
                )
                
                # Quantum layer
                self.quantum = QuantumLayer(n_qubits, n_layers)
                
                # Classical postprocessing (no sigmoid - use BCEWithLogitsLoss)
                self.classical_post = nn.Sequential(
                    nn.Linear(n_qubits, 8),
                    nn.ReLU(),
                    nn.Linear(8, 1)
                )
            
            def forward(self, x):
                x = self.classical_pre(x)
                x = self.quantum(x)
                x = self.classical_post(x)
                return x.squeeze(-1)
        
        return HybridNet(self.input_dim, self.n_qubits, self.n_layers)
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            verbose: bool = True) -> 'HybridQuantumClassifier':
        """
        Train the hybrid classifier.
        
        Args:
            X: Training features
            y: Training labels
            validation_data: Optional (X_val, y_val) tuple
            verbose: Whether to print progress
            
        Returns:
            Self
        """
        start_time = time.time()
        
        # Convert to tensors and move to GPU
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.training_history = {"loss": [], "val_loss": []}
        
        iterator = range(self.epochs)
        if verbose:
            iterator = tqdm(iterator, desc=f"Training Hybrid ({self.device})")
        
        for epoch in iterator:
            self.model.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                # Data already on correct device from DataLoader
                self.optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            epoch_loss /= len(dataloader)
            self.training_history["loss"].append(epoch_loss)
            
            # Validation
            if validation_data is not None:
                X_val, y_val = validation_data
                val_loss = self._compute_loss(X_val, y_val)
                self.training_history["val_loss"].append(val_loss)
            
            if verbose and (epoch + 1) % 5 == 0:
                msg = f"loss: {epoch_loss:.4f}"
                if validation_data is not None:
                    msg += f", val_loss: {self.training_history['val_loss'][-1]:.4f}"
                iterator.set_postfix_str(msg)
        
        self.training_time = time.time() - start_time
        self._is_fitted = True
        
        if verbose:
            print(f"Hybrid model trained in {self.training_time:.2f} seconds")
            print(f"Final loss: {self.training_history['loss'][-1]:.4f}")
        
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
            # Apply sigmoid to convert logits to probabilities
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
            random_state=checkpoint["params"]["random_state"]
        )
        
        classifier.model.load_state_dict(checkpoint["model_state_dict"])
        classifier.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        classifier.training_history = checkpoint["training_history"]
        classifier.training_time = checkpoint["training_time"]
        classifier._is_fitted = True
        
        return classifier
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


class ClassicalControlClassifier:
    """
    Classical control network for ablation study.
    
    This network has an identical architecture to HybridQuantumClassifier
    but replaces the quantum layer with a classical linear layer of 
    comparable dimensionality.
    
    Purpose: Isolate the contribution of the quantum layer by comparing:
    - Hybrid (with quantum): Classical → Quantum → Classical
    - Control (pure classical): Classical → Linear → Classical
    
    If both achieve similar performance, the quantum layer is redundant.
    If Hybrid significantly outperforms Control, quantum provides value.
    """
    
    def __init__(self, input_dim: int = 52,
                 n_hidden: int = N_QUBITS_HYBRID,
                 learning_rate: float = LEARNING_RATE,
                 epochs: int = HYBRID_EPOCHS,
                 batch_size: int = BATCH_SIZE,
                 pos_weight: float = None,
                 random_state: int = RANDOM_STATE):
        """
        Initialize the classical control classifier.
        
        Args:
            input_dim: Number of input features
            n_hidden: Hidden dimension (matches n_qubits of hybrid)
            learning_rate: Learning rate
            epochs: Number of training epochs
            batch_size: Batch size for training
            pos_weight: Weight for positive class
            random_state: Random seed
        """
        self.name = "Classical Control (Ablation)"
        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.pos_weight = pos_weight
        self.random_state = random_state
        
        # Set seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build model
        self.model = self._build_model().to(self.device)
        
        # Loss function
        if pos_weight is not None:
            pw_tensor = torch.tensor([pos_weight], device=self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.training_history: Dict = {"loss": [], "val_loss": []}
        self.training_time: float = 0
        self._is_fitted = False
    
    def _build_model(self) -> nn.Module:
        """Build the classical control model."""
        
        class ClassicalControlNet(nn.Module):
            """
            Classical network matching hybrid architecture.
            
            Replaces quantum layer (n_qubits inputs → n_qubits outputs)
            with a classical layer of same dimensionality.
            """
            
            def __init__(self, input_dim, n_hidden):
                super().__init__()
                
                # Classical preprocessing (same as hybrid)
                self.classical_pre = nn.Sequential(
                    nn.Linear(input_dim, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, n_hidden)
                )
                
                # Classical replacement for quantum layer
                # Uses same input/output dimensions as quantum layer
                self.classical_core = nn.Sequential(
                    nn.Linear(n_hidden, n_hidden * 2),
                    nn.Tanh(),  # Mimic quantum output range [-1, 1]
                    nn.Linear(n_hidden * 2, n_hidden)
                )
                
                # Classical postprocessing (same as hybrid)
                self.classical_post = nn.Sequential(
                    nn.Linear(n_hidden, 8),
                    nn.ReLU(),
                    nn.Linear(8, 1)
                )
            
            def forward(self, x):
                x = self.classical_pre(x)
                x = self.classical_core(x)
                x = self.classical_post(x)
                return x.squeeze(-1)
        
        return ClassicalControlNet(self.input_dim, self.n_hidden)
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            verbose: bool = True) -> 'ClassicalControlClassifier':
        """Train the classical control classifier."""
        start_time = time.time()
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.training_history = {"loss": [], "val_loss": []}
        
        iterator = range(self.epochs)
        if verbose:
            iterator = tqdm(iterator, desc=f"Training Control ({self.device})")
        
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
            
            if validation_data is not None:
                val_loss = self._compute_loss(*validation_data)
                self.training_history["val_loss"].append(val_loss)
            
            if verbose and (epoch + 1) % 5 == 0:
                msg = f"loss: {epoch_loss:.4f}"
                if validation_data is not None:
                    msg += f", val_loss: {self.training_history['val_loss'][-1]:.4f}"
                iterator.set_postfix_str(msg)
        
        self.training_time = time.time() - start_time
        self._is_fitted = True
        
        if verbose:
            print(f"Classical control trained in {self.training_time:.2f} seconds")
            print(f"Final loss: {self.training_history['loss'][-1]:.4f}")
        
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
        """Predict class labels."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        proba = self.predict_proba(X)
        return (proba[:, 1] > threshold).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
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
            "n_hidden": self.n_hidden,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "pos_weight": self.pos_weight,
            "random_state": self.random_state
        }
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


def run_ablation_study(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       input_dim: int = 8,
                       n_qubits: int = N_QUBITS_HYBRID,
                       verbose: bool = True) -> Dict:
    """
    Run ablation study comparing hybrid quantum vs classical control.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        input_dim: Input dimension
        n_qubits: Number of qubits/hidden units
        verbose: Whether to print progress
        
    Returns:
        Dictionary with comparison results
    """
    from sklearn.metrics import balanced_accuracy_score, f1_score
    
    results = {}
    
    # Train hybrid model
    if verbose:
        print("\n" + "="*60)
        print("ABLATION STUDY: Quantum vs Classical")
        print("="*60)
        print("\n1. Training Hybrid Quantum-Classical Model...")
    
    hybrid = HybridQuantumClassifier(
        input_dim=input_dim,
        n_qubits=n_qubits,
        epochs=30,
        random_state=42
    )
    hybrid.fit(X_train, y_train, verbose=verbose)
    y_pred_hybrid = hybrid.predict(X_test)
    
    results['hybrid'] = {
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_hybrid),
        'f1_score': f1_score(y_test, y_pred_hybrid, average='macro'),
        'training_time': hybrid.training_time,
        'n_parameters': hybrid.count_parameters()
    }
    
    # Train classical control
    if verbose:
        print("\n2. Training Classical Control Model...")
    
    control = ClassicalControlClassifier(
        input_dim=input_dim,
        n_hidden=n_qubits,
        epochs=30,
        random_state=42
    )
    control.fit(X_train, y_train, verbose=verbose)
    y_pred_control = control.predict(X_test)
    
    results['control'] = {
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_control),
        'f1_score': f1_score(y_test, y_pred_control, average='macro'),
        'training_time': control.training_time,
        'n_parameters': control.count_parameters()
    }
    
    # Comparison
    ba_diff = results['hybrid']['balanced_accuracy'] - results['control']['balanced_accuracy']
    f1_diff = results['hybrid']['f1_score'] - results['control']['f1_score']
    
    results['comparison'] = {
        'ba_difference': ba_diff,
        'f1_difference': f1_diff,
        'quantum_adds_value': ba_diff > 0.02,  # >2% improvement threshold
        'speedup_factor': results['control']['training_time'] / results['hybrid']['training_time']
    }
    
    if verbose:
        print("\n" + "="*60)
        print("ABLATION STUDY RESULTS")
        print("="*60)
        print(f"\nHybrid (Quantum):")
        print(f"  Balanced Accuracy: {results['hybrid']['balanced_accuracy']:.4f}")
        print(f"  F1 Score: {results['hybrid']['f1_score']:.4f}")
        print(f"  Training Time: {results['hybrid']['training_time']:.2f}s")
        print(f"  Parameters: {results['hybrid']['n_parameters']}")
        
        print(f"\nClassical Control:")
        print(f"  Balanced Accuracy: {results['control']['balanced_accuracy']:.4f}")
        print(f"  F1 Score: {results['control']['f1_score']:.4f}")
        print(f"  Training Time: {results['control']['training_time']:.2f}s")
        print(f"  Parameters: {results['control']['n_parameters']}")
        
        print(f"\nConclusion:")
        if results['comparison']['quantum_adds_value']:
            print(f"  ✅ Quantum layer provides {ba_diff*100:.1f}% improvement")
        else:
            print(f"  ⚠️ Quantum layer does not significantly outperform classical")
        print("="*60)
    
    return results

