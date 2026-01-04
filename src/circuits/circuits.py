"""
Quantum circuits for QML classification.

This module provides the essential circuit components used by the models,
Components:
- Angle Embedding: Encode classical data as rotation angles 
- Strongly Entangling Layers: Variational ansatz with entanglement 
- Quantum Kernel: Overlap-based kernel for QSVM 
"""
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from typing import List, Optional


def angle_embedding(x: np.ndarray, wires: List[int], rotation: str = "Y") -> None:
    """
    Encode classical features as rotation angles.
    
    Each feature x[i] becomes a rotation angle on qubit i.
    This is the standard encoding used in PL10, PL11, PL12.
    
    |ψ(x)⟩ = ⊗ᵢ R_rot(xᵢ)|0⟩
    
    Args:
        x: Feature vector (length must equal number of wires)
        wires: Qubit indices to use
        rotation: Rotation axis - "X", "Y", or "Z" (default "Y")
    
    Example:
        >>> @qml.qnode(dev)
        ... def circuit(x):
        ...     angle_embedding(x, wires=range(4))
        ...     return qml.probs()
    """
    qml.AngleEmbedding(x, wires=wires, rotation=rotation)


def strongly_entangling_layers(weights: np.ndarray, wires: List[int]) -> None:
    """
    Apply strongly entangling variational layers.
    
    Each layer consists of:
    1. Single-qubit rotations (RX, RY, RZ) on each qubit
    2. CNOT gates creating entanglement between qubits
    
    This is the standard ansatz used in PL11 and PL12.
    
    Args:
        weights: Parameter array of shape (n_layers, n_qubits, 3)
                 Use qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)
        wires: Qubit indices to use
    
    Example:
        >>> n_layers, n_qubits = 2, 4
        >>> shape = qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)
        >>> weights = np.random.randn(*shape) * 0.1
        >>> 
        >>> @qml.qnode(dev)
        ... def circuit(x, weights):
        ...     angle_embedding(x, wires=range(n_qubits))
        ...     strongly_entangling_layers(weights, wires=range(n_qubits))
        ...     return qml.expval(qml.PauliZ(0))
    """
    qml.StronglyEntanglingLayers(weights, wires=wires)


def create_kernel_circuit(dev, n_qubits: int, feature_map: str = "angle"):
    """
    Create a quantum kernel circuit for QSVM.
    
    The kernel measures overlap between two quantum states:
    K(x₁, x₂) = |⟨φ(x₁)|φ(x₂)⟩|² = P(|0...0⟩)
    
    This is achieved by:
    1. Apply feature map U(x₁) to encode x₁
    2. Apply inverse feature map U†(x₂) to "decode" x₂
    3. Measure probability of |0...0⟩ state
    
    kernel(a, b)[0]
    
    Args:
        dev: PennyLane device
        n_qubits: Number of qubits
        feature_map: Type of feature map
                    - "angle": Standard AngleEmbedding (default)
                    - "custom": H-RZ-CNOT-RY pattern
                    - "zz": ZZ feature map with entanglement
    
    Returns:
        QNode function kernel(x1, x2) that returns probabilities
    
    Example:
        >>> dev = qml.device("default.qubit", wires=4)
        >>> kernel = create_kernel_circuit(dev, n_qubits=4, feature_map="angle")
        >>> k_value = kernel(x1, x2)[0]  # Kernel value is probs[0]
    """
    wires = list(range(n_qubits))
    
    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        # Apply feature map for x1
        if feature_map == "angle":
            qml.AngleEmbedding(x1, wires=wires)
        elif feature_map == "custom":
            # Custom: H-RZ-CNOT-RY pattern
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(x1[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            for i in range(n_qubits):
                qml.RY(x1[i], wires=i)
        elif feature_map == "zz":
            # ZZ feature map
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(2 * x1[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(2 * x1[i] * x1[i + 1], wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
        
        # Apply adjoint (inverse) of feature map for x2
        if feature_map == "angle":
            qml.adjoint(qml.AngleEmbedding)(x2, wires=wires)
        elif feature_map == "custom":
            def custom_map(x):
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
                    qml.RZ(x[i], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(n_qubits):
                    qml.RY(x[i], wires=i)
            qml.adjoint(custom_map)(x2)
        elif feature_map == "zz":
            def zz_map(x):
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
                    qml.RZ(2 * x[i], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                    qml.RZ(2 * x[i] * x[i + 1], wires=i + 1)
                    qml.CNOT(wires=[i, i + 1])
            qml.adjoint(zz_map)(x2)
        
        return qml.probs()
    
    return kernel_circuit


def compute_kernel_value(kernel_circuit, x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Compute quantum kernel value between two data points.
    
    K(x₁, x₂) = |⟨φ(x₁)|φ(x₂)⟩|² = P(|0...0⟩)
    
    Args:
        kernel_circuit: QNode created by create_kernel_circuit
        x1: First data point
        x2: Second data point
    
    Returns:
        Kernel value (float in [0, 1])
    """
    probs = kernel_circuit(x1, x2)
    return float(probs[0])


def compute_kernel_matrix(kernel_circuit, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute kernel matrix between two sets of data points.
    
    K[i,j] = kernel(A[i], B[j])
    
    Args:
        kernel_circuit: QNode created by create_kernel_circuit
        A: First set of data points (n_a, n_features)
        B: Second set of data points (n_b, n_features)
    
    Returns:
        Kernel matrix of shape (n_a, n_b)
    """
    return np.array([[compute_kernel_value(kernel_circuit, a, b) for b in B] for a in A])


def hermitian_projector(wire: int = 0):
    """
    Create Hermitian projector |0⟩⟨0| for measurement.
    
    Returns expected value in [0, 1], directly usable as probability.
    This is the measurement style used in PL11.
    
    Args:
        wire: Qubit to measure (default: first qubit)
    
    Returns:
        Observable for qml.expval()
    
    Example:
        >>> @qml.qnode(dev)
        ... def circuit(x, weights):
        ...     angle_embedding(x, wires=range(n_qubits))
        ...     strongly_entangling_layers(weights, wires=range(n_qubits))
        ...     return qml.expval(hermitian_projector(0))
    """
    state_0 = [[1], [0]]
    projector = np.array(state_0) @ np.array(state_0).T.conj()
    return qml.Hermitian(projector, wires=[wire])

