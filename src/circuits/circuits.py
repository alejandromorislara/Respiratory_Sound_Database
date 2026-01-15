import pennylane as qml
import numpy as np
from typing import List


def angle_embedding(x: np.ndarray, wires: List[int], rotation: str = "Y") -> None:
    """Encode classical features as rotation angles on qubits."""
    qml.AngleEmbedding(x, wires=wires, rotation=rotation)


def strongly_entangling_layers(weights: np.ndarray, wires: List[int]) -> None:
    """Apply strongly entangling variational layers."""
    qml.StronglyEntanglingLayers(weights, wires=wires)


def create_kernel_circuit(dev, n_qubits: int, feature_map: str = "angle"):
    """
    Create a quantum kernel circuit for QSVM.
    K(x1, x2) = |<phi(x1)|phi(x2)>|^2 = P(|0...0>)
    """
    wires = list(range(n_qubits))
    
    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        if feature_map == "angle":
            qml.AngleEmbedding(x1, wires=wires)
        elif feature_map == "custom":
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(x1[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            for i in range(n_qubits):
                qml.RY(x1[i], wires=i)
        elif feature_map == "zz":
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(2 * x1[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(2 * x1[i] * x1[i + 1], wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
        
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
    """Compute quantum kernel value between two data points."""
    probs = kernel_circuit(x1, x2)
    return float(probs[0])


def hermitian_projector(wire: int = 0):
    """Create Hermitian projector |0><0| for measurement."""
    state_0 = [[1], [0]]
    projector = np.array(state_0) @ np.array(state_0).T.conj()
    return qml.Hermitian(projector, wires=[wire])
