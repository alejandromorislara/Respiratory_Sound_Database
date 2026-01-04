"""
Quantum circuits module for QML classification.
"""
from .circuits import (
    # Data encoding
    angle_embedding,
    
    # Variational ansatz
    strongly_entangling_layers,
    
    # Quantum kernel (QSVM)
    create_kernel_circuit,
    compute_kernel_value,
    compute_kernel_matrix,
    
    # Measurement
    hermitian_projector,
)

__all__ = [
    "angle_embedding",
    "strongly_entangling_layers",
    "create_kernel_circuit",
    "compute_kernel_value",
    "compute_kernel_matrix",
    "hermitian_projector",
]
