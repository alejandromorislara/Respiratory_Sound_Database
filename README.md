# Quantum Machine Learning for Respiratory Sound Classification

## Classification of COPD vs Healthy using Quantum Computing

---

## Overview

This project implements **Quantum Machine Learning (QML)** techniques for classifying respiratory sounds from the ICBHI Respiratory Sound Database. The goal is to distinguish between **Healthy** and **COPD** (Chronic Obstructive Pulmonary Disease) patients using quantum computing approaches.

## Project Structure

```
quantum-respiratory-classification/
│
├── config/                      # Configuration files
│   ├── settings.py              # Global constants (N_MFCC, N_QUBITS, etc.)
│   └── paths.py                 # Project paths
│
├── data/
│   ├── raw/                     # Original Kaggle data
│   ├── processed/               # Extracted features (CSV)
│   └── splits/                  # Train/test splits
│
├── src/                         # Source code (OOP)
│   ├── data/                    # Data loading and splitting
│   │   ├── loader.py            # DataLoader class
│   │   └── splitter.py          # PatientWiseSplitter class
│   │
│   ├── features/                # Feature extraction
│   │   ├── extractor.py         # AudioFeatureExtractor class
│   │   └── reducer.py           # DimensionalityReducer class
│   │
│   ├── models/                  # Classification models
│   │   ├── base.py              # BaseQuantumClassifier (abstract)
│   │   ├── classical.py         # ClassicalSVM (baseline)
│   │   ├── qsvm.py              # QuantumKernelSVM
│   │   ├── qnn.py               # QuantumNeuralNetwork
│   │   └── hybrid.py            # HybridQuantumClassifier
│   │
│   ├── circuits/                # Quantum circuits
│   │   ├── embeddings.py        # Data encoding circuits
│   │   ├── ansatze.py           # Variational ansatze
│   │   └── kernels.py           # Quantum kernel functions
│   │
│   ├── evaluation/              # Evaluation utilities
│   │   ├── metrics.py           # MedicalMetrics class
│   │   └── visualizer.py        # ResultsVisualizer class
│   │
│   └── utils/                   # Helper functions
│       └── helpers.py
│
├── notebooks/                   # Jupyter notebooks
│   └── 07_Final_Comparison.ipynb  # Main notebook for evaluation
│
├── results/                     # Output results
│   └── figures/                 # Generated plots
│
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Models Implemented

### 1. Classical SVM (Baseline)
- Standard Support Vector Machine with RBF kernel
- Uses scikit-learn implementation
- Serves as performance baseline

### 2. Quantum Kernel SVM (QSVM)
- Uses quantum circuit to compute kernel values
- Kernel: K(x₁, x₂) = |⟨φ(x₁)|φ(x₂)⟩|²
- Feature map with Hadamard, RZ, CNOT, and RY gates

### 3. Quantum Neural Network (QNN)
- Parameterized quantum circuit
- Angle embedding for data encoding
- StronglyEntanglingLayers as trainable ansatz
- Hermitian measurement projecting to |0⟩ state

### 4. Hybrid Quantum-Classical Network
- Classical preprocessing: Linear layers
- Quantum layer: 4 qubits with variational circuit
- Classical postprocessing: Output classification
- Trained end-to-end with PyTorch

## Dataset

**ICBHI 2017 Respiratory Sound Database**
- Source: Kaggle
- 920 audio files from 126 patients
- Diagnoses: COPD, Healthy, URTI, LRTI, Asthma, etc.
- Binary task: Healthy vs COPD (90 patients)

## Features

**MFCC Feature Extraction (52 dimensions)**
| Feature Type | Count |
|-------------|-------|
| MFCC Mean | 13 |
| MFCC Std | 13 |
| Delta MFCC Mean | 13 |
| Delta MFCC Std | 13 |

**Dimensionality Reduction**
- PCA: 52 → 8 components (for quantum compatibility)
- Explains ~95% variance

## Installation

### Quick Install (Recommended)

**Windows:**
```bash
# Run the installer script
install.bat
```

**Linux/Mac:**
```bash
chmod +x install.sh
./install.sh
```

### Manual Installation

```bash
# Create virtual environment (Python 3.10+ recommended)
python -m venv .venv

# Activate
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install PyTorch (choose one):
# CPU only:
pip install torch torchvision torchaudio

# CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

### Verify CUDA Installation

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

## Requirements

- **Python 3.10+** (required for PennyLane compatibility)
- PennyLane 0.35.1 (stable)
- PyTorch >= 2.0.0 (with optional CUDA support)
- librosa >= 0.10.0
- scikit-learn >= 1.2.0

### GPU Acceleration

The hybrid quantum-classical model can use CUDA for the classical layers:
- CUDA 11.8 or 12.1 recommended
- PennyLane uses `default.qubit` (CPU) for quantum simulation
- For GPU quantum simulation, use `pennylane-lightning-gpu` (requires cuQuantum)

## Usage

### 1. Feature Extraction

```python
from src.features.extractor import extract_features_pipeline

# Extract features for Healthy vs COPD classification
features_df = extract_features_pipeline(
    filter_classes=['Healthy', 'COPD'],
    save=True
)
```

### 2. Training Models

```python
from src.models.qsvm import QuantumKernelSVM
from src.models.qnn import QuantumNeuralNetwork

# QSVM
qsvm = QuantumKernelSVM(n_qubits=8, feature_map='custom')
qsvm.fit(X_train, y_train)

# QNN
qnn = QuantumNeuralNetwork(n_qubits=8, n_layers=2)
qnn.fit(X_train, y_train, epochs=100)
```

### 3. Run Complete Pipeline

```bash
# Open the final comparison notebook
jupyter notebook notebooks/07_Final_Comparison.ipynb
```

## Key Considerations

### Patient-Wise Split
To avoid data leakage, all samples from the same patient must be in the same split:
```python
from src.data.splitter import PatientWiseSplitter
splitter = PatientWiseSplitter(test_size=0.2)
train_df, test_df = splitter.split_dataframe(features_df)
```

### Quantum Simulation Limits
- Simulation scales exponentially with qubits
- 8 qubits is practical for classical simulation
- Subsample data for reasonable training times

## Medical Metrics

| Metric | Description | Clinical Importance |
|--------|-------------|---------------------|
| Sensitivity | TP/(TP+FN) | Detect sick patients |
| Specificity | TN/(TN+FP) | Identify healthy patients |
| AUC-ROC | Area under curve | Overall discrimination |

## Expected Results

| Model | Accuracy | Sensitivity | Training Time |
|-------|----------|-------------|---------------|
| Classical SVM | 75-85% | 70-80% | < 1 min |
| QSVM | 60-75% | 55-70% | 5-30 min |
| QNN | 60-75% | 55-70% | 5-15 min |
| Hybrid | 65-80% | 60-75% | 5-20 min |

*Note: Quantum models are limited by simulation speed. Performance may differ on real quantum hardware.*

## References

1. ICBHI 2017 Challenge: https://bhichallenge.med.auth.gr/
2. PennyLane Documentation: https://pennylane.ai/
3. Quantum Machine Learning: https://arxiv.org/abs/1307.0411

## License

MIT License

## Author

[Your Name]

---

**Note**: This project is for educational purposes demonstrating QML techniques. For clinical applications, proper validation and regulatory approval would be required.

