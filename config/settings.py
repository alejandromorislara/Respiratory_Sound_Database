"""
Global settings and constants for the Quantum Respiratory Classification project.
"""

# Audio processing settings
SAMPLE_RATE = 22050  # Hz
N_MFCC = 13  # Number of MFCC coefficients
HOP_LENGTH = 512  # Hop length for MFCC extraction
N_FFT = 2048  # FFT window size

# Feature extraction settings
FEATURE_NAMES = (
    [f"mfcc_{i}_mean" for i in range(N_MFCC)] +
    [f"mfcc_{i}_std" for i in range(N_MFCC)] +
    [f"delta_mfcc_{i}_mean" for i in range(N_MFCC)] +
    [f"delta_mfcc_{i}_std" for i in range(N_MFCC)]
)
N_FEATURES = len(FEATURE_NAMES)  # 52 features total

# Dimensionality reduction settings
N_COMPONENTS_PCA = 8  # Number of PCA components (= number of qubits)
PCA_VARIANCE_THRESHOLD = 0.95  # Minimum explained variance

# Quantum settings
N_QUBITS = 8  # Number of qubits for QSVM and VQC
N_QUBITS_HYBRID = 4  # Number of qubits for hybrid model
N_LAYERS_VQC = 3  # Number of variational layers for VQC
N_LAYERS_HYBRID = 2  # Number of variational layers for hybrid

# Training settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
LEARNING_RATE = 0.01
VQC_EPOCHS = 100
HYBRID_EPOCHS = 50
BATCH_SIZE = 32

# Cross-validation settings
N_OUTER_FOLDS = 5  # Outer folds for nested CV (performance estimation)
N_INNER_FOLDS = 3  # Inner folds for nested CV (hyperparameter tuning)
N_CV_REPETITIONS = 5  # Repetitions for 5x2cv statistical testing

# Ensemble settings
N_ENSEMBLE_ESTIMATORS = 'auto'  # 'auto' = ceil(n_majority / n_minority)
ENSEMBLE_VOTING = 'soft'  # 'soft' (probability averaging) or 'hard' (majority vote)

# Classification settings
TARGET_CLASSES = {
    "Healthy": 0,
    "COPD": 1
}
CLASS_NAMES = ["Healthy", "COPD"]

