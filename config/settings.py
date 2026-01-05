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
N_QUBITS = 8  # Number of qubits for QSVM and QNN
N_QUBITS_HYBRID = 4  # Number of qubits for hybrid model
N_LAYERS_QNN = 2  # Number of variational layers for QNN
N_LAYERS_HYBRID = 1  # Number of variational layers for hybrid

# Training settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
LEARNING_RATE = 0.01
QNN_EPOCHS = 100
HYBRID_EPOCHS = 50
BATCH_SIZE = 32

# Classification settings
TARGET_CLASSES = {
    "Healthy": 0,
    "COPD": 1
}
CLASS_NAMES = ["Healthy", "COPD"]

