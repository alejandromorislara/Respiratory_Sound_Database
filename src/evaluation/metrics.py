import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
    classification_report, balanced_accuracy_score
)
from typing import Dict, Tuple, Optional, List
import pandas as pd


class MedicalMetrics:
    """
    Computes and stores medical-relevant metrics for classification.
    
    In medical contexts:
    - Sensitivity (Recall): Ability to detect sick patients (avoid false negatives)
    - Specificity: Ability to correctly identify healthy patients
    - PPV (Precision): Probability that positive prediction is correct
    - NPV: Probability that negative prediction is correct
    """
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray,
                 y_prob: Optional[np.ndarray] = None,
                 class_names: list = ["Healthy", "COPD"]):
        """
        Initialize metrics calculator.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional, for ROC curve)
            class_names: Names of classes
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_prob = np.array(y_prob) if y_prob is not None else None
        self.class_names = class_names
        
        # Compute confusion matrix
        self.cm = confusion_matrix(y_true, y_pred)
        
        # Extract values (for binary classification)
        if len(self.cm) == 2:
            self.tn, self.fp, self.fn, self.tp = self.cm.ravel()
        else:
            self.tn = self.fp = self.fn = self.tp = None
    
    @property
    def accuracy(self) -> float:
        """Overall accuracy."""
        return accuracy_score(self.y_true, self.y_pred)
    
    @property
    def sensitivity(self) -> float:
        """
        Sensitivity (Recall, True Positive Rate).
        
        Answers: Of all sick patients, how many did we correctly identify?
        """
        return recall_score(self.y_true, self.y_pred, zero_division=0)
    
    @property
    def recall(self) -> float:
        """Alias for sensitivity."""
        return self.sensitivity
    
    @property
    def specificity(self) -> float:
        """
        Specificity (True Negative Rate).
        
        Answers: Of all healthy patients, how many did we correctly identify?
        """
        if self.tn is not None and (self.tn + self.fp) > 0:
            return self.tn / (self.tn + self.fp)
        return 0.0
    
    @property
    def precision(self) -> float:
        """
        Precision (Positive Predictive Value).
        
        Answers: Of all positive predictions, how many were correct?
        """
        return precision_score(self.y_true, self.y_pred, zero_division=0)
    
    @property
    def ppv(self) -> float:
        """Alias for precision (Positive Predictive Value)."""
        return self.precision
    
    @property
    def npv(self) -> float:
        """
        Negative Predictive Value.
        
        Answers: Of all negative predictions, how many were correct?
        """
        if self.tn is not None and (self.tn + self.fn) > 0:
            return self.tn / (self.tn + self.fn)
        return 0.0
    
    @property
    def f1(self) -> float:
        """F1 Score (harmonic mean of precision and recall)."""
        return f1_score(self.y_true, self.y_pred, zero_division=0)
    
    @property
    def auc_roc(self) -> float:
        """Area Under ROC Curve (requires probabilities)."""
        if self.y_prob is not None:
            try:
                return roc_auc_score(self.y_true, self.y_prob)
            except ValueError:
                return 0.0
        return 0.0
    
    def get_roc_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get ROC curve data.
        
        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        if self.y_prob is None:
            raise ValueError("Probabilities required for ROC curve")
        return roc_curve(self.y_true, self.y_prob)
    
    def get_all_metrics(self) -> Dict[str, float]:
        """Get all metrics as a dictionary."""
        metrics = {
            "Accuracy": self.accuracy,
            "Sensitivity": self.sensitivity,
            "Specificity": self.specificity,
            "Precision": self.precision,
            "NPV": self.npv,
            "F1-Score": self.f1,
        }
        
        if self.y_prob is not None:
            metrics["AUC-ROC"] = self.auc_roc
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        return self.cm
    
    def print_report(self, model_name: str = "Model") -> None:
        """Print a formatted metrics report."""
        print("\n" + "=" * 60)
        print(f"Classification Report: {model_name}")
        print("=" * 60)
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"              {self.class_names[0]:>10} {self.class_names[1]:>10}")
        print(f"Actual {self.class_names[0]:>6}  {self.cm[0,0]:>10} {self.cm[0,1]:>10}")
        print(f"       {self.class_names[1]:>6}  {self.cm[1,0]:>10} {self.cm[1,1]:>10}")
        
        # Metrics
        print("\nMetrics:")
        for name, value in self.get_all_metrics().items():
            print(f"  {name:15}: {value:.4f}")
        
        print("\nInterpretation:")
        print(f"  - Correctly identified {self.sensitivity*100:.1f}% of COPD patients")
        print(f"  - Correctly identified {self.specificity*100:.1f}% of Healthy patients")
        print("=" * 60 + "\n")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Return metrics as a DataFrame row."""
        return pd.DataFrame([self.get_all_metrics()])


def compare_models(results: Dict[str, MedicalMetrics]) -> pd.DataFrame:
    """
    Compare metrics from multiple models.
    
    Args:
        results: Dictionary mapping model names to MedicalMetrics objects
        
    Returns:
        DataFrame with comparison
    """
    data = []
    for model_name, metrics in results.items():
        row = {"Model": model_name}
        row.update(metrics.get_all_metrics())
        data.append(row)
    
    df = pd.DataFrame(data)
    df = df.set_index("Model")
    
    return df


def print_comparison_table(results: Dict[str, MedicalMetrics]) -> None:
    """Print a formatted comparison table."""
    df = compare_models(results)
    
    print("\n" + "=" * 80)
    print("Model Comparison")
    print("=" * 80)
    print(df.to_string())
    print("=" * 80)
    
    # Find best model for each metric
    print("\nBest models:")
    for col in df.columns:
        best_model = df[col].idxmax()
        print(f"  {col}: {best_model} ({df.loc[best_model, col]:.4f})")
    print()


    def to_dataframe(self) -> pd.DataFrame:
        """Convert fold results to DataFrame."""
        return pd.DataFrame(self.fold_metrics)
    
    def print_summary(self, confidence: float = 0.95) -> None:
        """Print formatted summary with confidence intervals."""
        summary = self.get_summary(confidence)
        
        print(f"\n{'='*60}")
        print(f"Results: {self.model_name}")
        print(f"{'='*60}")
        print(f"Number of folds: {summary['n_folds']}")
        print(f"\nMetrics (mean ± {int(confidence*100)}% CI):")
        
        for key, val in summary.items():
            if key in ['model', 'n_folds']:
                continue
            margin = (val['ci_high'] - val['ci_low']) / 2
            print(f"  {key}: {val['mean']:.4f} ± {margin:.4f}")
        
        print(f"{'='*60}\n")

