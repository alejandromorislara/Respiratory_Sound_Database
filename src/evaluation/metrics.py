import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from typing import Dict, Tuple, Optional
import pandas as pd


class MedicalMetrics:
    """Computes medical-relevant metrics for binary classification."""
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray,
                 y_prob: Optional[np.ndarray] = None,
                 class_names: list = ["Healthy", "COPD"]):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_prob = np.array(y_prob) if y_prob is not None else None
        self.class_names = class_names
        self.cm = confusion_matrix(y_true, y_pred)
        
        if len(self.cm) == 2:
            self.tn, self.fp, self.fn, self.tp = self.cm.ravel()
        else:
            self.tn = self.fp = self.fn = self.tp = None
    
    @property
    def accuracy(self) -> float:
        return accuracy_score(self.y_true, self.y_pred)
    
    @property
    def sensitivity(self) -> float:
        return recall_score(self.y_true, self.y_pred, zero_division=0)
    
    @property
    def recall(self) -> float:
        return self.sensitivity
    
    @property
    def specificity(self) -> float:
        if self.tn is not None and (self.tn + self.fp) > 0:
            return self.tn / (self.tn + self.fp)
        return 0.0
    
    @property
    def precision(self) -> float:
        return precision_score(self.y_true, self.y_pred, zero_division=0)
    
    @property
    def ppv(self) -> float:
        return self.precision
    
    @property
    def npv(self) -> float:
        if self.tn is not None and (self.tn + self.fn) > 0:
            return self.tn / (self.tn + self.fn)
        return 0.0
    
    @property
    def f1(self) -> float:
        return f1_score(self.y_true, self.y_pred, zero_division=0)
    
    @property
    def auc_roc(self) -> float:
        if self.y_prob is not None:
            try:
                return roc_auc_score(self.y_true, self.y_prob)
            except ValueError:
                return 0.0
        return 0.0
    
    def get_roc_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.y_prob is None:
            raise ValueError("Probabilities required for ROC curve")
        return roc_curve(self.y_true, self.y_prob)
    
    def get_all_metrics(self) -> Dict[str, float]:
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
        return self.cm
    
    def print_report(self, model_name: str = "Model") -> None:
        print("\n" + "=" * 60)
        print(f"Classification Report: {model_name}")
        print("=" * 60)
        print("\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"              {self.class_names[0]:>10} {self.class_names[1]:>10}")
        print(f"Actual {self.class_names[0]:>6}  {self.cm[0,0]:>10} {self.cm[0,1]:>10}")
        print(f"       {self.class_names[1]:>6}  {self.cm[1,0]:>10} {self.cm[1,1]:>10}")
        print("\nMetrics:")
        for name, value in self.get_all_metrics().items():
            print(f"  {name:15}: {value:.4f}")
        print("\nInterpretation:")
        print(f"  - Correctly identified {self.sensitivity*100:.1f}% of COPD patients")
        print(f"  - Correctly identified {self.specificity*100:.1f}% of Healthy patients")
        print("=" * 60 + "\n")
    
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.get_all_metrics()])


def compare_models(results: Dict[str, MedicalMetrics]) -> pd.DataFrame:
    data = []
    for model_name, metrics in results.items():
        row = {"Model": model_name}
        row.update(metrics.get_all_metrics())
        data.append(row)
    df = pd.DataFrame(data)
    df = df.set_index("Model")
    return df


def print_comparison_table(results: Dict[str, MedicalMetrics]) -> None:
    df = compare_models(results)
    print("\n" + "=" * 80)
    print("Model Comparison")
    print("=" * 80)
    print(df.to_string())
    print("=" * 80)
    print("\nBest models:")
    for col in df.columns:
        best_model = df[col].idxmax()
        print(f"  {col}: {best_model} ({df.loc[best_model, col]:.4f})")
    print()
