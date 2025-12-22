"""
Medical evaluation metrics for respiratory sound classification.

This module provides:
1. MedicalMetrics: Clinical metrics (sensitivity, specificity, PPV, NPV)
2. Statistical tests for algorithm comparison (McNemar, 5x2cv t-test)
3. Confidence interval estimation for rigorous reporting
"""
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


# =============================================================================
# STATISTICAL TESTS FOR ALGORITHM COMPARISON
# =============================================================================

def mcnemar_test(y_true: np.ndarray, 
                 y_pred_1: np.ndarray, 
                 y_pred_2: np.ndarray,
                 correction: bool = True) -> Dict:
    """
    McNemar's test for comparing two classifiers on the same dataset.
    
    This test evaluates whether the disagreements between two classifiers
    are statistically significant. It only considers samples where the
    classifiers disagree.
    
    Null hypothesis: Both classifiers have the same error rate.
    
    Args:
        y_true: True labels
        y_pred_1: Predictions from classifier 1
        y_pred_2: Predictions from classifier 2
        correction: Whether to apply continuity correction (Edwards)
        
    Returns:
        Dictionary with test results:
        - chi2: Chi-squared statistic
        - p_value: Two-sided p-value
        - n_01: Cases where 1 is wrong, 2 is correct
        - n_10: Cases where 1 is correct, 2 is wrong
        - significant: Whether difference is significant (p < 0.05)
        - interpretation: Text interpretation
    """
    # Compute correctness
    correct_1 = (y_pred_1 == y_true)
    correct_2 = (y_pred_2 == y_true)
    
    # Contingency table of disagreements
    # n_00: Both wrong
    # n_01: 1 wrong, 2 correct
    # n_10: 1 correct, 2 wrong
    # n_11: Both correct
    
    n_01 = np.sum(~correct_1 & correct_2)  # 1 wrong, 2 correct
    n_10 = np.sum(correct_1 & ~correct_2)  # 1 correct, 2 wrong
    
    # Check if test is applicable
    if n_01 + n_10 < 25:
        # Use exact binomial test for small samples
        from scipy.stats import binom_test
        n = n_01 + n_10
        k = min(n_01, n_10)
        p_value = 2 * stats.binom.cdf(k, n, 0.5) if n > 0 else 1.0
        
        return {
            'chi2': None,
            'p_value': p_value,
            'n_01': n_01,
            'n_10': n_10,
            'n_disagreements': n_01 + n_10,
            'significant': p_value < 0.05,
            'test_used': 'exact_binomial',
            'interpretation': _interpret_mcnemar(n_01, n_10, p_value)
        }
    
    # McNemar's chi-squared test
    if correction:
        # Edwards' continuity correction
        chi2 = (abs(n_01 - n_10) - 1) ** 2 / (n_01 + n_10)
    else:
        chi2 = (n_01 - n_10) ** 2 / (n_01 + n_10)
    
    # p-value from chi-squared distribution with df=1
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    
    return {
        'chi2': chi2,
        'p_value': p_value,
        'n_01': n_01,
        'n_10': n_10,
        'n_disagreements': n_01 + n_10,
        'significant': p_value < 0.05,
        'test_used': 'mcnemar_chi2',
        'interpretation': _interpret_mcnemar(n_01, n_10, p_value)
    }


def _interpret_mcnemar(n_01: int, n_10: int, p_value: float) -> str:
    """Generate human-readable interpretation of McNemar test."""
    if p_value >= 0.05:
        return ("No existe diferencia estadísticamente significativa entre "
                "los clasificadores (p >= 0.05). Las discrepancias observadas "
                "pueden atribuirse al azar.")
    else:
        if n_10 > n_01:
            better = "el primer clasificador"
        else:
            better = "el segundo clasificador"
        return (f"Existe diferencia estadísticamente significativa (p = {p_value:.4f}). "
                f"Los resultados sugieren que {better} tiene mejor rendimiento.")


def paired_5x2cv_ttest(scores_1: np.ndarray, 
                        scores_2: np.ndarray) -> Dict:
    """
    Paired 5×2cv t-test for comparing machine learning algorithms.
    
    Implementation of Dietterich's 5×2cv paired t-test (1998), which is
    recommended for comparing algorithms when computational cost is high.
    
    The test computes:
    t = p̄₁ / sqrt((1/5) × Σᵢ sᵢ²)
    
    where p̄₁ is the mean difference in the first fold and sᵢ² is the
    variance of differences in repetition i.
    
    Args:
        scores_1: Scores from algorithm 1, shape (5, 2) for 5 reps × 2 folds
        scores_2: Scores from algorithm 2, shape (5, 2)
        
    Returns:
        Dictionary with test results
    """
    scores_1 = np.array(scores_1)
    scores_2 = np.array(scores_2)
    
    if scores_1.shape != (5, 2) or scores_2.shape != (5, 2):
        raise ValueError("Scores must have shape (5, 2) for 5×2cv")
    
    # Compute differences
    diffs = scores_1 - scores_2
    
    # Mean of first fold differences
    p_bar_1 = diffs[:, 0].mean()
    
    # Variance for each repetition
    variances = np.zeros(5)
    for i in range(5):
        p_i1 = diffs[i, 0]
        p_i2 = diffs[i, 1]
        p_mean = (p_i1 + p_i2) / 2
        variances[i] = (p_i1 - p_mean)**2 + (p_i2 - p_mean)**2
    
    # Compute t-statistic
    denominator = np.sqrt(0.2 * variances.sum())
    
    if denominator == 0:
        return {
            't_statistic': 0.0,
            'p_value': 1.0,
            'df': 5,
            'significant': False,
            'mean_diff': diffs.mean(),
            'std_diff': diffs.std(),
            'interpretation': "No se puede calcular: varianza cero"
        }
    
    t_stat = p_bar_1 / denominator
    
    # Two-tailed p-value with df=5
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=5))
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'df': 5,
        'significant': p_value < 0.05,
        'mean_diff': diffs.mean(),
        'std_diff': diffs.std(),
        'algorithm_1_better': diffs.mean() > 0,
        'interpretation': _interpret_5x2cv(diffs.mean(), p_value)
    }


def _interpret_5x2cv(mean_diff: float, p_value: float) -> str:
    """Generate interpretation of 5x2cv t-test."""
    if p_value >= 0.05:
        return ("No existe diferencia estadísticamente significativa entre "
                "los algoritmos según el test 5×2cv (p >= 0.05).")
    else:
        direction = "superior" if mean_diff > 0 else "inferior"
        return (f"Diferencia estadísticamente significativa (p = {p_value:.4f}). "
                f"El primer algoritmo es {direction} al segundo.")


def compute_confidence_interval(scores: np.ndarray,
                                 confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute confidence interval for a set of scores.
    
    Uses t-distribution for small samples.
    
    Args:
        scores: Array of scores from CV folds
        confidence: Confidence level (default 0.95)
        
    Returns:
        Tuple of (mean, ci_low, ci_high)
    """
    n = len(scores)
    mean = scores.mean()
    se = scores.std(ddof=1) / np.sqrt(n)
    
    # t-value for confidence interval
    alpha = 1 - confidence
    t_val = stats.t.ppf(1 - alpha/2, df=n-1)
    
    ci_low = mean - t_val * se
    ci_high = mean + t_val * se
    
    return mean, ci_low, ci_high


def format_metric_with_ci(scores: np.ndarray, 
                          metric_name: str = "",
                          confidence: float = 0.95) -> str:
    """
    Format a metric with confidence interval for reporting.
    
    Args:
        scores: Array of scores from CV folds
        metric_name: Name of the metric
        confidence: Confidence level
        
    Returns:
        Formatted string: "mean ± margin (CI confidence%)"
    """
    mean, ci_low, ci_high = compute_confidence_interval(scores, confidence)
    margin = (ci_high - ci_low) / 2
    
    if metric_name:
        return f"{metric_name}: {mean:.4f} ± {margin:.4f} (IC {int(confidence*100)}%)"
    return f"{mean:.4f} ± {margin:.4f}"


class CrossValidationResults:
    """
    Container for cross-validation results with statistical analysis.
    
    Stores per-fold metrics and provides:
    - Mean ± CI for each metric
    - Statistical comparisons between models
    - Formatted reporting
    """
    
    def __init__(self, model_name: str):
        """Initialize results container."""
        self.model_name = model_name
        self.fold_metrics: List[Dict] = []
        
    def add_fold_result(self, metrics: Dict) -> None:
        """Add metrics from a single fold."""
        self.fold_metrics.append(metrics)
    
    def get_metric_scores(self, metric_name: str) -> np.ndarray:
        """Get array of scores for a specific metric across folds."""
        return np.array([fold[metric_name] for fold in self.fold_metrics])
    
    def get_summary(self, confidence: float = 0.95) -> Dict:
        """
        Get summary statistics for all metrics.
        
        Returns:
            Dictionary with mean, std, and CI for each metric
        """
        if not self.fold_metrics:
            return {}
        
        summary = {'model': self.model_name, 'n_folds': len(self.fold_metrics)}
        
        for metric in self.fold_metrics[0].keys():
            scores = self.get_metric_scores(metric)
            mean, ci_low, ci_high = compute_confidence_interval(scores, confidence)
            
            summary[metric] = {
                'mean': mean,
                'std': scores.std(),
                'ci_low': ci_low,
                'ci_high': ci_high
            }
        
        return summary
    
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


def compare_cv_results(results_1: CrossValidationResults,
                       results_2: CrossValidationResults,
                       metric: str = 'balanced_accuracy') -> Dict:
    """
    Compare two sets of CV results using paired t-test.
    
    Args:
        results_1: CV results from first model
        results_2: CV results from second model
        metric: Metric to compare
        
    Returns:
        Dictionary with comparison results
    """
    scores_1 = results_1.get_metric_scores(metric)
    scores_2 = results_2.get_metric_scores(metric)
    
    if len(scores_1) != len(scores_2):
        raise ValueError("Both models must have same number of folds")
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(scores_1, scores_2)
    
    return {
        'model_1': results_1.model_name,
        'model_2': results_2.model_name,
        'metric': metric,
        'mean_1': scores_1.mean(),
        'mean_2': scores_2.mean(),
        'mean_difference': (scores_1 - scores_2).mean(),
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'better_model': results_1.model_name if scores_1.mean() > scores_2.mean() else results_2.model_name
    }

