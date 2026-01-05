"""
Visualization utilities for model evaluation and results.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.paths import FIGURES_DIR
from .metrics import MedicalMetrics


class ResultsVisualizer:
    """
    Creates visualizations for model evaluation results.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6),
                 style: str = "whitegrid",
                 save_dir: Optional[Path] = None):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size
            style: Seaborn style
            save_dir: Directory to save figures
        """
        self.figsize = figsize
        self.save_dir = save_dir or FIGURES_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = figsize
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: List[str] = ["Healthy", "COPD"],
                             model_name: str = "Model",
                             save: bool = True) -> plt.Figure:
        """
        Plot confusion matrix as a heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            model_name: Name of the model
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14)
        
        plt.tight_layout()
        
        if save:
            filepath = self.save_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                      model_name: str = "Model",
                      save: bool = True) -> plt.Figure:
        """
        Plot ROC curve for a single model.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            model_name: Name of the model
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve - {model_name}', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.save_dir / f"roc_curve_{model_name.lower().replace(' ', '_')}.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        return fig
    
    def plot_roc_curves_comparison(self, results: Dict[str, Tuple[np.ndarray, np.ndarray]],
                                   save: bool = True) -> plt.Figure:
        """
        Plot ROC curves for multiple models on the same plot.
        
        Args:
            results: Dictionary mapping model names to (y_true, y_prob) tuples
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
        
        for (model_name, (y_true, y_prob)), color in zip(results.items(), colors):
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, color=color,
                   label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve Comparison', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.save_dir / "roc_curves_comparison.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        return fig
    
    def plot_training_history(self, history: List[float],
                             model_name: str = "Model",
                             metric_name: str = "Loss",
                             save: bool = True) -> plt.Figure:
        """
        Plot training history (loss or accuracy over epochs).
        
        Args:
            history: List of metric values per epoch
            model_name: Name of the model
            metric_name: Name of the metric
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(history) + 1)
        ax.plot(epochs, history, 'b-', lw=2, marker='o', markersize=3)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'Training {metric_name} - {model_name}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add min/max annotations
        min_val = min(history)
        min_epoch = history.index(min_val) + 1
        ax.annotate(f'Min: {min_val:.4f}', xy=(min_epoch, min_val),
                   xytext=(min_epoch + len(history)*0.1, min_val + (max(history)-min_val)*0.1),
                   arrowprops=dict(arrowstyle='->', color='red'),
                   fontsize=10, color='red')
        
        plt.tight_layout()
        
        if save:
            filepath = self.save_dir / f"training_{metric_name.lower()}_{model_name.lower().replace(' ', '_')}.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        return fig
    
    def plot_metrics_comparison(self, results: Dict[str, MedicalMetrics],
                               save: bool = True) -> plt.Figure:
        """
        Plot bar chart comparing metrics across models.
        
        Args:
            results: Dictionary mapping model names to MedicalMetrics objects
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Collect metrics
        models = list(results.keys())
        metrics_names = ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1-Score"]
        
        data = {metric: [] for metric in metrics_names}
        for model_name in models:
            metrics = results[model_name].get_all_metrics()
            for metric in metrics_names:
                data[metric].append(metrics.get(metric, 0))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(models))
        width = 0.15
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(metrics_names)))
        
        for i, (metric, color) in enumerate(zip(metrics_names, colors)):
            offset = (i - len(metrics_names)/2 + 0.5) * width
            bars = ax.bar(x + offset, data[metric], width, label=metric, color=color)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8, rotation=90)
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Comparison - Key Metrics', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        ax.set_ylim(0, 1.15)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            filepath = self.save_dir / "metrics_comparison.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        return fig
    
    def plot_pca_variance(self, explained_variance: np.ndarray,
                         n_selected: int,
                         save: bool = True) -> plt.Figure:
        """
        Plot PCA explained variance.
        
        Args:
            explained_variance: Explained variance ratio per component
            n_selected: Number of selected components
            save: Whether to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        components = range(1, len(explained_variance) + 1)
        cumulative = np.cumsum(explained_variance)
        
        # Individual variance
        ax1.bar(components, explained_variance, color='steelblue', alpha=0.8)
        ax1.axvline(x=n_selected, color='red', linestyle='--', 
                   label=f'Selected ({n_selected} components)')
        ax1.set_xlabel('Principal Component', fontsize=12)
        ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
        ax1.set_title('Individual Explained Variance', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Cumulative variance
        ax2.plot(components, cumulative, 'b-o', lw=2, markersize=6)
        ax2.axhline(y=cumulative[n_selected-1], color='red', linestyle='--',
                   label=f'{cumulative[n_selected-1]:.1%} variance')
        ax2.axvline(x=n_selected, color='red', linestyle='--')
        ax2.fill_between(components[:n_selected], cumulative[:n_selected], 
                        alpha=0.3, color='steelblue')
        ax2.set_xlabel('Number of Components', fontsize=12)
        ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
        ax2.set_title('Cumulative Explained Variance', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        
        plt.tight_layout()
        
        if save:
            filepath = self.save_dir / "pca_variance.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        return fig

