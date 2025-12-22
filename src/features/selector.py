"""
Feature selection using Recursive Feature Elimination (RFE).

This module provides interpretable feature selection that preserves the physical
meaning of acoustic features, in contrast to PCA which creates abstract components.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import pickle

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import N_COMPONENTS_PCA, RANDOM_STATE


class FeatureSelector:
    """
    Recursive Feature Elimination for interpretable feature selection.
    
    This approach preserves the physical meaning of acoustic features (e.g., 
    "mfcc_2_mean", "delta_mfcc_5_std") rather than transforming them into 
    abstract principal components.
    
    Theoretical advantages over PCA for bioacoustic signals:
    1. Preserves interpretability: Selected features have direct physical meaning
    2. Considers class discrimination: Selects features by classification power
    3. Handles non-linear relationships: Uses tree-based estimator
    4. Avoids discarding low-variance but discriminative features
    """
    
    def __init__(self, n_features: int = N_COMPONENTS_PCA,
                 estimator: str = 'random_forest',
                 n_estimators: int = 100,
                 use_cv: bool = False,
                 cv_folds: int = 5,
                 random_state: int = RANDOM_STATE):
        """
        Initialize the feature selector.
        
        Args:
            n_features: Target number of features to select (= number of qubits)
            estimator: Base estimator type ('random_forest', 'gradient_boosting')
            n_estimators: Number of trees in the forest
            use_cv: Whether to use cross-validation to find optimal n_features
            cv_folds: Number of CV folds if use_cv=True
            random_state: Random seed for reproducibility
        """
        self.n_features = n_features
        self.estimator_type = estimator
        self.n_estimators = n_estimators
        self.use_cv = use_cv
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Initialize scaler for normalization
        self.scaler = StandardScaler()
        
        # Create base estimator
        self._base_estimator = self._create_estimator()
        
        # Will be set after fitting
        self.rfe = None
        self.selected_indices_ = None
        self.selected_names_ = None
        self.feature_importances_ = None
        self.feature_ranking_ = None
        self._is_fitted = False
        
    def _create_estimator(self):
        """Create the base estimator for RFE."""
        if self.estimator_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=10,
                min_samples_split=5,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.estimator_type == 'gradient_boosting':
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=5,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown estimator type: {self.estimator_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None) -> 'FeatureSelector':
        """
        Fit the feature selector on training data.
        
        Args:
            X: Feature matrix (n_samples x n_features)
            y: Target labels
            feature_names: Optional list of feature names
            
        Returns:
            Self
        """
        # Store feature names
        if feature_names is not None:
            self._feature_names = list(feature_names)
        else:
            self._feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create RFE or RFECV
        if self.use_cv:
            self.rfe = RFECV(
                estimator=self._base_estimator,
                min_features_to_select=self.n_features,
                cv=self.cv_folds,
                scoring='balanced_accuracy',
                n_jobs=-1
            )
        else:
            self.rfe = RFE(
                estimator=self._base_estimator,
                n_features_to_select=self.n_features,
                step=1
            )
        
        # Fit RFE
        self.rfe.fit(X_scaled, y)
        
        # Store results
        self.selected_indices_ = np.where(self.rfe.support_)[0]
        self.selected_names_ = [self._feature_names[i] for i in self.selected_indices_]
        self.feature_ranking_ = self.rfe.ranking_
        
        # Get feature importances from the fitted estimator
        if hasattr(self.rfe.estimator_, 'feature_importances_'):
            # Map importances back to selected features
            importances = self.rfe.estimator_.feature_importances_
            self.feature_importances_ = dict(zip(self.selected_names_, importances))
        else:
            self.feature_importances_ = None
        
        self._is_fitted = True
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features to selected subset.
        
        Args:
            X: Feature matrix (n_samples x n_features)
            
        Returns:
            Reduced feature matrix (n_samples x n_selected_features)
        """
        if not self._is_fitted:
            raise ValueError("Selector must be fitted before transform")
        
        # Scale features using fitted scaler
        X_scaled = self.scaler.transform(X)
        
        # Select features
        X_selected = X_scaled[:, self.selected_indices_]
        
        return X_selected
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray,
                      feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Optional list of feature names
            
        Returns:
            Reduced feature matrix
        """
        self.fit(X, y, feature_names)
        return self.transform(X)
    
    def get_selected_feature_names(self) -> List[str]:
        """Get names of selected features."""
        if not self._is_fitted:
            raise ValueError("Selector must be fitted first")
        return self.selected_names_
    
    def get_selected_indices(self) -> np.ndarray:
        """Get indices of selected features."""
        if not self._is_fitted:
            raise ValueError("Selector must be fitted first")
        return self.selected_indices_
    
    def get_feature_importances(self) -> Dict[str, float]:
        """
        Get importance scores for selected features.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self._is_fitted:
            raise ValueError("Selector must be fitted first")
        if self.feature_importances_ is None:
            raise ValueError("Feature importances not available for this estimator")
        return self.feature_importances_
    
    def get_feature_ranking(self) -> Dict[str, int]:
        """
        Get ranking for all features (1 = selected, higher = eliminated earlier).
        
        Returns:
            Dictionary mapping feature names to rankings
        """
        if not self._is_fitted:
            raise ValueError("Selector must be fitted first")
        return dict(zip(self._feature_names, self.feature_ranking_))
    
    def print_summary(self) -> None:
        """Print a summary of the feature selection."""
        if not self._is_fitted:
            print("Selector not fitted yet")
            return
        
        print("\n" + "=" * 60)
        print("Feature Selection Summary (RFE)")
        print("=" * 60)
        print(f"Input features: {len(self._feature_names)}")
        print(f"Selected features: {self.n_features}")
        print(f"Base estimator: {self.estimator_type}")
        
        print("\nSelected features (by importance):")
        if self.feature_importances_:
            # Sort by importance
            sorted_feats = sorted(
                self.feature_importances_.items(),
                key=lambda x: x[1],
                reverse=True
            )
            for i, (name, importance) in enumerate(sorted_feats, 1):
                print(f"  {i}. {name}: {importance:.4f}")
        else:
            for i, name in enumerate(self.selected_names_, 1):
                print(f"  {i}. {name}")
        
        print("=" * 60 + "\n")
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the fitted selector to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "rfe": self.rfe,
                "n_features": self.n_features,
                "estimator_type": self.estimator_type,
                "selected_indices": self.selected_indices_,
                "selected_names": self.selected_names_,
                "feature_importances": self.feature_importances_,
                "feature_ranking": self.feature_ranking_,
                "feature_names": self._feature_names
            }, f)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'FeatureSelector':
        """Load a fitted selector from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        selector = cls(
            n_features=data["n_features"],
            estimator=data["estimator_type"]
        )
        selector.scaler = data["scaler"]
        selector.rfe = data["rfe"]
        selector.selected_indices_ = data["selected_indices"]
        selector.selected_names_ = data["selected_names"]
        selector.feature_importances_ = data["feature_importances"]
        selector.feature_ranking_ = data["feature_ranking"]
        selector._feature_names = data["feature_names"]
        selector._is_fitted = True
        
        return selector


def compare_pca_vs_rfe(X: np.ndarray, y: np.ndarray,
                       feature_names: List[str],
                       n_components: int = 8) -> Dict:
    """
    Compare PCA and RFE approaches for dimensionality reduction.
    
    This function provides a quantitative comparison to support the
    theoretical justification for using RFE over PCA.
    
    Args:
        X: Feature matrix
        y: Target labels
        feature_names: List of feature names
        n_components: Number of components/features to select
        
    Returns:
        Dictionary with comparison results
    """
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    # PCA approach
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    svm_pca = SVC(kernel='rbf', class_weight='balanced', random_state=42)
    scores_pca = cross_val_score(svm_pca, X_pca, y, cv=5, scoring='balanced_accuracy')
    
    results['pca'] = {
        'mean_score': scores_pca.mean(),
        'std_score': scores_pca.std(),
        'explained_variance': pca.explained_variance_ratio_.sum(),
        'interpretable': False,
        'component_names': [f"PC{i+1}" for i in range(n_components)]
    }
    
    # RFE approach
    selector = FeatureSelector(n_features=n_components)
    X_rfe = selector.fit_transform(X, y, feature_names)
    
    svm_rfe = SVC(kernel='rbf', class_weight='balanced', random_state=42)
    scores_rfe = cross_val_score(svm_rfe, X_rfe, y, cv=5, scoring='balanced_accuracy')
    
    results['rfe'] = {
        'mean_score': scores_rfe.mean(),
        'std_score': scores_rfe.std(),
        'selected_features': selector.get_selected_feature_names(),
        'feature_importances': selector.get_feature_importances(),
        'interpretable': True
    }
    
    # Statistical comparison
    from scipy.stats import ttest_rel
    t_stat, p_value = ttest_rel(scores_rfe, scores_pca)
    
    results['comparison'] = {
        't_statistic': t_stat,
        'p_value': p_value,
        'rfe_better': scores_rfe.mean() > scores_pca.mean(),
        'significant': p_value < 0.05
    }
    
    return results


def select_features_pipeline(df: pd.DataFrame,
                             feature_cols: List[str],
                             target_col: str = 'label',
                             n_features: int = N_COMPONENTS_PCA,
                             save_selector: bool = True) -> Tuple[np.ndarray, FeatureSelector]:
    """
    Complete feature selection pipeline.
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        target_col: Name of target column
        n_features: Number of features to select
        save_selector: Whether to save the selector
        
    Returns:
        Tuple of (selected features, fitted selector)
    """
    from config.paths import PROCESSED_DATA_DIR
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    selector = FeatureSelector(n_features=n_features)
    X_selected = selector.fit_transform(X, y, feature_cols)
    
    selector.print_summary()
    
    if save_selector:
        selector.save(PROCESSED_DATA_DIR / "selector.pkl")
    
    return X_selected, selector

