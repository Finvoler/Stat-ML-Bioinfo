"""
Cancer subtype classification using SVM and Random Forest.

Implements two canonical supervised learning approaches for high-dimensional
genomic classification:

1. Support Vector Machine (SVM) with optional RBF / linear kernel.
   - SVM-RFE: Recursive Feature Elimination to identify biomarker signatures.
   - One-vs-Rest (OvR) strategy for PAM50 5-class subtyping.

2. Random Forest (RF) with OOB error estimation and variable importance.
   - Double randomness (Bootstrap + feature subsampling) for variance reduction.
   - Native multi-class support without explicit OvR decomposition.

References
----------
Cortes & Vapnik (1995). Support-vector networks. Machine Learning, 20(3), 273-297.
Guyon et al. (2002). Gene selection for cancer classification using SVMs.
    Machine Learning, 46(1), 389-422.
Breiman (2001). Random forests. Machine Learning, 45(1), 5-32.
Parker et al. (2009). Supervised risk predictor of breast cancer based on
    intrinsic subtypes. Journal of Clinical Oncology, 27(8), 1160-1167.
"""

import numpy as np
import pandas as pd
from typing import Literal, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


PAM50_SUBTYPES = ["Luminal A", "Luminal B", "HER2-enriched", "Basal-like", "Normal-like"]


class BreastCancerClassifier:
    """
    Unified interface for cancer molecular subtype classification.

    Supports two model families:
    - 'svm_linear': Linear SVM via One-vs-Rest (efficient for p >> n).
    - 'svm_rbf'  : RBF-kernel SVM (captures non-linear gene interactions).
    - 'random_forest': Random Forest with OOB error estimation.

    Parameters
    ----------
    model_type : str
        One of {'svm_linear', 'svm_rbf', 'random_forest'}.
    n_estimators : int
        Number of trees (Random Forest only). Default 500.
    C : float
        SVM regularization parameter. Larger C → stricter margin. Default 1.0.
    gamma : str or float
        RBF kernel coefficient. Default 'scale' (1 / (n_features * X.var())).
    random_state : int
        Reproducibility seed. Default 42.
    """

    def __init__(
        self,
        model_type: Literal["svm_linear", "svm_rbf", "random_forest"] = "random_forest",
        n_estimators: int = 500,
        C: float = 1.0,
        gamma: str = "scale",
        random_state: int = 42,
    ):
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.C = C
        self.gamma = gamma
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.model = self._build_model()

    def _build_model(self):
        scaler = StandardScaler()

        if self.model_type == "svm_linear":
            clf = OneVsRestClassifier(
                LinearSVC(C=self.C, max_iter=5000, random_state=self.random_state),
                n_jobs=-1,
            )
        elif self.model_type == "svm_rbf":
            clf = OneVsRestClassifier(
                SVC(
                    kernel="rbf",
                    C=self.C,
                    gamma=self.gamma,
                    probability=True,
                    random_state=self.random_state,
                ),
                n_jobs=-1,
            )
        elif self.model_type == "random_forest":
            clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_features="sqrt",   # m ~ sqrt(p), standard for classification
                oob_score=True,
                n_jobs=-1,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type!r}")

        return Pipeline([("scaler", scaler), ("clf", clf)])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BreastCancerClassifier":
        """
        Fit the classifier.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (samples x genes), e.g., PAM50 log2-TPM values.
        y : pd.Series
            Subtype labels (string or integer).
        """
        y_enc = self.label_encoder.fit_transform(y)
        self.model.fit(X, y_enc)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted subtype labels (decoded back to original strings)."""
        y_pred_enc = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred_enc)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probability estimates (requires probability=True for SVM)."""
        clf = self.model.named_steps["clf"]
        if not hasattr(clf, "predict_proba"):
            raise AttributeError(
                "predict_proba requires SVM with probability=True or Random Forest."
            )
        return self.model.predict_proba(X)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Return overall accuracy on (X, y)."""
        y_enc = self.label_encoder.transform(y)
        y_pred = self.model.predict(X)
        return accuracy_score(y_enc, y_pred)

    def oob_score(self) -> Optional[float]:
        """
        Return Out-of-Bag error estimate (Random Forest only).

        OOB uses the ~36.8% of samples not included in each bootstrap
        replicate as an internal test set, avoiding the need for a
        separate validation split in small-n studies.
        """
        if self.model_type != "random_forest":
            return None
        return self.model.named_steps["clf"].oob_score_

    def feature_importances(self, feature_names: list) -> pd.Series:
        """
        Return gene importance scores (Random Forest only).

        Uses Mean Decrease in Gini Impurity aggregated across all trees.
        Genes with high importance are candidate diagnostic biomarkers.

        Parameters
        ----------
        feature_names : list
            Gene/feature names corresponding to columns of X.

        Returns
        -------
        pd.Series
            Importance scores sorted descending.
        """
        clf = self.model.named_steps["clf"]
        if not isinstance(clf, RandomForestClassifier):
            raise AttributeError("Feature importances are only available for Random Forest.")
        importances = clf.feature_importances_
        return pd.Series(importances, index=feature_names).sort_values(ascending=False)


def svm_rfe(
    X: pd.DataFrame,
    y: pd.Series,
    n_features_to_select: int = 20,
    step: int = 1,
) -> tuple[np.ndarray, pd.Series]:
    """
    SVM-Recursive Feature Elimination for genomic biomarker discovery.

    Algorithm:
    1. Train a linear SVM on current feature set.
    2. Rank features by |w_j|^2 (squared SVM weight).
    3. Remove the lowest-ranked feature(s).
    4. Repeat until target feature count is reached.

    The key insight: |w_j|^2 directly reflects a gene's contribution to
    the separating hyperplane. High-weight genes are the most discriminative
    biomarkers for clinical assay development.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples x genes).
    y : pd.Series
        Class labels.
    n_features_to_select : int
        Target number of genes after elimination.
    step : int
        Number of features removed per iteration.

    Returns
    -------
    selected_mask : np.ndarray
        Boolean mask of selected features.
    ranking : pd.Series
        Rank of each gene (1 = most important).
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    estimator = LinearSVC(C=1.0, max_iter=5000, random_state=42)
    selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=step)
    selector.fit(X_scaled, y_enc)

    ranking = pd.Series(selector.ranking_, index=X.columns, name="svm_rfe_rank")
    return selector.support_, ranking


def evaluate_multiclass(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list] = None,
) -> dict:
    """
    Comprehensive evaluation metrics for multi-class classification.

    Returns accuracy, Macro-F1, per-class F1, and the confusion matrix.
    For imbalanced PAM50 data (Luminal A >> HER2-enriched), Macro-F1 is
    the primary metric as it treats all classes equally regardless of size.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.
    class_names : list, optional
        Human-readable class names for the report.

    Returns
    -------
    dict with keys: accuracy, macro_f1, report (str), confusion_matrix (ndarray)
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "report": classification_report(y_true, y_pred, target_names=class_names),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
