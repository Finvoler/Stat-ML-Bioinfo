"""
Regularized regression for high-dimensional genomic data.

Addresses the p >> n problem where OLS fails (X^T X is singular):

  Ridge (L2):    beta = argmin ||Y - Xβ||² + λ||β||²
  Lasso (L1):    beta = argmin ||Y - Xβ||² + λ||β||₁
  Elastic Net:   beta = argmin ||Y - Xβ||² + λ₁||β||₁ + λ₂||β||²

Key biological insight: Lasso's L1 penalty produces SPARSE solutions —
it drives most β_j exactly to zero, automatically selecting a small set
of informative genes from tens of thousands. Elastic Net additionally
groups correlated genes (Ridge property), avoiding the instability of
Lasso when genes are co-expressed in pathways.

References
----------
Hoerl & Kennard (1970). Ridge regression. Technometrics, 12(1), 55-67.
Tibshirani (1996). Lasso regression. JRSS-B, 58(1), 267-288.
Zou & Hastie (2005). Elastic Net. JRSS-B, 67(2), 301-320.
"""

import numpy as np
import pandas as pd
from typing import Literal, Optional

from sklearn.linear_model import (
    ElasticNetCV,
    LassoCV,
    LogisticRegressionCV,
    RidgeCV,
)
from sklearn.preprocessing import StandardScaler


class RegularizedRegression:
    """
    Cross-validated regularized regression for genomics.

    Supports continuous outcomes (gene expression → phenotype) and binary
    classification (gene expression → disease status) via Logistic regression.
    Lambda (regularization strength) is selected by cross-validation.

    Parameters
    ----------
    model_type : str
        One of {'lasso', 'ridge', 'elastic_net', 'logistic_lasso'}.
    cv : int
        Number of CV folds for lambda selection. Default 5.
    n_alphas : int
        Number of lambda values on the regularization path. Default 100.
    l1_ratio : float or list
        Elastic Net mixing parameter ρ: penalty = ρ*L1 + (1-ρ)*L2.
        Default [0.1, 0.5, 0.7, 0.9, 0.95, 1.0].
    random_state : int
        Reproducibility seed.
    """

    def __init__(
        self,
        model_type: Literal["lasso", "ridge", "elastic_net", "logistic_lasso"] = "lasso",
        cv: int = 5,
        n_alphas: int = 100,
        l1_ratio: Optional[list] = None,
        random_state: int = 42,
    ):
        self.model_type = model_type
        self.cv = cv
        self.n_alphas = n_alphas
        self.l1_ratio = l1_ratio or [0.1, 0.5, 0.7, 0.9, 0.95, 1.0]
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self._feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RegularizedRegression":
        """
        Fit the regularized model with CV-selected lambda.

        Parameters
        ----------
        X : pd.DataFrame
            Gene expression matrix (samples x genes), log2-TPM recommended.
        y : pd.Series
            Continuous phenotype or binary disease status.
        """
        self._feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)

        if self.model_type == "lasso":
            self.model = LassoCV(cv=self.cv, n_alphas=self.n_alphas, n_jobs=-1)
        elif self.model_type == "ridge":
            alphas = np.logspace(-4, 4, self.n_alphas)
            self.model = RidgeCV(alphas=alphas, cv=self.cv)
        elif self.model_type == "elastic_net":
            self.model = ElasticNetCV(
                l1_ratio=self.l1_ratio,
                cv=self.cv,
                n_alphas=self.n_alphas,
                n_jobs=-1,
            )
        elif self.model_type == "logistic_lasso":
            self.model = LogisticRegressionCV(
                penalty="l1",
                solver="liblinear",
                cv=self.cv,
                random_state=self.random_state,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type!r}")

        self.model.fit(X_scaled, y)
        return self

    @property
    def coef_(self) -> pd.Series:
        """
        Return regression coefficients as a named Series.

        For Lasso/Elastic Net, many coefficients will be exactly zero —
        these genes are eliminated from the model. Non-zero genes are the
        selected biomarker candidates.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before accessing coefficients.")
        coef = getattr(self.model, "coef_", None)
        if coef is None:
            raise AttributeError("Model has no coef_ attribute.")
        if coef.ndim > 1:
            coef = coef.ravel()
        return pd.Series(coef, index=self._feature_names)

    @property
    def selected_genes(self) -> pd.Index:
        """Return genes with non-zero coefficients (Lasso/Elastic Net only)."""
        return self.coef_[self.coef_ != 0].index

    @property
    def n_selected(self) -> int:
        """Number of genes retained after regularization."""
        return (self.coef_ != 0).sum()

    @property
    def best_lambda(self) -> float:
        """Return the CV-selected regularization strength."""
        if hasattr(self.model, "alpha_"):
            return self.model.alpha_
        if hasattr(self.model, "C_"):
            return 1.0 / float(np.mean(self.model.C_))
        raise AttributeError("Could not retrieve lambda from this model.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def sparsity_path(self) -> pd.DataFrame:
        """
        Return coefficient path over the regularization path (Lasso/ElasticNet).

        Useful for visualizing how genes enter/exit the model as lambda varies,
        providing insight into the stability of biomarker selection.
        """
        if not hasattr(self.model, "coef_path_"):
            raise AttributeError("coef_path_ is only available for LassoCV/ElasticNetCV.")
        alphas = self.model.alphas_
        path = self.model.coef_path_.T   # shape: (n_alphas, n_genes)
        return pd.DataFrame(path, index=alphas, columns=self._feature_names)
