"""
Feature selection methods for high-dimensional genomics.

Three complementary strategies implemented:

1. Variance Filtering: Remove "housekeeping" genes with near-zero variance
   across samples — these carry no discriminative information.

2. Pearson Correlation Filtering: Remove highly redundant gene pairs to
   reduce multicollinearity before feeding into linear models.

3. SHAP-based Ranking: Use Shapley values from a trained tree ensemble to
   identify genes whose expression most impacts individual predictions.
   Unlike p-value ranking, SHAP captures non-linear and interaction effects.

Reference
---------
Lundberg & Lee (2017). A unified approach to interpreting model predictions.
NeurIPS, 30, 4765-4774.
"""

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from typing import Optional, Union


def variance_filter(
    expression: pd.DataFrame,
    min_variance: float = 0.1,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Remove genes with variance below a threshold (or keep top-N by variance).

    Low-variance genes (e.g., constitutively expressed housekeeping genes)
    contribute noise rather than signal to classifiers and are safely removed.

    Parameters
    ----------
    expression : pd.DataFrame
        Log2-transformed expression matrix (genes x samples).
    min_variance : float
        Minimum variance threshold. Genes below this are removed.
        Ignored if top_n is specified.
    top_n : int, optional
        If provided, keep only the top_n highest-variance genes.

    Returns
    -------
    pd.DataFrame
        Filtered expression matrix.
    """
    gene_var = expression.var(axis=1)
    if top_n is not None:
        selected = gene_var.nlargest(top_n).index
    else:
        selected = gene_var[gene_var >= min_variance].index
    return expression.loc[selected]


def pearson_filter(
    expression: pd.DataFrame,
    threshold: float = 0.95,
) -> pd.DataFrame:
    """
    Remove one gene from each highly correlated pair (|r| > threshold).

    Highly co-expressed genes (e.g., genes in the same signaling pathway)
    provide redundant information and inflate p-values in linear models.
    This greedy filter iteratively removes one member of each near-collinear pair.

    Parameters
    ----------
    expression : pd.DataFrame
        Expression matrix (genes x samples).
    threshold : float
        Pearson |r| above which one gene is dropped. Default 0.95.

    Returns
    -------
    pd.DataFrame
        De-correlated expression matrix.
    """
    # Transpose: correlate genes (rows) → compute gene x gene correlation
    corr = expression.T.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return expression.drop(index=to_drop)


def shap_feature_ranking(
    model,
    X: pd.DataFrame,
    top_k: int = 50,
    model_type: str = "tree",
) -> pd.DataFrame:
    """
    Rank features by mean |SHAP| value using a trained model.

    SHAP (SHapley Additive exPlanations) decomposes each prediction into
    additive contributions from individual features:

        f(x) = E[f(x)] + Σ_j φ_j(x)

    where φ_j is the Shapley value computed by marginalizing over all
    feature subsets. |φ_j| averaged across samples gives a global measure
    of gene importance that respects non-linearity and interactions —
    unlike simple coefficient magnitudes or p-values.

    Parameters
    ----------
    model : sklearn estimator
        Fitted tree-based model (RandomForestClassifier, GradientBoosting, etc.).
        For non-tree models, set model_type='kernel' (much slower).
    X : pd.DataFrame
        Feature matrix used for explanation (typically the test set).
    top_k : int
        Number of top genes to return. Default 50.
    model_type : str
        'tree' for TreeExplainer (fast); 'kernel' for model-agnostic explainer.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['gene', 'mean_abs_shap'] sorted descending.
    """
    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    else:
        background = shap.sample(X, min(100, len(X)))
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X)

    # For multi-class, average |SHAP| across all classes
    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0).mean(axis=0)
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)

    ranking = pd.DataFrame(
        {"gene": X.columns, "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False).head(top_k).reset_index(drop=True)

    return ranking


def differential_expression_filter(
    expression: pd.DataFrame,
    labels: pd.Series,
    fdr_threshold: float = 0.05,
    fold_change_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Filter genes by FDR-corrected Welch t-test (two-group comparison).

    Implements the Benjamini-Hochberg (BH) procedure to control False
    Discovery Rate at the specified level:

        P_(k) ≤ (k/m) * α

    Only genes passing BOTH the FDR threshold AND the fold-change threshold
    are retained. Fold-change threshold prevents selecting statistically
    significant but biologically trivial differences.

    Parameters
    ----------
    expression : pd.DataFrame
        Log2-transformed expression (genes x samples).
    labels : pd.Series
        Binary group labels (0/1 or two unique string values).
    fdr_threshold : float
        BH-corrected FDR threshold. Default 0.05.
    fold_change_threshold : float
        Minimum |log2 fold change| to retain a gene. Default 2.0 (4-fold).

    Returns
    -------
    pd.DataFrame
        Filtered expression matrix containing only DE genes.
    """
    from scipy import stats

    unique_groups = labels.unique()
    if len(unique_groups) != 2:
        raise ValueError("differential_expression_filter requires exactly 2 groups.")

    g1 = expression.loc[:, labels == unique_groups[0]]
    g2 = expression.loc[:, labels == unique_groups[1]]

    _, p_values = stats.ttest_ind(g1.T, g2.T, equal_var=False)
    log2fc = g1.mean(axis=1) - g2.mean(axis=1)

    # Benjamini-Hochberg correction
    m = len(p_values)
    order = np.argsort(p_values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, m + 1)
    bh_threshold = (ranks / m) * fdr_threshold
    is_significant = p_values <= bh_threshold

    is_de = is_significant & (log2fc.abs() >= fold_change_threshold)
    return expression.loc[is_de]
