"""
Dimensionality reduction: PCA and t-SNE for omics visualization.

High-dimensional gene expression data (p > 20,000) cannot be directly
visualized. Dimensionality reduction serves two purposes:

1. Quality Control: PCA scatter plots reveal batch effects, outliers,
   and whether samples separate by biological condition vs technical noise.

2. Exploratory Discovery: t-SNE reveals non-linear cluster structure
   (e.g., novel cell subtypes in scRNA-seq), which linear PCA misses.

References
----------
van der Maaten & Hinton (2008). Visualizing data using t-SNE.
    JMLR, 9(11), 2579-2605.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple


def pca_reduce(
    expression: pd.DataFrame,
    n_components: int = 50,
    scale: bool = True,
    return_variance: bool = False,
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """
    Perform PCA on expression data.

    PCA finds the directions w_k of maximum variance:

        w_1 = argmax_{||w||=1} Var(Xw) = argmax_{||w||=1} w^T C w

    where C = (1/(n-1)) X^T X is the gene covariance matrix.
    Solving via SVD: X = U Σ V^T, principal components are columns of V.

    Parameters
    ----------
    expression : pd.DataFrame
        Log2-transformed expression matrix (genes x samples).
    n_components : int
        Number of PCs to retain. Default 50 (covers ~80% variance typically).
    scale : bool
        Whether to z-score genes before PCA. Recommended when comparing
        genes with different dynamic ranges. Default True.
    return_variance : bool
        If True, also return explained variance ratio array.

    Returns
    -------
    pc_scores : pd.DataFrame
        Shape (n_samples, n_components) — rows are samples in PC space.
    explained_variance : np.ndarray or None
        Fraction of total variance explained by each PC (if requested).
    """
    X = expression.T.values.astype(float)  # (n_samples, n_genes)

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    n_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(X)

    col_names = [f"PC{i+1}" for i in range(n_components)]
    pc_df = pd.DataFrame(scores, index=expression.columns, columns=col_names)

    if return_variance:
        return pc_df, pca.explained_variance_ratio_
    return pc_df, None


def scree_data(expression: pd.DataFrame, n_components: int = 30) -> pd.DataFrame:
    """
    Compute variance explained for a scree plot.

    Used to determine the "elbow" — the number of PCs to retain before
    the marginal variance explained becomes negligible.

    Returns
    -------
    pd.DataFrame
        Columns: ['PC', 'explained_variance', 'cumulative_variance'].
    """
    X = StandardScaler().fit_transform(expression.T.values.astype(float))
    n_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X)
    ev = pca.explained_variance_ratio_
    return pd.DataFrame({
        "PC": range(1, n_components + 1),
        "explained_variance": ev,
        "cumulative_variance": np.cumsum(ev),
    })


def tsne_embed(
    expression: pd.DataFrame,
    n_components: int = 2,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    pca_init_components: int = 50,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Embed samples into 2D using t-SNE for cluster visualization.

    t-SNE minimizes the KL divergence between high-dimensional pairwise
    similarities (Gaussian kernel) and low-dimensional similarities
    (Student-t kernel with df=1, i.e., Cauchy):

        q_ij = (1 + ||y_i - y_j||²)^{-1} / Σ_{k≠l} (1 + ||y_k - y_l||²)^{-1}

    The heavy-tailed t-distribution prevents the "crowding problem" — distant
    points are pushed far apart, revealing cluster boundaries more clearly.

    Best practice: first reduce to 50 PCs, then apply t-SNE. This removes
    noise dimensions while preserving the dominant biological variation.

    Parameters
    ----------
    expression : pd.DataFrame
        Log2-transformed expression matrix (genes x samples).
    n_components : int
        Output dimensionality. 2 for 2D scatter, 3 for 3D. Default 2.
    perplexity : float
        Effective number of nearest neighbors (5–50 typical). Default 30.
    n_iter : int
        Number of optimization iterations. Default 1000.
    pca_init_components : int
        Pre-reduce to this many PCs before t-SNE. Default 50.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    pd.DataFrame
        Shape (n_samples, n_components) with columns ['tSNE1', 'tSNE2'].
    """
    # Step 1: PCA pre-reduction to remove noise and speed up t-SNE
    pc_scores, _ = pca_reduce(expression, n_components=pca_init_components, scale=True)
    X_pca = pc_scores.values

    # Step 2: t-SNE embedding
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    embedding = tsne.fit_transform(X_pca)

    col_names = [f"tSNE{i+1}" for i in range(n_components)]
    return pd.DataFrame(embedding, index=expression.columns, columns=col_names)
