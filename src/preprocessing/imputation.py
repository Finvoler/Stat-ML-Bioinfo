"""
KNN-based missing value imputation for omics data.

In microarray and mass-spectrometry proteomics datasets, missing values arise
from probe hybridization failure or values falling below the limit of detection
(LOD). Simple mean imputation distorts the variance structure; KNN imputation
leverages inter-gene co-expression to produce biologically coherent estimates.

Reference
---------
Troyanskaya et al. (2001). Missing value estimation methods for DNA
microarrays. Bioinformatics, 17(6), 520-525.
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


def knn_impute(
    expression: pd.DataFrame,
    n_neighbors: int = 10,
    weights: str = "distance",
    missing_values: float = np.nan,
) -> pd.DataFrame:
    """
    Impute missing values using K-Nearest Neighbors based on co-expression.

    For each gene g with a missing value in sample j, the algorithm:
    1. Finds K genes with the most similar expression profiles across
       all other samples (neighbours are defined in gene space).
    2. Estimates the missing value as the distance-weighted average of
       the K neighbours' values in sample j.

    x_hat_gj = sum_{k in N_K} w_k * x_kj / sum_{k in N_K} w_k

    where weights w_k are the inverse Euclidean distances between profiles.

    Parameters
    ----------
    expression : pd.DataFrame
        Expression matrix (genes x samples) containing NaN for missing entries.
    n_neighbors : int
        Number of neighboring genes used for imputation. Default 10.
    weights : str
        'uniform': equal weighting; 'distance': inverse-distance weighting
        (recommended — closer profiles contribute more).
    missing_values : float
        Value in the matrix representing missing data. Default np.nan.

    Returns
    -------
    pd.DataFrame
        Imputed expression matrix with the same shape and index/columns
        as the input. No NaN values remain.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> expr = pd.DataFrame(np.random.rand(100, 20))
    >>> expr.iloc[5, 3] = np.nan
    >>> expr_imputed = knn_impute(expr)
    >>> assert not expr_imputed.isna().any().any()
    """
    imputer = KNNImputer(
        n_neighbors=n_neighbors,
        weights=weights,
        missing_values=missing_values,
    )
    # KNNImputer operates on samples x features; transpose if genes x samples
    imputed_array = imputer.fit_transform(expression.T).T
    return pd.DataFrame(imputed_array, index=expression.index, columns=expression.columns)


def flag_missing(expression: pd.DataFrame) -> pd.DataFrame:
    """
    Return a boolean DataFrame indicating missing entries.

    Parameters
    ----------
    expression : pd.DataFrame
        Expression matrix potentially containing NaN values.

    Returns
    -------
    pd.DataFrame
        Boolean mask: True where values are missing.
    """
    return expression.isna()


def missing_summary(expression: pd.DataFrame) -> pd.Series:
    """
    Compute per-sample and per-gene missing value fractions.

    Parameters
    ----------
    expression : pd.DataFrame
        Expression matrix.

    Returns
    -------
    dict
        {'per_gene': Series of missing fraction per gene,
         'per_sample': Series of missing fraction per sample}
    """
    per_gene = expression.isna().mean(axis=1)
    per_sample = expression.isna().mean(axis=0)
    return {"per_gene": per_gene, "per_sample": per_sample}
