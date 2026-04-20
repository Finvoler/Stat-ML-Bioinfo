"""
RNA-seq expression normalization methods.

Implements RPKM, TPM, log2 transform, and Z-score standardization.
TPM is preferred for cross-sample comparison because all column sums
equal 10^6, making relative abundance directly comparable.

References
----------
Mortazavi et al. (2008). Mapping and quantifying mammalian transcriptomes
by RNA-Seq. Nature Methods, 5(7), 621-628.
"""

import numpy as np
import pandas as pd


def rpkm(counts: pd.DataFrame, gene_lengths: pd.Series) -> pd.DataFrame:
    """
    Compute RPKM (Reads Per Kilobase per Million mapped reads).

    RPKM_i = C_i / (L_i_kb * N_million)

    where C_i is the raw read count for gene i, L_i_kb is its length in
    kilobases, and N_million is the total library size in millions.

    Parameters
    ----------
    counts : pd.DataFrame
        Raw read count matrix, shape (n_genes, n_samples).
        Index must be gene identifiers matching gene_lengths.
    gene_lengths : pd.Series
        Gene lengths in base pairs, indexed by gene identifier.

    Returns
    -------
    pd.DataFrame
        RPKM-normalized matrix, same shape as counts.

    Notes
    -----
    RPKM values are NOT directly comparable across samples because
    per-sample RPKM sums differ. Prefer TPM for cross-sample analyses.
    """
    lengths_kb = gene_lengths.loc[counts.index] / 1_000.0
    library_sizes_million = counts.sum(axis=0) / 1_000_000.0
    rpkm_vals = counts.div(lengths_kb, axis=0).div(library_sizes_million, axis=1)
    return rpkm_vals


def tpm(counts: pd.DataFrame, gene_lengths: pd.Series) -> pd.DataFrame:
    """
    Compute TPM (Transcripts Per Million).

    TPM first normalizes by gene length, then scales so that each
    sample's total equals exactly 10^6. This ensures the sum-to-constant
    property:  sum_i TPM_i = 10^6  for every sample.

    TPM_i = (C_i / L_i_kb) / sum_j(C_j / L_j_kb) * 10^6

    Parameters
    ----------
    counts : pd.DataFrame
        Raw read count matrix, shape (n_genes, n_samples).
    gene_lengths : pd.Series
        Gene lengths in base pairs.

    Returns
    -------
    pd.DataFrame
        TPM-normalized matrix. Every column sums to 10^6, enabling
        direct cross-sample relative abundance comparison.
    """
    lengths_kb = gene_lengths.loc[counts.index] / 1_000.0
    # Step 1: length-normalized rates
    rate = counts.div(lengths_kb, axis=0)
    # Step 2: scale to per-million
    tpm_vals = rate.div(rate.sum(axis=0), axis=1) * 1_000_000.0
    return tpm_vals


def log2_transform(
    expression: pd.DataFrame, pseudocount: float = 1.0
) -> pd.DataFrame:
    """
    Apply log2 transformation with pseudocount for variance stabilization.

    x'_ij = log2(x_ij + pseudocount)

    Log2 transformation serves three purposes:
    1. Compresses the wide dynamic range of expression values (~10^5 fold).
    2. Approximately symmetrizes the distribution toward Gaussianity,
       satisfying assumptions of linear models and PCA.
    3. Stabilizes variance across the expression range (heteroscedasticity
       reduction).

    Parameters
    ----------
    expression : pd.DataFrame
        Expression matrix (genes x samples), typically TPM or RPKM values.
    pseudocount : float
        Small constant added before log to handle zero counts. Default 1.0.

    Returns
    -------
    pd.DataFrame
        Log2-transformed values, same shape as input.
    """
    return np.log2(expression + pseudocount)


def z_score_normalize(
    expression: pd.DataFrame, axis: int = 0
) -> pd.DataFrame:
    """
    Perform Z-score standardization (mean=0, std=1).

    z_ij = (x_ij - mu_i) / sigma_i   (axis=0: gene-wise)

    Used prior to heatmap generation to remove the confounding effect of
    different baseline expression levels across genes, ensuring that the
    color scale reflects relative variation rather than absolute magnitude.

    Parameters
    ----------
    expression : pd.DataFrame
        Log-transformed expression matrix (genes x samples).
    axis : int
        0 → normalize along columns (gene-wise, standard for heatmaps).
        1 → normalize along rows (sample-wise).

    Returns
    -------
    pd.DataFrame
        Z-score normalized matrix. Rows (axis=0) or columns (axis=1)
        have zero mean and unit standard deviation.
    """
    mu = expression.mean(axis=axis)
    sigma = expression.std(axis=axis).replace(0, 1)  # avoid div-by-zero for constant genes

    if axis == 0:
        return expression.subtract(mu, axis=1).divide(sigma, axis=1)
    else:
        return expression.subtract(mu, axis=0).divide(sigma, axis=0)
