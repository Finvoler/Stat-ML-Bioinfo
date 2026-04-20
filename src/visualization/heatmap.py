"""
Two-way clustered heatmap generation for gene expression visualization.

Heatmaps are the primary diagnostic tool for inspecting high-dimensional
omics data. Two-way clustering (genes AND samples simultaneously) reveals:
- Co-expressed gene modules (functional clusters)
- Sample subtypes with distinct molecular signatures

Key preprocessing: Z-score normalization row-wise (per gene) removes
the confounding effect of absolute expression magnitude, ensuring the
color encodes relative variation (high/low vs. mean).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from typing import Optional, Tuple


def clustered_heatmap(
    expression: pd.DataFrame,
    labels: Optional[pd.Series] = None,
    z_score: bool = True,
    method: str = "average",
    metric: str = "correlation",
    cmap: str = "RdBu_r",
    figsize: Tuple[int, int] = (14, 10),
    title: str = "Gene Expression Heatmap",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generate a two-way hierarchically clustered heatmap.

    Both rows (genes) and columns (samples) are reordered by hierarchical
    clustering, grouping similar genes and samples together. Correlation
    distance captures "shape" similarity (up/down co-regulation), which
    is more biologically relevant than Euclidean distance for expression data.

    Parameters
    ----------
    expression : pd.DataFrame
        Expression matrix (genes x samples). Should be log2-transformed.
    labels : pd.Series, optional
        Sample-level annotation (e.g., subtype or batch).
        Displayed as a color bar above the heatmap.
    z_score : bool
        Apply row-wise Z-score standardization before plotting. Default True.
    method : str
        Linkage method: 'average' (UPGMA), 'ward', 'complete'. Default 'average'.
    metric : str
        Distance metric: 'correlation' or 'euclidean'. Default 'correlation'.
    cmap : str
        Diverging colormap. 'RdBu_r' (red=high, blue=low). Default 'RdBu_r'.
    figsize : tuple
        Figure dimensions in inches. Default (14, 10).
    title : str
        Plot title.
    save_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    matplotlib Figure object.
    """
    data = expression.copy()

    # Z-score normalize rows (genes) to equalize expression scales
    if z_score:
        mu = data.mean(axis=1)
        sigma = data.std(axis=1).replace(0, 1)
        data = data.sub(mu, axis=0).div(sigma, axis=0)
        data = data.clip(-3, 3)  # Cap for visual clarity

    # Build sample annotation colors
    col_colors = None
    if labels is not None:
        unique_labels = labels.unique()
        palette = sns.color_palette("tab10", n_colors=len(unique_labels))
        color_map = dict(zip(sorted(unique_labels), palette))
        col_colors = labels.map(color_map)

    # Create seaborn clustermap
    g = sns.clustermap(
        data,
        method=method,
        metric=metric,
        col_colors=col_colors,
        cmap=cmap,
        figsize=figsize,
        xticklabels=False,
        yticklabels=data.shape[0] <= 50,  # Only show gene labels if small
        linewidths=0,
        cbar_kws={"label": "Z-score" if z_score else "log2 Expression"},
    )
    g.fig.suptitle(title, y=1.02, fontsize=14, fontweight="bold")

    if labels is not None:
        # Add legend for sample colors
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=c, label=str(l)) for l, c in color_map.items()]
        g.ax_heatmap.legend(
            handles=handles,
            title="Subtype",
            bbox_to_anchor=(1.15, 1),
            loc="upper left",
            fontsize=9,
        )

    if save_path:
        g.fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return g.fig


def expression_heatmap(
    expression: pd.DataFrame,
    genes: list,
    samples: Optional[list] = None,
    labels: Optional[pd.Series] = None,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Selected Gene Expression",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a heatmap for a specified subset of genes.

    Useful for visualizing the expression of known biomarker panels
    (e.g., PAM50 genes, top SHAP-ranked genes) across samples.

    Parameters
    ----------
    expression : pd.DataFrame
        Full expression matrix (genes x samples).
    genes : list
        Subset of gene names to display.
    samples : list, optional
        Subset of sample names. All samples used if None.
    labels : pd.Series, optional
        Sample annotations for column color bar.
    """
    subset = expression.loc[
        [g for g in genes if g in expression.index],
        samples if samples is not None else expression.columns,
    ]
    return clustered_heatmap(
        subset, labels=labels, z_score=True,
        figsize=figsize, title=title, save_path=save_path
    )
