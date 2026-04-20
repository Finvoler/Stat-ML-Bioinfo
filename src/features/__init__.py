"""Feature selection and SHAP-based interpretability."""

from .selection import shap_feature_ranking, variance_filter, pearson_filter
from .dimensionality import pca_reduce, tsne_embed

__all__ = [
    "shap_feature_ranking",
    "variance_filter",
    "pearson_filter",
    "pca_reduce",
    "tsne_embed",
]
