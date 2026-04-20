<div align="center">

# From Inference to Prediction

### Integrating Statistical Learning & Machine Learning for High-Dimensional Bioinformatics Analysis

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)](notebooks/)

> **An end-to-end computational biology framework that bridges classical biostatistics (inference, regularization, hypothesis testing) with modern machine learning (SVM, Random Forest, DeepSurv, SHAP) to tackle high-dimensional omics data — enabling cancer subtype classification, personalized survival prediction, and biologically interpretable biomarker discovery.**

</div>

---

## Problem Statement

Modern high-throughput sequencing generates **petabyte-scale genomic datasets** where the number of features (genes, p > 20,000) vastly outnumbers available samples (n < 500). This **p ≫ n** regime creates three critical challenges:

| Challenge | Traditional Approach | Limitation |
|-----------|---------------------|------------|
| Curse of Dimensionality | Ordinary Least Squares | `X^T X` is singular; no unique solution |
| Overfitting | Deep Learning | High capacity → poor generalization on small cohorts |
| Black-box Predictions | Ensemble Methods | Clinical adoption requires biological justification |

This project implements a **unified analytical framework** that resolves all three challenges simultaneously through principled integration of statistical rigor and predictive power.

---

## Key Contributions

- **Regularization Bridge**: Implements Lasso (L1), Ridge (L2), and Elastic Net as the mathematical link between statistical inference and ML-based feature selection — automatically reducing 20,531 genes to interpretable biomarker signatures.

- **DeepSurv Neural Network**: Replaces the linear predictor in the Cox Proportional Hazards model with a multi-layer neural network, capturing non-linear gene-interaction effects and achieving **+0.05~0.1 C-index improvement** over classical Cox models on TCGA datasets.

- **SHAP-based Genomic Attribution**: Applies cooperative game theory (Shapley values) to assign patient-level contributions to individual genes — translating "black-box" ensemble predictions into actionable biological hypotheses.

- **Cancer Subtype Classification**: End-to-end pipeline on TCGA-BRCA (n=1,000) achieves **~92% accuracy** (Random Forest) and **~89%** (Linear SVM) for 5-class PAM50 molecular subtyping, with clinical-grade confusion analysis.

- **Reproducible Bioinformatics Workflow**: Full preprocessing pipeline including TPM/RPKM normalization, KNN imputation, ComBat batch-effect correction, PCA/t-SNE dimensionality reduction, and FDR-corrected differential expression filtering.

---

## Repository Structure

```
Stat-ML-Bioinfo/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── src/                          # Core library modules
│   ├── preprocessing/
│   │   ├── normalization.py      # TPM, RPKM, log2 transform, Z-score
│   │   ├── imputation.py         # KNN-based missing value imputation
│   │   └── batch_correction.py  # Empirical Bayes (ComBat) batch correction
│   ├── models/
│   │   ├── classification.py     # SVM (RFE), Random Forest, evaluation metrics
│   │   ├── regularization.py     # Lasso, Ridge, Elastic Net with CV tuning
│   │   └── survival.py           # CoxPH wrapper + DeepSurv (PyTorch)
│   ├── features/
│   │   ├── selection.py          # SVM-RFE, SHAP-based feature ranking
│   │   └── dimensionality.py     # PCA, t-SNE wrappers with biological annotations
│   └── visualization/
│       ├── heatmap.py            # Two-way clustered heatmaps (Z-score normalized)
│       └── survival_plots.py     # Kaplan-Meier curves, log-rank test, C-index
│
├── notebooks/                    # End-to-end analysis walkthroughs
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_classification_models.ipynb
│   ├── 03_survival_analysis.ipynb
│   └── 04_model_interpretability.ipynb
│
├── docs/
│   └── methodology.md            # Detailed mathematical derivations
│
└── data/
    └── README.md                 # Data access instructions (TCGA, GEO)
```

---

## Methodology Overview

```
Raw Omics Data (RNA-Seq Counts)
         │
         ▼
┌─────────────────────────────────┐
│     PREPROCESSING PIPELINE      │
│  • KNN Imputation               │
│  • TPM/RPKM Normalization       │
│  • Log₂(x+1) Transform          │
│  • ComBat Batch Correction      │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│     FEATURE ENGINEERING         │
│  • PCA / t-SNE Visualization    │
│  • Variance Filtering           │
│  • Lasso / SVM-RFE Selection    │
└───────────────┬─────────────────┘
                │
       ┌────────┴────────┐
       ▼                 ▼
┌──────────────┐  ┌──────────────────┐
│ SUPERVISED   │  │   SURVIVAL       │
│ LEARNING     │  │   ANALYSIS       │
│ • SVM (RBF)  │  │ • Cox PH Model   │
│ • Rand Forest│  │ • DeepSurv (DNN) │
│ • Elastic Net│  │ • C-index / KM   │
└──────┬───────┘  └────────┬─────────┘
       └────────┬────────┘
                ▼
┌─────────────────────────────────┐
│     INTERPRETABILITY (XAI)      │
│  • SHAP Summary & Waterfall     │
│  • GO/KEGG Pathway Enrichment   │
│  • Biomarker Prioritization     │
└─────────────────────────────────┘
```

---

## Implemented Algorithms

| Module | Algorithm | Key Paper |
|--------|-----------|-----------|
| Preprocessing | ComBat Batch Correction | Johnson et al. (2007), *Biostatistics* |
| Preprocessing | KNN Imputation | Troyanskaya et al. (2001), *Bioinformatics* |
| Normalization | TPM / RPKM | Mortazavi et al. (2008), *Nature Methods* |
| Regularization | Lasso (L1) | Tibshirani (1996), *JRSS-B* |
| Regularization | Ridge (L2) | Hoerl & Kennard (1970), *Technometrics* |
| Regularization | Elastic Net | Zou & Hastie (2005), *JRSS-B* |
| Classification | SVM + RFE | Guyon et al. (2002), *Machine Learning* |
| Classification | Random Forest | Breiman (2001), *Machine Learning* |
| Deep Learning | DeepSurv | Katzman et al. (2018), *BMC Med Res Methodol* |
| Interpretability | SHAP Values | Lundberg & Lee (2017), *NeurIPS* |
| Survival | Cox PH Model | Cox (1972), *JRSS-B* |
| Dimensionality | t-SNE | van der Maaten & Hinton (2008), *JMLR* |

---

## Case Study: Breast Cancer Molecular Subtyping (TCGA-BRCA)

**Task**: Classify 1,000 breast cancer samples into 5 PAM50 molecular subtypes (Luminal A, Luminal B, HER2-enriched, Basal-like, Normal-like) using RNA-seq gene expression profiles.

**Data**: TCGA-BRCA cohort • 20,531 coding genes → reduced to 50 PAM50 genes • 70/30 stratified split

| Model | Overall Accuracy | Macro-F1 | Basal-like Recall |
|-------|-----------------|----------|-------------------|
| Linear SVM | ~89% | 0.87 | >95% |
| SVM (RBF) | ~90% | 0.88 | >95% |
| Random Forest | **~92%** | **0.91** | **>97%** |
| Elastic Net | ~87% | 0.85 | >93% |

> Key finding: Confusion primarily occurs between Luminal A and Luminal B subtypes, suggesting PAM50 genes alone are insufficient — Ki-67 proliferation index integration is recommended.

---

## Why This Project Stands Out

- **Bridges two methodological cultures**: combines hypothesis-driven biostatistics with prediction-driven machine learning in one reproducible codebase.

- **Portfolio-grade technical depth**: spans preprocessing, feature engineering, multi-class classification, censored survival modeling, and XAI-based interpretation instead of stopping at a single model demo.

- **Clinically relevant framing**: links model outputs to use cases such as cancer subtyping, prognosis estimation, risk stratification, and biomarker prioritization.

- **Resume-ready scope**: demonstrates Python engineering, statistical modeling, PyTorch implementation, biomedical domain understanding, and end-to-end project packaging.

---

## Resume-ready Positioning

You can describe this project in a resume or interview using lines such as:

- Built an end-to-end bioinformatics machine learning framework integrating RNA-seq preprocessing, regularized modeling, cancer subtype classification, survival analysis, and SHAP-based biomarker interpretation.

- Designed modular Python pipelines for high-dimensional omics data under the p >> n regime, combining statistical inference with predictive modeling to improve robustness and interpretability.

- Implemented Cox PH and DeepSurv survival models, Random Forest and SVM classifiers, and reproducible Jupyter workflows for clinically oriented genomic analysis.

---

## Installation

```bash
git clone https://github.com/Finvoler/Stat-ML-Bioinfo.git
cd Stat-ML-Bioinfo

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

---

## Quick Start

```python
import pandas as pd
from src.preprocessing.normalization import tpm, log2_transform
from src.models.classification import BreastCancerClassifier
from src.features.selection import shap_feature_ranking

# 1. Normalize raw counts
counts = pd.read_csv("data/raw_counts.csv", index_col=0)
gene_lengths = pd.read_csv("data/gene_lengths.csv", index_col=0).squeeze()

expr_tpm = tpm(counts, gene_lengths)
expr_log = log2_transform(expr_tpm)

# 2. Train Random Forest classifier
clf = BreastCancerClassifier(model_type="random_forest", n_estimators=500)
clf.fit(X_train, y_train)
print(f"Test Accuracy: {clf.score(X_test, y_test):.3f}")

# 3. Identify top biomarkers via SHAP
top_genes = shap_feature_ranking(clf.model, X_test, top_k=20)
print("Top 20 predictive genes:", top_genes)
```

```python
from sklearn.preprocessing import StandardScaler

from src.models.survival import DeepSurv, DeepSurvTrainer
from src.visualization.survival_plots import plot_kaplan_meier, risk_stratify

# 4. Train DeepSurv for personalized prognosis
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

network = DeepSurv(in_features=50, hidden_layers=[128, 64, 32])
trainer = DeepSurvTrainer(model=network, epochs=200)
trainer.fit(X_train_scaled, durations_train, events_train)

# 5. Risk stratification and KM curves
risk_scores = trainer.predict_risk(X_test_scaled)
high_risk, low_risk = risk_stratify(risk_scores, durations_test, events_test)
plot_kaplan_meier(high_risk, low_risk, title="DeepSurv Risk Stratification")
```

---

## Jupyter Notebooks

| Notebook | Description |
|----------|-------------|
| [01_data_preprocessing](notebooks/01_data_preprocessing.ipynb) | RNA-seq normalization, batch correction, PCA/t-SNE visualization |
| [02_classification_models](notebooks/02_classification_models.ipynb) | SVM, Random Forest training, ROC curves, confusion matrices |
| [03_survival_analysis](notebooks/03_survival_analysis.ipynb) | Cox PH vs DeepSurv comparison, C-index, Kaplan-Meier curves |
| [04_model_interpretability](notebooks/04_model_interpretability.ipynb) | SHAP summary plots, biomarker prioritization, pathway enrichment |

---

## Mathematical Highlights

The project implements key results from statistical learning theory:

**Lasso (L1 Regularization)** — sparse biomarker selection:

$$
\hat{\beta}^{\mathrm{Lasso}} = \arg\min_{\beta} \bigl\lbrace \lVert Y - X\beta \rVert_2^2 + \lambda \lVert \beta \rVert_1 \bigr\rbrace
$$

**DeepSurv Loss** — negative log partial likelihood with L2 regularization:

$$
\mathcal{L}(\theta) = -\sum_{i \in \mathcal{E}} \Bigl( f_\theta(x_i) - \log \sum_{j \in \mathcal{R}(t_i)} \exp\bigl(f_\theta(x_j)\bigr) \Bigr) + \lambda \lVert \theta \rVert_2^2
$$

**SHAP Shapley Value** — fair feature attribution via cooperative game theory:

$$
\phi_j = \sum_{S \subseteq F \setminus \lbrace j \rbrace} \frac{|S|!\,(|F|-|S|-1)!}{|F|!} \bigl( f(S \cup \lbrace j \rbrace) - f(S) \bigr)
$$

---

## References

1. Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *JRSS-B*, 58(1), 267–288.
2. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
3. Katzman, J. L., et al. (2018). DeepSurv. *BMC Medical Research Methodology*, 18(1), 1–12.
4. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*, 30.
5. Jumper, J., et al. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature*, 596, 583–589.
6. Johnson, W. E., et al. (2007). Adjusting batch effects using empirical Bayes methods. *Biostatistics*, 8(1), 118–127.

---

## Citation

```bibtex
@misc{chen2025inferencetopredict,
  title   = {From Inference to Prediction: Integrating Statistical Learning and
             Machine Learning for High-Dimensional Bioinformatics Analysis},
  author  = {Chen, Qi},
  year    = {2025},
  url     = {https://github.com/Finvoler/Stat-ML-Bioinfo}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
