# Methodology

## Project Positioning

This project operationalizes a core bioinformatics transition:

- Classical biostatistics focuses on inference, significance testing, confidence, and mechanism-oriented interpretation.
- Modern machine learning focuses on prediction, generalization, automation, and scalability.
- Real biomedical systems require both.

The repository is therefore designed as an end-to-end analytical framework that starts with noisy, high-dimensional omics measurements and ends with clinically meaningful outputs: subtype prediction, survival risk estimation, and interpretable biomarker prioritization.

## Core Pain Points

### 1. High-dimensional, small-sample regime

In omics research, the number of measured features often exceeds the number of patients by orders of magnitude:

- p > 20,000 genes
- n < 500 clinical samples
- p >> n leads to singular design matrices, unstable estimates, and severe overfitting risk

### 2. Heterogeneous and noisy data acquisition

Biological datasets are often integrated across:

- sequencing runs
- laboratories
- platforms
- patient cohorts

This introduces missingness, scale mismatch, and batch effects that can dominate downstream models if not corrected.

### 3. Prediction-interpretation tension

High-performing models such as Random Forests or neural networks can improve classification and prognosis, but they are hard to trust in a biomedical context unless their decisions can be mapped back to genes, pathways, or mechanisms.

## Innovation Framing

This repository is built around three innovation themes.

### Statistical rigor + predictive performance

Instead of treating statistics and machine learning as competing paradigms, the project combines them into a single workflow:

- statistical preprocessing for robust inputs
- regularization for stable high-dimensional learning
- machine learning for classification and survival prediction
- interpretability for mechanism-oriented post hoc explanation

### From feature significance to decision relevance

Traditional analysis often stops at p-values and fold changes. This project extends beyond significance by quantifying which genes actually influence model decisions through:

- Lasso-based sparsity
- Random Forest importance
- SHAP-based attribution

### Translation-oriented modeling

The repository is structured to resemble a research-to-deployment prototype rather than a one-off notebook:

- modular source code
- reproducible notebooks
- case-study framing
- portable project narrative suitable for GitHub portfolio and resume use

## Pipeline Overview

### Step 1. Data preprocessing

Modules in src/preprocessing handle:

- KNN-based missing value imputation
- RPKM and TPM normalization
- log2 variance stabilization
- ComBat-style batch correction

These steps reduce technical artifacts before feature learning begins.

### Step 2. Dimensionality reduction and filtering

Modules in src/features handle:

- variance filtering
- Pearson-correlation filtering
- PCA
- t-SNE
- differential expression filtering

This stage improves signal-to-noise ratio and supports exploratory analysis.

### Step 3. Predictive modeling

Modules in src/models implement:

- Linear SVM and RBF SVM for subtype classification
- Random Forest for non-linear classification and gene importance
- Lasso, Ridge, and Elastic Net for regularized high-dimensional regression
- Cox PH and DeepSurv for censored survival prediction

### Step 4. Interpretability and biological meaning

Modules in src/features and src/visualization support:

- SHAP-based feature ranking
- clustered heatmaps
- Kaplan-Meier curves
- C-index comparison

This stage closes the loop between model output and biological reasoning.

## Mathematical Rationale

### Ridge regression

Ridge stabilizes estimates when multicollinearity is high:

beta_hat = argmin ||Y - X beta||_2^2 + lambda ||beta||_2^2

Useful when correlated genes should remain in the model but coefficient variance must be controlled.

### Lasso regression

Lasso introduces sparsity:

beta_hat = argmin ||Y - X beta||_2^2 + lambda ||beta||_1

This is especially valuable for biomarker discovery because many coefficients are driven exactly to zero.

### Elastic Net

Elastic Net mixes sparsity and grouping behavior:

Loss = ||Y - X beta||_2^2 + lambda_1 ||beta||_1 + lambda_2 ||beta||_2^2

This is more stable than pure Lasso when genes are strongly co-expressed.

### Cox proportional hazards model

Cox models risk as:

h(t|x) = h0(t) exp(beta^T x)

It is the classical standard for censored outcomes but assumes a linear predictor.

### DeepSurv

DeepSurv replaces the linear term with a neural network:

h(t|x) = h0(t) exp(f_theta(x))

This enables learning non-linear interactions that often exist in gene regulation and disease progression.

## Practical Research Value

This project is suitable for:

- computational biology portfolio use
- graduate application showcase
- resume bullet conversion for data science / bioinformatics roles
- teaching demonstration of inference vs prediction in life sciences

## Resume-ready Summary

Suggested concise positioning:

Designed and implemented an end-to-end bioinformatics machine learning framework integrating statistical inference, regularization, ensemble learning, survival modeling, and SHAP-based interpretability for high-dimensional omics analysis; built reproducible pipelines for RNA-seq preprocessing, cancer subtype classification, prognostic modeling, and biomarker prioritization.

Suggested impact-oriented positioning:

Bridged classical biostatistics and modern machine learning in a portfolio-grade bioinformatics project, translating high-dimensional genomic data into clinically interpretable predictions for diagnosis and prognosis.
