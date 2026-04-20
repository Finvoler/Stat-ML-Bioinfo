# Data Access Guide

This repository does not ship raw biomedical datasets because:

- source files are large
- public repositories impose redistribution constraints
- patient-derived data may require controlled access or attribution

## Recommended Public Data Sources

### 1. TCGA-BRCA

Use for:

- breast cancer subtype classification
- survival modeling
- PAM50-style expression analysis

Access:

- Genomic Data Commons portal
- https://portal.gdc.cancer.gov/

Suggested assets:

- RNA-seq gene expression quantification
- clinical follow-up metadata
- subtype labels or PAM50 annotations when available

### 2. GEO

Use for:

- expression microarray studies
- smaller cohort replication tasks
- batch-effect correction demonstrations

Access:

- https://www.ncbi.nlm.nih.gov/geo/

Suggested search terms:

- breast cancer expression profiling
- PAM50
- RNA-seq survival
- cancer subtype microarray

## Expected Local Layout

Place datasets using the following structure:

data/
├── raw/
│   ├── counts/
│   ├── metadata/
│   └── clinical/
└── processed/
    ├── normalized/
    ├── corrected/
    └── features/

## Suggested File Formats

- raw counts: CSV or TSV with genes as rows and samples as columns
- gene lengths: CSV with gene identifier and length in base pairs
- sample metadata: CSV with sample_id, batch, condition, subtype
- survival metadata: CSV with sample_id, duration, event

## Minimal Example Schema

### counts matrix

- row index: gene_id
- columns: sample_id
- values: integer raw counts

### metadata table

- sample_id
- condition
- batch
- subtype
- duration
- event

## Notes

- Apply TPM normalization before cross-sample classification tasks.
- Apply log2 transform after normalization.
- Use ComBat only when batch labels are known and biological covariates should be preserved.
- Never perform feature selection on the full dataset before train/test split. That would introduce data leakage.
