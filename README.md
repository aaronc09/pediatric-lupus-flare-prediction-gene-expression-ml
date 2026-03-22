# Pediatric Lupus Flare Prediction using Gene Expression and Machine Learning
Predicting pre-flare states in pediatric SLE patients using whole-blood gene expression data (GSE65391) and machine learning models with SHAP-based interpretation.

## Overview
Systemic lupus erythematosus (SLE) is a chronic autoimmune disease characterized by unpredictable flares. Early prediction of flares could improve clinical management and patient outcomes.
This project develops a machine learning pipeline to predict pre-flare states in pediatric lupus patients using gene expression data from the GSE65391 dataset. The pipeline includes preprocessing, feature selection, nested cross-validation, and model interpretation using SHAP.

## Repository Structure
repo/
├── scripts/
│   ├── build_lupus_dataset.py        # Preprocessing pipeline
│   └── lupus_nested_cv_pipeline.py  # Machine learning pipeline
│
├── notebooks/
│   ├── 01_build_lupus_dataset_colab.ipynb   # Run preprocessing (Colab)
│   └── 02_run_ml_pipeline_colab.ipynb       # Run ML pipeline (Colab)
│
├── outputs/
│   ├── figures/   # Model performance plots, SHAP visualizations
│   └── tables/    # Metrics, feature importance tables
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Run in Google Colab
### Step 1: Preprocessing
[Open in Colab](https://colab.research.google.com/github/aaronc09/pediatric-lupus-flare-prediction-gene-expression-ml/blob/main/notebooks/01_build_lupus_dataset_colab.ipynb)
### Step 2: Machine Learning Pipeline
[Open in Colab](https://colab.research.google.com/github/aaronc09/pediatric-lupus-flare-prediction-gene-expression-ml/blob/main/notebooks/02_run_ml_pipeline_colab.ipynb)

## Dataset
- Source: GEO (GSE65391)
- Platform: Illumina HumanHT-12 v4.0
- Type: Whole-blood gene expression
- The dataset is publicly available via NCBI GEO.

## Methods
- Longitudinal pairing of patient visits
- Pre-flare labeling based on SLEDAI increase
- Feature selection:
  - Welch’s t-test
  - correlation filtering
  - L1-based selection
- Nested cross-validation with subject-level grouping
- Models:
  - Logistic Regression
  - SVM
  - Random Forest
  - XGBoost
- Evaluation metrics:
  - PR-AUC (primary)
  - ROC-AUC
  - F1 score
- Model interpretation using SHAP

## Outputs
The pipeline generates:
- figures/ → PR curves, confusion matrices, SHAP plots  
- tables/ → metrics and model results  
- shap/ → gene importance  
- logs/ → run metadata

## Results
The models demonstrated weak to moderate predictive signals.

## Acknowledgments
