# Machine Learning-Based Prediction of Pediatric Lupus Flares Using Gene-Expression Data

Leakage-safe machine learning pipeline to predict near-term (≤90 days) lupus flares in pediatric systemic lupus erythematosus (SLE) using gene-expression data (GSE65391).

## Dataset
- Source: GEO (GSE65391)  
- Platform: Illumina HumanHT-12 v4.0 (GPL10558)  
- Type: Whole-blood gene expression  
- The dataset is publicly available via NCBI GEO  

## Overview
- Dataset: GSE65391 (Illumina GPL10558)  
- Samples: ~440 paired visits (~104 patients)  
- Task: Predict pre-flare (SLEDAI increase ≥4 within 90 days)  
- Class prevalence: ~17%  

## Approach
- Subject-level nested cross-validation (5×5)  
- StratifiedGroupKFold (prevents data leakage)  
- Feature selection within folds (variance → ANOVA → model-based)  
- Models: Logistic Regression (L2), XGBoost  
- Hyperparameter tuning: Optuna (PR-AUC objective)  

## Modeling Pipeline

Preprocessed Data  

- Nested Cross-Validation (Outer 5-fold / Inner 5-fold)
  - Feature Selection
    - Variance Threshold  
    - ANOVA (SelectKBest)  
    - Model-based selection (L1 / XGBoost)  
  - Hyperparameter Optimization (Optuna, inner CV)  
  - Model Training
    - Logistic Regression (L2)  
    - XGBoost  
  - Threshold Selection (optimize F1)  
  - Outer-Fold Evaluation
    - Primary: PR-AUC  
    - Secondary: ROC-AUC, F1, Recall, Precision, Specificity, Balanced Accuracy, Brier Score  

- Model Performance (Cross-Validated)  
- SHAP Analysis  
- Permutation-Based Sanity Checks  

## Run in Google Colab

Step 1: Preprocessing  
https://colab.research.google.com/github/aaronc09/pediatric-lupus-flare-prediction-gene-expression-ml/blob/main/notebooks/01_build_lupus_dataset_colab.ipynb  

Step 2: Machine Learning Pipeline  
https://colab.research.google.com/github/aaronc09/pediatric-lupus-flare-prediction-gene-expression-ml/blob/main/notebooks/02_run_ml_pipeline_colab.ipynb  

Run Step 1 first to generate the dataset, then Step 2 to train and evaluate models.

## Results
- Logistic Regression PR-AUC: ~0.21  
- XGBoost PR-AUC: ~0.25  
- Baseline: ~0.17  

→ Modest predictive signal with high variability across folds  

## Notes
- Nested CV + grouped splits used to prevent data leakage  
- Gene-expression alone shows limited predictive power  
- Likely requires multimodal data and patient subtyping  

## Structure
- docs/ — pipeline diagrams and summary files  
- scripts/ — preprocessing and ML pipeline  
- notebooks/ — Colab notebooks  
- outputs/ — results (figures, tables, SHAP, logs)  

## Citation
Choi A. *Machine Learning-Based Prediction of Pediatric Lupus Flares Using Gene Expression Data*. 2026.

## License
MIT License
