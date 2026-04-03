# Pediatric Systemic Lupus Erythematosus Flare Prediction Using Gene-Expression Data and Machine Learning

- Leakage-safe machine learning pipeline to predict near-term (≤90 days) lupus flares in pediatric systemic lupus erythematosus (SLE) using gene-expression data (GSE65391).
- This study presents original primary research based on independent analysis of publicly available gene-expression data.

## Key Contribution
- First implementation of a leakage-safe, subject-level nested cross-validation pipeline for pediatric lupus flare prediction using gene-expression data, with comparison against clinical and baseline models.

## Dataset
- Source: GEO (GSE65391)  
- Platform: Illumina HumanHT-12 v4.0 (GPL10558)  
- Type: Whole-blood gene expression  
- Publicly available via NCBI GEO  

## Overview
- Dataset: GSE65391 (Illumina GPL10558)  
- Samples: ~440 paired visits (~104 patients)  
- Task: Predict pre-flare visits (SLEDAI increase ≥4 at next visit within ≤90 days)  
- Class prevalence: ~17%  

## Approach
- Subject-level nested cross-validation (5×5)  
- StratifiedGroupKFold (prevents subject leakage)  
- Feature selection strictly performed within training folds only (prevents data leakage)  
  - Variance threshold → ANOVA → model-based selection  
- Models:
  - Logistic Regression (L2)  
  - XGBoost  
- Comparator models:
  - Prevalence baseline  
  - SLEDAI-only model (clinical comparator)  
- Hyperparameter tuning:
  - Optuna (inner CV, PR-AUC objective)  

## Modeling Pipeline

Preprocessed Data  

- Nested Cross-Validation (Outer 5-fold / Inner 5-fold)
  - Gene-Expression Models:
    - Feature Selection (within folds)
    - Hyperparameter Optimization (Optuna)
    - Model Training (Logistic Regression, XGBoost)
    - Threshold Selection (optimize F1)
  - Comparator Models:
    - Prevalence baseline
    - SLEDAI-only model
  - Outer-Fold Evaluation (held-out subjects)
    - Primary: PR-AUC  
    - Secondary: ROC-AUC, F1, Recall, Precision, Specificity, Balanced Accuracy, Brier Score  

- Cross-Validated Model Comparison  
- SHAP Analysis (gene-expression models only)  
- Permutation-Based Sanity Checks  

## Results (Cross-Validated)

### Primary Metric (PR-AUC)
- Logistic Regression: 0.21 ± 0.06  
- XGBoost: 0.25 ± 0.09  
- SLEDAI-only: 0.25 ± 0.07  
- Baseline (prevalence): 0.16 ± 0.04  

### Key Observations
- XGBoost and SLEDAI-only achieve similar PR-AUC (~0.25)  
- Logistic regression shows lower performance (~0.21)  
- All models only modestly outperform baseline (~0.17 prevalence)  

### Secondary Metrics (selected)
- Best ROC-AUC: SLEDAI-only (~0.65 ± 0.11)  
- Best F1 score: SLEDAI-only (~0.32 ± 0.11)  
- Best Brier score (calibration): Baseline (~0.14 ± 0.02)  

### Interpretation
- Gene-expression models provide limited improvement over clinical signal  
- SLEDAI-only performs comparably or better across several metrics  
- Predictive signal is modest and highly variable across folds  

→ Gene-expression data alone is insufficient for strong flare prediction in this setting  

## Run in Google Colab

Step 1: Preprocessing  
https://colab.research.google.com/github/aaronc09/pediatric-lupus-flare-prediction-gene-expression-ml/blob/main/notebooks/01_build_lupus_dataset_colab.ipynb  

Step 2: Machine Learning Pipeline  
https://colab.research.google.com/github/aaronc09/pediatric-lupus-flare-prediction-gene-expression-ml/blob/main/notebooks/02_run_ml_pipeline_colab.ipynb  

Run Step 1 first, then Step 2.

## Environment
- Python: 3.12.x (tested on 3.12.13, Google Colab)  
- numpy: 2.0.2  
- pandas: 2.2.2  
- matplotlib: 3.9.2  
- scikit-learn: 1.5.2  
- xgboost: 2.1.1  
- shap: 0.46.0  
- optuna: 4.0.0  

- All experiments were conducted in Google Colab with fixed random seeds where applicable.

## Notes
- Strict leakage prevention via:
  - subject-level splits  
  - fold-contained feature selection  
- PR-AUC used due to class imbalance (~17%)  
- Gene-expression data alone shows limited predictive power  
- Future work should incorporate:
  - multimodal data (clinical + molecular)  
  - patient subtyping / stratification  

## Structure
- docs/ — pipeline diagrams and summary files  
- scripts/ — preprocessing and ML pipeline  
- notebooks/ — Colab notebooks  
- outputs/ — figures, tables, SHAP, logs  

## Citation
Choi A. *Pediatric Systemic Lupus Erythematosus Flare Prediction Using Gene-Expression Data and Machine Learning*. 2026.

## License
MIT License
