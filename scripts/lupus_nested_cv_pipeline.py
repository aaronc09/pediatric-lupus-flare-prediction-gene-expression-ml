# =========================================================
# Author: Aaron Choi
# Project: Pediatric Lupus Flare Prediction (GSE65391)
# File: lupus_nested_cv_pipeline.py
#
# Description:
#   Nested cross-validation machine learning pipeline for
#   predicting pre-flare states in pediatric SLE subjects
#   using whole-blood gene-expression data.
#
# Key methods:
#   - Subject-level grouped nested cross-validation
#   - Leakage-safe feature selection within inner folds
#   - Optuna hyperparameter optimization using PR-AUC
#   - SHAP-based gene importance and direction inference
#   - Optional permutation-label sanity checks
#
# Input:
#   - Preprocessed feature table: lupus_final_df.pkl
#
# Outputs:
#   - Figures, tables, SHAP summaries, and logs
#
# Notes:
#   - Designed for reproducible execution locally and in Google Colab
#   - Paths are configurable via LUPUS_DATA_PATH and LUPUS_OUTPUT_DIR
# =========================================================


from __future__ import annotations

import os
import re
import io
import sys
import json
import random
import logging
import platform
import warnings
import itertools
import textwrap
import requests
import datetime
from collections import Counter, defaultdict

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore", category=UserWarning,   module="shap")
warnings.filterwarnings("ignore", category=FutureWarning, module="shap")
warnings.filterwarnings("ignore", category=UserWarning,   module="xgboost")
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in divide",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
    category=FutureWarning,
)

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["savefig.dpi"] = 600

sns.set_theme(style="white", font_scale=1.4)
plt.rcParams.update({
    "font.size":        20,
    "axes.titlesize":   26,
    "axes.labelsize":   22,
    "xtick.labelsize":  20,
    "ytick.labelsize":  20,
    "legend.fontsize":  16,
    "figure.titlesize": 32,
})

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, SelectKBest, f_classif
from sklearn.metrics import (
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    brier_score_loss,
)

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None

import xgboost as xgb
import shap
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# Logging
# =============================================================================
logger = logging.getLogger("lupus_nested_cv_pipeline")
logger.setLevel(logging.INFO)
logger.handlers.clear()
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)
logger.propagate = False

logger.info("scikit-learn : %s", sklearn.__version__)
logger.info("xgboost      : %s", xgb.__version__)
logger.info("shap         : %s", shap.__version__)
logger.info("optuna       : %s", optuna.__version__)
logger.info("python       : %s", sys.version)
logger.info("platform     : %s", platform.platform())

# =============================================================================
# Configuration
# =============================================================================
GLOBAL_SEED = 42

DEFAULT_DATA_PATH  = "outputs/lupus_final_df.pkl"
DEFAULT_OUTPUT_DIR = "outputs/ml_outputs"

DATA_PATH  = os.environ.get("LUPUS_DATA_PATH", DEFAULT_DATA_PATH)
OUTPUT_DIR = os.environ.get("LUPUS_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)

N_OUTER_FOLDS     = 5
N_INNER_FOLDS     = 5
OUTER_MASTER_SEED = GLOBAL_SEED

VAR_THRESH  = 1e-5
PREFILTER_K = 300
TOP_K             = 20
TOP_K_CORRECT_OOF = 10

N_TRIALS_LR  = 40
N_TRIALS_XGB = 150

XGB_NJOBS = 1
XGB_TREE  = "hist"

SHAP_BG_MAX   = 200
SHAP_EVAL_MAX = 400

RUN_PERMUTATION_CHECK     = True
N_PERMUTATION_OUTER_FOLDS = N_OUTER_FOLDS
PERM_N_TRIALS_LR          = 20
PERM_N_TRIALS_XGB         = 25

CLASS_NAMES_CODE    = ["non_pre_flare", "pre_flare"]
CLASS_NAMES_DISPLAY = ["Non-pre-flare", "Pre-flare"]

NEG_CLASS_CODE    = CLASS_NAMES_CODE[0]
POS_CLASS_CODE    = CLASS_NAMES_CODE[1]
NEG_CLASS_DISPLAY = CLASS_NAMES_DISPLAY[0]
POS_CLASS_DISPLAY = CLASS_NAMES_DISPLAY[1]

POSITIVE_CLASS_NAME = POS_CLASS_DISPLAY
NEGATIVE_CLASS_NAME = NEG_CLASS_DISPLAY

MODEL_NAMES = ["LR_L2", "XGB"]
MODEL_DISPLAY_NAMES = {
    "LR_L2": "Logistic Regression L2",
    "XGB":   "XGBoost",
}

GPL10558_FULL_URL = (
    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
    "?acc=GPL10558&targ=self&form=text&view=full"
)

REQUIRE_STRATIFIED_GROUP_KFOLD = True
MIN_OOF_COVERAGE_WARN          = 0.80
MIN_FALLBACK_FEATURES          = 10

TABLE_BODY_FONTSIZE   = 24
TABLE_HEADER_FONTSIZE = 24
TABLE_TITLE_FONTSIZE  = 44

FIG_TITLE_SIZE     = 40
SUBPLOT_TITLE_SIZE = 32
AXIS_LABEL_SIZE    = 22
TICK_LABEL_SIZE    = 20
VALUE_LABEL_SIZE   = 18
LEGEND_SIZE        = 16


def _make_output_dirs(base: str) -> dict:
    dirs = {
        "figures": os.path.join(base, "figures"),
        "tables":  os.path.join(base, "tables"),
        "shap":    os.path.join(base, "shap"),
        "logs":    os.path.join(base, "logs"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


# =============================================================================
# Utility functions
# =============================================================================
def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


seed_everything(GLOBAL_SEED)


def get_model_display_name(model_name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def round_numeric_df(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].round(decimals)
    return out


def clean_gene_name(x: object) -> object:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s in ("", "nan", "NA", "None"):
        return np.nan
    s = re.split(r"[\|;,/]", s)[0].strip()
    return s if s else np.nan


def make_gene_label(probe_id: object, gene_name_map: dict) -> str:
    probe_id  = str(probe_id).strip()
    gene_name = gene_name_map.get(probe_id, probe_id)
    if pd.isna(gene_name) or str(gene_name).strip() == "" or str(gene_name).strip() == probe_id:
        return probe_id
    return f"{probe_id} ({gene_name})"


# =============================================================================
# Model construction helpers
# =============================================================================

def build_lr_selector(C: float, seed: int) -> LogisticRegression:
    """L1 logistic regression for the feature selection stage."""
    return LogisticRegression(
        penalty="l1", solver="liblinear", C=C,
        max_iter=2000, random_state=seed, class_weight="balanced",
    )


def build_lr_clf(C: float, seed: int) -> LogisticRegression:
    """L2 logistic regression — final classifier."""
    return LogisticRegression(
        penalty="l2", solver="lbfgs", C=C,
        max_iter=5000, random_state=seed, class_weight="balanced",
    )


def build_xgb_selector(seed: int, spw: float) -> xgb.XGBClassifier:
    """Lightweight XGBoost for feature ranking prior to full tuning."""
    return xgb.XGBClassifier(
        n_estimators=60, max_depth=2, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
        min_child_weight=5.0, scale_pos_weight=spw,
        tree_method=XGB_TREE, deterministic_histogram=True,
        eval_metric="aucpr", random_state=seed,
        n_jobs=XGB_NJOBS, verbosity=0,
    )


def build_xgb_clf(params: dict, seed: int, spw: float) -> xgb.XGBClassifier:
    """Build XGBoost model using tuned parameters."""
    return xgb.XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        min_child_weight=params["min_child_weight"],
        reg_lambda=params["reg_lambda"],
        reg_alpha=params["reg_alpha"],
        gamma=params["gamma"],
        max_delta_step=params["max_delta_step"],
        scale_pos_weight=spw,
        tree_method=XGB_TREE, deterministic_histogram=True,
        eval_metric="aucpr", random_state=seed,
        n_jobs=XGB_NJOBS, verbosity=0,
    )


# =============================================================================
# Hyperparameter search space
# =============================================================================
def get_search_space_tables():
    """Return DataFrames describing the Optuna search space for LR_L2 and XGB."""
    lr_table = pd.DataFrame({
        "Parameter":    ["lasso_c", "l2_c"],
        "Search space": [
            "float, log scale, [1e-5, 2e-1]",
            "float, log scale, [1e-6, 1e-1]",
        ]
    })
    xgb_table = pd.DataFrame({
        "Parameter": [
            "xgb_sel_topk", "n_estimators", "max_depth", "learning_rate",
            "subsample", "colsample_bytree", "min_child_weight", "gamma",
            "reg_lambda", "reg_alpha", "max_delta_step",
        ],
        "Search space": [
            "int, [10, 60], step=10",
            "int, [200, 1500], step=50",
            "int, [2, 3]",
            "float, log scale, [0.01, 0.20]",
            "float, [0.6, 0.8]",
            "float, [0.5, 0.8]",
            "float, [20.0, 40.0]",
            "float, [1.0, 5.0]",
            "float, log scale, [10.0, 200.0]",
            "float, log scale, [1e-3, 10.0]",
            "int, [0, 8]",
        ]
    })
    return {"LR_L2": lr_table, "XGB": xgb_table}


# =============================================================================
# Data validation and loading
# =============================================================================
def validate_data(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> None:
    errors = []

    if not (len(X) == len(y) == len(groups)):
        errors.append(f"Length mismatch: X={len(X)}, y={len(y)}, groups={len(groups)}.")

    if not (X.index.equals(y.index) and X.index.equals(groups.index)):
        errors.append(
            "Index mismatch between X, y, and/or groups. "
            "Re-align indices before running the pipeline."
        )

    unique_labels = set(y.dropna().unique())
    if unique_labels != {0, 1}:
        errors.append(f"Labels must be exactly {{0, 1}}. Found: {sorted(unique_labels)}.")

    if y.isna().any():
        errors.append(f"y contains {y.isna().sum()} NaN value(s).")

    nan_cols = int(X.isna().any(axis=0).sum())
    inf_cols = int(np.isinf(X.values).any(axis=0).sum())
    if nan_cols > 0:
        errors.append(f"X contains NaN values in {nan_cols} column(s).")
    if inf_cols > 0:
        errors.append(f"X contains Inf values in {inf_cols} column(s).")

    n_subjects   = groups.nunique()
    pos_mask     = y == 1
    pos_subjects = groups[pos_mask].nunique()
    neg_subjects = groups[~pos_mask].nunique()

    if n_subjects < N_OUTER_FOLDS:
        errors.append(
            f"Only {n_subjects} unique subjects — need at least {N_OUTER_FOLDS} "
            f"for {N_OUTER_FOLDS}-fold grouped CV."
        )
    if pos_subjects < N_OUTER_FOLDS:
        errors.append(
            f"Only {pos_subjects} unique {POS_CLASS_DISPLAY} subjects. "
            f"Need at least {N_OUTER_FOLDS} for StratifiedGroupKFold."
        )
    if neg_subjects < N_OUTER_FOLDS:
        errors.append(
            f"Only {neg_subjects} unique {NEG_CLASS_DISPLAY} subjects. "
            f"Need at least {N_OUTER_FOLDS} for StratifiedGroupKFold."
        )

    if X.index.duplicated().any():
        n_dup = int(X.index.duplicated().sum())
        errors.append(
            f"X has {n_dup} duplicated row index value(s). "
            f"This will corrupt index-based fingerprinting."
        )

    if errors:
        msg = "\n".join(f"  • {e}" for e in errors)
        raise ValueError(f"Data validation failed:\n{msg}")

    logger.info(
        "Data validation passed: %d samples | %d features | %d subjects | "
        "%d pre-flare subjects / %d non-pre-flare subjects",
        len(X), X.shape[1], n_subjects, pos_subjects, neg_subjects,
    )


def load_data(filepath: str):
    """
    Load the processed lupus DataFrame from a pickle file.

    Returns X (features), y (labels), groups (subject IDs), gene_cols (feature names),
    and sample_ids (original row index values, e.g. GSM IDs, before any reset).
    Raises FileNotFoundError if the file is missing, or ValueError on schema problems.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Could not find data file:\n  {filepath}\n\n"
            f"Common fixes:\n"
            f"  • Confirm the file exists at the path above.\n"
            f"  • Override via env var: os.environ['LUPUS_DATA_PATH'] = '/your/path.pkl'\n"
            f"  • Pass directly: run_pipeline('/your/path.pkl')\n"
        )

    df = pd.read_pickle(filepath)
    logger.info("Data loaded from: %s", filepath)
    logger.info("Columns: %d total", df.shape[1])

    if "preflare_bool" not in df.columns or "subject" not in df.columns:
        raise ValueError("Missing required columns: need 'preflare_bool' and 'subject'")

    gene_cols = [c for c in df.columns if str(c).startswith("ILMN_")]
    if len(gene_cols) == 0:
        num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
        gene_cols = [c for c in num_cols if c not in ("preflare_bool",)]
        logger.warning("No ILMN_* columns found — falling back to numeric columns.")

    X      = df[gene_cols].astype(np.float32)
    y      = df["preflare_bool"].astype(int)
    groups = df["subject"].astype(str)

    sample_ids = X.index.astype(str).tolist()

    if (not pd.api.types.is_integer_dtype(X.index)) or X.index.duplicated().any():
        logger.info("Resetting row index to 0..N-1 for compatibility.")
        X      = X.reset_index(drop=True)
        y      = y.reset_index(drop=True)
        groups = groups.reset_index(drop=True)

    validate_data(X, y, groups)

    logger.info("Loaded  : %d samples x %d features", df.shape[0], len(gene_cols))
    class_counts = y.value_counts().to_dict()
    logger.info(
        "Class counts: %s=%d, %s=%d",
        POS_CLASS_DISPLAY, class_counts.get(1, 0),
        NEG_CLASS_DISPLAY, class_counts.get(0, 0),
    )
    logger.info("Subjects: %d unique", groups.nunique())

    return X, y, groups, gene_cols, sample_ids


# =============================================================================
# Gene annotation
# =============================================================================
def download_gpl_annotation(save_csv_path: str):
    """
    Download GPL10558 probe-to-gene mapping from GEO and save as CSV.
    """
    if os.path.exists(save_csv_path):
        logger.info("Annotation file already exists: %s", save_csv_path)
        return save_csv_path

    logger.info("Downloading GPL10558 annotation from GEO (one-time setup)...")
    try:
        resp = requests.get(GPL10558_FULL_URL, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Failed to download GPL10558 annotation from GEO.\n"
            f"  URL: {GPL10558_FULL_URL}\n  Error: {e}"
        ) from e

    lines       = resp.text.splitlines()
    table_start = next((i for i, l in enumerate(lines) if l.startswith("ID\t")), None)
    if table_start is None:
        raise RuntimeError("Could not find GPL platform table in downloaded text.")

    table_end = next(
        (j for j in range(table_start + 1, len(lines)) if lines[j].startswith("!platform_table_end")),
        len(lines),
    )

    gpl_df     = pd.read_csv(io.StringIO("\n".join(lines[table_start:table_end])), sep="\t", low_memory=False)
    cols_lower = {c.lower(): c for c in gpl_df.columns}

    probe_col = next((cols_lower[c] for c in ["id", "probe_id", "probeid", "ilmn_id", "ilmnid", "array_address_id", "probe"] if c in cols_lower), None)
    gene_col  = next((cols_lower[c] for c in ["symbol", "gene symbol", "gene_symbol", "genesymbol", "gene", "gene_name", "symbol_interpreted"] if c in cols_lower), None)

    if probe_col is None:
        raise RuntimeError(f"Could not identify probe column. Columns: {gpl_df.columns.tolist()}")
    if gene_col is None:
        fallback = [c for c in gpl_df.columns if "symbol" in c.lower() or "gene" in c.lower()]
        if fallback:
            gene_col = fallback[0]
        else:
            raise RuntimeError(f"Could not identify gene-name column. Columns: {gpl_df.columns.tolist()}")

    out = gpl_df[[probe_col, gene_col]].copy()
    out.columns   = ["probe_id", "gene_name"]
    out["probe_id"]  = out["probe_id"].astype(str).str.strip()
    out["gene_name"] = out["gene_name"].map(clean_gene_name)
    out = out.dropna(subset=["probe_id"])
    out = out[out["probe_id"] != ""].drop_duplicates(subset=["probe_id"], keep="first")

    out.to_csv(save_csv_path, index=False)
    logger.info("Saved annotation mapping to: %s", save_csv_path)
    return save_csv_path


def load_gene_name_map(probe_ids: list, annotation_path: str | None) -> dict:
    mapping = {str(p).strip(): str(p).strip() for p in probe_ids}

    if annotation_path is None or not os.path.exists(annotation_path):
        logger.warning("Annotation file missing — showing probe IDs only.")
        return mapping

    ann              = pd.read_csv(annotation_path)
    ann["probe_id"]  = ann["probe_id"].astype(str).str.strip()
    ann["gene_name"] = ann["gene_name"].map(clean_gene_name)
    lookup           = dict(zip(ann["probe_id"], ann["gene_name"]))

    matched = 0
    for p in probe_ids:
        key = str(p).strip()
        if key in lookup and pd.notna(lookup[key]) and str(lookup[key]).strip() != "":
            mapping[key] = str(lookup[key]).strip()
            matched += 1

    logger.info("Gene annotation: matched %d / %d probe IDs.", matched, len(probe_ids))
    return mapping


def dataset_fingerprint(X: pd.DataFrame, groups: pd.Series, y: pd.Series) -> str:
    """SHA-256 hash of index, subject labels, and class labels — used to validate fold cache."""
    import hashlib
    payload = pd.DataFrame({
        "row_index": X.index.astype(str),
        "subject":   groups.astype(str).values,
        "label":     y.astype(int).values,
    })
    header = "|".join(map(str, X.columns))
    txt    = header + "\n" + payload.to_csv(index=False)
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


# =============================================================================
# Cross-validation helpers
# =============================================================================
def make_outer_cv():
    """Instantiate outer CV splitter, preferring StratifiedGroupKFold."""
    if StratifiedGroupKFold is not None:
        return StratifiedGroupKFold(
            n_splits=N_OUTER_FOLDS, shuffle=True, random_state=OUTER_MASTER_SEED,
        ), True

    if REQUIRE_STRATIFIED_GROUP_KFOLD:
        raise ImportError(
            "StratifiedGroupKFold is not available in this scikit-learn environment.\n"
            "Please upgrade scikit-learn (1.2+ recommended) and rerun."
        )

    logger.warning(
        "StratifiedGroupKFold unavailable. Falling back to GroupKFold "
        "(no shuffle, no random_state, weaker class-balance control)."
    )
    return GroupKFold(n_splits=N_OUTER_FOLDS), False


# =============================================================================
# Feature selection
# =============================================================================
def fit_variance_filter(Xtr_np, thresh=VAR_THRESH):
    vt      = VarianceThreshold(threshold=thresh)
    Xtr_v   = vt.fit_transform(Xtr_np)
    return vt, vt.get_support(), Xtr_v


def apply_variance_filter(vt, X_np):
    return vt.transform(X_np)


def fit_univariate_prefilter(Xtr_np, ytr_np, k=PREFILTER_K):
    k       = min(int(k), Xtr_np.shape[1])
    skb     = SelectKBest(score_func=f_classif, k=k)
    Xtr_pf  = skb.fit_transform(Xtr_np, ytr_np)
    return skb, skb.get_support(), Xtr_pf


def apply_univariate_prefilter(skb, X_np):
    return skb.transform(X_np)


def lasso_select_mask(Xtr_pf, ytr, seed, lasso_c, min_features=MIN_FALLBACK_FEATURES):
    """
    L1-based feature selection; falls back to top-min_features by |coef| if L1
    zeroes all coefficients
    """
    scaler = StandardScaler()
    Xtr_s  = scaler.fit_transform(Xtr_pf)

    lasso = build_lr_selector(C=lasso_c, seed=seed)
    lasso.fit(Xtr_s, ytr)

    sel  = SelectFromModel(lasso, prefit=True)
    mask = sel.get_support()

    if mask.sum() == 0:
        coef    = np.abs(lasso.coef_[0])
        k       = min(min_features, Xtr_pf.shape[1])
        top_idx = np.argsort(coef)[::-1][:k]
        mask    = np.zeros(Xtr_pf.shape[1], dtype=bool)
        mask[top_idx] = True

    return scaler, mask


def xgb_select_mask(Xtr_pf, ytr, seed, top_k):
    pos = int((ytr == 1).sum())
    neg = int((ytr == 0).sum())
    spw = neg / max(pos, 1)

    sel_model = build_xgb_selector(seed=seed, spw=spw)
    sel_model.fit(Xtr_pf, ytr)

    imp = getattr(sel_model, "feature_importances_", None)
    if imp is None or len(imp) == 0:
        top_idx = np.arange(min(MIN_FALLBACK_FEATURES, Xtr_pf.shape[1]))
        mask    = np.zeros(Xtr_pf.shape[1], dtype=bool)
        mask[top_idx] = True
        return mask

    k       = min(int(top_k), Xtr_pf.shape[1])
    top_idx = np.argsort(imp)[::-1][:k]
    mask    = np.zeros(Xtr_pf.shape[1], dtype=bool)
    mask[top_idx] = True

    if mask.sum() == 0:
        top_idx = np.argsort(imp)[::-1][:min(MIN_FALLBACK_FEATURES, Xtr_pf.shape[1])]
        mask    = np.zeros(Xtr_pf.shape[1], dtype=bool)
        mask[top_idx] = True

    return mask


def choose_threshold_max_f1(y_true, probs):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1      = (2 * precision[:-1] * recall[:-1]) / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    best    = int(np.nanargmax(f1)) if np.isfinite(f1).any() else 0
    return float(thresholds[best]), float(f1[best])


def _make_inner_cv(seed: int):
    if StratifiedGroupKFold is not None:
        return StratifiedGroupKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=seed), True
    return GroupKFold(n_splits=N_INNER_FOLDS), False


# =============================================================================
# Fold checks, caching, and inner-CV scoring
# =============================================================================
def _log_fold_class_counts(split_label: str, y_arr: np.ndarray, fold_idx: int) -> bool:
    n_pos        = int((y_arr == 1).sum())
    n_neg        = int((y_arr == 0).sum())
    both_present = (n_pos > 0) and (n_neg > 0)
    if not both_present:
        logger.warning(
            "[%s fold %d] Single-class split detected: pre-flare=%d, non-pre-flare=%d — skipping.",
            split_label, fold_idx, n_pos, n_neg,
        )
    else:
        logger.info("[%s fold %d] pre-flare=%d, non-pre-flare=%d", split_label, fold_idx, n_pos, n_neg)
    return both_present


def build_inner_fold_cache(X_train, y_train, g_train, seed):
    """
    Pre-compute inner CV folds with variance filtering, univariate pre-filtering,
    and scaling applied to each split. Returns a list of fold dicts (or None for
    single-class splits) that tuning functions consume without re-doing preprocessing.
    """
    cv, _ = _make_inner_cv(seed)

    X_np = X_train.values
    y_np = y_train.values
    g_np = g_train.values

    fold_cache = []
    n_usable   = 0

    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_np, y_np, g_np)):
        ytr = y_np[tr_idx]
        yva = y_np[va_idx]

        if not (_log_fold_class_counts("inner-train", ytr, fold_idx) and
                _log_fold_class_counts("inner-val",   yva, fold_idx)):
            fold_cache.append(None)
            continue

        Xtr = X_np[tr_idx]
        Xva = X_np[va_idx]

        vt, vt_support, Xtr_v = fit_variance_filter(Xtr, thresh=VAR_THRESH)
        if Xtr_v.shape[1] == 0:
            vt         = None
            vt_support = np.ones(Xtr.shape[1], dtype=bool)
            Xtr_v      = Xtr
            Xva_v      = Xva
        else:
            Xva_v = apply_variance_filter(vt, Xva)

        skb, pf_support, Xtr_pf = fit_univariate_prefilter(Xtr_v, ytr, k=PREFILTER_K)
        Xva_pf = apply_univariate_prefilter(skb, Xva_v)
        assert Xtr_pf.shape[1] == pf_support.sum(), f"Inner fold {fold_idx}: prefilter column mismatch"

        vt_pos             = np.where(vt_support)[0]
        pf_pos_in_original = vt_pos[np.where(pf_support)[0]]

        scaler_pf     = StandardScaler()
        Xtr_pf_scaled = scaler_pf.fit_transform(Xtr_pf)
        Xva_pf_scaled = scaler_pf.transform(Xva_pf)

        fold_cache.append({
            "fold_idx": fold_idx, "tr_idx": tr_idx, "va_idx": va_idx,
            "Xtr_pf": Xtr_pf, "Xva_pf": Xva_pf,
            "Xtr_pf_scaled": Xtr_pf_scaled, "Xva_pf_scaled": Xva_pf_scaled,
            "scaler_pf": scaler_pf, "ytr": ytr, "yva": yva,
            "pf_pos_in_original": pf_pos_in_original,
            "pf_support": pf_support, "vt_support": vt_support,
        })
        n_usable += 1

    logger.info("Inner fold cache built: %d/%d usable (seed=%d)", n_usable, len(fold_cache), seed)
    return fold_cache


def cv_oof_probs_pr_auc_from_cache(fold_cache, y_train, model_name, seed, params, trial=None):
    """
    Run inner-CV OOF prediction using a pre-built fold cache and return
    (oof_probs, oof_pr_auc, contributed_mask). Supports Optuna pruning via
    the optional trial argument (XGB only).
    """
    y_np        = y_train.values
    oof_probs   = np.zeros(len(y_np), dtype=float)
    contributed = np.zeros(len(y_np), dtype=bool)
    prune_step  = 0

    for fold_entry in fold_cache:
        if fold_entry is None:
            continue

        Xtr_pf        = fold_entry["Xtr_pf"]
        Xva_pf        = fold_entry["Xva_pf"]
        Xtr_pf_scaled = fold_entry["Xtr_pf_scaled"]
        Xva_pf_scaled = fold_entry["Xva_pf_scaled"]
        ytr           = fold_entry["ytr"]
        yva           = fold_entry["yva"]
        va_idx        = fold_entry["va_idx"]

        if model_name == "LR_L2":
            lasso = build_lr_selector(C=params["lasso_c"], seed=seed)
            lasso.fit(Xtr_pf_scaled, ytr)
            sel      = SelectFromModel(lasso, prefit=True)
            sel_mask = sel.get_support()

            if sel_mask.sum() == 0:
                coef    = np.abs(lasso.coef_[0])
                k       = min(MIN_FALLBACK_FEATURES, Xtr_pf_scaled.shape[1])
                top_idx = np.argsort(coef)[::-1][:k]
                sel_mask = np.zeros(Xtr_pf_scaled.shape[1], dtype=bool)
                sel_mask[top_idx] = True

            clf   = build_lr_clf(C=params["l2_c"], seed=seed)
            clf.fit(Xtr_pf_scaled[:, sel_mask], ytr)
            probs = clf.predict_proba(Xva_pf_scaled[:, sel_mask])[:, 1]

        else:
            sel_mask = xgb_select_mask(Xtr_pf, ytr, seed=seed, top_k=params["xgb_sel_topk"])
            pos      = int((ytr == 1).sum())
            neg      = int((ytr == 0).sum())
            clf      = build_xgb_clf(params=params, seed=seed, spw=neg / max(pos, 1))
            clf.fit(Xtr_pf[:, sel_mask], ytr)
            probs = clf.predict_proba(Xva_pf[:, sel_mask])[:, 1]

        oof_probs[va_idx]   = probs
        contributed[va_idx] = True

        if trial is not None:
            if contributed.sum() >= 2 and len(np.unique(y_np[contributed])) >= 2:
                prune_step += 1
                trial.report(average_precision_score(y_np[contributed], oof_probs[contributed]), step=prune_step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

    coverage = contributed.mean()
    if coverage < MIN_OOF_COVERAGE_WARN:
        logger.warning(
            "Low inner OOF coverage for %s at seed=%d: %d/%d rows (%.1f%%).",
            model_name, seed, contributed.sum(), len(contributed), coverage * 100,
        )

    if contributed.sum() == 0:
        return oof_probs, 0.0, contributed

    oof_pr = average_precision_score(y_np[contributed], oof_probs[contributed])
    return oof_probs, float(oof_pr), contributed


def tune_lr_for_pr_auc(X_train, y_train, g_train, seed, n_trials, fold_cache=None):
    """
    Run Optuna hyperparameter search for LR_L2, optimising inner-CV OOF PR-AUC.
    Returns (best_params, best_pr_auc, best_oof_probs, best_contributed, best_threshold).
    """
    if fold_cache is None:
        fold_cache = build_inner_fold_cache(X_train, y_train, g_train, seed)

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))

    def objective(trial):
        p = {
            "lasso_c": trial.suggest_float("lasso_c", 1e-5, 2e-1, log=True),
            "l2_c":    trial.suggest_float("l2_c",    1e-6, 1e-1, log=True),
        }
        oof_probs, oof_pr, contributed = cv_oof_probs_pr_auc_from_cache(fold_cache, y_train, "LR_L2", seed, p)
        trial.set_user_attr("oof_probs",   oof_probs.tolist())
        trial.set_user_attr("contributed", contributed.tolist())
        return oof_pr

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best             = study.best_params
    best_oof_probs   = np.array(study.best_trial.user_attrs["oof_probs"])
    best_contributed = np.array(study.best_trial.user_attrs["contributed"])
    y_np             = y_train.values

    if best_contributed.sum() >= 2 and len(np.unique(y_np[best_contributed])) >= 2:
        best_threshold, _ = choose_threshold_max_f1(y_np[best_contributed], best_oof_probs[best_contributed])
    else:
        best_threshold = 0.5

    return (
        {"lasso_c": round(float(best["lasso_c"]), 6), "l2_c": round(float(best["l2_c"]), 6)},
        float(study.best_value),
        best_oof_probs,
        best_contributed,
        best_threshold,
    )


def tune_xgb_for_pr_auc(X_train, y_train, g_train, seed, n_trials, fold_cache=None):
    """
    Run Optuna hyperparameter search for XGBoost with MedianPruner, optimising
    inner-CV OOF PR-AUC. Returns (best_params, best_pr_auc, best_oof_probs,
    best_contributed, best_threshold).
    """
    if fold_cache is None:
        fold_cache = build_inner_fold_cache(X_train, y_train, g_train, seed)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=8, n_warmup_steps=2),
    )

    def objective(trial):
        p = {
            "xgb_sel_topk":     trial.suggest_int("xgb_sel_topk", 10, 60, step=10),
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1500, step=50),
            "max_depth":        trial.suggest_int("max_depth", 2, 3),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 0.8),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
            "min_child_weight": trial.suggest_float("min_child_weight", 20.0, 40.0),
            "gamma":            trial.suggest_float("gamma", 1.0, 5.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 10.0, 200.0, log=True),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "max_delta_step":   trial.suggest_int("max_delta_step", 0, 8),
        }
        oof_probs, oof_pr, contributed = cv_oof_probs_pr_auc_from_cache(
            fold_cache, y_train, "XGB", seed, p, trial=trial
        )
        trial.set_user_attr("oof_probs",   oof_probs.tolist())
        trial.set_user_attr("contributed", contributed.tolist())
        return oof_pr

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best      = study.best_params
    int_keys  = {"xgb_sel_topk", "n_estimators", "max_depth", "max_delta_step"}
    best_params      = {k: int(v) if k in int_keys else float(v) for k, v in best.items()}
    best_oof_probs   = np.array(study.best_trial.user_attrs["oof_probs"])
    best_contributed = np.array(study.best_trial.user_attrs["contributed"])
    y_np             = y_train.values

    if best_contributed.sum() >= 2 and len(np.unique(y_np[best_contributed])) >= 2:
        best_threshold, _ = choose_threshold_max_f1(y_np[best_contributed], best_oof_probs[best_contributed])
    else:
        best_threshold = 0.5

    return best_params, float(study.best_value), best_oof_probs, best_contributed, best_threshold


def fit_final_model_on_full_train(X_train, y_train, model_name, seed, params):
    """
    Fit the full preprocessing pipeline and classifier on the entire outer training
    set using tuned hyperparameters. Returns a pipe dict suitable for transform_with_pipe.
    """
    Xtr = X_train.values
    ytr = y_train.values

    vt, vt_support, Xtr_v = fit_variance_filter(Xtr, thresh=VAR_THRESH)
    if Xtr_v.shape[1] == 0:
        vt = None; vt_support = np.ones(Xtr.shape[1], dtype=bool); Xtr_v = Xtr

    skb, pf_support, Xtr_pf = fit_univariate_prefilter(Xtr_v, ytr, k=PREFILTER_K)

    if model_name == "LR_L2":
        scaler, sel_mask = lasso_select_mask(Xtr_pf, ytr, seed=seed, lasso_c=params["lasso_c"])
        Xtr_s = scaler.transform(Xtr_pf)[:, sel_mask]
        clf   = build_lr_clf(C=params["l2_c"], seed=seed)
        clf.fit(Xtr_s, ytr)
        return {
            "model": model_name, "vt": vt, "vt_support": vt_support,
            "skb": skb, "pf_support": pf_support,
            "scaler": scaler, "sel_mask": sel_mask, "clf": clf,
        }

    sel_mask = xgb_select_mask(Xtr_pf, ytr, seed=seed, top_k=params["xgb_sel_topk"])
    pos = int((ytr == 1).sum()); neg = int((ytr == 0).sum())
    clf = build_xgb_clf(params=params, seed=seed, spw=neg / max(pos, 1))
    clf.fit(Xtr_pf[:, sel_mask], ytr)
    return {
        "model": model_name, "vt": vt, "vt_support": vt_support,
        "skb": skb, "pf_support": pf_support,
        "scaler": None, "sel_mask": sel_mask, "clf": clf,
    }


def transform_with_pipe(X, pipe):
    X_np       = X.values if hasattr(X, "values") else X
    vt_support = pipe["vt_support"]
    vt_pos     = np.where(vt_support)[0]

    Xv  = pipe["vt"].transform(X_np) if pipe["vt"] is not None else X_np[:, vt_support]
    Xpf = pipe["skb"].transform(Xv)

    pf_pos_in_original  = vt_pos[np.where(pipe["pf_support"])[0]]
    sel_mask            = pipe["sel_mask"]
    sel_pos_in_original = pf_pos_in_original[sel_mask]

    Xs = (pipe["scaler"].transform(Xpf)[:, sel_mask] if pipe["scaler"] is not None
          else Xpf[:, sel_mask])

    comb_mask = np.zeros(X_np.shape[1], dtype=bool)
    comb_mask[sel_pos_in_original] = True

    assert Xs.shape[1] == sel_mask.sum(), (
        f"transform_with_pipe: Xs columns ({Xs.shape[1]}) != sel_mask sum ({sel_mask.sum()})"
    )
    assert Xs.shape[1] == comb_mask.sum(), (
        f"transform_with_pipe: Xs columns ({Xs.shape[1]}) != comb_mask sum ({comb_mask.sum()})"
    )
    return Xs, comb_mask


def evaluate_on_outer_test(X_train, X_test, y_train, y_test, model_name, seed, params, threshold):
    """
    Fit a final model on the outer training split and evaluate it on the held-out
    outer test split. Returns (metrics dict, confusion matrix, pipe, comb_mask,
    predicted probabilities, predicted labels).
    """
    pipe = fit_final_model_on_full_train(X_train, y_train, model_name, seed, params)

    Xte_s, comb_mask = transform_with_pipe(X_test, pipe)
    probs = pipe["clf"].predict_proba(Xte_s)[:, 1]
    yte   = y_test.values

    pr_auc = average_precision_score(yte, probs) if len(np.unique(yte)) >= 2 else np.nan
    aucroc = roc_auc_score(yte, probs)           if len(np.unique(yte)) >= 2 else np.nan
    brier  = brier_score_loss(yte, probs)
    preds  = (probs >= threshold).astype(int)
    cm     = confusion_matrix(yte, preds, labels=[0, 1])

    metrics = {
        "PR-AUC (PRIMARY)":  pr_auc,
        "AUC-ROC":           aucroc,
        "Sensitivity":       recall_score(yte, preds, zero_division=0),
        "Precision":         precision_score(yte, preds, zero_division=0),
        "Specificity":       recall_score(yte, preds, pos_label=0, zero_division=0),
        "F1":                f1_score(yte, preds, zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(yte, preds),
        "Brier Score":       brier,
        "Threshold":         float(threshold),
        "Features selected": int(comb_mask.sum()),
    }
    return metrics, cm, pipe, comb_mask, probs, preds


# =============================================================================
# Table and figure helpers
# =============================================================================
def wrap_text(x, width=18):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s == "":
        return ""
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False, break_on_hyphens=False))


def prettify_header(col):
    col = str(col)
    mapping = {
        "PR-AUC (PRIMARY)": "PR-AUC\n(PRIMARY)",
        "OOF PR-AUC": "OOF\nPR-AUC",
        "AUC-ROC": "AUC-\nROC",
        "Brier Score": "Brier\nScore",
        "Balanced Accuracy": "Balanced\nAccuracy",
        "Features selected": "Features\nselected",
        "learning_rate": "learning_\nrate",
        "colsample_bytree": "colsample_\nbytree",
        "min_child_weight": "min_child_\nweight",
        "max_delta_step": "max_delta_\nstep",
        "n_estimators": "n_\nestimators",
        "reg_lambda": "reg_\nlambda",
        "reg_alpha": "reg_\nalpha",
        "xgb_sel_topk": "xgb_sel_\ntopk",
        "Search space": "Search\nspace",
        "Outer Fold": "Outer\nFold",
        "Splits Appearing": "Folds\nAppearing",
        "Stability Score": "Stability\nScore",
        "Mean |SHAP| (OOF Correct Train)": "Mean\n|SHAP|\n(OOF Correct\nTrain)",
        "Mean |SHAP| (Global)": "Mean\n|SHAP|\n(Global)",
    }
    return mapping.get(col, wrap_text(col, width=14))


def prepare_table_df(df, wrap_cell_width=18):
    out = df.copy()
    out.columns = [prettify_header(c) for c in out.columns]
    for c in out.columns:
        colname = str(c).replace("\n", " ")
        width   = 24 if "Direction" in colname else 26 if "Search space" in colname else wrap_cell_width
        out[c]  = out[c].apply(lambda x: wrap_text(x, width=width) if isinstance(x, str) else x)
    return out


def compute_col_widths(df, min_w=0.08, max_w=0.32):
    lengths = []
    for c in df.columns:
        max_len = max(len(line) for line in str(c).split("\n"))
        for v in df[c].tolist():
            max_len = max(max_len, max(len(line) for line in str(v).split("\n")))
        lengths.append(max_len)

    lengths = np.array(lengths, dtype=float)
    if lengths.sum() == 0:
        return [1 / len(df.columns)] * len(df.columns)

    widths = np.clip(lengths / lengths.sum(), min_w, max_w)
    return (widths / widths.sum()).tolist()


def get_metrics_table_col_widths(df):
    cols      = list(df.columns)
    width_map = {
        "Model": 0.17, "PR-AUC\n(PRIMARY)": 0.095, "OOF\nPR-AUC": 0.095,
        "AUC-\nROC": 0.095, "Brier\nScore": 0.095, "F1": 0.08,
        "Sensitivity": 0.085, "Specificity": 0.085, "Precision": 0.085,
        "Balanced\nAccuracy": 0.095, "Features\nselected": 0.095, "Threshold": 0.08,
    }
    widths = [width_map.get(c, 0.09) for c in cols]
    total  = sum(widths)
    return [w / total for w in widths]


def get_custom_col_widths(df):
    cols = list(df.columns)

    if "Stability\nScore" in cols and "Mean\n|SHAP|\n(Global)" in cols and "Gene" in cols:
        widths = []
        for c in cols:
            if c == "Gene":                   widths.append(0.33)
            elif c == "Folds\nAppearing":     widths.append(0.14)
            elif c == "Stability\nScore":     widths.append(0.14)
            elif c == "Mean\n|SHAP|\n(Global)": widths.append(0.17)
            elif c == "Direction":            widths.append(0.22)
            else:                             widths.append(0.12)
        return [w / sum(widths) for w in widths]

    if "Stability\nScore" in cols and "Mean\n|SHAP|\n(OOF Correct\nTrain)" in cols and "Gene" in cols:
        widths = []
        for c in cols:
            if c == "Gene":                   widths.append(0.33)
            elif c == "Folds\nAppearing":     widths.append(0.14)
            elif c == "Stability\nScore":     widths.append(0.14)
            elif c == "Mean\n|SHAP|\n(OOF Correct\nTrain)": widths.append(0.17)
            elif c == "Direction":            widths.append(0.22)
            else:                             widths.append(0.12)
        return [w / sum(widths) for w in widths]

    if "Direction" in cols and "Folds\nAppearing" in cols and "Gene" in cols:
        widths = []
        for c in cols:
            if c == "Gene":               widths.append(0.42)
            elif c == "Folds\nAppearing": widths.append(0.18)
            elif c == "Direction":        widths.append(0.40)
            else:                         widths.append(0.12)
        return [w / sum(widths) for w in widths]

    if "Direction" in cols:
        widths = []
        for c in cols:
            if c == "Direction":   widths.append(0.34)
            elif c == "Gene_Name": widths.append(0.18)
            elif c == "Probe_ID":  widths.append(0.19)
            elif c in ["Feature_Label", "Search\nspace"]: widths.append(0.24)
            else:                  widths.append(0.12)
        return [w / sum(widths) for w in widths]

    if "Model" in cols:
        return get_metrics_table_col_widths(df)
    if "Search\nspace" in cols:
        return [0.38, 0.62]
    if len(cols) >= 8:
        return compute_col_widths(df, min_w=0.07, max_w=0.20)
    return compute_col_widths(df, min_w=0.10, max_w=0.30)


def style_table(tbl, fontsize=TABLE_BODY_FONTSIZE, header_fontsize=TABLE_HEADER_FONTSIZE):
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_linewidth(0.8)
        cell.set_edgecolor("gray")
        if row == 0:
            cell.set_text_props(weight="bold", fontsize=header_fontsize, ha="center", va="center")
        else:
            cell.set_text_props(fontsize=fontsize, ha="center", va="center")


def set_row_heights_by_text(
    tbl, df, base_height=0.13, header_base_height=0.18, header_extra_per_line=0.055
):
    ncols            = len(df.columns)
    header_max_lines = max(str(df.columns[c]).count("\n") + 1 for c in range(ncols))
    header_height    = header_base_height + (header_max_lines - 1) * header_extra_per_line

    for c in range(ncols):
        if (0, c) in tbl.get_celld():
            tbl[(0, c)].set_height(header_height)

    for r in range(len(df)):
        max_lines = max(str(df.iloc[r, c]).count("\n") + 1 for c in range(ncols))
        row_h     = base_height * max_lines
        for c in range(ncols):
            if (r + 1, c) in tbl.get_celld():
                tbl[(r + 1, c)].set_height(row_h)


def draw_single_table(
    ax, df, title,
    fontsize=TABLE_BODY_FONTSIZE, header_fontsize=TABLE_HEADER_FONTSIZE,
    title_fontsize=TABLE_TITLE_FONTSIZE, wrap_cell_width=18,
    bbox=(0.01, 0.05, 0.98, 0.90), col_widths=None,
    body_base_height=0.10, header_base_height=0.16,
):
    ax.axis("off")
    df_disp = prepare_table_df(df, wrap_cell_width=wrap_cell_width)

    if col_widths is None:
        col_widths = get_custom_col_widths(df_disp)

    tbl = ax.table(
        cellText=df_disp.values.tolist(), colLabels=list(df_disp.columns),
        cellLoc="center", bbox=bbox, colWidths=col_widths,
    )
    style_table(tbl, fontsize=fontsize, header_fontsize=header_fontsize)
    set_row_heights_by_text(
        tbl, df_disp, base_height=body_base_height,
        header_base_height=header_base_height, header_extra_per_line=0.055,
    )
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=4)


def plot_two_tables_side_by_side_with_gap(
    df_left, title_left, df_right, title_right, save_path, fig_title=None
):
    n_rows     = max(len(df_left), len(df_right), 1)
    fig_height = max(8.5, 0.95 * n_rows + 3.8)

    fig     = plt.figure(figsize=(24, fig_height))
    gs      = fig.add_gridspec(1, 3, width_ratios=[1, 0.08, 1])
    ax_left = fig.add_subplot(gs[0, 0])
    ax_gap  = fig.add_subplot(gs[0, 1])
    ax_right = fig.add_subplot(gs[0, 2])
    ax_gap.axis("off")

    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=48, fontweight="bold", y=0.99)

    for ax, df, title in [(ax_left, df_left, title_left), (ax_right, df_right, title_right)]:
        draw_single_table(
            ax, df, title,
            fontsize=TABLE_BODY_FONTSIZE, header_fontsize=TABLE_HEADER_FONTSIZE,
            title_fontsize=32, wrap_cell_width=24 if ax is ax_left else 26,
            bbox=(0.01, 0.09, 0.98, 0.84), body_base_height=0.10, header_base_height=0.16,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.965], pad=0.6, w_pad=1.2)
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


def plot_confusion_matrices_one_model(all_cms, fold_ids, model_name, save_path):
    fold_ids = sorted([fid for fid in fold_ids if (fid, model_name) in all_cms])
    if not fold_ids:
        logger.warning("plot_confusion_matrices_one_model: no data for %s — skipping.", model_name)
        return

    n_folds     = len(fold_ids)
    n_cols      = 2
    n_rows      = int(np.ceil(n_folds / n_cols))
    all_vals    = [v for fid in fold_ids for v in all_cms[(fid, model_name)].ravel()]
    global_vmax = max(all_vals) if all_vals else 1

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7.2 * n_cols + 1.5, 6.2 * n_rows + 1.8),
        constrained_layout=False,
    )
    axes = np.array(axes).reshape(n_rows, n_cols)
    fig.suptitle(
        f"Confusion Matrices Across Held-Out Outer Test Folds\n{get_model_display_name(model_name)}",
        fontsize=38, fontweight="bold", y=0.98,
    )

    cbar_ax    = fig.add_axes([0.92, 0.18, 0.022, 0.62])
    cbar_drawn = False

    for idx, fold_id in enumerate(fold_ids):
        r, c = idx // n_cols, idx % n_cols
        ax   = axes[r, c]
        cm   = all_cms[(fold_id, model_name)]

        sns.heatmap(
            cm, annot=False, fmt="d", ax=ax,
            cmap="Blues", linewidths=0.5, linecolor="white",
            vmin=0, vmax=global_vmax,
            cbar=not cbar_drawn, cbar_ax=cbar_ax if not cbar_drawn else None,
            square=True,
        )
        if not cbar_drawn:
            cbar_drawn = True

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(
                j + 0.5, i + 0.5, f"{cm[i, j]}",
                ha="center", va="center", fontsize=17, fontweight="bold",
                color="white" if cm[i, j] > global_vmax / 2 else "black",
            )

        ax.set_title(f"Outer Fold {fold_id}", fontsize=26, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted", fontsize=18, labelpad=8)
        ax.set_ylabel("Actual",    fontsize=18, labelpad=8)
        ax.set_xticklabels(CLASS_NAMES_DISPLAY, fontsize=17)
        ax.set_yticklabels(CLASS_NAMES_DISPLAY, fontsize=17, rotation=0)
        ax.tick_params(axis="both", length=0)

    for idx in range(n_folds, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    if cbar_drawn:
        cbar_ax.set_ylabel("Count", rotation=270, labelpad=18, fontsize=17)
        cbar_ax.tick_params(labelsize=16)
    else:
        cbar_ax.axis("off")

    plt.subplots_adjust(left=0.08, right=0.89, top=0.90, bottom=0.10, hspace=0.42, wspace=0.30)
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


def plot_metrics_table(summary_df, save_path):
    """Render mean ± SD and median metrics for both models as a formatted table."""
    METRIC_DISPLAY = {
        "PR-AUC (PRIMARY)":  "PR-AUC (PRIMARY)",
        "AUC-ROC":           "AUC-ROC",
        "Sensitivity":       "Recall (Sensitivity)",
        "Precision":         "Precision",
        "Specificity":       "Specificity",
        "F1":                "F1",
        "Balanced Accuracy": "Balanced Accuracy",
        "OOF PR-AUC":        "OOF PR-AUC",
        "Brier Score":       "Brier Score",
        "Threshold":         "Threshold",
        "Features selected": "Features Selected",
    }

    def _split(val):
        s = str(val).strip()
        if "\n" in s:
            top, bot = s.split("\n", 1)
            return top.strip(), bot.strip().strip("()")
        return s, ""

    lr_name  = "Logistic Regression L2"
    xgb_name = "XGBoost"

    columns = [
        "Metric",
        "Logistic Regression L2\nMean \u00b1 SD",
        "Logistic Regression L2\nMedian",
        "XGBoost\nMean \u00b1 SD",
        "XGBoost\nMedian",
    ]

    data = []
    for col in summary_df.columns:
        if col not in METRIC_DISPLAY:
            continue
        lr_mean,  lr_med  = _split(summary_df.loc[lr_name,  col]) if lr_name  in summary_df.index else ("", "")
        xgb_mean, xgb_med = _split(summary_df.loc[xgb_name, col]) if xgb_name in summary_df.index else ("", "")
        data.append([METRIC_DISPLAY[col], lr_mean, lr_med, xgb_mean, xgb_med])

    fig, ax = plt.subplots(figsize=(28, 18))
    ax.axis("off")

    col_widths = [0.22, 0.24, 0.15, 0.24, 0.15]

    tbl = ax.table(
        cellText=data,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
        bbox=[0, 0.02, 1, 0.90],
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(26)
    tbl.scale(1.0, 3.6)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#BBBBBB")
        cell.get_text().set_wrap(True)

        if row == 0:
            cell.set_text_props(weight="bold", fontsize=28, va="center", ha="center")
            cell.set_facecolor("#E8EEF6")
            cell.set_height(0.16)
        else:
            cell.set_text_props(fontsize=24, va="center", ha="center")
            cell.set_facecolor("#F6F6F6" if row % 2 == 0 else "white")
            cell.set_height(0.075)

    ax.set_title(
        "Average Classification Metrics Across Held-Out Outer Test Folds",
        fontsize=50,
        fontweight="bold",
        pad=8,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


def plot_per_fold_metrics_side_by_side(supp_df, save_path):
    METRIC_DISPLAY = {
        "PR-AUC (PRIMARY)":  "PR-AUC",
        "OOF PR-AUC":        "OOF PR-AUC",
        "AUC-ROC":           "AUC-ROC",
        "Brier Score":       "Brier Score",
        "F1":                "F1",
        "Sensitivity":       "Sensitivity",
        "Specificity":       "Specificity",
        "Precision":         "Precision",
        "Balanced Accuracy": "Balanced Accuracy",
        "Threshold":         "Threshold",
        "Features selected": "Features Selected",
        "Test Subjects":     "Test Subjects",
        "Pre-flare (n)":     "Pre-flare (n)",
        "Non-pre-flare (n)": "Non-pre-flare (n)",
    }
    metric_order = [
        "Test Subjects", "Pre-flare (n)", "Non-pre-flare (n)",
        "PR-AUC (PRIMARY)", "OOF PR-AUC", "AUC-ROC", "Brier Score",
        "F1", "Sensitivity", "Specificity", "Precision",
        "Balanced Accuracy", "Threshold", "Features selected",
    ]
    count_metrics = {"Test Subjects", "Pre-flare (n)", "Non-pre-flare (n)", "Features selected"}
    models   = list(supp_df["Model"].unique())
    fold_ids = sorted(supp_df["Outer Fold"].unique())

    def _build_model_table(model_name):
        sub  = supp_df[supp_df["Model"] == model_name].sort_values("Outer Fold")
        rows = []
        for metric in metric_order:
            if metric not in sub.columns:
                continue
            row = {"Metric": METRIC_DISPLAY.get(metric, metric)}
            for fold_id in fold_ids:
                tmp = sub[sub["Outer Fold"] == fold_id]
                val = ""
                if not tmp.empty:
                    v = tmp.iloc[0][metric]
                    if pd.notna(v):
                        val = str(int(round(float(v)))) if metric in count_metrics else f"{float(v):.3f}"
                row[f"Fold {fold_id}"] = val
            rows.append(row)
        return pd.DataFrame(rows)

    def _set_row_heights(tbl, df):
        ncols = len(df.columns)
        for c in range(ncols):
            if (0, c) in tbl.get_celld():
                tbl[(0, c)].set_height(0.13)
        for r in range(len(df)):
            max_lines = max(str(df.iloc[r, c]).count("\n") + 1 for c in range(ncols))
            row_h     = 0.072 + (max_lines - 1) * 0.028
            for c in range(ncols):
                if (r + 1, c) in tbl.get_celld():
                    tbl[(r + 1, c)].set_height(row_h)

    def _render_table(ax, model_name):
        ax.axis("off")
        df_model   = _build_model_table(model_name)
        n_folds    = len([c for c in df_model.columns if c.startswith("Fold ")])
        col_widths = [0.42] + [0.58 / max(n_folds, 1)] * n_folds
        tbl = ax.table(
            cellText=df_model.values.tolist(), colLabels=list(df_model.columns),
            cellLoc="center", colWidths=col_widths, bbox=[0, 0, 1, 0.95],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(24)
        for (row, col), cell in tbl.get_celld().items():
            cell.set_linewidth(0.7); cell.set_edgecolor("gray")
            cell.set_text_props(ha="center", va="center", wrap=True)
            if row == 0:
                cell.set_text_props(weight="bold", fontsize=20); cell.set_facecolor("#EEEEEE")
            else:
                cell.set_text_props(fontsize=18)
                cell.set_facecolor("#F7F7F7" if col == 0 else "#FFFFFF")
        _set_row_heights(tbl, df_model)
        tbl.scale(1.1, 1.6)
        ax.set_title(model_name, fontsize=38, fontweight="bold", pad=12)

    fig_height = max(12.0, 0.60 * len(metric_order) + 4.0)
    fig, axes  = plt.subplots(len(models), 1, figsize=(22, fig_height * 1.4))
    if len(models) == 1:
        axes = [axes]
    fig.suptitle(
        "Supplementary Table: Per-Fold Held-Out Test Metrics Across Outer Folds",
        fontsize=44, fontweight="bold", y=1.02,
    )
    for ax, model_name in zip(axes, models):
        _render_table(ax, model_name)
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.0)
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


# =============================================================================
# Permutation sanity check plotting
# =============================================================================
def plot_permutation_table_top_bottom(perm_df, save_path):
    """
    Permutation sanity-check results as two stacked tables (LR top, XGB bottom),
    with metrics as rows and folds as columns.
    """
    METRIC_DISPLAY = {
        "OOF PR-AUC (Permuted Train Labels)": "OOF PR-AUC\n(Permuted Labels)",
        "Held-Out Test PR-AUC":               "Held-Out\nTest PR-AUC",
        "AUC-ROC":                            "AUC-ROC",
        "Brier Score":                        "Brier Score",
        "F1":                                 "F1",
        "Threshold":                          "Threshold",
        "Features selected":                  "Features Selected",
    }
    metric_order = list(METRIC_DISPLAY.keys())
    models       = list(perm_df["Model"].unique())
    fold_ids     = sorted(perm_df["Outer Fold"].unique())

    def _build_model_table(model_name):
        sub  = perm_df[perm_df["Model"] == model_name].sort_values("Outer Fold")
        rows = []
        for metric in metric_order:
            if metric not in sub.columns:
                continue
            row = {"Metric": METRIC_DISPLAY.get(metric, metric)}
            for fold_id in fold_ids:
                tmp = sub[sub["Outer Fold"] == fold_id]
                val = ""
                if not tmp.empty:
                    v = tmp.iloc[0][metric]
                    if pd.notna(v):
                        val = str(int(round(float(v)))) if metric == "Features selected" else f"{float(v):.3f}"
                row[f"Fold {fold_id}"] = val
            rows.append(row)
        return pd.DataFrame(rows)

    def _set_row_heights(tbl, df):
        ncols = len(df.columns)
        for c in range(ncols):
            if (0, c) in tbl.get_celld():
                tbl[(0, c)].set_height(0.16)
        for r in range(len(df)):
            max_lines = max(str(df.iloc[r, c]).count("\n") + 1 for c in range(ncols))
            row_h     = 0.082 + (max_lines - 1) * 0.034
            for c in range(ncols):
                if (r + 1, c) in tbl.get_celld():
                    tbl[(r + 1, c)].set_height(row_h)

    def _render_table(ax, model_name):
        ax.axis("off")
        df_model   = _build_model_table(model_name)
        n_folds    = len([c for c in df_model.columns if c.startswith("Fold ")])
        col_widths = [0.44] + [0.56 / max(n_folds, 1)] * n_folds
        tbl = ax.table(
            cellText=df_model.values.tolist(), colLabels=list(df_model.columns),
            cellLoc="center", colWidths=col_widths, bbox=[0, 0, 1, 0.93],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(21)
        for (row, col), cell in tbl.get_celld().items():
            cell.set_linewidth(0.7); cell.set_edgecolor("gray")
            cell.set_text_props(ha="center", va="center", wrap=True)
            if row == 0:
                cell.set_text_props(weight="bold", fontsize=19); cell.set_facecolor("#EEEEEE")
            else:
                cell.set_text_props(fontsize=18)
                cell.set_facecolor("#F7F7F7" if col == 0 else "#FFFFFF")
        _set_row_heights(tbl, df_model)
        tbl.scale(1.1, 1.5)
        ax.set_title(get_model_display_name(model_name), fontsize=34, fontweight="bold", pad=12)

    n_metrics  = len(metric_order)
    fig_height = max(16.0, 1.05 * n_metrics + 10.0)
    fig, axes  = plt.subplots(2, 1, figsize=(20, fig_height * 1.15))
    fig.suptitle(
        "Permutation-Label Sanity Check Across Held-Out Outer Test Folds\n(Both Models — All Outer Folds)",
        fontsize=42, fontweight="bold", y=0.995,
    )

    for ax, model_name in zip(axes, ["LR_L2", "XGB"]):
        if model_name in models:
            _render_table(ax, model_name)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=20, transform=ax.transAxes)
            ax.set_title(get_model_display_name(model_name), fontsize=34, fontweight="bold", pad=12)

    plt.tight_layout(rect=[0, 0, 1, 0.965], h_pad=2.8)
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


# =============================================================================
# PR-curve and SHAP analysis
# =============================================================================
def _draw_pr_auc_on_ax(ax, run_store, model_name, fold_ids):
    for fold_id in fold_ids:
        key = (fold_id, model_name)
        if key not in run_store:
            continue
        probs  = run_store[key]["test_probs"]
        y_test = run_store[key]["y_test"]
        precision, recall, _ = precision_recall_curve(y_test.values, probs)
        pr_auc = average_precision_score(y_test.values, probs)
        ax.plot(recall, precision, linewidth=2, label=f"Fold {fold_id} (PR-AUC={pr_auc:.3f})")

    positive_rates = [
        run_store[(fid, model_name)]["y_test"].mean()
        for fid in fold_ids if (fid, model_name) in run_store
    ]
    baseline = float(np.mean(positive_rates)) if positive_rates else 0.5
    ax.axhline(y=baseline, linestyle="--", linewidth=1.5, label=f"Baseline={baseline:.3f}")
    ax.set_xlabel("Recall",    fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Precision", fontsize=AXIS_LABEL_SIZE)
    ax.set_title(get_model_display_name(model_name), fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold")
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, loc="best")
    ax.grid(alpha=0.25)


def plot_pr_auc_curves_side_by_side(run_store, fold_ids, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(
        "Precision-Recall Curves Across Held-Out Outer Test Folds",
        fontsize=FIG_TITLE_SIZE, fontweight="bold",
    )
    for ax, model_name in zip(axes, ["LR_L2", "XGB"]):
        _draw_pr_auc_on_ax(ax, run_store, model_name, fold_ids)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    return save_path


def plot_shap_bar_side_by_side(lr_shap_df, xgb_shap_df, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(18, max(16, 0.85 * TOP_K + 8)))
    fig.suptitle(
        "Top 20 Influential Genes by Mean |SHAP| Across Models and Outer Training Folds",
        fontsize=FIG_TITLE_SIZE, fontweight="bold", y=0.98,
    )
    for ax, df, model_name in zip(axes, [lr_shap_df, xgb_shap_df], ["LR_L2", "XGB"]):
        if df is None or df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=VALUE_LABEL_SIZE)
            ax.set_title(get_model_display_name(model_name), fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold")
            continue

        plot_df = df.sort_values("Mean |SHAP| (Global)", ascending=True).copy()
        plot_df["Gene"] = plot_df["Gene"].apply(lambda s: wrap_text(s, width=28))
        vals = plot_df["Mean |SHAP| (Global)"].values
        ax.barh(plot_df["Gene"], vals)

        xmax = float(np.nanmax(vals)) if len(vals) else 1.0
        pad  = max(0.03, xmax * 0.18)
        ax.set_xlim(0, xmax + pad)
        for i, v in enumerate(vals):
            ax.text(v + pad * 0.08, i, f"{v:.4f}", va="center", ha="left", fontsize=15, clip_on=True)

        ax.set_xlabel("Mean |SHAP| Across Outer Folds", fontsize=AXIS_LABEL_SIZE)
        ax.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)
        ax.tick_params(axis="y", labelsize=17)
        ax.set_title(get_model_display_name(model_name), fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold")

    plt.subplots_adjust(left=0.16, right=0.97, top=0.92, bottom=0.06, hspace=0.15)
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    return save_path


def compute_shap_values_for_df(model_name, clf, bg_df, eval_df):
    if model_name == "LR_L2":
        explainer = shap.LinearExplainer(clf, bg_df)
    else:
        explainer = shap.TreeExplainer(clf)

    sv = explainer.shap_values(eval_df)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv


def infer_direction(x, s):
    """
    Infer association direction from SHAP correlation.
    Returns a string describing whether higher expression pushes toward pre-flare
    or non-pre-flare; falls back to "unclear" when correlation is undefined.
    """
    x    = np.asarray(x)
    s    = np.asarray(s)
    if len(x) < 2 or np.std(x) == 0 or np.std(s) == 0:
        return "unclear"
    corr = np.corrcoef(x, s)[0, 1]
    if np.isnan(corr):
        return "unclear"
    if corr > 0:
        return f"Higher expression -> pushes toward {POSITIVE_CLASS_NAME}"
    if corr < 0:
        return f"Higher expression -> pushes toward {NEGATIVE_CLASS_NAME}"
    return "unclear"


def get_top_global_shap_genes_for_one_outer_fold(
    X_train, pipe, gene_cols, comb_mask, model_name, fold_seed, gene_name_map, top_k=TOP_K,
):
    selected_genes = np.array(gene_cols)[comb_mask]
    if len(selected_genes) == 0:
        return pd.DataFrame(columns=["Gene", "Mean |SHAP| (Global)", "Direction"])

    Xtr_s, _ = transform_with_pipe(X_train, pipe)
    Xtr_df   = pd.DataFrame(Xtr_s, columns=selected_genes)

    if Xtr_df.shape[0] > SHAP_BG_MAX + 20:
        bg_df     = Xtr_df.sample(n=SHAP_BG_MAX, random_state=fold_seed)
        remaining = Xtr_df.drop(bg_df.index)
        X_eval    = remaining if remaining.shape[0] <= SHAP_EVAL_MAX else remaining.sample(n=SHAP_EVAL_MAX, random_state=fold_seed)
    else:
        bg_df  = Xtr_df if Xtr_df.shape[0] <= SHAP_BG_MAX  else Xtr_df.sample(n=SHAP_BG_MAX,  random_state=fold_seed)
        X_eval = Xtr_df if Xtr_df.shape[0] <= SHAP_EVAL_MAX else Xtr_df.sample(n=SHAP_EVAL_MAX, random_state=fold_seed)

    sv       = compute_shap_values_for_df(model_name=model_name, clf=pipe["clf"], bg_df=bg_df, eval_df=X_eval)
    mean_abs = np.abs(sv).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:top_k]

    rows = []
    for idx in top_idx:
        gene = selected_genes[idx]
        rows.append({
            "Gene": make_gene_label(gene, gene_name_map),
            "Mean |SHAP| (Global)": float(mean_abs[idx]),
            "Direction": infer_direction(X_eval[gene].values, sv[:, idx]),
        })
    return pd.DataFrame(rows)


def aggregate_standard_shap_genes_across_outer_folds(run_store, model_name, gene_name_map, top_k=TOP_K):
    gene_fold_sets       = defaultdict(set)
    gene_direction_votes = defaultdict(list)
    gene_shap_values     = defaultdict(list)

    for (fold_id, mn), item in run_store.items():
        if mn != model_name:
            continue
        df_fold = get_top_global_shap_genes_for_one_outer_fold(
            X_train=item["X_train"], pipe=item["pipe"],
            gene_cols=item["gene_cols"], comb_mask=item["comb_mask"],
            model_name=model_name, fold_seed=item["fold_seed"],
            gene_name_map=gene_name_map, top_k=top_k,
        )
        if df_fold.empty:
            continue
        for _, row in df_fold.iterrows():
            gene, direction, shap_mag = row["Gene"], row["Direction"], row["Mean |SHAP| (Global)"]
            gene_fold_sets[gene].add(fold_id)
            if isinstance(direction, str) and direction.strip():
                gene_direction_votes[gene].append(direction)
            if pd.notna(shap_mag):
                gene_shap_values[gene].append(float(shap_mag))

    total_folds = len({fold_id for fold_id, _ in run_store.keys()})
    rows = []
    for gene, fold_set in gene_fold_sets.items():
        direction_list  = [d for d in gene_direction_votes.get(gene, []) if d != "unclear"]
        final_direction = "unclear" if not direction_list else Counter(direction_list).most_common(1)[0][0]
        shap_list       = gene_shap_values.get(gene, [])
        rows.append({
            "Gene": gene,
            "Folds Appearing": len(fold_set),
            "Stability Score": len(fold_set) / max(total_folds, 1),
            "Mean |SHAP| (Global)": float(np.mean(shap_list)) if shap_list else np.nan,
            "Direction": final_direction,
        })

    if not rows:
        return pd.DataFrame(columns=["Gene", "Folds Appearing", "Stability Score", "Mean |SHAP| (Global)", "Direction"])

    agg_df = (
        pd.DataFrame(rows)
        .sort_values(by=["Folds Appearing", "Mean |SHAP| (Global)", "Gene"], ascending=[False, False, True])
        .head(top_k).reset_index(drop=True)
    )
    agg_df["Stability Score"]       = agg_df["Stability Score"].round(3)
    agg_df["Mean |SHAP| (Global)"] = agg_df["Mean |SHAP| (Global)"].round(4)
    return agg_df


def plot_beeswarm_side_by_side(run_store, lr_gene_labels, xgb_gene_labels, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(18, 18))
    fig.suptitle(
        "SHAP Beeswarm Plot of Top 20 Influential Genes Across Models and Outer Training Folds",
        fontsize=48, fontweight="bold", y=1.02,
    )
    for ax, model_name, gene_labels in zip(axes, ["LR_L2", "XGB"], [lr_gene_labels, xgb_gene_labels]):
        shap_blocks    = []
        feature_blocks = []

        for (fold_id, mn), item in run_store.items():
            if mn != model_name:
                continue
            selected_genes = np.array(item["gene_cols"])[item["comb_mask"]]
            if len(selected_genes) == 0:
                continue

            Xtr_s, _ = transform_with_pipe(item["X_train"], item["pipe"])
            Xtr_df   = pd.DataFrame(Xtr_s, columns=[make_gene_label(g, item["gene_name_map"]) for g in selected_genes])
            X_eval   = Xtr_df if Xtr_df.shape[0] <= SHAP_EVAL_MAX else Xtr_df.sample(n=SHAP_EVAL_MAX, random_state=item["fold_seed"])
            bg_df    = Xtr_df if Xtr_df.shape[0] <= SHAP_BG_MAX   else Xtr_df.sample(n=SHAP_BG_MAX,   random_state=item["fold_seed"])

            sv    = compute_shap_values_for_df(model_name=model_name, clf=item["pipe"]["clf"], bg_df=bg_df, eval_df=X_eval)
            sv_df = pd.DataFrame(sv, columns=Xtr_df.columns).reindex(columns=gene_labels, fill_value=0.0)
            shap_blocks.append(sv_df)
            feature_blocks.append(X_eval.reindex(columns=gene_labels))

        ax.set_title(get_model_display_name(model_name), fontsize=36, fontweight="bold", pad=12)
        if not shap_blocks:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=22)
            continue

        shap_all = pd.concat(shap_blocks,    axis=0, ignore_index=True)
        feat_all = pd.concat(feature_blocks, axis=0, ignore_index=True)
        ordered  = list(gene_labels)[::-1]

        plt.sca(ax)
        with plt.rc_context({"font.size": 20, "axes.labelsize": 24, "xtick.labelsize": 20, "ytick.labelsize": 20}):
            shap.summary_plot(
                shap_all[ordered].values, feat_all[ordered],
                feature_names=ordered, max_display=len(ordered),
                sort=False, show=False, plot_size=None,
            )
        ax.tick_params(axis="both", labelsize=20)
        ax.xaxis.label.set_size(24); ax.yaxis.label.set_size(24)
        for extra_ax in fig.axes:
            if extra_ax is not ax:
                extra_ax.tick_params(labelsize=18)
                if extra_ax.get_ylabel():
                    extra_ax.set_ylabel(extra_ax.get_ylabel(), fontsize=20)

    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=3.0)
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    return save_path


def plot_correct_oof_genes_side_by_side(lr_df, xgb_df, save_path):
    def _wrap_cell_text(val, width):
        if pd.isna(val):
            return ""
        return "\n".join(textwrap.wrap(str(val), width=width, break_long_words=False, break_on_hyphens=False))

    def _prepare_df(df):
        if df is None or df.empty:
            return df
        display_cols = [c for c in ["Gene", "Folds Appearing", "Stability Score", "Mean |SHAP| (OOF Correct Train)", "Direction"] if c in df.columns]
        plot_df = df[display_cols].copy()
        for c in plot_df.select_dtypes(include=[float]).columns:
            plot_df[c] = plot_df[c].round(4)
        plot_df = plot_df.rename(columns={
            "Folds Appearing":            "Folds\nAppearing",
            "Stability Score":            "Stability\nScore",
            "Mean |SHAP| (OOF Correct Train)": "Mean |SHAP|\n(OOF Correct\nTrain)",
        })
        if "Gene"      in plot_df.columns: plot_df["Gene"]      = plot_df["Gene"].apply(lambda x: _wrap_cell_text(x, 30))
        if "Direction" in plot_df.columns: plot_df["Direction"] = plot_df["Direction"].apply(lambda x: _wrap_cell_text(x, 34))
        return plot_df

    def _set_row_heights(tbl, df):
        ncols = len(df.columns)
        for c in range(ncols):
            if (0, c) in tbl.get_celld():
                tbl[(0, c)].set_height(0.20)
        for r in range(len(df)):
            max_lines = max(str(df.iloc[r, c]).count("\n") + 1 for c in range(ncols))
            row_h     = 0.068 + (max_lines - 1) * 0.036
            for c in range(ncols):
                if (r + 1, c) in tbl.get_celld():
                    tbl[(r + 1, c)].set_height(row_h)

    def _render_table(ax, df, title):
        ax.axis("off")
        if df is None or df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=26, transform=ax.transAxes)
            ax.set_title(title, fontsize=40, fontweight="bold", pad=14)
            return
        plot_df = _prepare_df(df)
        widths  = []
        for c in plot_df.columns:
            if c == "Gene":               widths.append(0.30)
            elif c == "Folds\nAppearing": widths.append(0.11)
            elif c == "Stability\nScore": widths.append(0.11)
            elif c == "Mean |SHAP|\n(OOF Correct\nTrain)": widths.append(0.15)
            elif c == "Direction":        widths.append(0.33)
            else:                         widths.append(0.14)
        col_widths = [w / sum(widths) for w in widths]
        tbl = ax.table(
            cellText=plot_df.values.tolist(), colLabels=list(plot_df.columns),
            cellLoc="center", colWidths=col_widths, bbox=[0, 0, 1, 0.95],
        )
        tbl.auto_set_font_size(False); tbl.set_fontsize(21)
        for (row, col), cell in tbl.get_celld().items():
            cell.set_linewidth(0.7); cell.set_edgecolor("gray")
            cell.set_text_props(ha="center", va="center", wrap=True)
            if row == 0:
                cell.set_text_props(weight="bold", fontsize=22)
            else:
                cell.set_text_props(fontsize=21)
        _set_row_heights(tbl, plot_df)
        ax.set_title(title, fontsize=40, fontweight="bold", pad=14)

    n_rows     = max(len(lr_df) if lr_df is not None and not lr_df.empty else 1,
                     len(xgb_df) if xgb_df is not None and not xgb_df.empty else 1)
    fig_height = max(10.0, (0.068 + 0.036) * n_rows * 8.5 + 3.5) * 2 + 4.0
    fig, axes  = plt.subplots(2, 1, figsize=(22, fig_height))
    fig.suptitle(
        "Top 10 Genes Associated with Correct Inner-Validation Predictions\nAcross Outer Training Folds",
        fontsize=44, fontweight="bold", y=0.995,
    )
    _render_table(axes[0], lr_df,  get_model_display_name("LR_L2"))
    _render_table(axes[1], xgb_df, get_model_display_name("XGB"))
    plt.tight_layout(rect=[0, 0, 1, 0.975], h_pad=3.5)
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    return save_path


# =============================================================================
# OOF correct training genes
# =============================================================================
def get_top_correct_oof_training_genes_for_one_outer_fold(
    X_train, y_train, g_train, gene_cols, gene_name_map,
    model_name, seed, params, top_k=TOP_K,
):
    fold_cache           = build_inner_fold_cache(X_train, y_train, g_train, seed)
    gene_abs_shap_values = defaultdict(list)
    gene_direction_votes = defaultdict(list)
    any_correct          = False

    for fold_entry in fold_cache:
        if fold_entry is None:
            continue

        Xtr_pf        = fold_entry["Xtr_pf"]
        Xva_pf        = fold_entry["Xva_pf"]
        Xtr_pf_scaled = fold_entry["Xtr_pf_scaled"]
        Xva_pf_scaled = fold_entry["Xva_pf_scaled"]
        ytr           = fold_entry["ytr"]
        yva           = fold_entry["yva"]
        pf_pos_in_original = fold_entry["pf_pos_in_original"]

        if model_name == "LR_L2":
            lasso = build_lr_selector(C=params["lasso_c"], seed=seed)
            lasso.fit(Xtr_pf_scaled, ytr)
            sel      = SelectFromModel(lasso, prefit=True)
            sel_mask = sel.get_support()
            if sel_mask.sum() == 0:
                coef    = np.abs(lasso.coef_[0])
                k       = min(MIN_FALLBACK_FEATURES, Xtr_pf_scaled.shape[1])
                top_idx = np.argsort(coef)[::-1][:k]
                sel_mask = np.zeros(Xtr_pf_scaled.shape[1], dtype=bool)
                sel_mask[top_idx] = True
            Xtr_s = Xtr_pf_scaled[:, sel_mask]
            Xva_s = Xva_pf_scaled[:, sel_mask]
            clf   = build_lr_clf(C=params["l2_c"], seed=seed)
            clf.fit(Xtr_s, ytr)
        else:
            sel_mask = xgb_select_mask(Xtr_pf, ytr, seed=seed, top_k=params["xgb_sel_topk"])
            Xtr_s = Xtr_pf[:, sel_mask]
            Xva_s = Xva_pf[:, sel_mask]
            pos   = int((ytr == 1).sum()); neg = int((ytr == 0).sum())
            clf   = build_xgb_clf(params=params, seed=seed, spw=neg / max(pos, 1))
            clf.fit(Xtr_s, ytr)

        probs          = clf.predict_proba(Xva_s)[:, 1]
        fold_threshold, _ = choose_threshold_max_f1(yva, probs)
        correct_mask   = (probs >= fold_threshold).astype(int) == yva
        if correct_mask.sum() == 0:
            continue
        any_correct = True

        selected_genes = np.array(gene_cols)[pf_pos_in_original[sel_mask]]
        Xtr_df         = pd.DataFrame(Xtr_s, columns=selected_genes)
        Xva_df         = pd.DataFrame(Xva_s, columns=selected_genes)
        X_correct      = Xva_df.loc[correct_mask].copy()
        if X_correct.shape[0] > SHAP_EVAL_MAX:
            X_correct = X_correct.sample(n=SHAP_EVAL_MAX, random_state=seed)
        bg_df = Xtr_df if Xtr_df.shape[0] <= SHAP_BG_MAX else Xtr_df.sample(n=SHAP_BG_MAX, random_state=seed)

        sv_correct = compute_shap_values_for_df(model_name=model_name, clf=clf, bg_df=bg_df, eval_df=X_correct)
        for idx, gene in enumerate(selected_genes):
            gene_label = make_gene_label(gene, gene_name_map)
            gene_abs_shap_values[gene_label].extend(np.abs(sv_correct[:, idx]).tolist())
            direction = infer_direction(X_correct[gene].values, sv_correct[:, idx])
            if isinstance(direction, str) and direction.strip():
                gene_direction_votes[gene_label].append(direction)

    if not any_correct:
        return pd.DataFrame(columns=["Gene", "Mean |SHAP| (OOF Correct Train)", "Direction"])

    rows = []
    for gene, shap_list in gene_abs_shap_values.items():
        direction_list  = [d for d in gene_direction_votes.get(gene, []) if d != "unclear"]
        final_direction = "unclear" if not direction_list else Counter(direction_list).most_common(1)[0][0]
        rows.append({"Gene": gene, "Mean |SHAP| (OOF Correct Train)": float(np.mean(shap_list)), "Direction": final_direction})

    df = (
        pd.DataFrame(rows)
        .sort_values(by=["Mean |SHAP| (OOF Correct Train)", "Gene"], ascending=[False, True])
        .head(top_k).reset_index(drop=True)
    )
    df["Mean |SHAP| (OOF Correct Train)"] = df["Mean |SHAP| (OOF Correct Train)"].round(4)
    return df


def aggregate_top_correct_oof_training_genes_across_outer_folds(run_store, model_name, gene_name_map, top_k=TOP_K):
    gene_fold_sets       = defaultdict(set)
    gene_direction_votes = defaultdict(list)
    gene_shap_values     = defaultdict(list)

    for (fold_id, mn), item in run_store.items():
        if mn != model_name:
            continue
        df_fold = get_top_correct_oof_training_genes_for_one_outer_fold(
            X_train=item["X_train"], y_train=item["y_train"], g_train=item["g_train"],
            gene_cols=item["gene_cols"], gene_name_map=gene_name_map,
            model_name=model_name, seed=item["fold_seed"], params=item["params"], top_k=top_k,
        )
        if df_fold.empty:
            continue
        for _, row in df_fold.iterrows():
            gene, direction, shap_mag = row["Gene"], row["Direction"], row["Mean |SHAP| (OOF Correct Train)"]
            gene_fold_sets[gene].add(fold_id)
            if isinstance(direction, str) and direction.strip():
                gene_direction_votes[gene].append(direction)
            if pd.notna(shap_mag):
                gene_shap_values[gene].append(float(shap_mag))

    total_folds = len({fold_id for fold_id, _ in run_store.keys()})
    rows = []
    for gene, fold_set in gene_fold_sets.items():
        direction_list  = [d for d in gene_direction_votes.get(gene, []) if d != "unclear"]
        final_direction = "unclear" if not direction_list else Counter(direction_list).most_common(1)[0][0]
        shap_list       = gene_shap_values.get(gene, [])
        rows.append({
            "Gene": gene,
            "Folds Appearing": len(fold_set),
            "Stability Score": len(fold_set) / max(total_folds, 1),
            "Mean |SHAP| (OOF Correct Train)": float(np.mean(shap_list)) if shap_list else np.nan,
            "Direction": final_direction,
        })

    if not rows:
        return pd.DataFrame(columns=["Gene", "Folds Appearing", "Stability Score", "Mean |SHAP| (OOF Correct Train)", "Direction"])

    agg_df = (
        pd.DataFrame(rows)
        .sort_values(by=["Folds Appearing", "Mean |SHAP| (OOF Correct Train)", "Gene"], ascending=[False, False, True])
        .head(top_k).reset_index(drop=True)
    )
    agg_df["Stability Score"]                 = agg_df["Stability Score"].round(3)
    agg_df["Mean |SHAP| (OOF Correct Train)"] = agg_df["Mean |SHAP| (OOF Correct Train)"].round(4)
    return agg_df


# =============================================================================
# Permutation sanity check
# =============================================================================
def run_permutation_sanity_check_on_outer_test_folds(outer_summaries):

    rows = []

    for i, outer_item in enumerate(outer_summaries[:N_PERMUTATION_OUTER_FOLDS], start=1):
        fold_id  = outer_item["outer_fold"]
        fold_seed = outer_item["fold_seed"]

        X_train = outer_item["X_train"]; y_train = outer_item["y_train"]
        g_train = outer_item["g_train"]; X_test  = outer_item["X_test"]
        y_test  = outer_item["y_test"]

        logger.info("Permutation check — outer fold %d (%d/%d)", fold_id, i, min(len(outer_summaries), N_PERMUTATION_OUTER_FOLDS))

        rng    = np.random.RandomState(fold_seed + 10000)
        y_perm = pd.Series(rng.permutation(y_train.values), index=y_train.index).astype(int)

        perm_fold_cache = build_inner_fold_cache(X_train, y_perm, g_train, fold_seed)

        for model_name in MODEL_NAMES:
            if model_name == "LR_L2":
                best_params, oof_pr, _, _, thr = tune_lr_for_pr_auc(
                    X_train, y_perm, g_train, seed=fold_seed, n_trials=PERM_N_TRIALS_LR, fold_cache=perm_fold_cache,
                )
            else:
                best_params, oof_pr, _, _, thr = tune_xgb_for_pr_auc(
                    X_train, y_perm, g_train, seed=fold_seed, n_trials=PERM_N_TRIALS_XGB, fold_cache=perm_fold_cache,
                )

            metrics, _, _, _, _, _ = evaluate_on_outer_test(
                X_train, X_test, y_perm, y_test,
                model_name=model_name, seed=fold_seed, params=best_params, threshold=thr,
            )

            rows.append({
                "Outer Fold":                         fold_id,
                "Model":                              model_name,
                "OOF PR-AUC (Permuted Train Labels)": float(oof_pr),
                "Held-Out Test PR-AUC":               float(metrics["PR-AUC (PRIMARY)"]),
                "AUC-ROC":                            float(metrics["AUC-ROC"]),
                "Brier Score":                        float(metrics["Brier Score"]),
                "F1":                                 float(metrics["F1"]),
                "Threshold":                          float(metrics["Threshold"]),
                "Features selected":                  int(metrics["Features selected"]),
            })

    return pd.DataFrame(rows)


# =============================================================================
# Notebook display helpers
# =============================================================================
def show_final_outputs_in_order(
    search_space_png, best_params_png,
    cm_lr_png, cm_xgb_png,
    metrics_png, permutation_png, per_fold_supp_png,
    pr_curves_side_png=None, shap_bar_side_png=None,
    beeswarm_side_png=None, correct_oof_side_png=None,
    output_dir=None,
):
    if not _is_jupyter():
        if output_dir:
            logger.info("Outputs saved to: %s", output_dir)
        return

    from IPython.display import display, Image
    ordered_paths = [
        ("1) Hyperparameter search space",                       search_space_png),
        ("2) Best hyperparameters selected for each outer fold", best_params_png),
        ("3A) Confusion matrices — Logistic Regression L2",      cm_lr_png),
        ("3B) Confusion matrices — XGBoost",                     cm_xgb_png),
        ("4) PR-AUC curves — LR L2 vs XGB",                      pr_curves_side_png),
        ("5) Final average classification metrics",              metrics_png),
        ("6) SHAP bar — LR L2 vs XGB",                           shap_bar_side_png),
        ("7) SHAP beeswarm — LR L2 vs XGB",                      beeswarm_side_png),
        ("8) Top 10 correct-OOF genes — LR L2 vs XGB",          correct_oof_side_png),
        ("9) Permutation-label sanity check (both models)",      permutation_png),
        ("10) Supplementary: per-fold held-out test metrics",    per_fold_supp_png),
    ]
    for title, path in ordered_paths:
        if path is None:
            continue
        logger.info("=== %s ===", title)
        if os.path.exists(path):
            display(Image(filename=path))
        else:
            logger.warning("Figure not available: %s", title)

    if output_dir:
        logger.info("=== OUTPUTS SAVED TO: %s ===", output_dir)


# =============================================================================
# Outer-fold preparation
# =============================================================================
def _prepare_outer_folds(outer_cv, X, y, groups, outer_is_stratified, logs_dir):

    import pickle

    outer_folds_pkl = os.path.join(logs_dir, "outer_folds.pkl")
    fold_meta_json  = os.path.join(logs_dir, "outer_folds_meta.json")

    X_np = X.values; y_np = y.values; g_np = groups.values

    current_meta = {
        "n_rows":            int(len(X)),
        "n_subjects":        int(groups.nunique()),
        "fingerprint":       dataset_fingerprint(X, groups, y),
        "n_outer_folds":     N_OUTER_FOLDS,
        "outer_master_seed": OUTER_MASTER_SEED,
        "cv_strategy":       "StratifiedGroupKFold" if outer_is_stratified else "GroupKFold",
    }

    if os.path.exists(outer_folds_pkl) and os.path.exists(fold_meta_json):
        with open(fold_meta_json, "r") as f:
            saved_meta = json.load(f)
        if saved_meta == current_meta:
            with open(outer_folds_pkl, "rb") as f:
                fold_indices = pickle.load(f)
            logger.info("Loaded cached outer fold indices from: %s", outer_folds_pkl)
        else:
            changed = {k: (saved_meta.get(k), current_meta[k]) for k in current_meta if saved_meta.get(k) != current_meta[k]}
            logger.warning("Fold metadata mismatch — regenerating. Changed: %s", changed)
            fold_indices = list(outer_cv.split(X_np, y_np, g_np))
            with open(outer_folds_pkl, "wb") as f: pickle.dump(fold_indices, f)
            with open(fold_meta_json,  "w") as f: json.dump(current_meta, f, indent=2)
    else:
        fold_indices = list(outer_cv.split(X_np, y_np, g_np))
        with open(outer_folds_pkl, "wb") as f: pickle.dump(fold_indices, f)
        with open(fold_meta_json,  "w") as f: json.dump(current_meta, f, indent=2)
        logger.info("Outer fold indices saved to: %s", outer_folds_pkl)

    return fold_indices, outer_folds_pkl, fold_meta_json


# =============================================================================
# Outer-fold execution
# =============================================================================
def _run_one_outer_split(outer_idx, tr_idx, te_idx, X, y, groups, gene_cols, gene_name_map):

    fold_seed = GLOBAL_SEED + outer_idx * 100

    logger.info("=" * 70)
    logger.info("OUTER SPLIT %d | seed=%d", outer_idx, fold_seed)
    logger.info("=" * 70)

    X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
    y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]
    g_train, g_test = groups.iloc[tr_idx], groups.iloc[te_idx]

    logger.info("Train: %d | Test: %d", X_train.shape[0], X_test.shape[0])
    train_counts = y_train.value_counts().to_dict(); test_counts = y_test.value_counts().to_dict()
    logger.info("Train labels: %s=%d, %s=%d", POS_CLASS_DISPLAY, train_counts.get(1, 0), NEG_CLASS_DISPLAY, train_counts.get(0, 0))
    logger.info("Test  labels: %s=%d, %s=%d", POS_CLASS_DISPLAY, test_counts.get(1, 0),  NEG_CLASS_DISPLAY, test_counts.get(0, 0))
    logger.info("Train subjects: %d | Test subjects: %d", g_train.nunique(), g_test.nunique())

    _log_fold_class_counts("outer-train", y_train.values, outer_idx)
    _log_fold_class_counts("outer-test",  y_test.values,  outer_idx)

    overlap = set(g_train.values) & set(g_test.values)
    if overlap:
        raise RuntimeError(f"Subject overlap in outer split {outer_idx}: {sorted(overlap)}")
    logger.info("Subject overlap check passed.")

    logger.info("Building inner fold cache for outer split %d ...", outer_idx)
    fold_cache = build_inner_fold_cache(X_train, y_train, g_train, fold_seed)
    usable     = sum(f is not None for f in fold_cache)
    logger.info("Cache built (%d/%d usable inner folds).", usable, len(fold_cache))

    metrics_list      = []
    cms_list          = []
    params_rows_list  = []
    run_store_entries = {}

    for model_name in MODEL_NAMES:
        logger.info("Running %s on outer split %d ...", get_model_display_name(model_name), outer_idx)

        if model_name == "LR_L2":
            best_params, best_oof_pr, oof_probs, contributed, thr = tune_lr_for_pr_auc(
                X_train, y_train, g_train, seed=fold_seed, n_trials=N_TRIALS_LR, fold_cache=fold_cache,
            )
        else:
            best_params, best_oof_pr, oof_probs, contributed, thr = tune_xgb_for_pr_auc(
                X_train, y_train, g_train, seed=fold_seed, n_trials=N_TRIALS_XGB, fold_cache=fold_cache,
            )

        metrics, cm, pipe, comb_mask, test_probs, test_preds = evaluate_on_outer_test(
            X_train, X_test, y_train, y_test,
            model_name=model_name, seed=fold_seed, params=best_params, threshold=thr,
        )
        metrics.update({
            "Model": model_name, "Outer Fold": outer_idx, "Fold Seed": fold_seed,
            "OOF PR-AUC": float(best_oof_pr), "Params": str(best_params),
            "OOF Coverage": float(contributed.mean()),
            "Test Subjects":     int(g_test.nunique()),
            "Pre-flare (n)":     int((y_test == 1).sum()),
            "Non-pre-flare (n)": int((y_test == 0).sum()),
        })
        metrics_list.append(metrics)
        cms_list.append(((outer_idx, model_name), cm))

        run_store_entries[(outer_idx, model_name)] = {
            "params": best_params, "threshold": thr, "pipe": pipe,
            "comb_mask": comb_mask, "test_probs": test_probs, "test_preds": test_preds,
            "cm": cm, "oof_pr": float(best_oof_pr), "oof_probs": oof_probs,
            "oof_contributed": contributed, "y_test": y_test,
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "g_train": g_train, "g_test": g_test,
            "gene_cols": gene_cols, "gene_name_map": gene_name_map,
            "fold_seed": fold_seed, "model_name": model_name,
        }

        row = {"Outer Fold": outer_idx, "Fold Seed": fold_seed}
        row.update(best_params)
        params_rows_list.append({"model_name": model_name, "row": row})

        logger.info(
            "%s | split %d: TEST PR-AUC=%.4f | OOF PR-AUC=%.4f | ROC-AUC=%.4f | F1=%.4f | coverage=%.1f%% | thr=%.3f",
            get_model_display_name(model_name), outer_idx,
            metrics["PR-AUC (PRIMARY)"], metrics["OOF PR-AUC"],
            metrics["AUC-ROC"], metrics["F1"], metrics["OOF Coverage"] * 100, thr,
        )

    outer_summary = {
        "outer_fold": outer_idx, "fold_seed": fold_seed,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "g_train": g_train, "g_test": g_test,
    }
    return metrics_list, cms_list, params_rows_list, run_store_entries, outer_summary


# =============================================================================
# SHAP output generation
# =============================================================================
def _generate_shap_outputs(run_store, gene_name_map, dirs):

    fold_ids = list(range(1, N_OUTER_FOLDS + 1))
    results  = {}

    for model_name in MODEL_NAMES:
        shap_df = aggregate_standard_shap_genes_across_outer_folds(
            run_store=run_store, model_name=model_name, gene_name_map=gene_name_map, top_k=TOP_K,
        )
        shap_csv = os.path.join(dirs["shap"], f"top{TOP_K}_standard_shap_{model_name}.csv")
        shap_df.to_csv(shap_csv, index=False)

        correct_df  = aggregate_top_correct_oof_training_genes_across_outer_folds(
            run_store=run_store, model_name=model_name, gene_name_map=gene_name_map, top_k=TOP_K_CORRECT_OOF,
        )
        correct_csv = os.path.join(dirs["shap"], f"top{TOP_K_CORRECT_OOF}_correct_oof_genes_{model_name}.csv")
        correct_df.to_csv(correct_csv, index=False)

        results[model_name] = {
            "shap_df": shap_df, "shap_csv": shap_csv,
            "correct_df": correct_df, "correct_csv": correct_csv,
        }

    lr_shap_df  = results["LR_L2"]["shap_df"];  xgb_shap_df = results["XGB"]["shap_df"]
    lr_correct  = results["LR_L2"]["correct_df"]; xgb_correct = results["XGB"]["correct_df"]
    lr_gene_labels  = lr_shap_df["Gene"].tolist()  if not lr_shap_df.empty  else []
    xgb_gene_labels = xgb_shap_df["Gene"].tolist() if not xgb_shap_df.empty else []

    pr_curves_side_png = os.path.join(dirs["figures"], "04_pr_auc_curves_side_by_side.png")
    plot_pr_auc_curves_side_by_side(run_store, fold_ids, pr_curves_side_png)

    shap_bar_side_png = os.path.join(dirs["figures"], "06_standard_shap_bar_side_by_side.png")
    plot_shap_bar_side_by_side(lr_shap_df, xgb_shap_df, shap_bar_side_png)

    beeswarm_side_png = os.path.join(dirs["figures"], "07_shap_beeswarm_side_by_side.png")
    if lr_gene_labels or xgb_gene_labels:
        plot_beeswarm_side_by_side(run_store, lr_gene_labels, xgb_gene_labels, beeswarm_side_png)
    else:
        beeswarm_side_png = None

    correct_oof_side_png = os.path.join(dirs["figures"], "08_correct_oof_genes_top10.png")
    plot_correct_oof_genes_side_by_side(lr_correct, xgb_correct, correct_oof_side_png)

    return {
        "pr_curves_side_png":   pr_curves_side_png,
        "shap_bar_side_png":    shap_bar_side_png,
        "beeswarm_side_png":    beeswarm_side_png,
        "correct_oof_side_png": correct_oof_side_png,
        "lr_shap_df":           lr_shap_df, "xgb_shap_df":    xgb_shap_df,
        "lr_correct_df":        lr_correct,  "xgb_correct_df": xgb_correct,
        "lr_shap_csv":          results["LR_L2"]["shap_csv"],
        "xgb_shap_csv":         results["XGB"]["shap_csv"],
        "lr_correct_csv":       results["LR_L2"]["correct_csv"],
        "xgb_correct_csv":      results["XGB"]["correct_csv"],
    }


# =============================================================================
# Summary output generation
# =============================================================================
def _export_summary_outputs(
    metrics_df, run_store, best_model_name, outer_summaries,
    best_params_rows, all_cms, gene_name_map,
    search_space_png, outer_is_stratified,
    outer_folds_pkl, fold_meta_json,
    X_shape, dirs, data_filepath, annotation_path,
):
    """Save summary tables/figures and display them in order."""
    lr_rows  = [x["row"] for x in best_params_rows if x["model_name"] == "LR_L2"]
    xgb_rows = [x["row"] for x in best_params_rows if x["model_name"] == "XGB"]

    lr_best_params_df  = pd.DataFrame(lr_rows)
    xgb_best_params_df = pd.DataFrame(xgb_rows)

    if not lr_best_params_df.empty:
        lr_best_params_df = lr_best_params_df.sort_values("Outer Fold").reset_index(drop=True)
    if not xgb_best_params_df.empty:
        xgb_best_params_df = xgb_best_params_df.sort_values("Outer Fold").reset_index(drop=True)
        xgb_best_params_df = round_numeric_df(xgb_best_params_df, decimals=3)

    lr_best_params_df.to_csv( os.path.join(dirs["tables"], "best_params_lr_l2.csv"),  index=False)
    xgb_best_params_df.to_csv(os.path.join(dirs["tables"], "best_params_xgboost.csv"), index=False)

    def _transpose_params_df(df):
        plot_df = df.drop(columns=["Fold Seed"], errors="ignore").set_index("Outer Fold").T.reset_index()
        plot_df.columns = ["Parameter"] + [f"Fold {int(c)}" for c in plot_df.columns[1:]]
        return plot_df

    best_params_png = os.path.join(dirs["figures"], "02_best_hyperparameters_side_by_side.png")
    plot_two_tables_side_by_side_with_gap(
        _transpose_params_df(lr_best_params_df),  "Logistic Regression L2",
        _transpose_params_df(xgb_best_params_df), "XGBoost",
        save_path=best_params_png,
        fig_title="Best Hyperparameters Selected for Each Outer Training Fold",
    )

    cm_lr_png  = os.path.join(dirs["figures"], "03_confusion_matrices_outer_test_lr_l2.png")
    cm_xgb_png = os.path.join(dirs["figures"], "03_confusion_matrices_outer_test_xgb.png")
    plot_confusion_matrices_one_model(all_cms, list(range(1, N_OUTER_FOLDS + 1)), "LR_L2", cm_lr_png)
    plot_confusion_matrices_one_model(all_cms, list(range(1, N_OUTER_FOLDS + 1)), "XGB",   cm_xgb_png)

    def _ms(s):
        s = pd.to_numeric(s, errors="coerce")
        mean_val, std_val, med_val = s.mean(skipna=True), s.std(skipna=True), s.median(skipna=True)
        if pd.isna(mean_val):
            return "NA"
        return f"{mean_val:.3f} ± {std_val if not pd.isna(std_val) else 0.0:.3f}\n({med_val:.3f})"

    summary = {}
    for m in MODEL_NAMES:
        sub    = metrics_df[metrics_df["Model"] == m]
        feat_s = pd.to_numeric(sub["Features selected"], errors="coerce")
        summary[m] = {
            "PR-AUC (PRIMARY)":  _ms(sub["PR-AUC (PRIMARY)"]),
            "AUC-ROC":           _ms(sub["AUC-ROC"]),
            "Sensitivity":       _ms(sub["Sensitivity"]),
            "Precision":         _ms(sub["Precision"]),
            "Specificity":       _ms(sub["Specificity"]),
            "F1":                _ms(sub["F1"]),
            "Balanced Accuracy": _ms(sub["Balanced Accuracy"]),
            "OOF PR-AUC":        _ms(sub["OOF PR-AUC"]),
            "Brier Score":       _ms(sub["Brier Score"]),
            "Features selected": f"{feat_s.mean(skipna=True):.3f} ± {feat_s.std(skipna=True) if not pd.isna(feat_s.std(skipna=True)) else 0.0:.3f}\n({feat_s.median(skipna=True):.3f})",
            "Threshold":         _ms(sub["Threshold"]),
        }

    summary_df = pd.DataFrame(summary).T
    summary_df.index = [get_model_display_name(idx) for idx in summary_df.index]

    logger.info("=== SUMMARY — Held-Out Test Set (mean ± std [median] across %d outer folds) ===", N_OUTER_FOLDS)
    logger.info("\n%s", summary_df.to_string())

    mean_test_pr_auc = metrics_df.groupby("Model")["PR-AUC (PRIMARY)"].mean()
    mean_oof         = metrics_df.groupby("Model")["OOF PR-AUC"].mean()
    logger.info("=== MODEL PERFORMANCE SUMMARY ===")
    for m in MODEL_NAMES:
        logger.info("  %s: held-out PR-AUC=%.4f | inner OOF PR-AUC=%.4f", get_model_display_name(m), mean_test_pr_auc[m], mean_oof[m])
    logger.info("  Best model (held-out PR-AUC): %s", get_model_display_name(best_model_name))

    metrics_png = os.path.join(dirs["figures"], "05_final_average_classification_metrics.png")
    plot_metrics_table(summary_df, metrics_png)
    summary_df.to_csv(os.path.join(dirs["tables"], "average_metrics_summary.csv"), index=True)

    shap_paths = _generate_shap_outputs(run_store, gene_name_map, dirs)

    if RUN_PERMUTATION_CHECK:
        permutation_df  = run_permutation_sanity_check_on_outer_test_folds(outer_summaries)
        permutation_csv = os.path.join(dirs["tables"],  "permutation_sanity_check.csv")
        permutation_png = os.path.join(dirs["figures"], "09_permutation_sanity_check_both_models.png")
        permutation_df.to_csv(permutation_csv, index=False)
        plot_permutation_table_top_bottom(permutation_df, permutation_png)
    else:
        permutation_df = permutation_csv = permutation_png = None

    supp_cols = [
        "Outer Fold", "Model", "Test Subjects", "Pre-flare (n)", "Non-pre-flare (n)",
        "PR-AUC (PRIMARY)", "OOF PR-AUC", "AUC-ROC", "Brier Score",
        "F1", "Sensitivity", "Specificity", "Precision",
        "Balanced Accuracy", "Threshold", "Features selected",
    ]
    supp_df = metrics_df[[c for c in supp_cols if c in metrics_df.columns]].copy()
    supp_df = supp_df.sort_values(["Model", "Outer Fold"]).reset_index(drop=True)
    supp_df["Model"] = supp_df["Model"].map(get_model_display_name)
    for col in [c for c in supp_df.columns if c not in ("Outer Fold", "Model", "Test Subjects", "Pre-flare (n)", "Non-pre-flare (n)")]:
        supp_df[col] = pd.to_numeric(supp_df[col], errors="coerce").round(3)

    per_fold_supp_png = os.path.join(dirs["figures"], "10_supplementary_per_fold_test_metrics.png")
    plot_per_fold_metrics_side_by_side(supp_df, save_path=per_fold_supp_png)

    output_dir = os.path.dirname(dirs["figures"])
    manifest   = {
        "generated_at": datetime.datetime.now().isoformat(),
        "environment": {
            "python_version": sys.version, "platform": platform.platform(),
            "scikit_learn": sklearn.__version__, "xgboost": xgb.__version__,
            "shap": shap.__version__, "optuna": optuna.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
        },
        "config": {
            "GLOBAL_SEED": GLOBAL_SEED, "DATA_PATH": data_filepath,
            "N_OUTER_FOLDS": N_OUTER_FOLDS, "N_INNER_FOLDS": N_INNER_FOLDS,
            "VAR_THRESH": VAR_THRESH, "PREFILTER_K": PREFILTER_K,
            "TOP_K": TOP_K, "TOP_K_CORRECT_OOF": TOP_K_CORRECT_OOF,
            "N_TRIALS_LR": N_TRIALS_LR, "N_TRIALS_XGB": N_TRIALS_XGB,
            "XGB_TREE": XGB_TREE, "SHAP_BG_MAX": SHAP_BG_MAX, "SHAP_EVAL_MAX": SHAP_EVAL_MAX,
            "RUN_PERMUTATION_CHECK": RUN_PERMUTATION_CHECK,
            "N_PERMUTATION_OUTER_FOLDS": N_PERMUTATION_OUTER_FOLDS,
            "PERM_N_TRIALS_LR": PERM_N_TRIALS_LR, "PERM_N_TRIALS_XGB": PERM_N_TRIALS_XGB,
            "REQUIRE_STRATIFIED_GROUP_KFOLD": REQUIRE_STRATIFIED_GROUP_KFOLD,
            "MIN_OOF_COVERAGE_WARN": MIN_OOF_COVERAGE_WARN,
            "MIN_FALLBACK_FEATURES": MIN_FALLBACK_FEATURES,
        },
        "dataset": {
            "n_samples": X_shape[0], "n_features": X_shape[1],
            "data_path": data_filepath,
            "class_label_mapping": {"0": NEG_CLASS_DISPLAY, "1": POS_CLASS_DISPLAY},
        },
        "outer_cv_strategy":              "StratifiedGroupKFold" if outer_is_stratified else "GroupKFold",
        "threshold_method":               "max-F1 from best trial's combined inner OOF predictions — no access to outer test set",
        "pruning":                        "MedianPruner active for XGB only; LR omits pruning (fast objective, overhead not justified)",
        "early_stopping":                 "not implemented — n_estimators tuned by Optuna",
        "l1_fallback":                    f"top-{MIN_FALLBACK_FEATURES} by |coef|",
        "highest_mean_pr_auc_model":      best_model_name,
        "annotation_path":                annotation_path,
        "outputs": {
            "search_space_png":           search_space_png,
            "best_params_png":            best_params_png,
            "cm_lr_png":                  cm_lr_png,
            "cm_xgb_png":                 cm_xgb_png,
            "pr_auc_curves_png":          shap_paths["pr_curves_side_png"],
            "metrics_png":                metrics_png,
            "shap_bar_png":               shap_paths["shap_bar_side_png"],
            "shap_beeswarm_png":          shap_paths["beeswarm_side_png"],
            "correct_oof_genes_png":      shap_paths["correct_oof_side_png"],
            "permutation_png":            permutation_png,
            "per_fold_supp_png":          per_fold_supp_png,
            "lr_shap_csv":                shap_paths["lr_shap_csv"],
            "xgb_shap_csv":               shap_paths["xgb_shap_csv"],
            "lr_correct_oof_csv":         shap_paths["lr_correct_csv"],
            "xgb_correct_oof_csv":        shap_paths["xgb_correct_csv"],
            "permutation_csv":            permutation_csv,
            "outer_folds_pkl":            outer_folds_pkl,
            "outer_folds_meta_json":      fold_meta_json,
            "fold_assignments_csv":       os.path.join(dirs["tables"], "outer_fold_membership_all_splits.csv"),
            "fold_test_assignments_csv":  os.path.join(dirs["tables"], "outer_fold_test_assignments.csv"),
        },
    }
    with open(os.path.join(dirs["logs"], "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    show_final_outputs_in_order(
        search_space_png=search_space_png, best_params_png=best_params_png,
        cm_lr_png=cm_lr_png, cm_xgb_png=cm_xgb_png,
        metrics_png=metrics_png, permutation_png=permutation_png,
        per_fold_supp_png=per_fold_supp_png,
        pr_curves_side_png=shap_paths["pr_curves_side_png"],
        shap_bar_side_png=shap_paths["shap_bar_side_png"],
        beeswarm_side_png=shap_paths["beeswarm_side_png"],
        correct_oof_side_png=shap_paths["correct_oof_side_png"],
        output_dir=output_dir,
    )

    logger.info("Done. Outputs saved to: %s", output_dir)

    return {
        "metrics_df":                metrics_df,
        "summary_df":                summary_df,
        "lr_best_params_df":         lr_best_params_df,
        "xgb_best_params_df":        xgb_best_params_df,
        "permutation_df":            permutation_df,
        "outer_cv_strategy":         "StratifiedGroupKFold" if outer_is_stratified else "GroupKFold",
        "highest_mean_pr_auc_model": best_model_name,
        "annotation_path":           annotation_path,
    }


# =============================================================================
# Main entry point
# =============================================================================
def run_pipeline(data_filepath: str, output_dir: str = OUTPUT_DIR):
    """
    Execute the full nested cross-validation pipeline and return a results dict.
    All outputs are written to output_dir/{figures,tables,shap,logs}/.
    """
    sns.set_theme(style="white", font_scale=1.4)
    plt.rcParams.update({
        "font.size":        TICK_LABEL_SIZE,
        "axes.titlesize":   SUBPLOT_TITLE_SIZE,
        "axes.labelsize":   AXIS_LABEL_SIZE,
        "xtick.labelsize":  TICK_LABEL_SIZE,
        "ytick.labelsize":  TICK_LABEL_SIZE,
        "legend.fontsize":  LEGEND_SIZE,
        "figure.titlesize": FIG_TITLE_SIZE,
    })

    seed_everything(GLOBAL_SEED)
    dirs = _make_output_dirs(output_dir)

    import time
    for dir_path in dirs.values():
        test_file = os.path.join(dir_path, '.write_test')
        for attempt in range(12):
            try:
                with open(test_file, 'w') as f:
                    f.write('ok')
                os.remove(test_file)
                break
            except OSError:
                if attempt == 11:
                    raise RuntimeError(
                        f'Output directory not writable after 60s: {dir_path}\n'
                        f'Check that the output directory exists and is writable. '
                        f'If using Google Drive, make sure Drive is fully mounted and synced.'
                    )
                logger.info('Waiting for output directory to become writable... (%ds)', (attempt + 1) * 5)
                time.sleep(5)

    outer_cv, outer_is_stratified = make_outer_cv()
    logger.info("Outer CV: %s", "StratifiedGroupKFold" if outer_is_stratified else "GroupKFold")

    annotation_path = os.path.join(dirs["logs"], "GPL10558_probe_to_gene_symbol.csv")
    if not os.path.exists(annotation_path):
        logger.info("Annotation file not found — downloading (one-time setup).")
        download_gpl_annotation(annotation_path)
    else:
        logger.info("Annotation file found: %s", annotation_path)

    X, y, groups, gene_cols, sample_ids = load_data(data_filepath)
    gene_name_map = load_gene_name_map(gene_cols, annotation_path)

    search_spaces    = get_search_space_tables()
    search_space_png = os.path.join(dirs["figures"], "01_hyperparameter_search_space_tables.png")
    plot_two_tables_side_by_side_with_gap(
        search_spaces["LR_L2"], "Logistic Regression L2",
        search_spaces["XGB"],   "XGBoost",
        save_path=search_space_png,
        fig_title="Hyperparameter Search Space",
    )

    fold_indices, outer_folds_pkl, fold_meta_json = _prepare_outer_folds(
        outer_cv, X, y, groups, outer_is_stratified, dirs["logs"],
    )

    fold_assignment_rows = []
    test_assignment_rows = []
    for outer_idx, (tr_idx, te_idx) in enumerate(fold_indices, start=1):
        for split_label, idx_arr in [("train", tr_idx), ("test", te_idx)]:
            for i in idx_arr:
                fold_assignment_rows.append({
                    "sample_index": sample_ids[i],
                    "subject": str(groups.iloc[i]),
                    "label": int(y.iloc[i]),
                    "outer_fold": outer_idx,
                    "split": split_label,
                })
    pd.DataFrame(fold_assignment_rows).to_csv(
        os.path.join(dirs["tables"], "outer_fold_membership_all_splits.csv"), index=False,
    )

    for outer_idx, (_, te_idx) in enumerate(fold_indices, start=1):
        for i in te_idx:
            test_assignment_rows.append({
                "sample_index": sample_ids[i],
                "subject": str(groups.iloc[i]),
                "label": int(y.iloc[i]),
                "outer_fold_test": outer_idx,
            })
    pd.DataFrame(test_assignment_rows).to_csv(
        os.path.join(dirs["tables"], "outer_fold_test_assignments.csv"), index=False,
    )

    all_metrics      = []
    all_cms          = {}
    best_params_rows = []
    run_store        = {}
    outer_summaries  = []

    for outer_idx, (tr_idx, te_idx) in enumerate(fold_indices, start=1):
        (metrics_list, cms_list, params_rows_list,
         run_store_entries, outer_summary) = _run_one_outer_split(
            outer_idx, tr_idx, te_idx, X, y, groups, gene_cols, gene_name_map,
        )
        all_metrics.extend(metrics_list)
        all_cms.update(dict(cms_list))
        best_params_rows.extend(params_rows_list)
        run_store.update(run_store_entries)
        outer_summaries.append(outer_summary)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(dirs["tables"], "metrics_outer_test_folds.csv"), index=False)

    low_coverage = metrics_df[metrics_df["OOF Coverage"] < MIN_OOF_COVERAGE_WARN]
    if not low_coverage.empty:
        logger.warning(
            "Low OOF coverage in %d fold(s):\n%s",
            len(low_coverage),
            low_coverage[["Model", "Outer Fold", "OOF Coverage"]].to_string(index=False),
        )
    else:
        logger.info("OOF coverage check passed: all folds >= %.0f%%", MIN_OOF_COVERAGE_WARN * 100)

    mean_test_pr_auc = metrics_df.groupby("Model")["PR-AUC (PRIMARY)"].mean()
    best_model_name  = mean_test_pr_auc.sort_values(ascending=False).index[0]

    return _export_summary_outputs(
        metrics_df, run_store, best_model_name, outer_summaries,
        best_params_rows, all_cms, gene_name_map,
        search_space_png, outer_is_stratified,
        outer_folds_pkl, fold_meta_json,
        X_shape=X.shape, dirs=dirs,
        data_filepath=data_filepath,
        annotation_path=annotation_path,
    )


def _is_jupyter() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def main():
    if _is_jupyter():
        results = run_pipeline(DATA_PATH, output_dir=OUTPUT_DIR)
    else:
        import argparse
        parser = argparse.ArgumentParser(
            description="Nested CV pipeline for lupus pre-flare prediction (GSE65391)."
        )
        parser.add_argument("--data", type=str, default=DATA_PATH,
                            help="Path to lupus_final_df.pkl (default: %(default)s)")
        parser.add_argument("--out",  type=str, default=OUTPUT_DIR,
                            help="Root output directory (default: %(default)s)")
        args = parser.parse_args()
        results = run_pipeline(args.data, output_dir=args.out)
    return results


if __name__ == "__main__":
    main()
