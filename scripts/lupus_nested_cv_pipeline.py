# =========================================================
# Author: Aaron Choi
# Project: Pediatric Lupus Flare Prediction (GSE65391)
#
# Nested cross-validation pipeline for predicting pre-flare
# states in pediatric SLE using gene-expression data,
#
# Main components:
# - subject-level grouped nested cross-validation
# - leakage-safe feature selection and hyperparameter tuning
# - Logistic Regression and XGBoost gene-expression models
# - prevalence baseline and SLEDAI-only logistic regression comparators
# - SHAP analysis for gene-expression models
# - permutation-label sanity checks
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

plt.ioff()  # disable automatic matplotlib rendering during figure creation


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
N_TRIALS_XGB = 80

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

MODEL_NAMES = ["LR_L2", "XGB"]  # main gene-expression models
COMPARATOR_MODEL_NAMES = ["BASELINE", "SLEDAI_ONLY"]  # comparator models
ALL_REPORT_MODEL_NAMES = MODEL_NAMES + COMPARATOR_MODEL_NAMES

MODEL_DISPLAY_NAMES = {
    "LR_L2":       "Logistic Regression L2",
    "XGB":         "XGBoost",
    "BASELINE":    "Prevalence baseline",
    "SLEDAI_ONLY": "SLEDAI-only",
}

GPL10558_FULL_URL = (
    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
    "?acc=GPL10558&targ=self&form=text&view=full"
)

REQUIRE_STRATIFIED_GROUP_KFOLD = True
MIN_OOF_COVERAGE_WARN          = 0.80
MIN_FALLBACK_FEATURES          = 10

TABLE_BODY_FONTSIZE   = 20
TABLE_HEADER_FONTSIZE = 20
TABLE_TITLE_FONTSIZE  = 32

FIG_TITLE_SIZE     = 42
SUBPLOT_TITLE_SIZE = 28
AXIS_LABEL_SIZE    = 22
TICK_LABEL_SIZE    = 20
VALUE_LABEL_SIZE   = 18
LEGEND_SIZE        = 16

ALLOW_ANNOTATION_DOWNLOAD = os.environ.get("LUPUS_ALLOW_ANNOTATION_DOWNLOAD", "1") == "1"
ANNOTATION_PATH = os.environ.get("LUPUS_ANNOTATION_PATH", "")

DISPLAY_FIGURES_IN_NOTEBOOK = False # If True, display figures inline in Jupyter/Colab


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


def make_run_output_dir(base_output_dir: str) -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def make_cache_dir(base_output_dir: str) -> str:
    cache_dir = os.path.join(base_output_dir, "_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def make_and_save_figure(fig_path: str | None, plot_func, *args, **kwargs):
    """
    Call a plotting function that saves its own figure to disk.
    This wrapper only checks whether the expected output file exists.
    It does not display figures here.
    """
    plot_func(*args, **kwargs)
    if fig_path is not None and os.path.exists(fig_path):
        logger.info("Saved figure: %s", fig_path)
        return fig_path
    logger.warning("Expected figure was not created: %s", fig_path)
    return None


# Basic helper functions
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


def round_shap_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].round(4)
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
# Model-building helper functions
# =============================================================================

def build_lr_selector(C: float, seed: int) -> LogisticRegression:
    """L1 logistic regression for feature selection."""
    return LogisticRegression(
        penalty="l1", solver="liblinear", C=C,
        max_iter=2000, random_state=seed, class_weight="balanced",
    )


def build_lr_clf(C: float, seed: int) -> LogisticRegression:
    """L2 logistic regression for the final model."""
    return LogisticRegression(
        penalty="l2", solver="lbfgs", C=C,
        max_iter=5000, random_state=seed, class_weight="balanced",
    )


def build_xgb_selector(seed: int, spw: float) -> xgb.XGBClassifier:
    """XGBoost model for feature ranking before final fitting."""
    return xgb.XGBClassifier(
        n_estimators=60, max_depth=2, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
        min_child_weight=5.0, scale_pos_weight=spw,
        tree_method=XGB_TREE, deterministic_histogram=True,
        eval_metric="aucpr", random_state=seed,
        n_jobs=XGB_NJOBS, verbosity=0,
    )


def build_xgb_clf(params: dict, seed: int, spw: float) -> xgb.XGBClassifier:
    """Build the final XGBoost model with tuned parameters."""
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


# Hyperparameter search-space tables
def get_search_space_tables():
    """Create tables showing the Optuna search space for LR_L2 and XGB."""
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
            "int, [2, 4]",
            "float, log scale, [0.01, 0.20]",
            "float, [0.6, 1.0]",
            "float, [0.6, 1.0]",
            "float, [10.0, 30.0]",
            "float, [0.0, 5.0]",
            "float, log scale, [1.0, 100.0]",
            "float, log scale, [1e-6, 10.0]",
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


def load_raw_dataframe(filepath: str) -> pd.DataFrame:
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
    return df


def load_data(df: pd.DataFrame):
    """
    Load the processed lupus dataset from a dataframe.

    Returns X, y, groups, gene_cols, and sample_ids.
    Raises ValueError if required columns are missing.
    """

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

    sample_ids = X.index.astype(str).tolist()  # keep original sample IDs for fold export

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


def load_sledai_feature(df: pd.DataFrame, candidate_cols=None, preferred_col=None):
    """
    Load one SLEDAI column from the dataset.

    The function first checks exact candidate column names.
    If none are found, it looks for other numeric columns
    whose names contain 'sledai'. If more than one possible
    match is found, preferred_col must be provided.
    """
    if candidate_cols is None:
        candidate_cols = [
            "SLEDAI",
            "sledai",
            "SLEDAI_score",
            "sledai_score",
            "current_SLEDAI",
            "current_sledai",
            "SLEDAI_current",
        ]

    forbidden_tokens = [
        "next", "future", "follow", "delta", "change", "diff",
        "increase", "decrease", "post", "outcome", "label", "target",
    ]

    if "preflare_bool" not in df.columns or "subject" not in df.columns:
        raise ValueError("Missing required columns: need 'preflare_bool' and 'subject'")

    non_ilmn_cols = [c for c in df.columns if not str(c).startswith("ILMN_")]

    def _is_numeric_like(series: pd.Series, min_non_na_fraction: float = 0.80) -> bool:
        converted = pd.to_numeric(series, errors="coerce")
        non_na_fraction = converted.notna().mean()
        return non_na_fraction >= min_non_na_fraction

    def _looks_forbidden(colname: str) -> bool:
        c = str(colname).lower()
        return any(tok in c for tok in forbidden_tokens)

    found = None

    if preferred_col is not None:
        if preferred_col not in df.columns:
            raise ValueError(
                f"preferred_col='{preferred_col}' was requested, but that column does not exist.\n"
                f"Available non-ILMN columns: {non_ilmn_cols}"
            )
        if _looks_forbidden(preferred_col):
            raise ValueError(
                f"preferred_col='{preferred_col}' looks like a future/derived column and was rejected."
            )
        if not _is_numeric_like(df[preferred_col]):
            raise ValueError(
                f"preferred_col='{preferred_col}' exists, but it is not sufficiently numeric."
            )
        found = preferred_col

    if found is None:
        for col in candidate_cols:
            if col in df.columns and not _looks_forbidden(col):
                if _is_numeric_like(df[col]):
                    found = col
                    break

    if found is None:
        fuzzy_hits = [c for c in non_ilmn_cols if "sledai" in str(c).lower() and not _looks_forbidden(c)]
        numeric_fuzzy_hits = [c for c in fuzzy_hits if _is_numeric_like(df[c])]

        if len(numeric_fuzzy_hits) == 1:
            found = numeric_fuzzy_hits[0]
            logger.info("Using fuzzy-matched numeric SLEDAI column: %s", found)
        elif len(numeric_fuzzy_hits) > 1:
            raise ValueError(
                "Multiple possible numeric SLEDAI columns found. Please specify preferred_col.\n"
                f"Candidates: {numeric_fuzzy_hits}"
            )

    if found is None:
        logger.warning(
            "No valid SLEDAI column found — SLEDAI-only comparator will be skipped."
        )
        return None, None, None, None

    X_sledai = df[[found]].copy()
    y = df["preflare_bool"].astype(int)
    groups = df["subject"].astype(str)

    if (not pd.api.types.is_integer_dtype(X_sledai.index)) or X_sledai.index.duplicated().any():
        X_sledai = X_sledai.reset_index(drop=True)
        y = y.reset_index(drop=True)
        groups = groups.reset_index(drop=True)

    X_sledai[found] = pd.to_numeric(X_sledai[found], errors="coerce")

    logger.info("Loaded SLEDAI-only feature column: %s", found)

    missing_frac = X_sledai[found].isna().mean()
    if missing_frac > 0:
        logger.warning(
            "SLEDAI column '%s' contains %.1f%% missing values; median imputation will be applied within folds.",
            found,
            missing_frac * 100,
        )

    return X_sledai.astype(np.float32), y, groups, found


# Probe annotation helpers
def download_gpl_annotation(save_csv_path: str):
    """
    Download the GPL10558 probe-to-gene annotation from GEO and save it as a CSV file.

    Notes:
    - This is optional and only used when LUPUS_ALLOW_ANNOTATION_DOWNLOAD=1.
    - Requires internet access and the 'requests' package to be installed.
    """
    try:
        import requests
    except ImportError as e:
        raise RuntimeError(
            "Optional annotation download requested, but the 'requests' package is not installed."
        ) from e

    if os.path.exists(save_csv_path):
        logger.info("Annotation file already exists: %s", save_csv_path)
        return save_csv_path

    logger.info("Downloading GPL10558 annotation from GEO...")
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

    probe_col = next(
        (cols_lower[c] for c in ["id", "probe_id", "probeid", "ilmn_id", "ilmnid", "array_address_id", "probe"] if c in cols_lower),
        None,
    )
    gene_col = next(
        (cols_lower[c] for c in ["symbol", "gene symbol", "gene_symbol", "genesymbol", "gene", "gene_name", "symbol_interpreted"] if c in cols_lower),
        None,
    )

    if probe_col is None:
        raise RuntimeError(f"Could not identify probe column. Columns: {gpl_df.columns.tolist()}")
    if gene_col is None:
        fallback = [c for c in gpl_df.columns if "symbol" in c.lower() or "gene" in c.lower()]
        if fallback:
            gene_col = fallback[0]
        else:
            raise RuntimeError(f"Could not identify gene-name column. Columns: {gpl_df.columns.tolist()}")

    out = gpl_df[[probe_col, gene_col]].copy()
    out.columns      = ["probe_id", "gene_name"]
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
        logger.warning("Annotation file missing. Using probe IDs only.")
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
    """Build a hash so cached outer folds can be checked against the current dataset."""
    import hashlib
    payload = pd.DataFrame({
        "row_index": X.index.astype(str),
        "subject":   groups.astype(str).values,
        "label":     y.astype(int).values,
    })
    header = "|".join(map(str, X.columns))
    txt    = header + "\n" + payload.to_csv(index=False)
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


# Cross-validation setup helpers
def make_outer_cv():
    """Create the outer cross-validation splitter, using StratifiedGroupKFold when available."""
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
        "StratifiedGroupKFold is not available. Falling back to GroupKFold "
        "(no shuffle and weaker class balance control)."
    )
    return GroupKFold(n_splits=N_OUTER_FOLDS), False


# Feature-selection helpers
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
    Run L1 feature selection.

    If all coefficients are zero, keep the features with the
    largest absolute coefficients instead.
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


def check_inner_cv_feasibility(y_train: pd.Series, g_train: pd.Series, n_splits: int, context: str = "") -> None:
    """
    Make sure the current outer-training split can support grouped inner CV.
    """
    n_subjects   = g_train.nunique()
    pos_subjects = g_train[y_train == 1].nunique()
    neg_subjects = g_train[y_train == 0].nunique()

    errors = []
    if n_subjects < n_splits:
        errors.append(f"subjects={n_subjects} < n_splits={n_splits}")
    if pos_subjects < n_splits:
        errors.append(f"pre-flare subjects={pos_subjects} < n_splits={n_splits}")
    if neg_subjects < n_splits:
        errors.append(f"non-pre-flare subjects={neg_subjects} < n_splits={n_splits}")

    if errors:
        prefix = f"{context}: " if context else ""
        raise ValueError(
            prefix
            + "Inner CV infeasible for this outer-training split. "
            + "; ".join(errors)
            + ". Reduce N_INNER_FOLDS or use a fallback CV strategy."
        )


# Inner-fold checks, caching, and scoring
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
    Build cached inner-fold objects after variance filtering,
    univariate prefiltering, and scaling.

    Returns a list of per-fold dictionaries. If an inner split
    has only one class, that fold is stored as None.
    """
    cv, _ = _make_inner_cv(seed)

    X_np = X_train.values
    y_np = y_train.values
    g_np = g_train.values

    try:
        split_list = list(cv.split(X_np, y_np, g_np))
    except Exception as e:
        logger.warning("Failed to create inner CV splits (seed=%d): %s", seed, e)
        return []

    fold_cache = []
    n_usable   = 0

    for fold_idx, (tr_idx, va_idx) in enumerate(split_list):
        ytr = y_np[tr_idx]
        yva = y_np[va_idx]

        if not (
            _log_fold_class_counts("inner-train", ytr, fold_idx)
            and _log_fold_class_counts("inner-val", yva, fold_idx)
        ):
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
            "fold_idx": fold_idx,
            "tr_idx": tr_idx,
            "va_idx": va_idx,
            "Xtr_pf": Xtr_pf,
            "Xva_pf": Xva_pf,
            "Xtr_pf_scaled": Xtr_pf_scaled,
            "Xva_pf_scaled": Xva_pf_scaled,
            "scaler_pf": scaler_pf,
            "ytr": ytr,
            "yva": yva,
            "pf_pos_in_original": pf_pos_in_original,
            "pf_support": pf_support,
            "vt_support": vt_support,
        })
        n_usable += 1

    logger.info("Inner fold cache built: %d/%d usable (seed=%d)", n_usable, len(fold_cache), seed)
    return fold_cache


def cv_oof_probs_pr_auc_from_cache(fold_cache, y_train, model_name, seed, params, trial=None):
    """
    Get inner-fold OOF predicted probabilities from cached folds.

    Returns:
        oof_probs, oof_pr_auc, contributed_mask

    If an Optuna trial is provided, pruning can be applied during XGBoost tuning.
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
            "Low inner OOF coverage for %s at seed=%d: %d/%d rows (%.1f%% covered).",
            model_name, seed, contributed.sum(), len(contributed), coverage * 100,
        )

    if contributed.sum() == 0:
        return oof_probs, 0.0, contributed

    oof_pr = average_precision_score(y_np[contributed], oof_probs[contributed])
    return oof_probs, float(oof_pr), contributed


def tune_lr_for_pr_auc(X_train, y_train, g_train, seed, n_trials, fold_cache=None):
    """
    Tune LR_L2 using inner-CV OOF PR-AUC.

    Returns:
        best_params, best_pr_auc, best_oof_probs, best_contributed, best_threshold
    """
    if fold_cache is None:
        fold_cache = build_inner_fold_cache(X_train, y_train, g_train, seed)

    if len(fold_cache) == 0 or all(f is None for f in fold_cache):
        raise ValueError("LR_L2 tuning failed: no usable inner folds available.")

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

    # Threshold is selected from inner OOF predictions by maximizing F1.
    # This threshold is then fixed before evaluation on the outer test fold.
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
    Tune XGBoost using inner-CV OOF PR-AUC.

    Returns:
        best_params, best_pr_auc, best_oof_probs, best_contributed, best_threshold
    """
    if fold_cache is None:
        fold_cache = build_inner_fold_cache(X_train, y_train, g_train, seed)

    if len(fold_cache) == 0 or all(f is None for f in fold_cache):
        raise ValueError("XGB tuning failed: no usable inner folds available.")

    # Use MedianPruner for XGBoost trials only.
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=8, n_warmup_steps=2),
    )

    def objective(trial):
        p = {
            "xgb_sel_topk":     trial.suggest_int("xgb_sel_topk", 10, 60, step=10),
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1500, step=50),
            "max_depth":        trial.suggest_int("max_depth", 2, 4),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 10.0, 30.0),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1.0, 100.0, log=True),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
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

    # Threshold is selected from inner OOF predictions by maximizing F1.
    # This threshold is then fixed before evaluation on the outer test fold.
    if best_contributed.sum() >= 2 and len(np.unique(y_np[best_contributed])) >= 2:
        best_threshold, _ = choose_threshold_max_f1(y_np[best_contributed], best_oof_probs[best_contributed])
    else:
        best_threshold = 0.5

    return best_params, float(study.best_value), best_oof_probs, best_contributed, best_threshold


def fit_final_model_on_full_train(X_train, y_train, model_name, seed, params):
    """Fit the final model on the full outer-train split."""
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
    """Fit the model on the outer-training split and evaluate it on the outer-test split."""
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


def compute_binary_metrics_from_probs(y_true, probs, threshold, n_features=0):
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)

    pr_auc = average_precision_score(y_true, probs) if len(np.unique(y_true)) >= 2 else np.nan
    if len(np.unique(y_true)) >= 2:
        aucroc = roc_auc_score(y_true, probs)
    else:
        aucroc = np.nan
    brier = brier_score_loss(y_true, probs)
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])

    metrics = {
        "PR-AUC (PRIMARY)":  pr_auc,
        "AUC-ROC":           aucroc,
        "Sensitivity":       recall_score(y_true, preds, zero_division=0),
        "Precision":         precision_score(y_true, preds, zero_division=0),
        "Specificity":       recall_score(y_true, preds, pos_label=0, zero_division=0),
        "F1":                f1_score(y_true, preds, zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(y_true, preds),
        "Brier Score":       brier,
        "Threshold":         float(threshold),
        "Features selected": int(n_features),
    }
    return metrics, cm, preds


def evaluate_baseline_on_outer_test(y_train, y_test):
    """
    Constant-probability prevalence baseline.

    Each test sample receives the same predicted probability, equal to the
    pre-flare prevalence in the outer-training split. This is a deterministic
    prevalence baseline, not a random classifier.
    """
    ytr = np.asarray(y_train).astype(int)
    yte = np.asarray(y_test).astype(int)

    train_prevalence = float(np.mean(ytr))
    probs = np.full(shape=len(yte), fill_value=train_prevalence, dtype=float)

    threshold = 0.5
    metrics, cm, preds = compute_binary_metrics_from_probs(
        y_true=yte,
        probs=probs,
        threshold=threshold,
        n_features=0,
    )

    return metrics, cm, probs, preds, train_prevalence


def median_impute_train_test(train_df, test_df):
    """Median-impute train and test data using medians from the training split only."""
    train_df = train_df.copy()
    test_df = test_df.copy()
    medians = train_df.median(axis=0)
    train_df = train_df.fillna(medians)
    test_df = test_df.fillna(medians)
    train_df = train_df.fillna(0.0)
    test_df = test_df.fillna(0.0)
    return train_df, test_df, medians


def tune_sledai_only_model(X_train_sledai, y_train, g_train, seed, c_grid=None):
    """
    Tune a logistic regression model using only the SLEDAI feature with grouped inner CV.

    Returns:
        best_C, best_oof_probs, best_contributed, best_threshold, best_oof_pr
    """
    if c_grid is None:
        c_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    Xtr_df = X_train_sledai.copy()
    ytr = y_train.values.astype(int)
    gtr = g_train.values

    if len(np.unique(ytr)) < 2:
        return None, None, None, 0.5, np.nan

    try:
        check_inner_cv_feasibility(
            y_train=pd.Series(ytr),
            g_train=pd.Series(gtr),
            n_splits=N_INNER_FOLDS,
            context="SLEDAI-only inner CV"
        )
    except ValueError as e:
        logger.warning("%s", e)
        return None, None, None, 0.5, np.nan

    inner_cv, _ = _make_inner_cv(seed)

    best_c = None
    best_oof_pr = -np.inf
    best_oof_probs = None
    best_contributed = None

    for c_val in c_grid:
        oof_probs = np.zeros(len(ytr), dtype=float)
        contributed = np.zeros(len(ytr), dtype=bool)

        for inner_fold_idx, (itr_idx, iva_idx) in enumerate(inner_cv.split(Xtr_df.values, ytr, gtr)):
            y_inner_tr = ytr[itr_idx]
            y_inner_va = ytr[iva_idx]

            if not (_log_fold_class_counts("sledai-inner-train", y_inner_tr, inner_fold_idx) and
                    _log_fold_class_counts("sledai-inner-val", y_inner_va, inner_fold_idx)):
                continue

            X_inner_tr_df = Xtr_df.iloc[itr_idx].copy()
            X_inner_va_df = Xtr_df.iloc[iva_idx].copy()

            inner_medians = X_inner_tr_df.median(axis=0)
            X_inner_tr_df = X_inner_tr_df.fillna(inner_medians)
            X_inner_va_df = X_inner_va_df.fillna(inner_medians)

            X_inner_tr = X_inner_tr_df.values
            X_inner_va = X_inner_va_df.values

            scaler = StandardScaler()
            X_inner_tr_s = scaler.fit_transform(X_inner_tr)
            X_inner_va_s = scaler.transform(X_inner_va)

            clf = LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                C=c_val,
                max_iter=5000,
                random_state=seed,
                class_weight="balanced",
            )
            clf.fit(X_inner_tr_s, y_inner_tr)

            fold_probs = clf.predict_proba(X_inner_va_s)[:, 1]
            oof_probs[iva_idx] = fold_probs
            contributed[iva_idx] = True

        if contributed.sum() >= 2 and len(np.unique(ytr[contributed])) >= 2:
            oof_pr = average_precision_score(ytr[contributed], oof_probs[contributed])
        else:
            oof_pr = -np.inf

        if oof_pr > best_oof_pr:
            best_oof_pr = oof_pr
            best_c = c_val
            best_oof_probs = oof_probs.copy()
            best_contributed = contributed.copy()

    if best_contributed is not None and best_contributed.sum() >= 2 and len(np.unique(ytr[best_contributed])) >= 2:
        best_threshold, _ = choose_threshold_max_f1(ytr[best_contributed], best_oof_probs[best_contributed])
    else:
        best_threshold = 0.5

    return best_c, best_oof_probs, best_contributed, best_threshold, best_oof_pr


def evaluate_sledai_only_model(X_train_sledai, X_test_sledai, y_train, y_test, g_train, seed):
    """Evaluate the SLEDAI-only model with grouped nested CV."""
    Xtr_df = X_train_sledai.copy()
    Xte_df = X_test_sledai.copy()
    ytr = y_train.values.astype(int)
    yte = y_test.values.astype(int)

    best_c, oof_probs, contributed, threshold, best_oof_pr = tune_sledai_only_model(
        X_train_sledai=Xtr_df,
        y_train=y_train,
        g_train=g_train,
        seed=seed,
    )

    if best_c is None:
        logger.warning(
            "SLEDAI-only tuning was infeasible for this outer split; "
            "using fallback final model with C=1.0 and threshold=0.5."
        )
        best_c = 1.0
        threshold = 0.5
        oof_probs = np.zeros(len(ytr), dtype=float)
        contributed = np.zeros(len(ytr), dtype=bool)
        best_oof_pr = np.nan

    Xtr_df_imp, Xte_df_imp, _ = median_impute_train_test(Xtr_df, Xte_df)

    if Xtr_df_imp.isna().any().any() or Xte_df_imp.isna().any().any():
        raise ValueError("NaNs remain in SLEDAI data after imputation.")

    final_scaler = StandardScaler()
    Xtr_s = final_scaler.fit_transform(Xtr_df_imp.values)
    Xte_s = final_scaler.transform(Xte_df_imp.values)

    final_clf = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        C=best_c,
        max_iter=5000,
        random_state=seed,
        class_weight="balanced",
    )
    final_clf.fit(Xtr_s, ytr)

    test_probs = final_clf.predict_proba(Xte_s)[:, 1]
    metrics, cm, preds = compute_binary_metrics_from_probs(
        y_true=yte,
        probs=test_probs,
        threshold=threshold,
        n_features=1,
    )

    return metrics, cm, test_probs, preds, threshold, oof_probs, contributed, best_c, best_oof_pr


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
    bbox=(0.01, 0.05, 0.98, 0.92), col_widths=None,
    body_base_height=0.11, header_base_height=0.18,
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
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=2)


def plot_two_tables(
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
        fig.suptitle(fig_title, fontsize=FIG_TITLE_SIZE, fontweight="bold", y=0.99)

    for ax, df, title in [(ax_left, df_left, title_left), (ax_right, df_right, title_right)]:
        df = round_numeric_df(df, 2)
        draw_single_table(
            ax, df, title,
            fontsize=TABLE_BODY_FONTSIZE, header_fontsize=TABLE_HEADER_FONTSIZE,
            title_fontsize=32, wrap_cell_width=24 if ax is ax_left else 26,
            bbox=(0.01, 0.09, 0.98, 0.84), body_base_height=0.10, header_base_height=0.16,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.965], pad=0.6, w_pad=1.2)
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


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
        fontsize=FIG_TITLE_SIZE, fontweight="bold", y=0.98,
    )

    cbar_ax    = fig.add_axes([0.92, 0.18, 0.022, 0.62])
    cbar_drawn = False

    for idx, fold_id in enumerate(fold_ids):
        r, c = idx // n_cols, idx % n_cols
        ax   = axes[r, c]
        cm   = all_cms[(fold_id, model_name)]

        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=global_vmax, aspect="equal")

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xticks(np.arange(-0.5, cm.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, cm.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)

        if not cbar_drawn:
            fig.colorbar(im, cax=cbar_ax)
            cbar_drawn = True

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center", fontsize=17, fontweight="bold",
                color="white" if cm[i, j] > global_vmax / 2 else "black",
            )

        ax.set_title(f"Outer Fold {fold_id}", fontsize=26, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted", fontsize=18, labelpad=8)
        ax.set_ylabel("Actual",    fontsize=18, labelpad=8)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
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
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_table(summary_df, save_path):
    """Render mean ± SD and median metrics for all models in summary_df as a formatted table."""

    # -----------------------------
    # Clean display values
    # -----------------------------
    summary_df = summary_df.copy()

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
        if s in ("nan", "NaN", "", "—", "NA"):
            return "NA", "NA"
        if "\n" in s:
            top, bot = s.split("\n", 1)
            return top.strip(), bot.strip().strip("()")
        return s, "—"

    model_display_names = list(summary_df.index)

    data = []
    for col in summary_df.columns:
        if col not in METRIC_DISPLAY:
            continue
        row = [METRIC_DISPLAY[col]]
        for name in model_display_names:
            mean_val, med_val = _split(summary_df.loc[name, col])
            row.append(mean_val)
            row.append(med_val)
        data.append(row)

    n_models = len(model_display_names)

    # Use blank placeholders for colLabels — real multi-line text set manually after table creation
    columns = ["Metric"] + [" "] * (n_models * 2)

    n_data_rows = len(data)
    fig_width  = min(22, 5 + n_models * 4.2)
    fig_height = max(10, 0.7 * n_data_rows + 4)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    metric_w = 0.16
    pair_w = (1.0 - metric_w) / max(n_models, 1)
    col_widths = [metric_w]
    for _ in model_display_names:
        col_widths.append(pair_w * 0.62)
        col_widths.append(pair_w * 0.38)

    tbl = ax.table(
        cellText=data,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
        bbox=[0.00, 0.00, 1.00, 0.88],
    )

    tbl.auto_set_font_size(False)
    body_fs   = 21
    header_fs = 21
    tbl.set_fontsize(21)

    tbl.scale(1.1, 1.5)

    n_cols = len(columns)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_linewidth(0.7)
        cell.set_edgecolor("gray")
        cell.set_text_props(ha="center", va="center", wrap=True)
        cell.PAD = 0.12

        if row == 0:
            cell.set_text_props(weight="bold", fontsize=header_fs)
            cell.set_facecolor("#EEEEEE")
            cell.set_height(0.26)
        else:
            if col == 0:
                cell.set_text_props(weight="bold", fontsize=body_fs)
            else:
                cell.set_text_props(fontsize=body_fs)
            cell.set_facecolor("#F7F7F7" if col == 0 else "#FFFFFF")
            cell.set_height(0.082)

    # Abbreviated header names — split across 3 lines so text wraps inside narrow cells
    ABBREV = {
        "Logistic Regression L2": "LR (L2)",
        "XGBoost":                "XGB",
        "Prevalence baseline":    "Baseline",
        "SLEDAI-only":            "SLEDAI",
    }

    # Set header text manually so \n renders as a real newline in the cell
    tbl[0, 0].get_text().set_text("Metric")
    for i, name in enumerate(model_display_names):
        mean_col   = 1 + i * 2
        median_col = 2 + i * 2
        short = ABBREV.get(name, name)
        tbl[0, mean_col].get_text().set_text(f"{short}\nMean\n\u00b1 SD")
        tbl[0, median_col].get_text().set_text(f"{short}\nMedian")
        for c in (mean_col, median_col):
            tbl[0, c].get_text().set_fontsize(header_fs)
            tbl[0, c].get_text().set_fontweight("bold")
            tbl[0, c].get_text().set_va("center")
            tbl[0, c].get_text().set_ha("center")
            tbl[0, c].get_text().set_multialignment("center")
    tbl[0, 0].get_text().set_fontsize(header_fs)
    tbl[0, 0].get_text().set_fontweight("bold")
    tbl[0, 0].get_text().set_va("center")
    tbl[0, 0].get_text().set_ha("center")

    # Increase header row height to fit 3 lines of text
    for c in range(len(columns)):
        if (0, c) in tbl.get_celld():
            tbl[0, c].set_height(0.26)

    ax.set_title(
        "Average Classification Metrics Across Outer Test Folds",
        fontsize=FIG_TITLE_SIZE,
        fontweight="bold",
        pad=6,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_per_fold_metrics(supp_df, save_path):
    """
    Plot per-fold held-out test metrics for all models.
    Gene-expression models are shown first, followed by comparator models.
    """
    supp_df = round_numeric_df(supp_df, 2)
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

    # Display order: gene-expression models first, then comparators
    MODEL_ORDER_FOR_FIGURE = [
        get_model_display_name("LR_L2"),
        get_model_display_name("XGB"),
        get_model_display_name("BASELINE"),
        get_model_display_name("SLEDAI_ONLY"),
    ]
    present_in_df = set(supp_df["Model"].unique())
    models   = [m for m in MODEL_ORDER_FOR_FIGURE if m in present_in_df]
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
                        val = str(int(round(float(v)))) if metric in count_metrics else f"{float(v):.2f}"
                    else:
                        val = "NA"
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
            cellText=df_model.values.tolist(),
            colLabels=list(df_model.columns),
            cellLoc="center",
            colWidths=col_widths,
            bbox=[0.0, 0.0, 1.0, 0.82],
        )

        tbl.auto_set_font_size(False)
        tbl.set_fontsize(14)
        tbl.scale(1.25, 1.6)

        for (row, col), cell in tbl.get_celld().items():
            cell.set_linewidth(0.7)
            cell.set_edgecolor("gray")
            cell.set_text_props(ha="center", va="center", wrap=True)
            if row == 0:
                cell.set_text_props(weight="bold", fontsize=15)
                cell.set_facecolor("#EEEEEE")
            else:
                cell.set_text_props(fontsize=14)
                cell.set_facecolor("#F7F7F7" if col == 0 else "#FFFFFF")

        _set_row_heights(tbl, df_model)

        ax.text(
            0.5, 0.92, model_name,
            ha="center", va="bottom",
            fontsize=22, fontweight="bold",
            transform=ax.transAxes,
        )

    fig_height = max(10.0, 0.50 * len(metric_order) + 3.0)
    fig, axes  = plt.subplots(len(models), 1, figsize=(18, 4.8 * len(models)))
    if len(models) == 1:
        axes = [axes]
    fig.suptitle(
        "Per-Fold Held-Out Test Metrics Across Outer Folds",
        fontsize=FIG_TITLE_SIZE, fontweight="bold", y=0.995,
    )
    for ax, model_name in zip(axes, models):
        _render_table(ax, model_name)
    plt.tight_layout(rect=[0, 0, 1, 0.985], h_pad=0.35)
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


# Permutation sanity check plotting
def plot_permutation_table_top_bottom(perm_df, save_path):
    """
    Plot permutation-label sanity check results as two tables
    (LR_L2 above XGB), with metrics as rows and folds as columns.
    """
    perm_df = round_numeric_df(perm_df, 2)
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
                        val = str(int(round(float(v)))) if metric == "Features selected" else f"{float(v):.2f}"
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
        "Permutation-Label Sanity Check Across Held-Out Outer Folds",
        fontsize=FIG_TITLE_SIZE, fontweight="bold", y=0.995,
    )

    for ax, model_name in zip(axes, ["LR_L2", "XGB"]):
        if model_name in models:
            _render_table(ax, model_name)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=20, transform=ax.transAxes)
            ax.set_title(get_model_display_name(model_name), fontsize=34, fontweight="bold", pad=12)

    plt.tight_layout(rect=[0, 0, 1, 0.965], h_pad=2.8)
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# PR-curve and SHAP analysis
# =============================================================================
def _draw_pr_auc_on_ax(ax, run_store, model_name, fold_ids):
    all_y_true = []
    for fold_id in fold_ids:
        key = (fold_id, model_name)
        if key not in run_store:
            continue
        probs  = run_store[key]["test_probs"]
        y_test = run_store[key]["y_test"]
        precision, recall, _ = precision_recall_curve(y_test.values, probs)
        pr_auc = average_precision_score(y_test.values, probs)
        ax.plot(recall, precision, linewidth=2, label=f"Fold {fold_id} (PR-AUC={pr_auc:.2f})")
        all_y_true.extend(y_test.values.tolist())

    if all_y_true:
        prevalence = float(np.mean(all_y_true))
        ax.axhline(
            y=prevalence,
            linestyle="--",
            linewidth=1.5,
            label=f"Baseline (Prevalence = {prevalence:.2f})",
        )

    ax.set_xlabel("Recall",    fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Precision", fontsize=AXIS_LABEL_SIZE)
    ax.set_title(get_model_display_name(model_name), fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold")
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, loc="best")
    ax.grid(alpha=0.25)


def plot_pr_auc_curves(run_store, fold_ids, save_path):
    """
    Plot held-out outer-test precision-recall curves for the selected models.

    Layout:
        top-left: LR_L2
        top-right: XGB
        bottom-left: SLEDAI_ONLY
        bottom-right: empty
    """
    # Plot PR curves for LR_L2, XGB, and SLEDAI_ONLY.
    models_to_plot = [m for m in MODEL_NAMES + ["SLEDAI_ONLY"] if any((fid, m) in run_store for fid in fold_ids)]

    if len(models_to_plot) == 0:
        logger.warning("No PR-curve data available.")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax_map = {
        "LR_L2":       axes[0, 0],
        "XGB":         axes[0, 1],
        "SLEDAI_ONLY": axes[1, 0],
    }

    # Leave the lower-right panel empty to preserve the 2x2 layout.
    axes[1, 1].axis("off")

    for model_name in models_to_plot:
        ax = ax_map.get(model_name)
        if ax is not None:
            _draw_pr_auc_on_ax(ax, run_store, model_name, fold_ids)

    fig.suptitle(
        "Precision-Recall Curves Across Held-Out Outer Test Folds",
        fontsize=FIG_TITLE_SIZE,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_shap_bar(lr_shap_df, xgb_shap_df, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(18, max(16, 0.85 * TOP_K + 8)))
    fig.suptitle(
        "Top 20 Genes by Mean |SHAP| Across Models",
        fontsize=FIG_TITLE_SIZE, fontweight="bold", y=0.98,
    )
    for ax, df, model_name in zip(axes, [lr_shap_df, xgb_shap_df], ["LR_L2", "XGB"]):
        if df is None or df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=VALUE_LABEL_SIZE)
            ax.set_title(get_model_display_name(model_name), fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold")
            continue

        df = round_shap_df(df)
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
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def compute_shap_values_for_df(model_name, clf, bg_df, eval_df):
    """
    Compute SHAP values on an evaluation set.
    Uses LinearExplainer for LR_L2 and TreeExplainer for XGB.
    """
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
    Check whether higher feature values tend to increase or decrease predictions.
    """
    x    = np.asarray(x)
    s    = np.asarray(s)
    if len(x) < 2 or np.std(x) == 0 or np.std(s) == 0:
        return "unclear"
    corr = np.corrcoef(x, s)[0, 1]
    if np.isnan(corr):
        return "unclear"
    if corr > 0:
        return "Higher → pre-flare"
    if corr < 0:
        return "Higher → non-pre-flare"
    return "unclear"


def get_top_global_shap_genes_for_one_outer_fold(
    X_train, pipe, gene_cols, comb_mask, model_name, fold_seed, gene_name_map, top_k=TOP_K,
):
    """
    Get the top genes for one outer fold by mean absolute SHAP value.
    """
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


def aggregate_global_shap_genes_across_outer_folds(run_store, model_name, gene_name_map, top_k=TOP_K):
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


def plot_beeswarm(run_store, lr_gene_labels, xgb_gene_labels, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(18, 20))
    fig.suptitle(
        "SHAP Beeswarm of Top 20 Genes Across Models",
        fontsize=FIG_TITLE_SIZE, fontweight="bold", y=0.98,
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

    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3.0)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_correct_oof_genes(lr_df, xgb_df, save_path):
    def _wrap_cell_text(val, width):
        if pd.isna(val):
            return ""
        return "\n".join(textwrap.wrap(str(val), width=width, break_long_words=False, break_on_hyphens=False))

    def _prepare_df(df):
        if df is None or df.empty:
            return df
        display_cols = [c for c in ["Gene", "Folds Appearing", "Stability Score", "Mean |SHAP| (OOF Correct Train)", "Direction"] if c in df.columns]
        plot_df = df[display_cols].copy()
        plot_df = round_shap_df(plot_df)
        plot_df = plot_df.rename(columns={
            "Folds Appearing":            "Folds\nAppearing",
            "Stability Score":            "Stability\nScore",
            "Mean |SHAP| (OOF Correct Train)": "Mean |SHAP|",
        })
        if "Gene" in plot_df.columns:
            plot_df["Gene"] = plot_df["Gene"].apply(lambda x: _wrap_cell_text(x, 24))
        return plot_df

    def _set_row_heights(tbl, df):
        ncols = len(df.columns)
        header_max_lines = max(
            str(df.columns[c]).count("\n") + 1 for c in range(ncols)
        )
        header_h = 0.10 + header_max_lines * 0.055
        for c in range(ncols):
            if (0, c) in tbl.get_celld():
                tbl[(0, c)].set_height(header_h)
        for r in range(len(df)):
            max_lines = max(str(df.iloc[r, c]).count("\n") + 1 for c in range(ncols))
            row_h     = 0.080 + (max_lines - 1) * 0.065
            for c in range(ncols):
                if (r + 1, c) in tbl.get_celld():
                    tbl[(r + 1, c)].set_height(row_h)

    def _render_table(ax, df, title):
        ax.axis("off")
        if df is None or df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=26, transform=ax.transAxes)
            ax.set_title(title, fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold", pad=8)
            return
        plot_df = _prepare_df(df)
        widths  = []
        for c in plot_df.columns:
            if c == "Gene":                              widths.append(0.26)
            elif c == "Folds\nAppearing":                widths.append(0.09)
            elif c == "Stability\nScore":                widths.append(0.09)
            elif c == "Mean |SHAP|":                      widths.append(0.14)
            elif c == "Direction":                       widths.append(0.42)
            else:                                        widths.append(0.12)
        col_widths = [w / sum(widths) for w in widths]
        tbl = ax.table(
            cellText=plot_df.values.tolist(), colLabels=list(plot_df.columns),
            cellLoc="center", colWidths=col_widths, bbox=[0.01, 0.02, 0.98, 0.92],
        )
        tbl.auto_set_font_size(False); tbl.set_fontsize(14)
        for (row, col), cell in tbl.get_celld().items():
            cell.set_linewidth(0.7); cell.set_edgecolor("gray")
            cell.set_text_props(ha="center", va="center", wrap=True)
            if row == 0:
                cell.set_text_props(weight="bold", fontsize=14)
            else:
                cell.set_text_props(fontsize=13)
        _set_row_heights(tbl, plot_df)
        ax.set_title(title, fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold", pad=8)

    n_rows     = max(len(lr_df) if lr_df is not None and not lr_df.empty else 1,
                     len(xgb_df) if xgb_df is not None and not xgb_df.empty else 1)
    fig_height = max(9.5, 0.68 * n_rows + 3.8)
    fig, axes  = plt.subplots(2, 1, figsize=(16.5, fig_height))
    fig.suptitle(
        f"Top {TOP_K_CORRECT_OOF} Genes Linked to Correct Inner-Validation Predictions",
        fontsize=24, fontweight="bold", y=0.995,
    )
    _render_table(axes[0], lr_df,  get_model_display_name("LR_L2"))
    _render_table(axes[1], xgb_df, get_model_display_name("XGB"))
    plt.tight_layout(rect=[0, 0, 1, 0.985], h_pad=2.2)
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return save_path


# SHAP analysis using correctly predicted inner-validation samples
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
# Permutation sanity check (LR_L2 and XGB only)
# =============================================================================
def run_permutation_sanity_check_on_outer_test_folds(outer_summaries):
    """Run a permutation-label sanity check for LR_L2 and XGB across outer folds."""

    rows = []

    usable_outer_summaries = [
        item for item in outer_summaries
        if not item.get("skipped", False)
    ]

    if not usable_outer_summaries:
        logger.warning("Permutation check skipped: no usable outer folds available.")
        return pd.DataFrame()

    for i, outer_item in enumerate(usable_outer_summaries[:N_PERMUTATION_OUTER_FOLDS], start=1):
        fold_id   = outer_item["outer_fold"]
        fold_seed = outer_item["fold_seed"]

        X_train = outer_item["X_train"]
        y_train = outer_item["y_train"]
        g_train = outer_item["g_train"]
        X_test  = outer_item["X_test"]
        y_test  = outer_item["y_test"]

        logger.info(
            "Permutation check — outer fold %d (%d/%d)",
            fold_id, i, min(len(usable_outer_summaries), N_PERMUTATION_OUTER_FOLDS)
        )

        rng    = np.random.RandomState(fold_seed + 10000)
        y_perm = pd.Series(rng.permutation(y_train.values), index=y_train.index).astype(int)

        try:
            check_inner_cv_feasibility(
                y_train=y_perm,
                g_train=g_train,
                n_splits=N_INNER_FOLDS,
                context=f"Permutation check outer fold {fold_id}"
            )
        except ValueError as e:
            logger.warning("Skipping permutation check for outer fold %d: %s", fold_id, e)
            continue

        perm_fold_cache    = build_inner_fold_cache(X_train, y_perm, g_train, fold_seed)
        usable_perm_folds  = sum(f is not None for f in perm_fold_cache)

        if usable_perm_folds == 0:
            logger.warning(
                "Skipping permutation check for outer fold %d: no usable inner folds after cache build.",
                fold_id,
            )
            continue

        for model_name in MODEL_NAMES:
            try:
                if model_name == "LR_L2":
                    best_params, oof_pr, _, _, thr = tune_lr_for_pr_auc(
                        X_train, y_perm, g_train,
                        seed=fold_seed,
                        n_trials=PERM_N_TRIALS_LR,
                        fold_cache=perm_fold_cache,
                    )
                else:
                    best_params, oof_pr, _, _, thr = tune_xgb_for_pr_auc(
                        X_train, y_perm, g_train,
                        seed=fold_seed,
                        n_trials=PERM_N_TRIALS_XGB,
                        fold_cache=perm_fold_cache,
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

            except Exception as e:
                logger.warning(
                    "Permutation check failed for model %s on outer fold %d: %s",
                    model_name, fold_id, e,
                )

    if not rows:
        logger.warning("Permutation check produced no usable results.")
        return pd.DataFrame()

    return pd.DataFrame(rows)


# Notebook display helpers
def show_final_outputs_in_order(
    search_space_png, best_params_png,
    cm_lr_png, cm_xgb_png,
    metrics_png, permutation_png, per_fold_supp_png,
    pr_curves_png=None, shap_bar_png=None,
    beeswarm_png=None, correct_oof_png=None,
    output_dir=None,
):
    ordered_paths = [
        ("1) Hyperparameter search space",                       search_space_png),
        ("2) Best hyperparameters selected for each outer fold", best_params_png),
        ("3a) Confusion matrices — Logistic Regression L2",      cm_lr_png),
        ("3b) Confusion matrices — XGBoost",                     cm_xgb_png),
        ("4) Per-fold held-out test metrics",                    per_fold_supp_png),
        ("5) PR-AUC curves",                                     pr_curves_png),
        ("6) Final average classification metrics",              metrics_png),
        ("7) SHAP bar plots",                                   shap_bar_png),
        ("8) SHAP beeswarm",                                     beeswarm_png),
        (f"9) Top {TOP_K_CORRECT_OOF} correct-OOF genes",       correct_oof_png),
        ("10) Permutation-label sanity check results",           permutation_png),
    ]

    if output_dir:
        logger.info("Outputs saved to: %s", output_dir)

    return ordered_paths


# Outer-fold preparation
def _prepare_outer_folds(outer_cv, X, y, groups, outer_is_stratified, cache_dir):

    import pickle

    outer_folds_pkl = os.path.join(cache_dir, "outer_folds.pkl")
    fold_meta_json  = os.path.join(cache_dir, "outer_folds_meta.json")

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
def _run_one_outer_split(
    outer_idx, tr_idx, te_idx,
    X, y, groups,
    gene_cols, gene_name_map,
    X_sledai,
):

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

    try:
        check_inner_cv_feasibility(
            y_train=y_train,
            g_train=g_train,
            n_splits=N_INNER_FOLDS,
            context=f"Outer split {outer_idx}"
        )
    except ValueError as e:
        logger.warning(
            "Skipping outer split %d because inner CV is infeasible: %s",
            outer_idx, e,
        )
        outer_summary = {
            "outer_fold": outer_idx,
            "fold_seed": fold_seed,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "g_train": g_train,
            "g_test": g_test,
            "skipped": True,
            "skip_reason": str(e),
        }
        return [], [], [], {}, outer_summary

    logger.info("Building inner-fold cache for outer split %d...", outer_idx)
    fold_cache = build_inner_fold_cache(X_train, y_train, g_train, fold_seed)
    usable     = sum(f is not None for f in fold_cache)
    logger.info("Cache built (%d/%d usable inner folds).", usable, len(fold_cache))

    if usable == 0:
        reason = f"Outer split {outer_idx}: inner fold cache contained 0 usable folds."
        logger.warning(
            "Skipping outer split %d because no usable inner folds remained after cache build: %s",
            outer_idx,
            reason,
        )
        outer_summary = {
            "outer_fold": outer_idx,
            "fold_seed": fold_seed,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "g_train": g_train,
            "g_test": g_test,
            "skipped": True,
            "skip_reason": reason,
        }
        return [], [], [], {}, outer_summary

    metrics_list      = []
    cms_list          = []
    params_rows_list  = []
    run_store_entries = {}

    for model_name in MODEL_NAMES:
        logger.info("Running %s on outer split %d ...", get_model_display_name(model_name), outer_idx)

        try:
            if model_name == "LR_L2":
                best_params, best_oof_pr, oof_probs, contributed, thr = tune_lr_for_pr_auc(
                    X_train, y_train, g_train,
                    seed=fold_seed,
                    n_trials=N_TRIALS_LR,
                    fold_cache=fold_cache,
                )
            else:
                best_params, best_oof_pr, oof_probs, contributed, thr = tune_xgb_for_pr_auc(
                    X_train, y_train, g_train,
                    seed=fold_seed,
                    n_trials=N_TRIALS_XGB,
                    fold_cache=fold_cache,
                )

            metrics, cm, pipe, comb_mask, test_probs, test_preds = evaluate_on_outer_test(
                X_train, X_test, y_train, y_test,
                model_name=model_name, seed=fold_seed, params=best_params, threshold=thr,
            )

            metrics.update({
                "Model": model_name,
                "Outer Fold": outer_idx,
                "Fold Seed": fold_seed,
                "OOF PR-AUC": float(best_oof_pr),
                "Params": str(best_params),
                "OOF Coverage": float(contributed.mean()),
                "Test Subjects":     int(g_test.nunique()),
                "Pre-flare (n)":     int((y_test == 1).sum()),
                "Non-pre-flare (n)": int((y_test == 0).sum()),
            })
            metrics_list.append(metrics)
            cms_list.append(((outer_idx, model_name), cm))

            run_store_entries[(outer_idx, model_name)] = {
                "params": best_params,
                "threshold": thr,
                "pipe": pipe,
                "comb_mask": comb_mask,
                "test_probs": test_probs,
                "test_preds": test_preds,
                "cm": cm,
                "oof_pr": float(best_oof_pr),
                "oof_probs": oof_probs,
                "oof_contributed": contributed,
                "y_test": y_test,
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "g_train": g_train,
                "g_test": g_test,
                "gene_cols": gene_cols,
                "gene_name_map": gene_name_map,
                "fold_seed": fold_seed,
                "model_name": model_name,
            }

            row = {"Outer Fold": outer_idx, "Fold Seed": fold_seed}
            row.update(best_params)
            params_rows_list.append({"model_name": model_name, "row": row})

            logger.info(
                "%s | split %d: TEST PR-AUC=%.4f | OOF PR-AUC=%.4f | ROC-AUC=%.4f | F1=%.4f | coverage=%.1f%% | thr=%.2f",
                get_model_display_name(model_name), outer_idx,
                metrics["PR-AUC (PRIMARY)"], metrics["OOF PR-AUC"],
                metrics["AUC-ROC"], metrics["F1"], metrics["OOF Coverage"] * 100, thr,
            )

        except Exception as e:
            logger.warning(
                "%s failed on outer split %d and will be skipped: %s",
                get_model_display_name(model_name), outer_idx, e,
            )

    # Evaluate comparator models for this outer split:
    #   1) prevalence baseline
    #   2) SLEDAI-only logistic regression

    # -------------------------
    # Prevalence baseline
    # -------------------------
    baseline_metrics, _, baseline_probs, _, baseline_prev = evaluate_baseline_on_outer_test(
        y_train=y_train,
        y_test=y_test,
    )

    baseline_metrics_row = {
        "Outer Fold": outer_idx,
        "Model": "BASELINE",
        "Fold Seed": fold_seed,
        "OOF PR-AUC": np.nan,
        "OOF Coverage": np.nan,
        "Params": f"Constant probability = outer-train pre-flare prevalence ({baseline_prev:.4f})",
        "PR-AUC (PRIMARY)": baseline_metrics["PR-AUC (PRIMARY)"],
        "AUC-ROC": baseline_metrics["AUC-ROC"],
        "Sensitivity": baseline_metrics["Sensitivity"],
        "Precision": baseline_metrics["Precision"],
        "Specificity": baseline_metrics["Specificity"],
        "F1": baseline_metrics["F1"],
        "Balanced Accuracy": baseline_metrics["Balanced Accuracy"],
        "Brier Score": baseline_metrics["Brier Score"],
        "Threshold": baseline_metrics["Threshold"],
        "Features selected": baseline_metrics["Features selected"],
        "Test Subjects": int(g_test.nunique()),
        "Pre-flare (n)": int((y_test == 1).sum()),
        "Non-pre-flare (n)": int((y_test == 0).sum()),
    }
    metrics_list.append(baseline_metrics_row)

    run_store_entries[(outer_idx, "BASELINE")] = {
        "test_probs": baseline_probs,
        "y_test": y_test.copy(),
    }

    logger.info(
        "%s | split %d: TEST PR-AUC=%.4f | ROC-AUC=%.4f | F1=%.4f | thr=%.2f",
        "Prevalence baseline", outer_idx,
        baseline_metrics["PR-AUC (PRIMARY)"],
        baseline_metrics["AUC-ROC"],
        baseline_metrics["F1"],
        baseline_metrics["Threshold"],
    )

    # -------------------------
    # SLEDAI-only comparator
    # -------------------------
    if X_sledai is not None:
        X_train_sledai = X_sledai.iloc[tr_idx]
        X_test_sledai  = X_sledai.iloc[te_idx]

        try:
            sledai_metrics, _, sledai_probs, _, _, sledai_oof_probs, sledai_contributed, sledai_best_c, sledai_best_oof_pr = evaluate_sledai_only_model(
                X_train_sledai=X_train_sledai,
                X_test_sledai=X_test_sledai,
                y_train=y_train,
                y_test=y_test,
                g_train=g_train,
                seed=fold_seed,
            )

            sledai_metrics_row = {
                "Outer Fold": outer_idx,
                "Model": "SLEDAI_ONLY",
                "Fold Seed": fold_seed,
                "OOF PR-AUC": float(sledai_best_oof_pr) if pd.notna(sledai_best_oof_pr) else np.nan,
                "OOF Coverage": (
                    float(sledai_contributed.mean())
                    if sledai_contributed is not None and len(sledai_contributed) > 0
                    else np.nan
                ),
                "Params": f"L2 logistic regression; C={float(sledai_best_c):.4g}; class_weight=balanced; single SLEDAI feature",
                "PR-AUC (PRIMARY)": sledai_metrics["PR-AUC (PRIMARY)"],
                "AUC-ROC": sledai_metrics["AUC-ROC"],
                "Sensitivity": sledai_metrics["Sensitivity"],
                "Precision": sledai_metrics["Precision"],
                "Specificity": sledai_metrics["Specificity"],
                "F1": sledai_metrics["F1"],
                "Balanced Accuracy": sledai_metrics["Balanced Accuracy"],
                "Brier Score": sledai_metrics["Brier Score"],
                "Threshold": sledai_metrics["Threshold"],
                "Features selected": sledai_metrics["Features selected"],
                "Test Subjects": int(g_test.nunique()),
                "Pre-flare (n)": int((y_test == 1).sum()),
                "Non-pre-flare (n)": int((y_test == 0).sum()),
            }
            metrics_list.append(sledai_metrics_row)

            run_store_entries[(outer_idx, "SLEDAI_ONLY")] = {
                "test_probs": sledai_probs,
                "y_test": y_test.copy(),
            }

            logger.info(
                "%s | split %d: TEST PR-AUC=%.4f | ROC-AUC=%.4f | F1=%.4f | thr=%.2f",
                "SLEDAI-only", outer_idx,
                sledai_metrics["PR-AUC (PRIMARY)"],
                sledai_metrics["AUC-ROC"],
                sledai_metrics["F1"],
                sledai_metrics["Threshold"],
            )

        except Exception as e:
            logger.warning(
                "SLEDAI-only comparator skipped for outer split %d due to error: %s",
                outer_idx, e,
            )
    else:
        logger.info("SLEDAI-only comparator skipped for outer split %d (no SLEDAI column).", outer_idx)

    outer_summary = {
        "outer_fold": outer_idx, "fold_seed": fold_seed,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "g_train": g_train, "g_test": g_test,
    }
    return metrics_list, cms_list, params_rows_list, run_store_entries, outer_summary


# =============================================================================
# SHAP outputs (gene-expression models only)
# =============================================================================
def _generate_shap_outputs(run_store, gene_name_map, dirs):

    gene_model_keys = [(fold_id, model_name) for (fold_id, model_name) in run_store.keys() if model_name in MODEL_NAMES]
    if not gene_model_keys:
        logger.warning("No gene-expression model outputs available; skipping SHAP outputs.")
        return {
            "pr_curves_png": None,
            "shap_bar_png": None,
            "beeswarm_png": None,
            "correct_oof_png": None,
            "lr_shap_df": pd.DataFrame(),
            "xgb_shap_df": pd.DataFrame(),
            "lr_correct_df": pd.DataFrame(),
            "xgb_correct_df": pd.DataFrame(),
            "lr_shap_csv": None,
            "xgb_shap_csv": None,
            "lr_correct_csv": None,
            "xgb_correct_csv": None,
        }

    fold_ids = sorted({fold_id for (fold_id, model_name) in run_store.keys()})
    results  = {}

    # Save SHAP-related CSVs for gene-expression models and generate summary figures.
    for model_name in MODEL_NAMES:
        shap_df = aggregate_global_shap_genes_across_outer_folds(
            run_store=run_store, model_name=model_name, gene_name_map=gene_name_map, top_k=TOP_K,
        )
        shap_csv = os.path.join(dirs["shap"], f"top{TOP_K}_global_shap_{model_name}.csv")
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

    pr_curves_png = os.path.join(dirs["figures"], "05_pr_auc_curves.png")
    pr_curves_png = make_and_save_figure(
        pr_curves_png,
        plot_pr_auc_curves,
        run_store, fold_ids, pr_curves_png,
    )

    shap_bar_png = os.path.join(dirs["figures"], "07_shap_bar_lr_vs_xgb.png")
    shap_bar_png = make_and_save_figure(
        shap_bar_png,
        plot_shap_bar,
        lr_shap_df, xgb_shap_df, shap_bar_png,
    )

    beeswarm_png = os.path.join(dirs["figures"], "08_shap_beeswarm.png")
    if lr_gene_labels or xgb_gene_labels:
        beeswarm_png = make_and_save_figure(
            beeswarm_png,
            plot_beeswarm,
            run_store, lr_gene_labels, xgb_gene_labels, beeswarm_png,
        )
    else:
        beeswarm_png = None

    correct_oof_png = os.path.join(dirs["figures"], "09_top10_correct_oof_genes.png")
    correct_oof_png = make_and_save_figure(
        correct_oof_png,
        plot_correct_oof_genes,
        lr_correct, xgb_correct, correct_oof_png,
    )

    return {
        "pr_curves_png":   pr_curves_png,
        "shap_bar_png":    shap_bar_png,
        "beeswarm_png":    beeswarm_png,
        "correct_oof_png": correct_oof_png,
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
    sledai_col, best_gene_expression_model_name,
):
    """Save summary tables/figures and display them in order."""
    lr_rows  = [x["row"] for x in best_params_rows if x["model_name"] == "LR_L2"]
    xgb_rows = [x["row"] for x in best_params_rows if x["model_name"] == "XGB"]

    lr_best_params_df  = pd.DataFrame(lr_rows)
    xgb_best_params_df = pd.DataFrame(xgb_rows)

    if not lr_best_params_df.empty:
        lr_best_params_df = lr_best_params_df.sort_values("Outer Fold").reset_index(drop=True)
        lr_best_params_df = round_numeric_df(lr_best_params_df, decimals=2)
    if not xgb_best_params_df.empty:
        xgb_best_params_df = xgb_best_params_df.sort_values("Outer Fold").reset_index(drop=True)
        xgb_best_params_df = round_numeric_df(xgb_best_params_df, decimals=2)

    lr_best_params_df.to_csv( os.path.join(dirs["tables"], "best_params_lr_l2.csv"),  index=False)
    xgb_best_params_df.to_csv(os.path.join(dirs["tables"], "best_params_xgboost.csv"), index=False)

    def _transpose_params_df(df):
        plot_df = df.drop(columns=["Fold Seed"], errors="ignore").set_index("Outer Fold").T.reset_index()
        plot_df.columns = ["Parameter"] + [f"Fold {int(c)}" for c in plot_df.columns[1:]]
        return plot_df

    # Export best-parameter tables for the tuned gene-expression models.
    best_params_png = os.path.join(dirs["figures"], "02_best_hyperparameters_selected.png")

    if lr_best_params_df.empty or xgb_best_params_df.empty:
        logger.warning(
            "Best-hyperparameter figure skipped because one or more parameter tables are empty "
            "(LR rows=%d, XGB rows=%d).",
            len(lr_best_params_df),
            len(xgb_best_params_df),
        )
        best_params_png = None
    else:
        best_params_png = make_and_save_figure(
            best_params_png,
            plot_two_tables,
            _transpose_params_df(lr_best_params_df),  "Logistic Regression L2",
            _transpose_params_df(xgb_best_params_df), "XGBoost",
            save_path=best_params_png,
            fig_title="Best Hyperparameters Selected for Each Outer Training Fold",
        )

    cm_lr_png  = os.path.join(dirs["figures"], "03a_confusion_matrix_lr_l2.png")
    cm_xgb_png = os.path.join(dirs["figures"], "03b_confusion_matrix_xgb.png")
    # Export confusion-matrix figures for the gene-expression models.
    lr_cm_exists  = any(k[1] == "LR_L2" for k in all_cms.keys())
    xgb_cm_exists = any(k[1] == "XGB"   for k in all_cms.keys())

    if lr_cm_exists:
        cm_lr_png = make_and_save_figure(
            cm_lr_png,
            plot_confusion_matrices_one_model,
            all_cms, list(range(1, N_OUTER_FOLDS + 1)), "LR_L2", cm_lr_png,
        )
    else:
        logger.warning("LR_L2 confusion-matrix figure skipped: no confusion matrices available.")
        cm_lr_png = None

    if xgb_cm_exists:
        cm_xgb_png = make_and_save_figure(
            cm_xgb_png,
            plot_confusion_matrices_one_model,
            all_cms, list(range(1, N_OUTER_FOLDS + 1)), "XGB", cm_xgb_png,
        )
    else:
        logger.warning("XGB confusion-matrix figure skipped: no confusion matrices available.")
        cm_xgb_png = None

    def _ms(s):
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return "NA"
        mean_val = s.mean()
        std_val  = s.std()
        med_val  = s.median()
        if pd.isna(std_val):
            std_val = 0.0
        return f"{mean_val:.2f} ± {std_val:.2f}\n({med_val:.2f})"

    present_models = metrics_df["Model"].dropna().astype(str).unique().tolist()
    ordered_present_models = [m for m in ALL_REPORT_MODEL_NAMES if m in present_models]

    logger.info("=== PRESENT MODELS IN metrics_df === %s", present_models)
    logger.info("=== ORDERED PRESENT MODELS FOR SUMMARY === %s", ordered_present_models)

    summary_rows = {}
    for m in ordered_present_models:
        sub = metrics_df.loc[metrics_df["Model"] == m].copy()
        if sub.empty:
            logger.warning("Model missing from summary build: %s", m)
            continue

        feat_s = pd.to_numeric(sub["Features selected"], errors="coerce")

        summary_rows[m] = {
            "PR-AUC (PRIMARY)":  _ms(sub["PR-AUC (PRIMARY)"]),
            "AUC-ROC":           _ms(sub["AUC-ROC"]),
            "Sensitivity":       _ms(sub["Sensitivity"]),
            "Precision":         _ms(sub["Precision"]),
            "Specificity":       _ms(sub["Specificity"]),
            "F1":                _ms(sub["F1"]),
            "Balanced Accuracy": _ms(sub["Balanced Accuracy"]),
            "OOF PR-AUC":        _ms(sub["OOF PR-AUC"]),
            "Brier Score":       _ms(sub["Brier Score"]),
            "Features selected": (
                f"{feat_s.mean(skipna=True):.2f} ± "
                f"{feat_s.std(skipna=True) if not pd.isna(feat_s.std(skipna=True)) else 0.0:.2f}\n"
                f"({feat_s.median(skipna=True):.2f})"
            ),
            "Threshold":         _ms(sub["Threshold"]),
        }

    summary_df = pd.DataFrame.from_dict(summary_rows, orient="index")
    summary_df = summary_df.reindex(ordered_present_models)
    summary_df.index = [get_model_display_name(m) for m in summary_df.index]

    desired_display_order = [
        get_model_display_name("LR_L2"),
        get_model_display_name("XGB"),
        get_model_display_name("BASELINE"),
        get_model_display_name("SLEDAI_ONLY"),
    ]
    summary_df = summary_df.reindex([m for m in desired_display_order if m in summary_df.index])

    logger.info("=== SUMMARY_DF INDEX BEFORE PLOTTING === %s", summary_df.index.tolist())
    logger.info("=== SUMMARY_DF SHAPE === %s", summary_df.shape)
    logger.info("\n%s", summary_df.to_string())

    mean_test_pr_auc = metrics_df.groupby("Model")["PR-AUC (PRIMARY)"].mean()
    mean_oof         = metrics_df.groupby("Model")["OOF PR-AUC"].mean()
    logger.info("=== MODEL PERFORMANCE SUMMARY ===")
    for m in ALL_REPORT_MODEL_NAMES:
        if m in mean_test_pr_auc.index:
            oof_val = mean_oof[m] if m in mean_oof.index else np.nan
            logger.info(
                "  %s: held-out PR-AUC=%.4f | inner OOF PR-AUC=%.4f",
                get_model_display_name(m),
                mean_test_pr_auc[m],
                oof_val,
            )
    logger.info("  Best model overall (held-out PR-AUC): %s", get_model_display_name(best_model_name))

    if best_gene_expression_model_name is not None:
        logger.info(
            "  Best gene-expression model (held-out PR-AUC): %s",
            get_model_display_name(best_gene_expression_model_name),
        )
    else:
        logger.info("  Best gene-expression model (held-out PR-AUC): not available")

    metrics_png = os.path.join(dirs["figures"], "06_final_average_classification_metrics.png")
    logger.info("=== SUMMARY TABLE MODELS: %s ===", list(summary_df.index))
    metrics_png = make_and_save_figure(
        metrics_png,
        plot_metrics_table,
        summary_df, metrics_png,
    )
    summary_df.to_csv(os.path.join(dirs["tables"], "average_metrics_summary.csv"), index=True)

    shap_paths = _generate_shap_outputs(run_store, gene_name_map, dirs)

    if RUN_PERMUTATION_CHECK:
        permutation_df = run_permutation_sanity_check_on_outer_test_folds(outer_summaries)

        if permutation_df is not None and not permutation_df.empty:
            permutation_df  = round_numeric_df(permutation_df, 2)
            permutation_csv = os.path.join(dirs["tables"],  "permutation_sanity_check.csv")
            permutation_png = os.path.join(dirs["figures"], "10_permutation_sanity_check.png")
            permutation_df.to_csv(permutation_csv, index=False)
            permutation_png = make_and_save_figure(
                permutation_png,
                plot_permutation_table_top_bottom,
                permutation_df, permutation_png,
            )
        else:
            logger.warning("Permutation outputs skipped because no usable permutation results were produced.")
            permutation_df  = pd.DataFrame()
            permutation_csv = None
            permutation_png = None
    else:
        permutation_df  = None
        permutation_csv = None
        permutation_png = None

    supp_cols = [
        "Outer Fold", "Model", "Test Subjects", "Pre-flare (n)", "Non-pre-flare (n)",
        "PR-AUC (PRIMARY)", "OOF PR-AUC", "AUC-ROC", "Brier Score",
        "F1", "Sensitivity", "Specificity", "Precision",
        "Balanced Accuracy", "Threshold", "Features selected",
    ]
    supp_df = metrics_df[[c for c in supp_cols if c in metrics_df.columns]].copy()
    model_order_map = {
        "LR_L2": 0,
        "XGB": 1,
        "BASELINE": 2,
        "SLEDAI_ONLY": 3,
    }
    supp_df["_model_order"] = supp_df["Model"].map(model_order_map)
    supp_df = supp_df.sort_values(["_model_order", "Outer Fold"]).drop(columns="_model_order").reset_index(drop=True)
    supp_df["Model"] = supp_df["Model"].map(lambda x: get_model_display_name(x) if x in MODEL_DISPLAY_NAMES else x)
    for col in [c for c in supp_df.columns if c not in ("Outer Fold", "Model", "Test Subjects", "Pre-flare (n)", "Non-pre-flare (n)")]:
        supp_df[col] = pd.to_numeric(supp_df[col], errors="coerce").round(2)
    supp_df = round_numeric_df(supp_df, 2)

    if supp_df.empty:
        logger.warning("Per-fold metrics figure skipped: supplementary metrics table is empty.")
        per_fold_supp_png = None
    else:
        per_fold_supp_png = os.path.join(dirs["figures"], "04_per_fold_held_out_test_metrics.png")
        per_fold_supp_png = make_and_save_figure(
            per_fold_supp_png,
            plot_per_fold_metrics,
            supp_df,
            save_path=per_fold_supp_png,
        )

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
            "SLEDAI_FEATURE_COL": sledai_col,
        },
        "outer_cv_strategy":              "StratifiedGroupKFold" if outer_is_stratified else "GroupKFold",
        "threshold_method":               "max-F1 on combined inner OOF predictions from the best trial; outer test set not used",
        "pruning":                        "MedianPruner active for XGB only; LR omits pruning (fast objective, overhead not justified)",
        "early_stopping":                 "not implemented — n_estimators tuned by Optuna",
        "l1_fallback":                    f"top-{MIN_FALLBACK_FEATURES} by |coef|",
        "highest_mean_pr_auc_model":      best_model_name,
        "highest_mean_pr_auc_gene_model": best_gene_expression_model_name,
        "annotation_path":                annotation_path,
        "outputs": {
            "search_space_png":           search_space_png,
            "best_params_png":            best_params_png,
            "cm_lr_png":                  cm_lr_png,
            "cm_xgb_png":                 cm_xgb_png,
            "pr_auc_curves_png":          shap_paths["pr_curves_png"],
            "metrics_png":                metrics_png,
            "shap_bar_png":               shap_paths["shap_bar_png"],
            "shap_beeswarm_png":          shap_paths["beeswarm_png"],
            "correct_oof_genes_png":      shap_paths["correct_oof_png"],
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

    ordered_paths = show_final_outputs_in_order(
        search_space_png=search_space_png, best_params_png=best_params_png,
        cm_lr_png=cm_lr_png, cm_xgb_png=cm_xgb_png,
        metrics_png=metrics_png, permutation_png=permutation_png,
        per_fold_supp_png=per_fold_supp_png,
        pr_curves_png=shap_paths["pr_curves_png"],
        shap_bar_png=shap_paths["shap_bar_png"],
        beeswarm_png=shap_paths["beeswarm_png"],
        correct_oof_png=shap_paths["correct_oof_png"],
        output_dir=output_dir,
    )

    logger.info("Done. Outputs saved to: %s", output_dir)

    if DISPLAY_FIGURES_IN_NOTEBOOK and _is_jupyter():
        import IPython.display as ipd

        logger.info("\nDisplaying saved figures at end of run...\n")

        for title, path in ordered_paths:
            if path is None:
                continue
            if os.path.exists(path):
                logger.info("=== %s ===", title)
                ipd.display(ipd.HTML(f"<h4>{title}</h4>"))
                ipd.display(ipd.Image(filename=path))
            else:
                logger.warning("Figure not available: %s", title)

    return {
        "metrics_df":                metrics_df,
        "summary_df":                summary_df,
        "lr_best_params_df":         lr_best_params_df,
        "xgb_best_params_df":        xgb_best_params_df,
        "permutation_df":            permutation_df,
        "outer_cv_strategy":         "StratifiedGroupKFold" if outer_is_stratified else "GroupKFold",
        "highest_mean_pr_auc_model": best_model_name,
        "highest_mean_pr_auc_gene_model": best_gene_expression_model_name,
        "annotation_path":           annotation_path,
    }


# =============================================================================
# Main entry point
# =============================================================================
def run_pipeline(data_filepath: str, output_dir: str = OUTPUT_DIR):
    """
    Run the full nested cross-validation pipeline and return a results dictionary.

    Outputs are saved under a timestamped run directory:
        output_dir/run_YYYYMMDD_HHMMSS/figures
        output_dir/run_YYYYMMDD_HHMMSS/tables
        output_dir/run_YYYYMMDD_HHMMSS/shap
        output_dir/run_YYYYMMDD_HHMMSS/logs
    """
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
    run_output_dir = make_run_output_dir(output_dir)
    dirs = _make_output_dirs(run_output_dir)
    logger.info("Run output directory: %s", run_output_dir)

    cache_dir = make_cache_dir(output_dir)

    import time
    for dir_path in dirs.values():
        test_file = os.path.join(dir_path, '.write_test')
        max_attempts = 6
        wait_seconds = 3

        for attempt in range(max_attempts):
            try:
                with open(test_file, "w") as f:
                    f.write("ok")
                os.remove(test_file)
                break
            except OSError:
                if attempt == max_attempts - 1:
                    raise RuntimeError(
                        f"Output directory is not writable: {dir_path}\n"
                        f"Check that the directory exists and that you have write permission. "
                        f"If using Google Drive, make sure Drive is mounted and available."
                    )
                logger.info("Output directory not ready yet; retrying...")
                time.sleep(wait_seconds)

    outer_cv, outer_is_stratified = make_outer_cv()
    logger.info("Outer CV: %s", "StratifiedGroupKFold" if outer_is_stratified else "GroupKFold")

    annotation_path = ANNOTATION_PATH if ANNOTATION_PATH else os.path.join(
        dirs["logs"], "GPL10558_probe_to_gene_symbol.csv"
    )

    if os.path.exists(annotation_path):
        logger.info("Annotation file found locally: %s", annotation_path)
    elif ALLOW_ANNOTATION_DOWNLOAD:
        logger.info(
            "Local annotation file not found. Optional GPL10558 annotation download is enabled; "
            "if download fails, the pipeline will continue using probe IDs only."
        )
        try:
            download_gpl_annotation(annotation_path)
        except Exception as e:
            logger.warning(
                "Optional annotation download failed; proceeding with probe IDs only. Error: %s",
                e,
            )
            annotation_path = None
    else:
        logger.info(
            "No local annotation file found. Proceeding with probe IDs only "
            "(annotation download is disabled)."
        )
        annotation_path = None

    df = load_raw_dataframe(data_filepath)
    X, y, groups, gene_cols, sample_ids = load_data(df)
    X_sledai, y_sledai, groups_sledai, sledai_col = load_sledai_feature(df)
    if sledai_col is not None:
        logger.info("SLEDAI comparator will use column: %s", sledai_col)
    else:
        logger.info("SLEDAI-only model disabled: no valid SLEDAI column found.")
    if X_sledai is not None:
        if not y_sledai.reset_index(drop=True).equals(y.reset_index(drop=True)):
            raise ValueError("SLEDAI labels do not align with the main dataset labels.")
        if not groups_sledai.reset_index(drop=True).equals(groups.reset_index(drop=True)):
            raise ValueError("SLEDAI groups do not align with the main dataset groups.")
    gene_name_map = load_gene_name_map(gene_cols, annotation_path)

    search_spaces    = get_search_space_tables()
    search_space_png = os.path.join(dirs["figures"], "01_hyperparameter_search_space.png")
    search_space_png = make_and_save_figure(
        search_space_png,
        plot_two_tables,
        search_spaces["LR_L2"], "Logistic Regression L2",
        search_spaces["XGB"],   "XGBoost",
        save_path=search_space_png,
        fig_title="Hyperparameter Search Space",
    )

    fold_indices, outer_folds_pkl, fold_meta_json = _prepare_outer_folds(
        outer_cv, X, y, groups, outer_is_stratified, cache_dir,
    )

    # Preserve original sample IDs in the fold assignment exports.
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
    if len(fold_assignment_rows) == 0:
        raise ValueError("No outer fold assignments were generated.")
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
    if len(test_assignment_rows) == 0:
        raise ValueError("No outer test assignments were generated.")
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
            outer_idx, tr_idx, te_idx,
            X, y, groups,
            gene_cols, gene_name_map,
            X_sledai,
        )
        all_metrics.extend(metrics_list)
        all_cms.update(dict(cms_list))
        best_params_rows.extend(params_rows_list)
        run_store.update(run_store_entries)
        outer_summaries.append(outer_summary)

    metrics_df = pd.DataFrame(all_metrics)
    if metrics_df.empty:
        skipped_msgs = []
        for item in outer_summaries:
            if item.get("skipped", False):
                skipped_msgs.append(
                    f"Outer fold {item.get('outer_fold')}: {item.get('skip_reason', 'unknown reason')}"
                )

        details = "\n".join(skipped_msgs) if skipped_msgs else "No additional skip details available."

        raise ValueError(
            "No usable outer folds were completed. "
            "All folds were skipped or failed.\n\n"
            "Possible fix: reduce N_INNER_FOLDS.\n\n"
            f"Details:\n{details}"
        )
    logger.info("=== MODELS IN metrics_df: %s ===", sorted(metrics_df["Model"].unique().tolist()))
    metrics_df.to_csv(os.path.join(dirs["tables"], "metrics_outer_test_folds.csv"), index=False)

    skipped_outer_folds = [
        {
            "outer_fold": item.get("outer_fold"),
            "fold_seed": item.get("fold_seed"),
            "skipped": item.get("skipped", False),
            "skip_reason": item.get("skip_reason", ""),
        }
        for item in outer_summaries
        if item.get("skipped", False)
    ]

    if skipped_outer_folds:
        pd.DataFrame(skipped_outer_folds).to_csv(
            os.path.join(dirs["tables"], "skipped_outer_folds.csv"),
            index=False,
        )

    # Check OOF coverage for models that use inner-fold validation.
    coverage_models = MODEL_NAMES + ["SLEDAI_ONLY"]

    low_coverage = metrics_df[
        metrics_df["Model"].isin(coverage_models) &
        (pd.to_numeric(metrics_df["OOF Coverage"], errors="coerce") < MIN_OOF_COVERAGE_WARN)
    ]

    if not low_coverage.empty:
        logger.warning(
            "Low OOF coverage in %d fold(s):\n%s",
            len(low_coverage),
            low_coverage[["Model", "Outer Fold", "OOF Coverage"]].to_string(index=False),
        )
    else:
        logger.info(
            "OOF coverage check passed for tuned models: all folds >= %.0f%%",
            MIN_OOF_COVERAGE_WARN * 100,
        )

    # Track both the best overall model and the best gene-expression model.
    mean_test_pr_auc_all = (
        metrics_df[metrics_df["Model"].isin(ALL_REPORT_MODEL_NAMES)]
        .groupby("Model")["PR-AUC (PRIMARY)"]
        .mean()
    ).dropna()

    if mean_test_pr_auc_all.empty:
        raise ValueError("Could not determine best overall model because all mean PR-AUC values are NaN.")

    best_model_name = mean_test_pr_auc_all.sort_values(ascending=False).index[0]

    mean_test_pr_auc_gene = (
        metrics_df[metrics_df["Model"].isin(MODEL_NAMES)]
        .groupby("Model")["PR-AUC (PRIMARY)"]
        .mean()
    ).dropna()

    if mean_test_pr_auc_gene.empty:
        logger.warning(
            "No usable gene-expression model results were available. "
            "Gene-model-specific ranking will be reported as unavailable."
        )
        best_gene_expression_model_name = None
    else:
        best_gene_expression_model_name = mean_test_pr_auc_gene.sort_values(ascending=False).index[0]

    return _export_summary_outputs(
        metrics_df, run_store, best_model_name, outer_summaries,
        best_params_rows, all_cms, gene_name_map,
        search_space_png, outer_is_stratified,
        outer_folds_pkl, fold_meta_json,
        X_shape=X.shape, dirs=dirs,
        data_filepath=data_filepath,
        annotation_path=annotation_path,
        sledai_col=sledai_col,
        best_gene_expression_model_name=best_gene_expression_model_name,
    )


def _is_jupyter() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        shell_type = type(shell).__name__
        # Covers standard Jupyter environments, including Colab.
        return shell_type in ("ZMQInteractiveShell", "Shell") or "colab" in str(type(shell)).lower()
    except ImportError:
        return False


def main():
    if _is_jupyter():
        results = run_pipeline(DATA_PATH, output_dir=OUTPUT_DIR)
    else:
        import argparse
        parser = argparse.ArgumentParser(
            description="Nested CV pipeline for pediatric lupus flare prediction (GSE65391)."
        )
        parser.add_argument("--data", type=str, default=DATA_PATH,
                            help="Path to lupus_final_df.pkl (default: %(default)s)")
        parser.add_argument("--out",  type=str, default=OUTPUT_DIR,
                            help="Root output directory (default: %(default)s)")
        args = parser.parse_args()
        results = run_pipeline(args.data, output_dir=args.out)
    logger.info("Pipeline completed successfully.")
    if isinstance(results, dict):
        logger.info(
            "Best overall model: %s",
            get_model_display_name(results.get("highest_mean_pr_auc_model", "unknown"))
        )
        best_gene = results.get("highest_mean_pr_auc_gene_model", None)
        if best_gene is not None:
            logger.info(
                "Best gene-expression model: %s",
                get_model_display_name(best_gene)
            )
    return results


if __name__ == "__main__":
    main()
