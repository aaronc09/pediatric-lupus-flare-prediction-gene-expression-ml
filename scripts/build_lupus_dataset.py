"""
Author: Aaron Choi
Project: Pediatric Lupus Flare Prediction (GSE65391)
File: build_lupus_dataframe.py

Build an analysis-ready longitudinal dataset from the pediatric lupus
gene-expression study GSE65391.

Overview:
    This script downloads or loads GEO data using GEOparse and generates
    a cleaned, paired, and labeled dataset for machine-learning analysis
    of pre-flare states in pediatric systemic lupus erythematosus (SLE)
    subjects.

Steps:
    1. Download or load GEO dataset GSE65391 using GEOparse
    2. Build metadata and gene-expression matrices
    3. Filter to lupus visit samples only (remove healthy control samples)
    4. Pair each visit with the subsequent visit from the same subject
    5. Filter paired visits to the specified follow-up window and label visits
       as "pre-flare" based on SLEDAI increase at follow-up
    6. Save output datasets and a pipeline summary JSON

Default output directory:
    outputs/

Primary ML-ready output:
    outputs/lupus_final_df.pkl

Outputs:
    - full_df
        Merged metadata and expression matrix
    - joined_df
        Within-subject visit pairs with an available subsequent visit
    - lupus_final_df
        Filtered and labeled dataset for modeling
    - pipeline_summary.json
        Dataset statistics and reproducibility information
"""


# --- Environment setup ---

import sys
import json
import argparse
import logging
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import GEOparse


logging.getLogger("GEOparse").setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.ERROR)

logger = logging.getLogger("lupus_pipeline")
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# --- Configuration ---

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build an analysis-ready longitudinal lupus dataset from GEO."
    )
    parser.add_argument(
        "--geo-accession",
        default="GSE65391",
        help="GEO accession number (default: GSE65391)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <cwd>/outputs)",
    )
    parser.add_argument(
        "--max-followup-days",
        type=int,
        default=90,
        help="Maximum follow-up window in days for visit pairing (default: 90)",
    )
    parser.add_argument(
        "--preflare-threshold",
        type=int,
        default=4,
        help="Minimum SLEDAI increase to label a visit as pre-flare (default: 4)",
    )
    args, _ = parser.parse_known_args()
    return args


def resolve_output_dir(args: argparse.Namespace) -> Path:
    """Resolve the output directory from args, falling back to <cwd>/outputs."""
    if args.out is not None:
        return args.out
    return Path.cwd() / "outputs"


# --- Utility functions ---

def print_reproducibility_info() -> None:
    """Print Python and key library versions for reproducibility."""
    print("=" * 80)
    print("REPRODUCIBILITY / VERSION INFO")
    print("=" * 80)
    print(f"Python version   : {sys.version}")
    print(f"pandas version   : {pd.__version__}")
    print(f"GEOparse version : {GEOparse.__version__}")
    print("=" * 80)


def safe_join_metadata_value(value: object) -> str:
    """Join list metadata values into a semicolon-separated string."""
    if isinstance(value, list):
        return "; ".join(str(x) for x in value)
    if pd.isna(value):
        return ""
    return str(value)


def parse_characteristics_list(characteristics_list: list[object]) -> dict[str, str]:
    """Parse a GEO characteristics_ch1 list into a key/value dict."""
    parsed: dict[str, str] = {}
    unparsed_counter = 0

    for item in characteristics_list:
        item = str(item)
        if ": " in item:
            key, value = item.split(": ", 1)
            key = key.strip()
            value = value.strip()
            if key in parsed and pd.notna(parsed[key]) and parsed[key] != "":
                parsed[key] = f"{parsed[key]}; {value}"
            else:
                parsed[key] = value
        else:
            unparsed_counter += 1
            parsed[f"unparsed_characteristic_{unparsed_counter}"] = item

    return parsed


def validate_columns(df: pd.DataFrame, required_cols: list[str], df_name: str = "dataframe") -> None:
    """Check that required columns are present."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing required columns: {missing}")


def save_dataframe_multiple_formats(
    df: pd.DataFrame,
    pkl_path: Optional[Path] = None,
    parquet_path: Optional[Path] = None,
    csv_gz_path: Optional[Path] = None,
    index: bool = True,
) -> None:
    """Save df to any combination of pickle, parquet, and compressed CSV.

    index=True is intentional: the index carries gsm_id, which is the primary
    row identity for all saved dataframes. Pass index=False only for frames
    that have a meaningless integer index.
    """
    if pkl_path is not None:
        df.to_pickle(pkl_path)
        logger.info(f"Saved pickle: {pkl_path}")

    if parquet_path is not None:
        try:
            df.to_parquet(parquet_path, index=index)
            logger.info(f"Saved parquet: {parquet_path}")
        except Exception as e:
            logger.warning(f"Could not save parquet: {e}")

    if csv_gz_path is not None:
        try:
            df.to_csv(csv_gz_path, compression="gzip", index=index)
            logger.info(f"Saved compressed CSV: {csv_gz_path}")
        except Exception as e:
            logger.warning(f"Could not save CSV: {e}")


def validate_final_ml_input(df: pd.DataFrame) -> None:
    """Validate that the final dataframe matches the ML pipeline input contract."""
    if df.empty:
        raise ValueError("Final ML dataset is empty.")

    required_cols = ["subject", "preflare_bool"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Final ML dataset missing required columns: {missing}")

    gene_cols = [c for c in df.columns if str(c).startswith("ILMN_")]
    if len(gene_cols) == 0:
        raise ValueError("Final ML dataset contains no ILMN_* gene-expression columns.")

    if df.index.duplicated().any():
        raise ValueError("Final ML dataset has duplicated sample index values.")

    if df["preflare_bool"].isna().any():
        raise ValueError("Final ML dataset contains NaN values in 'preflare_bool'.")

    if df["subject"].isna().any():
        raise ValueError("Final ML dataset contains NaN values in 'subject'.")

    logger.info(
        "Final ML input validation passed: %d samples | %d ILMN_* features | %d subjects",
        df.shape[0], len(gene_cols), df["subject"].nunique()
    )


# --- Load GEO data ---

def load_geo_dataset(geo_accession: str):
    """Download (or load from local cache) a GEO dataset by accession number."""
    logger.info(f"Downloading/loading GEO dataset: {geo_accession}")
    try:
        gse = GEOparse.get_GEO(geo=geo_accession)
    except Exception as e:
        raise RuntimeError(f"Failed to download/load GEO dataset {geo_accession}: {e}")
    logger.info(f"Loaded GEO dataset: {geo_accession}")
    return gse


# --- Build Gene-Expression matrix ---

def build_expression_matrix(gse) -> pd.DataFrame:
    """Build a samples x probes expression matrix from GEO sample tables."""
    logger.info("Building expression matrix...")

    all_expr = []

    for gsm_id, gsm in gse.gsms.items():
        if gsm.table is None:
            continue

        required_expr_cols = {"ID_REF", "VALUE"}
        missing_cols = required_expr_cols - set(gsm.table.columns)
        if missing_cols:
            raise ValueError(
                f"{gsm_id} is missing required expression columns: {sorted(missing_cols)}"
            )

        df = gsm.table[["ID_REF", "VALUE"]].copy()
        df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")

        n_dupes = df["ID_REF"].duplicated().sum()
        if n_dupes > 0:
            logger.warning(f"{gsm_id}: {n_dupes} duplicate probe IDs — keeping first occurrence")
            df = df[~df["ID_REF"].duplicated(keep="first")]

        df = df.rename(columns={"VALUE": gsm_id})
        df.set_index("ID_REF", inplace=True)
        all_expr.append(df)

    if not all_expr:
        raise ValueError("No expression tables were found in the GEO samples.")

    expr_matrix = pd.concat(all_expr, axis=1).T
    expr_matrix.index.name = "gsm_id"

    n_dup_samples = expr_matrix.index.duplicated().sum()
    if n_dup_samples > 0:
        raise ValueError(f"Expression matrix has {n_dup_samples} duplicate gsm_id rows")

    logger.info(f"Expression matrix shape: {expr_matrix.shape}")
    return expr_matrix


# --- Build Metadata table ---

def build_metadata_dataframe(gse) -> pd.DataFrame:
    """Build a metadata table and expand characteristics_ch1 into separate columns."""
    logger.info("Building metadata dataframe...")

    sample_records = []

    for gsm_id, gsm in gse.gsms.items():
        record = {"gsm_id": gsm_id}
        for key, value in gsm.metadata.items():
            if key == "characteristics_ch1":
                if isinstance(value, list):
                    parsed_chars = parse_characteristics_list(value)
                    record.update(parsed_chars)
                else:
                    record["characteristics_ch1"] = safe_join_metadata_value(value)
            else:
                record[key] = safe_join_metadata_value(value)
        sample_records.append(record)

    metadata_df = pd.DataFrame(sample_records).set_index("gsm_id")

    n_dup = metadata_df.index.duplicated().sum()
    if n_dup > 0:
        raise ValueError(f"Metadata dataframe has {n_dup} duplicate gsm_id rows")

    if "disease state" in metadata_df.columns:
        label_counts = metadata_df["disease state"].value_counts(dropna=False)
        logger.info(f"'disease state' label variants:\n{label_counts.to_string()}")

    logger.info(f"Metadata dataframe shape: {metadata_df.shape}")
    return metadata_df


# --- Merge metadata and gene-expression data ---

def merge_metadata_and_expression(metadata_df: pd.DataFrame, expr_matrix: pd.DataFrame) -> pd.DataFrame:
    """Merge metadata and expression data by gsm_id."""
    logger.info("Merging metadata and expression data...")

    full_df = pd.merge(
        metadata_df,
        expr_matrix,
        left_index=True,
        right_index=True,
        how="inner"
    )

    logger.info(f"Merged dataframe shape: {full_df.shape}")

    if full_df.empty:
        raise ValueError("Merged dataframe is empty. Check sample IDs and input tables.")

    return full_df


# --- Filter to lupus (SLE) subject visits only and prepare visits ---

def prepare_lupus_visits(full_df: pd.DataFrame) -> pd.DataFrame:
    """Filter to SLE samples and assign within-subject visit order."""
    logger.info("Preparing SLE visit data...")

    validate_columns(
        full_df,
        ["disease state", "subject", "days_since_diagnosis", "sledai"],
        "full_df"
    )

    disease_state_clean = full_df["disease state"].astype(str).str.strip().str.upper()
    lup_df = full_df[disease_state_clean == "SLE"].copy()

    if lup_df.empty:
        raise ValueError("No SLE samples found after filtering on 'disease state'.")

    lup_df["days_since_diagnosis"] = pd.to_numeric(
        lup_df["days_since_diagnosis"], errors="coerce"
    )
    lup_df["sledai"] = pd.to_numeric(lup_df["sledai"], errors="coerce")

    required_fields = ["subject", "days_since_diagnosis", "sledai"]
    for col in required_fields:
        n_missing = lup_df[col].isna().sum()
        if n_missing > 0:
            logger.warning(f"'{col}' missing in {n_missing} SLE samples (will be dropped)")

    lup_df = lup_df.dropna(subset=required_fields)

    if lup_df.empty:
        raise ValueError("No valid SLE samples remain after dropping rows with missing required fields.")

    lup_df = lup_df.sort_values(["subject", "days_since_diagnosis"])
    lup_df["calculated_visit_num"] = lup_df.groupby("subject").cumcount() + 1

    dup_mask = lup_df.duplicated(subset=["subject", "days_since_diagnosis"], keep=False)
    n_dup_visits = dup_mask.sum()
    if n_dup_visits > 0:
        logger.warning(f"{n_dup_visits} rows share identical subject/days_since_diagnosis — review carefully")

    logger.info(f"Lupus-only samples: {lup_df.shape[0]}")
    logger.info(f"Unique lupus subjects: {lup_df['subject'].nunique()}")
    return lup_df


# --- Visit pairing ---

def build_longitudinal_pairs(lup_df: pd.DataFrame) -> pd.DataFrame:
    """Pair each visit with the next visit for the same subject."""
    logger.info("Pairing visits within subject...")

    cols_to_shift = ["days_since_diagnosis", "sledai"]
    shifted = (
        lup_df
        .sort_values(["subject", "days_since_diagnosis"])
        .groupby("subject")[cols_to_shift]
        .shift(-1)
        .rename(columns={c: f"{c}_subsequent" for c in cols_to_shift})
    )

    joined = pd.concat([lup_df, shifted], axis=1).dropna(
        subset=["days_since_diagnosis_subsequent", "sledai_subsequent"]
    ).copy()

    n_dup_pairs = joined.duplicated(
        subset=["subject", "days_since_diagnosis", "days_since_diagnosis_subsequent"]
    ).sum()
    if n_dup_pairs > 0:
        logger.warning(f"{n_dup_pairs} duplicate visit-pair rows after merging")

    logger.info(f"Paired samples: {joined.shape[0]}")
    return joined


# --- Filter follow-up pairs and label pre-flare ---

def filter_and_label_preflare(
    joined_df: pd.DataFrame,
    max_followup_days: int = 90,
    preflare_threshold: int = 4,
) -> pd.DataFrame:
    """Filter to pairs within the follow-up window and label pre-flare visits."""
    logger.info(f"Filtering <={max_followup_days} day follow-up visits...")

    delta = (
        joined_df["days_since_diagnosis_subsequent"]
        - joined_df["days_since_diagnosis"]
    )

    lupus_final_df = joined_df[(delta > 0) & (delta <= max_followup_days)].copy()

    lupus_final_df["sledai_delta"] = (
        lupus_final_df["sledai_subsequent"] - lupus_final_df["sledai"]
    )
    lupus_final_df["preflare_bool"] = lupus_final_df["sledai_delta"] >= preflare_threshold

    n_pos = lupus_final_df["preflare_bool"].sum()
    n_neg = (~lupus_final_df["preflare_bool"]).sum()
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"preflare_bool has only one class (pos={n_pos}, neg={n_neg}). "
            "Check SLEDAI thresholds or data filtering."
        )

    logger.info(f"Final samples: {lupus_final_df.shape[0]}")
    logger.info(f"preflare_bool — True: {n_pos}, False: {n_neg}")
    return lupus_final_df


# --- Reporting ---

def print_subject_counts_by_stage(
    full_df: pd.DataFrame,
    lup_df: pd.DataFrame,
    joined_df: pd.DataFrame,
    lupus_final_df: pd.DataFrame,
    max_followup_days: int,
) -> None:
    """Print cohort counts and follow-up statistics."""
    print("\n" + "=" * 80)
    print("COHORT COUNTS BY STAGE")
    print("=" * 80)

    sle_only = full_df[full_df["disease state"].astype(str).str.strip().str.upper() == "SLE"]

    print("\nSLE samples before filtering")
    print(f"  samples          : {sle_only.shape[0]}")
    print(f"  unique subjects  : {sle_only['subject'].nunique()}")

    print("\nAfter dropna filtering")
    print(f"  samples          : {lup_df.shape[0]}")
    print(f"  unique subjects  : {lup_df['subject'].nunique()}")

    print("\nAfter pairing visits")
    print(f"  samples          : {joined_df.shape[0]}")
    print(f"  unique subjects  : {joined_df['subject'].nunique()}")

    print(f"\nAfter <={max_followup_days} day filter")
    print(f"  samples          : {lupus_final_df.shape[0]}")
    print(f"  unique subjects  : {lupus_final_df['subject'].nunique()}")

    gap = (
        lupus_final_df["days_since_diagnosis_subsequent"]
        - lupus_final_df["days_since_diagnosis"]
    )
    print("\nFollow-up gap statistics (days)")
    print(f"  mean   : {gap.mean():.1f}")
    print(f"  median : {gap.median():.1f}")
    print(f"  min    : {gap.min():.1f}")
    print(f"  max    : {gap.max():.1f}")

    print("\npreflare_bool class distribution")
    vc = lupus_final_df["preflare_bool"].value_counts()
    print(f"  True (pre-flare)      : {vc.get(True, 0)}")
    print(f"  False (non-pre-flare) : {vc.get(False, 0)}")

    print("=" * 80)


# --- Pipeline summary ---

def save_pipeline_summary(
    full_df: pd.DataFrame,
    lup_df: pd.DataFrame,
    joined_df: pd.DataFrame,
    lupus_final_df: pd.DataFrame,
    geo_accession: str,
    max_followup_days: int,
    preflare_threshold: int,
    summary_path: Path,
) -> dict[str, object]:
    """Save a summary JSON with cohort counts and follow-up statistics."""
    sle_only = full_df[full_df["disease state"].astype(str).str.strip().str.upper() == "SLE"]

    gap = (
        lupus_final_df["days_since_diagnosis_subsequent"]
        - lupus_final_df["days_since_diagnosis"]
    )

    pairs_per_subject = (
        lupus_final_df.groupby("subject").size().describe().to_dict()
    )

    summary = {
        "geo_accession": geo_accession,
        "geo_access_date": str(date.today()),
        "geoparse_version": GEOparse.__version__,
        "pandas_version": pd.__version__,
        "python_version": sys.version,

        "n_total_samples": int(full_df.shape[0]),
        "n_sle_samples_raw": int(sle_only.shape[0]),
        "n_subjects_sle_raw": int(sle_only["subject"].nunique()),

        "n_after_dropna": int(lup_df.shape[0]),
        "n_subjects_after_dropna": int(lup_df["subject"].nunique()),

        "n_after_pairing": int(joined_df.shape[0]),
        "n_subjects_after_pairing": int(joined_df["subject"].nunique()),

        "n_final": int(lupus_final_df.shape[0]),
        "n_subjects_final": int(lupus_final_df["subject"].nunique()),

        "preflare_true": int(lupus_final_df["preflare_bool"].sum()),
        "preflare_false": int((~lupus_final_df["preflare_bool"]).sum()),
        "preflare_prevalence": round(float(lupus_final_df["preflare_bool"].mean()), 4),

        "max_followup_days": max_followup_days,
        "preflare_sledai_delta_threshold": preflare_threshold,

        "followup_gap_mean_days": round(float(gap.mean()), 2),
        "followup_gap_median_days": round(float(gap.median()), 2),
        "followup_gap_min_days": round(float(gap.min()), 2),
        "followup_gap_max_days": round(float(gap.max()), 2),

        "pairs_per_subject_stats": {k: round(v, 2) for k, v in pairs_per_subject.items()},
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved pipeline summary: {summary_path}")
    return summary


# --- Main execution ---

def main(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    """Run the full dataset build from GEO download to final output files."""
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_df_pkl = output_dir / "full_df.pkl"
    full_df_parquet = output_dir / "full_df.parquet"
    joined_df_pkl = output_dir / "joined_df.pkl"
    joined_df_parquet = output_dir / "joined_df.parquet"
    lupus_final_pkl = output_dir / "lupus_final_df.pkl"
    lupus_final_parquet = output_dir / "lupus_final_df.parquet"
    lupus_final_csv_gz = output_dir / "lupus_final_df.csv.gz"
    pipeline_summary_json = output_dir / "pipeline_summary.json"

    print_reproducibility_info()

    gse = load_geo_dataset(args.geo_accession)

    expr_matrix = build_expression_matrix(gse)
    metadata_df = build_metadata_dataframe(gse)
    full_df = merge_metadata_and_expression(metadata_df, expr_matrix)

    save_dataframe_multiple_formats(
        full_df,
        pkl_path=full_df_pkl,
        parquet_path=full_df_parquet,
    )

    lup_df = prepare_lupus_visits(full_df)
    joined_df = build_longitudinal_pairs(lup_df)

    save_dataframe_multiple_formats(
        joined_df,
        pkl_path=joined_df_pkl,
        parquet_path=joined_df_parquet,
    )

    lupus_final_df = filter_and_label_preflare(
        joined_df,
        max_followup_days=args.max_followup_days,
        preflare_threshold=args.preflare_threshold,
    )

    validate_final_ml_input(lupus_final_df)

    save_dataframe_multiple_formats(
        lupus_final_df,
        pkl_path=lupus_final_pkl,
        parquet_path=lupus_final_parquet,
        csv_gz_path=lupus_final_csv_gz,
    )

    save_pipeline_summary(
        full_df=full_df,
        lup_df=lup_df,
        joined_df=joined_df,
        lupus_final_df=lupus_final_df,
        geo_accession=args.geo_accession,
        max_followup_days=args.max_followup_days,
        preflare_threshold=args.preflare_threshold,
        summary_path=pipeline_summary_json,
    )

    print_subject_counts_by_stage(
        full_df=full_df,
        lup_df=lup_df,
        joined_df=joined_df,
        lupus_final_df=lupus_final_df,
        max_followup_days=args.max_followup_days,
    )

    return {
        "expr_matrix": expr_matrix,
        "metadata_df": metadata_df,
        "full_df": full_df,
        "lup_df": lup_df,
        "joined_df": joined_df,
        "lupus_final_df": lupus_final_df,
    }


if __name__ == "__main__":
    args = parse_args()
    results = main(args)

    full_df = results["full_df"]
    lupus_final_df = results["lupus_final_df"]

    print("\nfull_df preview:")
    print(full_df.head())

    print("\nlupus_final_df preview:")
    print(lupus_final_df.head())
