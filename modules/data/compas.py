"""COMPAS two-year recidivism data; protected attributes are race and sex.

Predicts the recorded outcome, not the proprietary COMPAS score.
Only FEATURES are eligible predictor inputs. No score or follow-up fields enter X.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

FEATURES = ("age", "juv_fel_count", "juv_misd_count", "juv_other_count",
            "priors_count", "c_charge_degree")
RACES = ("Caucasian", "African-American", "Hispanic", "Asian", "Native American", "Other")
SEX_CODES = {"Male": 0, "Female": 1}
FILTER_COLUMNS = ("days_b_screening_arrest", "is_recid", "score_text")
SOURCE_URL = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"



def preprocess_compas(df, *, race_mode="black-white", apply_filters=True, drop_na=True):
    """Return (frame, group-name-to-ID mapping, metadata).

    race_mode='all': preserve six source categories.
    race_mode='black_white': restrict to Caucasian/African-American (codes 0/1).
    Sex: Male=0, Female=1. Label 1 denotes recorded two-year recidivism.
    IDs use 2*race_code+sex_code, so absent groups do not renumber others.
    Missing required fields are dropped only when drop_na=True; malformed values
    and unexpected categories always raise. Missing unused columns are irrelevant.
    """
    if race_mode not in ("all", "black_white"):
        raise ValueError("race_mode must be 'all' or 'black_white'.")
    df = df.copy()
    df.columns = df.columns.str.strip()
    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names.")
    required = list(FEATURES) + ["race", "sex", "two_year_recid"]
    if apply_filters:
        required += list(FILTER_COLUMNS)
    missing = set(required) - set(df)
    if missing:
        raise ValueError(f"Missing COMPAS columns: {sorted(missing)}")
    for col in required:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype("string").str.strip().replace({"": pd.NA, "?": pd.NA})
    counts = {"input": len(df)}
    incomplete = df[required].isna().any(axis=1)
    if incomplete.any() and not drop_na:
        raise ValueError("Missing required COMPAS values; use drop_na=True.")
    df = df.loc[~incomplete].copy()
    counts["after_missing"] = len(df)
    numeric = list(FEATURES[:-1]) + ["two_year_recid"]
    if apply_filters:
        numeric += ["days_b_screening_arrest", "is_recid"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite values in {col}.")
    if not df["two_year_recid"].isin([0, 1]).all():
        raise ValueError("two_year_recid must be 0 or 1.")
    if not df["race"].isin(RACES).all() or not df["sex"].isin(SEX_CODES).all():
        raise ValueError("Unexpected race or sex category.")
    if not df["c_charge_degree"].isin(["F", "M", "O"]).all():
        raise ValueError("Unexpected c_charge_degree.")
    if not df["age"].between(0, 120).all():
        raise ValueError("Invalid age.")
    for col in FEATURES[1:-1]:
        if ((df[col] < 0) | (df[col] % 1 != 0)).any():
            raise ValueError(f"{col} must contain nonnegative integer counts.")
    if apply_filters:
        if not df["is_recid"].isin([-1, 0, 1]).all():
            raise ValueError("Unexpected is_recid value.")
        if not df["score_text"].isin(["Low", "Medium", "High", "N/A"]).all():
            raise ValueError("Unexpected score_text.")
        keep = (df["days_b_screening_arrest"].between(-30, 30)
                & df["is_recid"].ne(-1)
                & df["c_charge_degree"].ne("O")
                & df["score_text"].ne("N/A"))
        df = df.loc[keep].copy()
    counts["after_filters"] = len(df)
    if race_mode == "black_white":
        df = df.loc[df["race"].isin(RACES[:2])].copy()
    counts["after_race_selection"] = len(df)
    if df.empty:
        raise ValueError("No COMPAS observations remain.")
    race_codes = {name: i for i, name in enumerate(RACES if race_mode == "all" else RACES[:2])}
    # Allowlist prevents identifiers, risk scores, and outcomes leaking into X.
    out = df[list(FEATURES) + ["race", "sex", "two_year_recid"]].copy()
    out["race"] = out["race"].map(race_codes).astype(int)
    out["sex"] = out["sex"].map(SEX_CODES).astype(int)
    out["two_year_recid"] = out["two_year_recid"].astype(int)
    out["group_id"] = 2 * out["race"] + out["sex"]
    out["group"] = out["race"].astype(str) + "_" + out["sex"].astype(str)
    mapping = dict(sorted(zip(out["group"], out["group_id"])))
    out = out.reset_index(drop=True)
    metadata = {
        "race_mode": race_mode, "race_codes": race_codes, "sex_codes": SEX_CODES,
        "apply_filters": apply_filters, "drop_na": drop_na, "row_counts": counts,
        "positive_label": "recorded two-year recidivism",
        "feature_columns": list(FEATURES),
        "group_names": {str(2*r+s): f"{race} / {sex}"
                        for race, r in race_codes.items() for sex, s in SEX_CODES.items()
                        if 2*r+s in set(out["group_id"])},
    }
    return out, mapping, metadata


def load_dataset(*, csv_path="data/compas-scores-two-years.csv", race_mode="black-white",
                 apply_filters=True, drop_na=True, seed=42, compute_pg=False,
                 pg_alpha=1.0, pg_num_draws=20000, pg_ci=0.95):
    """Adult-compatible loader. CSV must be downloaded explicitly before running."""
    from modules.utils.dataset_utils import DatasetSpec, LoadedDataset, compute_pg_dirichlet
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Download {SOURCE_URL} to {path}, or set dataset.kwargs.csv_path.")
    # Preserve literal N/A for the explicit score filter.
    raw = pd.read_csv(path, keep_default_na=False, na_values=[""], skipinitialspace=True)
    frame, mapping, metadata = preprocess_compas(
        raw, race_mode=race_mode, apply_filters=apply_filters, drop_na=drop_na)
    metadata.update(source_url=SOURCE_URL, csv_path=str(path.resolve()),
                    source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    spec=DatasetSpec(name="compas", protected_cols=("race", "sex"),
                                     label_col="two_year_recid"))
    pg_table = pg = None
    if compute_pg:
        pg_table, pg = compute_pg_dirichlet(
            df=frame, group_col="group", alpha=pg_alpha, return_intervals=True,
            num_draws=pg_num_draws, ci=pg_ci, seed=seed)
    return LoadedDataset(df=frame, original_columns=list(raw.columns), group_id=mapping,
                         pg_table=pg_table, pg=pg, metadata=metadata)