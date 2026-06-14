"""Shared helpers for the PRISM Python model scripts."""

from __future__ import annotations

import math
import re
import tempfile
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb


GITHUB_XLSX_URL = (
    "https://raw.githubusercontent.com/ndesai777777/prism_repo/main/"
    "DataSets/PRP_1000_full_pretreatment.xlsx"
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def clean_names_simple(columns: Iterable[object]) -> list[str]:
    clean: list[str] = []
    for column in columns:
        name = str(column).lower()
        name = re.sub(r"[^a-z0-9]+", "_", name)
        name = re.sub(r"^_+|_+$", "", name)
        name = re.sub(r"_+", "_", name)
        clean.append(name)
    return clean


def read_prism_excel(source_url: str = GITHUB_XLSX_URL) -> pd.DataFrame:
    """Read the source Excel file, falling back to the repo copy if offline."""
    local_path = project_root() / "DataSets" / "PRP_1000_full_pretreatment.xlsx"

    try:
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        urllib.request.urlretrieve(source_url, temp_path)
        print(f"Imported data from:\n{source_url}\n")
        return pd.read_excel(temp_path)
    except Exception as exc:
        if local_path.exists():
            print(
                "Could not download the Excel file; using local repo copy instead:\n"
                f"{local_path}\n"
                f"Download error: {exc}\n"
            )
            return pd.read_excel(local_path)
        raise


def ensure_output_folder(folder: str | Path) -> Path:
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_binary(values: pd.Series) -> pd.Series:
    series = pd.Series(values).copy()
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype(float)

    text = series.astype("string").str.strip().str.lower()
    numeric = pd.to_numeric(text, errors="coerce")
    out = numeric.astype(float)
    out = out.mask(text.isin(["1", "y", "yes", "true", "t"]), 1.0)
    out = out.mask(text.isin(["0", "n", "no", "false", "f"]), 0.0)
    return out.astype(float)


def safe_as_date(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce")


def add_date_features(df: pd.DataFrame, include_duration: bool = False) -> pd.DataFrame:
    df = df.copy()
    for column in ["index_date", "intervention_start_date", "intervention_end_date"]:
        if column in df.columns:
            df[column] = safe_as_date(df[column])

    if "intervention_start_date" in df.columns:
        start_date = df["intervention_start_date"]
        df["intervention_start_month"] = start_date.dt.month.astype(float)
        df["intervention_start_wday"] = (((start_date.dt.dayofweek + 1) % 7) + 1).astype(
            float
        )
    else:
        df["intervention_start_month"] = np.nan
        df["intervention_start_wday"] = np.nan

    if {"index_date", "intervention_start_date"}.issubset(df.columns):
        df["days_to_intervention_start"] = (
            df["intervention_start_date"] - df["index_date"]
        ).dt.days.astype(float)
    else:
        df["days_to_intervention_start"] = np.nan

    if include_duration:
        if {"intervention_start_date", "intervention_end_date"}.issubset(df.columns):
            df["intervention_duration_calc"] = (
                df["intervention_end_date"] - df["intervention_start_date"]
            ).dt.days.astype(float)
        else:
            df["intervention_duration_calc"] = np.nan

        if "intervention_days_active" not in df.columns:
            df["intervention_days_active"] = df["intervention_duration_calc"]

    return df


def impute_numeric(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.median(skipna=True)
    if pd.isna(median):
        median = 0
    return numeric.fillna(median)


def impute_categorical(values: pd.Series) -> pd.Series:
    text = values.astype("string")
    text = text.where(text.notna() & (text.str.strip() != ""), "Missing")
    return text.astype("category")


def present_columns(columns: Sequence[str], df: pd.DataFrame) -> list[str]:
    return [column for column in columns if column in df.columns]


def clean_feature_names(columns: Iterable[object]) -> list[str]:
    clean: list[str] = []
    seen: dict[str, int] = {}
    for column in columns:
        name = str(column)
        name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        if not name:
            name = "feature"
        count = seen.get(name, 0)
        seen[name] = count + 1
        clean.append(name if count == 0 else f"{name}_{count + 1}")
    return clean


def require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def prepare_model_frame(
    df: pd.DataFrame,
    predictor_vars: Sequence[str],
    numeric_vars: Sequence[str],
    binary_extra: Sequence[str] = (),
) -> pd.DataFrame:
    predictor_vars = present_columns(predictor_vars, df)
    model_df = df[["outcome_ed_90d", "intervention_flag", *predictor_vars]].copy()
    model_df = model_df[
        model_df["outcome_ed_90d"].notna() & model_df["intervention_flag"].notna()
    ].copy()

    flag_columns = [column for column in model_df.columns if column.endswith("_flag")]
    for column in flag_columns:
        model_df[column] = to_binary(model_df[column])

    for column in present_columns(binary_extra, model_df):
        model_df[column] = to_binary(model_df[column])

    for column in present_columns(numeric_vars, model_df):
        model_df[column] = pd.to_numeric(model_df[column], errors="coerce")

    for column in model_df.columns:
        if column in ["outcome_ed_90d", "intervention_flag"]:
            continue
        if pd.api.types.is_numeric_dtype(model_df[column]):
            model_df[column] = impute_numeric(model_df[column])
        else:
            model_df[column] = impute_categorical(model_df[column])

    feature_columns = [
        column
        for column in model_df.columns
        if column not in ["outcome_ed_90d", "intervention_flag"]
    ]
    keep_features = [
        column for column in feature_columns if model_df[column].dropna().nunique() > 1
    ]

    return model_df[["outcome_ed_90d", "intervention_flag", *keep_features]].copy().reset_index(drop=True)


def split_train_test(
    df: pd.DataFrame, train_fraction: float = 0.70, seed: int = 123
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_rows = len(df)
    train_size = math.floor(train_fraction * n_rows)
    train_positions = rng.choice(n_rows, size=train_size, replace=False)
    train_mask = np.zeros(n_rows, dtype=bool)
    train_mask[train_positions] = True
    return df.iloc[train_mask].copy(), df.iloc[~train_mask].copy()


def make_design_matrix(
    frames: Sequence[pd.DataFrame],
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    lengths = [len(frame) for frame in frames]
    combined = pd.concat(frames, axis=0, ignore_index=True)
    matrix = pd.get_dummies(combined, drop_first=False, dtype=float)
    matrix.columns = clean_feature_names(matrix.columns)
    matrix = matrix.astype(float)

    split_frames: list[pd.DataFrame] = []
    start = 0
    for length in lengths:
        stop = start + length
        split_frames.append(matrix.iloc[start:stop].copy())
        start = stop
    return matrix, split_frames


def align_to_columns(matrix: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    aligned = matrix.copy()
    for column in columns:
        if column not in aligned.columns:
            aligned[column] = 0.0
    return aligned.loc[:, list(columns)].astype(float)


def fit_xgb_binary(
    x_matrix: pd.DataFrame,
    y: Sequence[float],
    nrounds: int = 150,
    seed: int = 123,
) -> xgb.Booster:
    y_array = np.asarray(y, dtype=float)
    observed = set(pd.Series(y_array).dropna().unique())
    if not observed.issubset({0.0, 1.0}):
        raise ValueError(f"Binary target contains values other than 0 and 1: {observed}")

    dtrain = xgb.DMatrix(
        x_matrix,
        label=y_array,
        feature_names=list(x_matrix.columns),
    )
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": seed,
        "verbosity": 0,
    }
    return xgb.train(params=params, dtrain=dtrain, num_boost_round=nrounds, verbose_eval=False)


def predict_xgb(model: xgb.Booster, x_matrix: pd.DataFrame) -> np.ndarray:
    dmatrix = xgb.DMatrix(x_matrix, feature_names=list(x_matrix.columns))
    return model.predict(dmatrix)


def ntile_desc(values: pd.Series, n: int = 10) -> pd.Series:
    if len(values) == 0:
        return pd.Series(dtype="Int64", index=values.index)
    rank = values.rank(method="first", ascending=False).astype(int)
    tiles = ((rank - 1) * n // len(values)) + 1
    return tiles.astype(int)


def clip_probs(values: Sequence[float], lower: float = 0.05, upper: float = 0.95) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), lower, upper)


def xgb_importance_frame(model: xgb.Booster) -> pd.DataFrame:
    gain = model.get_score(importance_type="gain")
    cover = model.get_score(importance_type="cover")
    weight = model.get_score(importance_type="weight")
    features = sorted(set(gain) | set(cover) | set(weight), key=lambda item: gain.get(item, 0), reverse=True)
    return pd.DataFrame(
        {
            "feature": features,
            "gain": [gain.get(feature, 0.0) for feature in features],
            "cover": [cover.get(feature, 0.0) for feature in features],
            "frequency": [weight.get(feature, 0.0) for feature in features],
        }
    )


def shap_importance_frame(
    model: xgb.Booster,
    x_matrix: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    dmatrix = xgb.DMatrix(x_matrix, feature_names=list(x_matrix.columns))
    contributions = model.predict(dmatrix, pred_contribs=True)
    columns = [*x_matrix.columns, "BIAS"]
    shap_df = pd.DataFrame(contributions, columns=columns)
    no_bias = shap_df.drop(columns=["BIAS"], errors="ignore")
    return (
        no_bias.abs()
        .mean(axis=0)
        .rename("mean_abs_shap")
        .reset_index()
        .rename(columns={"index": "feature"})
        .assign(model=label)
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def print_distribution(label: str, values: pd.Series) -> None:
    print(f"{label}:")
    print(values.value_counts(dropna=False).sort_index())
    print()
