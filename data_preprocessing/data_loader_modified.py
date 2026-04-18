"""
Slimmed-down data loader for the current CO2 capture workflow.

The repository now trains on model-ready datasets exported by
`prepare_co2_capture_datasets.py`. Those datasets already encode the main
preprocessing policy, so the loader focuses on:

- metadata-aware CSV loading
- optional duplicate-input aggregation / diagnostics
- numeric feature validation
- optional one-hot encoding for explicitly requested text columns
- optional log transform on numeric features

Legacy promoter/material featurization and KDE-based imputation were removed
from the main path because they are no longer part of the carbon workflow.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple
import os

import numpy as np
import pandas as pd


MISSING_STRINGS = {
    "",
    " ",
    "none",
    "nan",
    "na",
    "n/a",
    "NONE",
    "NaN",
    "NA",
    "N/A",
    "None",
    "-",
    "–",
    "—",
}

DEFAULT_TARGET_CANDIDATES = (
    "Uptake (mmolg-1)",
    "CO selectivity (%)",
    "Methanol selectivity (%)",
    "CO2 conversion efficiency (%)",
)


def _normalize_missing(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in MISSING_STRINGS:
            return np.nan
        return stripped
    return value


def _read_csv_with_missing(csv_path: str, preserve_null: bool = True) -> pd.DataFrame:
    del preserve_null
    df = pd.read_csv(csv_path, dtype=object)
    for col in df.columns:
        df[col] = df[col].map(_normalize_missing)
    return df


def _strip_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _drop_unnamed_cols(df: pd.DataFrame) -> pd.DataFrame:
    unnamed = [col for col in df.columns if str(col).strip().lower().startswith("unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def _resolve_y_cols(df: pd.DataFrame, y_cols: Optional[Sequence[str]]) -> List[str]:
    if y_cols:
        resolved = [str(col).strip() for col in y_cols]
        missing = [col for col in resolved if col not in df.columns]
        if missing:
            raise KeyError(f"Missing target columns: {missing}")
        return resolved

    for col in DEFAULT_TARGET_CANDIDATES:
        if col in df.columns:
            return [col]

    raise ValueError(
        "Could not infer target columns. Please set `data_loader.y_cols` explicitly."
    )


def _first_non_null(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[0]


def _aggregate_target_series(series: pd.Series, agg: str):
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return np.nan
    key = str(agg).strip().lower()
    if key == "mean":
        return float(numeric.mean())
    if key == "first":
        return float(numeric.iloc[0])
    if key != "median":
        raise ValueError(f"Unsupported duplicate_target_agg='{agg}'.")
    return float(numeric.median())


def _resolve_feature_cols(
    df: pd.DataFrame,
    y_cols: Sequence[str],
    drop_metadata_cols: Sequence[str],
) -> List[str]:
    excluded = set(y_cols) | set(drop_metadata_cols)
    return [col for col in df.columns if col not in excluded]


def _aggregate_duplicate_input_rows(
    df: pd.DataFrame,
    *,
    y_cols: Sequence[str],
    drop_metadata_cols: Sequence[str],
    target_agg: str = "median",
) -> tuple[pd.DataFrame, Dict[str, int]]:
    feature_cols = _resolve_feature_cols(df, y_cols, drop_metadata_cols)
    if not feature_cols:
        return df.copy(), {"groups": 0, "rows": 0}

    records: List[Dict[str, Any]] = []
    duplicate_groups = 0
    duplicate_rows = 0
    grouped = df.groupby(feature_cols, dropna=False, sort=False)
    for _, group in grouped:
        if len(group) > 1:
            duplicate_groups += 1
            duplicate_rows += len(group)

        row: Dict[str, Any] = {}
        for col in df.columns:
            if col in feature_cols:
                row[col] = group.iloc[0][col]
            elif col in y_cols:
                row[col] = _aggregate_target_series(group[col], target_agg)
            else:
                row[col] = _first_non_null(group[col])
        records.append(row)

    out = pd.DataFrame(records, columns=list(df.columns))
    return out, {"groups": duplicate_groups, "rows": duplicate_rows}


def _coerce_numeric_frame(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame(index=df.index)
    out = pd.DataFrame(index=df.index)
    for col in cols:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    return out


def _summarize_missing(frame: pd.DataFrame) -> Dict[str, int]:
    if frame.empty:
        return {}
    counts = frame.isna().sum()
    return {str(col): int(cnt) for col, cnt in counts.items() if int(cnt) > 0}


def _fill_numeric_with_median(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    medians = frame.median(numeric_only=True)
    return frame.fillna(medians)


def _apply_log_transform_to_frame(
    frame: pd.DataFrame,
    *,
    log_transform_cols: Optional[Sequence[str]],
    log_transform_eps: float = 1e-8,
) -> pd.DataFrame:
    if not log_transform_cols:
        return frame

    out = frame.copy()
    try:
        eps = float(log_transform_eps)
    except (TypeError, ValueError):
        eps = 1e-8
    if eps <= 0:
        eps = 1e-8

    for col in log_transform_cols:
        if col not in out.columns:
            continue
        series = pd.to_numeric(out[col], errors="coerce")
        if series.isna().any():
            raise ValueError(
                f"Cannot log-transform '{col}' because it still contains missing values."
            )
        if (series <= 0).any():
            print(f"[WARN] log_transform '{col}': non-positive values found, clamping to {eps}.")
        out[col] = np.log(np.clip(series.to_numpy(dtype=np.float32), eps, None))
    return out


def _validate_legacy_args(
    *,
    element_cols: Sequence[str],
    promoter_ratio_cols: Optional[Sequence[str]],
) -> None:
    if any(str(col).strip() for col in element_cols):
        raise ValueError(
            "Element/material featurization was removed from the main carbon workflow. "
            "Use model-ready numeric datasets instead."
        )
    if promoter_ratio_cols:
        raise ValueError(
            "Legacy promoter ratio columns are no longer supported in the slimmed loader."
        )


def save_duplicate_input_conflict_report(
    csv_path: str,
    y_cols: Optional[Sequence[str]] = None,
    drop_metadata_cols: Sequence[str] = ("DOI", "Name", "Year"),
    output_dir: Optional[str] = None,
    output_prefix: str = "duplicate_input",
    preserve_null: bool = True,
) -> tuple[Optional[str], Optional[str], int, int]:
    df = _read_csv_with_missing(csv_path, preserve_null=preserve_null)
    df = _strip_colnames(df)
    df = _drop_unnamed_cols(df)
    y_cols_resolved = _resolve_y_cols(df, y_cols)
    feature_cols = _resolve_feature_cols(df, y_cols_resolved, drop_metadata_cols)
    if not feature_cols:
        return None, None, 0, 0

    conflicts: List[pd.DataFrame] = []
    aggregated: List[Dict[str, Any]] = []
    group_count = 0
    row_count = 0

    for group_id, (_, group) in enumerate(df.groupby(feature_cols, dropna=False, sort=False), start=1):
        if len(group) < 2:
            continue
        target_nunique = {
            col: int(pd.to_numeric(group[col], errors="coerce").nunique(dropna=True))
            for col in y_cols_resolved
        }
        if not any(count > 1 for count in target_nunique.values()):
            continue

        tagged = group.copy()
        tagged.insert(0, "__duplicate_group_id__", group_id)
        conflicts.append(tagged)
        group_count += 1
        row_count += len(group)

        agg_row: Dict[str, Any] = {"__duplicate_group_id__": group_id, "__rows__": int(len(group))}
        for col in df.columns:
            if col in feature_cols:
                agg_row[col] = group.iloc[0][col]
            elif col in y_cols_resolved:
                agg_row[col] = _aggregate_target_series(group[col], "median")
                agg_row[f"__{col}_nunique__"] = target_nunique[col]
            else:
                agg_row[col] = _first_non_null(group[col])
        aggregated.append(agg_row)

    if not conflicts:
        return None, None, 0, 0

    out_dir = output_dir or os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(out_dir, exist_ok=True)
    conflict_path = os.path.join(out_dir, f"{output_prefix}_conflicts.csv")
    aggregated_path = os.path.join(out_dir, f"{output_prefix}_aggregated.csv")
    pd.concat(conflicts, axis=0, ignore_index=True).to_csv(conflict_path, index=False)
    pd.DataFrame(aggregated).to_csv(aggregated_path, index=False)
    return conflict_path, aggregated_path, group_count, row_count


def load_smart_data_simple(
    csv_path: str,
    element_cols: tuple[str, ...] = (),
    text_cols: Tuple[str, ...] = (),
    y_cols: Optional[Sequence[str]] = None,
    promoter_ratio_cols: Optional[Sequence[str]] = None,
    promoter_onehot: bool = False,
    promoter_interaction_features: bool = False,
    promoter_pair_onehot: bool = False,
    promoter_pair_onehot_min_count: int = 2,
    promoter_pair_onehot_max_categories: int = 64,
    promoter_interaction_eps: float = 1e-8,
    log_transform_cols: Optional[Sequence[str]] = None,
    log_transform_eps: float = 1e-8,
    element_embedding: str = "disabled",
    drop_metadata_cols: Tuple[str, ...] = ("DOI", "Name", "Year"),
    fill_numeric: str = "median",
    missing_text_token: str = "__MISSING__",
    impute_missing: bool = False,
    impute_method: str = "simple",
    impute_seed: int = 42,
    preserve_null: bool = True,
    impute_type_substring: str = "Type",
    impute_skip_substring: str = "ame",
    aggregate_duplicate_inputs: bool = False,
    duplicate_target_agg: str = "median",
    return_dataframe: bool = False,
):
    del promoter_onehot
    del promoter_interaction_features
    del promoter_pair_onehot
    del promoter_pair_onehot_min_count
    del promoter_pair_onehot_max_categories
    del promoter_interaction_eps
    del element_embedding
    del fill_numeric
    del impute_seed
    del impute_type_substring
    del impute_skip_substring
    del return_dataframe

    _validate_legacy_args(element_cols=element_cols, promoter_ratio_cols=promoter_ratio_cols)

    df = _read_csv_with_missing(csv_path, preserve_null=preserve_null)
    df = _strip_colnames(df)
    df = _drop_unnamed_cols(df)

    y_cols_resolved = _resolve_y_cols(df, y_cols)
    if aggregate_duplicate_inputs:
        df, dup_stats = _aggregate_duplicate_input_rows(
            df,
            y_cols=y_cols_resolved,
            drop_metadata_cols=drop_metadata_cols,
            target_agg=duplicate_target_agg,
        )
        print(
            f"[INFO] Aggregated duplicate inputs: groups={dup_stats['groups']}, rows={dup_stats['rows']}."
        )

    text_cols = tuple(str(col).strip() for col in text_cols if str(col).strip())
    feature_cols = _resolve_feature_cols(df, y_cols_resolved, drop_metadata_cols)
    missing_text_cols = [col for col in text_cols if col not in feature_cols]
    if missing_text_cols:
        raise KeyError(f"text_cols not found in feature columns: {missing_text_cols}")

    text_set = set(text_cols)
    numeric_cols = [col for col in feature_cols if col not in text_set]
    X_num_df = _coerce_numeric_frame(df, numeric_cols)
    Y_df = _coerce_numeric_frame(df, y_cols_resolved)

    if Y_df.isna().any().any():
        missing = _summarize_missing(Y_df)
        raise ValueError(f"Target columns contain missing/non-numeric values: {missing}")

    if impute_missing:
        if str(impute_method).strip().lower() != "simple":
            print(
                f"[WARN] impute_method='{impute_method}' is no longer supported; fallback to simple median fill."
            )
        X_num_df = _fill_numeric_with_median(X_num_df)
    else:
        missing_numeric = _summarize_missing(X_num_df)
        if missing_numeric:
            raise ValueError(
                "Feature columns contain missing/non-numeric values. "
                "Rebuild the model-ready dataset or enable simple imputation. "
                f"Details: {missing_numeric}"
            )

    X_num_df = _apply_log_transform_to_frame(
        X_num_df,
        log_transform_cols=log_transform_cols,
        log_transform_eps=log_transform_eps,
    )

    x_parts: List[np.ndarray] = []
    x_col_names: List[str] = []
    numeric_cols_idx: List[int] = []
    onehot_groups: List[List[int]] = []
    feature_group_map: List[int] = []
    observed_values: Dict[str, List[str]] = {}
    observed_value_counts: Dict[str, Dict[str, int]] = {}
    observed_value_ratios: Dict[str, Dict[str, float]] = {}

    if numeric_cols:
        X_num = X_num_df[numeric_cols].to_numpy(dtype=np.float32)
        x_parts.append(X_num)
        x_col_names.extend(numeric_cols)
        numeric_cols_idx.extend(list(range(X_num.shape[1])))
        feature_group_map.extend([-1] * X_num.shape[1])

    cur_idx = len(x_col_names)
    for group_idx, col in enumerate(text_cols):
        series = df[col].astype("object").where(~df[col].isna(), missing_text_token)
        values = sorted(str(value).strip() for value in series.tolist())
        values = sorted(set(values))
        counts = Counter(str(value).strip() for value in series.tolist())
        matrix = np.zeros((len(df), len(values)), dtype=np.float32)
        value_to_idx = {value: idx for idx, value in enumerate(values)}
        for row_idx, raw_value in enumerate(series.tolist()):
            value = str(raw_value).strip()
            matrix[row_idx, value_to_idx[value]] = 1.0

        cols_encoded = [f"{col}__{value}" for value in values]
        x_parts.append(matrix)
        x_col_names.extend(cols_encoded)
        group_cols = list(range(cur_idx, cur_idx + len(values)))
        onehot_groups.append(group_cols)
        feature_group_map.extend([group_idx] * len(values))
        cur_idx += len(values)

        observed_values[col] = values
        observed_value_counts[col] = {value: int(counts.get(value, 0)) for value in values}
        total = float(sum(observed_value_counts[col].values())) or 1.0
        observed_value_ratios[col] = {
            value: float(observed_value_counts[col][value]) / total for value in values
        }

    X = np.hstack(x_parts) if x_parts else np.empty((len(df), 0), dtype=np.float32)
    Y = Y_df[y_cols_resolved].to_numpy(dtype=np.float32)

    return (
        X,
        Y,
        numeric_cols_idx,
        x_col_names,
        y_cols_resolved,
        observed_values,
        observed_value_counts,
        observed_value_ratios,
        onehot_groups,
        feature_group_map,
    )


def load_raw_data_for_correlation(
    csv_path: str,
    input_len: Optional[int] = None,
    output_len: Optional[int] = None,
    drop_nan: bool = True,
    fill_same_as_train: bool = True,
    element_cols: tuple[str, ...] = (),
    promoter_ratio_cols: Optional[Sequence[str]] = None,
    text_cols: Tuple[str, ...] = (),
    y_cols: Optional[Sequence[str]] = None,
    drop_metadata_cols: Tuple[str, ...] = ("DOI", "Name", "Year"),
    impute_seed: int = 42,
    impute_type_substring: str = "Type",
    impute_skip_substring: str = "ame",
    missing_text_token: str = "__MISSING__",
    impute_method: str = "simple",
    aggregate_duplicate_inputs: bool = False,
    duplicate_target_agg: str = "median",
    preserve_null: bool = True,
) -> pd.DataFrame:
    del fill_same_as_train
    del impute_seed
    del impute_type_substring
    del impute_skip_substring
    del impute_method
    _validate_legacy_args(element_cols=element_cols, promoter_ratio_cols=promoter_ratio_cols)

    df = _read_csv_with_missing(csv_path, preserve_null=preserve_null)
    df = _strip_colnames(df)
    df = _drop_unnamed_cols(df)
    y_cols_resolved = _resolve_y_cols(df, y_cols)

    if aggregate_duplicate_inputs:
        df, _ = _aggregate_duplicate_input_rows(
            df,
            y_cols=y_cols_resolved,
            drop_metadata_cols=drop_metadata_cols,
            target_agg=duplicate_target_agg,
        )

    text_cols = tuple(str(col).strip() for col in text_cols if str(col).strip())
    feature_cols = _resolve_feature_cols(df, y_cols_resolved, drop_metadata_cols)
    missing_text_cols = [col for col in text_cols if col not in feature_cols]
    if missing_text_cols:
        raise KeyError(f"text_cols not found in feature columns: {missing_text_cols}")

    numeric_cols = [col for col in feature_cols if col not in set(text_cols)]
    df_out = df[feature_cols + list(y_cols_resolved)].copy()
    for col in numeric_cols + list(y_cols_resolved):
        df_out[col] = pd.to_numeric(df_out[col], errors="coerce")
    for col in text_cols:
        df_out[col] = df_out[col].astype("object").where(~df_out[col].isna(), missing_text_token)

    if input_len is not None and len(feature_cols) != int(input_len):
        print(
            f"[WARN] Raw-correlation feature count={len(feature_cols)} "
            f"but config input_len={input_len}; using resolved columns."
        )
    if output_len is not None and len(y_cols_resolved) != int(output_len):
        print(
            f"[WARN] Raw-correlation target count={len(y_cols_resolved)} "
            f"but config output_len={output_len}; using resolved columns."
        )

    if drop_nan:
        df_out = df_out.dropna(axis=0, how="any")
    return df_out


def extract_data_statistics(
    X: np.ndarray,
    x_col_names: Sequence[str],
    numeric_cols_idx: Sequence[int],
    Y: Optional[np.ndarray] = None,
    y_col_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {"continuous_cols": {}, "onehot_groups": []}

    for idx in numeric_cols_idx:
        cname = x_col_names[idx]
        col_data = X[:, idx]
        stats["continuous_cols"][cname] = {
            "min": float(np.nanmin(col_data)),
            "max": float(np.nanmax(col_data)),
            "mean": float(np.nanmean(col_data)),
        }

    if Y is not None and y_col_names is not None:
        for idx, cname in enumerate(y_col_names):
            col_data = Y[:, idx]
            stats["continuous_cols"][cname] = {
                "min": float(np.nanmin(col_data)),
                "max": float(np.nanmax(col_data)),
                "mean": float(np.nanmean(col_data)),
            }

    return stats


def build_group_value_vectors(
    observed_values: Dict[str, List[str]],
    element_cols: Sequence[str],
    text_cols: Sequence[str],
    element_embedding: str = "disabled",
    observed_value_counts: Optional[Dict[str, Dict[str, int]]] = None,
    observed_value_ratios: Optional[Dict[str, Dict[str, float]]] = None,
    promoter_onehot: bool = False,
) -> Dict[str, Dict[str, Any]]:
    del element_embedding
    del observed_value_ratios
    del promoter_onehot
    if any(str(col).strip() for col in element_cols):
        raise ValueError(
            "Element/material group vectors are no longer supported in the slimmed loader."
        )

    group_vectors: Dict[str, Dict[str, Any]] = {}
    for col in text_cols:
        values = observed_values.get(col, [])
        if not values:
            continue
        vecs = np.eye(len(values), dtype=np.float32)
        weights = None
        if observed_value_counts and col in observed_value_counts:
            counts = observed_value_counts[col]
            weights_arr = np.array([counts.get(value, 0) for value in values], dtype=float)
            if weights_arr.sum() > 0:
                weights = (weights_arr / weights_arr.sum()).tolist()
        group_vectors[col] = {
            "values": list(values),
            "vectors": vecs,
            "weights": weights,
        }
    return group_vectors
