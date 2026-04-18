"""
Layered preprocessing for literature-curated CO2 adsorption data.

This module builds intermediate datasets that follow a conservative strategy:
1. Forward-fill paper-level metadata.
2. Drop rows with missing target values.
3. Drop rows with missing experimental conditions (P/T).
4. Build complete-case model-ready datasets for the structure baseline and
   surface-chemistry subsets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

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
}

TARGET_COL = "Uptake (mmolg-1)"
CONDITION_COLS = ["P (bar)", "T (K)"]
METADATA_FFILL_COLS = ["Ref number", "Carbon precursors", "DOI", "Year", "Title"]
METADATA_KEEP_COLS = ["Ref number", "Carbon precursors", "Sample name", "DOI", "Year", "Title"]
NUMERIC_COLS = [
    "Sbet (m2g-1)",
    "Vtotal (cm3g-1)",
    "Vmicro (cm3g-1)",
    "P (bar)",
    "T (K)",
    "C (%)",
    "O (%)",
    "N (%)",
    TARGET_COL,
]

DATASET_SPECS = {
    "structure_baseline": {
        "required_numeric_cols": [
            "Sbet (m2g-1)",
            "Vtotal (cm3g-1)",
            "Vmicro (cm3g-1)",
            "P (bar)",
            "T (K)",
        ],
        "notes": "Complete-case baseline with structure descriptors and conditions only.",
    },
    "surface_chemistry_on": {
        "required_numeric_cols": [
            "Sbet (m2g-1)",
            "Vtotal (cm3g-1)",
            "Vmicro (cm3g-1)",
            "P (bar)",
            "T (K)",
            "O (%)",
            "N (%)",
        ],
        "notes": "O/N-only surface-chemistry variant retained for comparison; currently redundant with the C/O/N row set.",
    },
    "surface_chemistry_con": {
        "required_numeric_cols": [
            "Sbet (m2g-1)",
            "Vtotal (cm3g-1)",
            "Vmicro (cm3g-1)",
            "P (bar)",
            "T (K)",
            "C (%)",
            "O (%)",
            "N (%)",
        ],
        "notes": "Recommended chemistry-augmented training set with complete C/O/N descriptors.",
    },
}


def _normalize_missing(value):
    if pd.isna(value):
        return pd.NA
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in MISSING_STRINGS:
            return pd.NA
        return stripped
    return value


def _unique_in_order(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _ensure_columns(df: pd.DataFrame, required_cols: Sequence[str]) -> None:
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing expected columns: {missing_cols}")


def load_normalized_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=object)
    df.columns = [str(col).strip() for col in df.columns]

    for col in df.columns:
        df[col] = df[col].map(_normalize_missing)

    numeric_cols = [col for col in NUMERIC_COLS if col in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def forward_fill_metadata(df: pd.DataFrame, metadata_cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in metadata_cols:
        if col in out.columns:
            out[col] = out[col].ffill()
    return out


def drop_missing_rows(df: pd.DataFrame, subset: Sequence[str]) -> pd.DataFrame:
    valid_subset = [col for col in subset if col in df.columns]
    return df.dropna(subset=valid_subset, how="any").reset_index(drop=True)


def build_model_ready_dataset(df: pd.DataFrame, required_numeric_cols: Sequence[str]) -> pd.DataFrame:
    required_cols = list(required_numeric_cols) + [TARGET_COL]
    complete_df = drop_missing_rows(df, required_cols)
    keep_cols = _unique_in_order(METADATA_KEEP_COLS + list(required_numeric_cols) + [TARGET_COL])
    keep_cols = [col for col in keep_cols if col in complete_df.columns]
    return complete_df.loc[:, keep_cols].copy()


def compute_missingness(df: pd.DataFrame, stage_name: str) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    row_count = int(len(df))
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        records.append(
            {
                "stage": stage_name,
                "column": col,
                "missing_count": missing_count,
                "missing_ratio": float(missing_count / row_count) if row_count else 0.0,
            }
        )
    return records


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_layered_datasets(input_csv: Path, output_root: Path) -> Dict[str, object]:
    df_raw = load_normalized_csv(input_csv)
    _ensure_columns(
        df_raw,
        list(METADATA_KEEP_COLS)
        + ["Sbet (m2g-1)", "Vtotal (cm3g-1)", "Vmicro (cm3g-1)", "P (bar)", "T (K)", TARGET_COL],
    )

    output_root.mkdir(parents=True, exist_ok=True)
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    stages: Dict[str, pd.DataFrame] = {
        "raw": df_raw,
        "metadata_ffill": forward_fill_metadata(df_raw, METADATA_FFILL_COLS),
    }
    stages["target_complete"] = drop_missing_rows(stages["metadata_ffill"], [TARGET_COL])
    stages["target_condition_complete"] = drop_missing_rows(
        stages["target_complete"],
        CONDITION_COLS,
    )

    stage_paths = {
        "metadata_ffill": output_root / "01_metadata_ffill" / "co2_capture_carbon_metadata_ffill.csv",
        "target_complete": output_root / "02_quality_filtered" / "co2_capture_carbon_target_complete.csv",
        "target_condition_complete": (
            output_root
            / "02_quality_filtered"
            / "co2_capture_carbon_target_condition_complete.csv"
        ),
    }
    for stage_name, path in stage_paths.items():
        save_dataframe(stages[stage_name], path)

    model_ready_paths: Dict[str, Path] = {}
    dataset_manifest: Dict[str, Dict[str, object]] = {}
    for dataset_name, spec in DATASET_SPECS.items():
        dataset_df = build_model_ready_dataset(
            stages["target_condition_complete"],
            spec["required_numeric_cols"],
        )
        numeric_cols = list(spec["required_numeric_cols"])
        dataset_path = output_root / "03_model_ready" / f"co2_capture_carbon_{dataset_name}.csv"
        save_dataframe(dataset_df, dataset_path)
        stages[dataset_name] = dataset_df
        model_ready_paths[dataset_name] = dataset_path
        dataset_manifest[dataset_name] = {
            "path": str(dataset_path),
            "rows": int(len(dataset_df)),
            "feature_cols": list(numeric_cols),
            "text_cols": [],
            "numeric_cols": list(numeric_cols),
            "metadata_cols": list(METADATA_KEEP_COLS),
            "target_cols": [TARGET_COL],
            "drop_metadata_cols": ["Ref number", "Carbon precursors", "Sample name", "DOI", "Year", "Title"],
            "input_len": int(len(numeric_cols)),
            "output_len": 1,
            "notes": spec["notes"],
        }

    ordered_stage_names = [
        "raw",
        "metadata_ffill",
        "target_complete",
        "target_condition_complete",
        "structure_baseline",
        "surface_chemistry_on",
        "surface_chemistry_con",
    ]
    parent_stage = {
        "raw": None,
        "metadata_ffill": "raw",
        "target_complete": "metadata_ffill",
        "target_condition_complete": "target_complete",
        "structure_baseline": "target_condition_complete",
        "surface_chemistry_on": "target_condition_complete",
        "surface_chemistry_con": "target_condition_complete",
    }
    row_summary = []
    for stage_name in ordered_stage_names:
        row_count = int(len(stages[stage_name]))
        parent_name = parent_stage[stage_name]
        parent_rows = row_count if parent_name is None else int(len(stages[parent_name]))
        row_summary.append(
            {
                "stage": stage_name,
                "rows": row_count,
                "parent_stage": parent_name,
                "dropped_from_parent": int(parent_rows - row_count),
            }
        )

    missingness_records: List[Dict[str, object]] = []
    for stage_name in ordered_stage_names:
        missingness_records.extend(compute_missingness(stages[stage_name], stage_name))

    row_summary_path = reports_dir / "stage_row_counts.csv"
    missingness_path = reports_dir / "stage_missingness.csv"
    manifest_path = reports_dir / "dataset_manifest.json"

    pd.DataFrame(row_summary).to_csv(row_summary_path, index=False)
    pd.DataFrame(missingness_records).to_csv(missingness_path, index=False)

    manifest = {
        "input_csv": str(input_csv),
        "output_root": str(output_root),
        "metadata_forward_fill_cols": METADATA_FFILL_COLS,
        "stages": {
            "metadata_ffill": str(stage_paths["metadata_ffill"]),
            "target_complete": str(stage_paths["target_complete"]),
            "target_condition_complete": str(stage_paths["target_condition_complete"]),
        },
        "datasets": dataset_manifest,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "row_summary_path": row_summary_path,
        "missingness_path": missingness_path,
        "manifest_path": manifest_path,
        "row_summary": row_summary,
        "datasets": dataset_manifest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build layered CO2 capture datasets.")
    parser.add_argument(
        "--input",
        default="data/co2_capture_carbon.csv",
        help="Path to the normalized source CSV.",
    )
    parser.add_argument(
        "--output-root",
        default="data_preprocessing/preprocessed/co2_capture_carbon",
        help="Root directory for intermediate and model-ready outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_layered_datasets(
        input_csv=Path(args.input),
        output_root=Path(args.output_root),
    )
    print("Saved layered preprocessing outputs:")
    for item in result["row_summary"]:
        print(
            f"  - {item['stage']}: rows={item['rows']}, "
            f"dropped_from_parent={item['dropped_from_parent']}"
        )
    print(f"Manifest: {result['manifest_path']}")


if __name__ == "__main__":
    main()
