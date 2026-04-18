#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path(__file__).resolve().parent
PROJECT_DATA_DIR = DATA_DIR.parent
OUTPUT_BASENAME = "kde_numeric_distributions"

CANONICAL_ALIASES = {
    "Sbet (m2g-1)": [
        "sbet",
        "sbetm2g1",
        "betsurfacearea",
        "betsurfaceaream2g1",
        "surfacearea",
    ],
    "Vtotal (cm3g-1)": [
        "vtotal",
        "vtotalcm3g1",
        "totalporevolume",
        "totalporevolumecm3g1",
    ],
    "Vmicro (cm3g-1)": [
        "vmicro",
        "vmicrocm3g1",
        "microporevolume",
        "microporevolumecm3g1",
    ],
    "P (bar)": ["p", "pbar", "pressure", "pressurebar"],
    "T (K)": ["t", "tk", "temperature", "temperaturek"],
    "C (%)": ["c", "cpct", "carbonpct"],
    "O (%)": ["o", "opct", "oxygenpct"],
    "N (%)": ["n", "npct", "nitrogenpct"],
    "Uptake (mmolg-1)": ["uptake", "uptakemmolg1", "co2uptake", "co2uptakemmolg1"],
}

ALIAS_LOOKUP = {
    alias: canonical
    for canonical, aliases in CANONICAL_ALIASES.items()
    for alias in aliases
}

PLOT_CONFIGS = [
    ("Sbet (m2g-1)", "S$_{\\mathrm{BET}}$ (m$^2$ g$^{-1}$)", "#0077C8"),
    ("Vtotal (cm3g-1)", "V$_{\\mathrm{total}}$ (cm$^3$ g$^{-1}$)", "#00A087"),
    ("Vmicro (cm3g-1)", "V$_{\\mathrm{micro}}$ (cm$^3$ g$^{-1}$)", "#D55E00"),
    ("P (bar)", "Pressure (bar)", "#56B4E9"),
    ("T (K)", "Temperature (K)", "#CC79A7"),
    ("C (%)", "C (%)", "#E69F00"),
    ("O (%)", "O (%)", "#1F77B4"),
    ("N (%)", "N (%)", "#009E73"),
    ("Uptake (mmolg-1)", "CO$_2$ uptake (mmol g$^{-1}$)", "#C44E52"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot KDE distributions for numeric columns.")
    parser.add_argument("--input", type=str, default=None, help="Input CSV path.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DATA_DIR),
        help="Directory for figure outputs.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="Output DPI.")
    return parser.parse_args()


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def simplify_key(value: object) -> str:
    text = clean_text(value).lower()
    text = text.replace("%", "pct")
    text = text.replace("^", "")
    text = re.sub(r"[\s_/\\-]+", "", text)
    text = re.sub(r"[(){}\[\],.:;]", "", text)
    return text


def canonicalize_column_name(name: object) -> str:
    clean_name = clean_text(name)
    key = simplify_key(clean_name)
    if key in ALIAS_LOOKUP:
        return ALIAS_LOOKUP[key]
    return clean_name


def safe_to_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.map(clean_text)
    cleaned = cleaned.replace({"": np.nan, "-": np.nan, "–": np.nan, "—": np.nan})
    cleaned = cleaned.str.replace(",", "", regex=False)
    cleaned = cleaned.str.replace("−", "-", regex=False)
    cleaned = cleaned.str.replace(r"(?<=\d)\.\.(?=\d)", ".", regex=True)
    cleaned = cleaned.str.replace(r"^[~≈<>≤≥]+", "", regex=True)
    cleaned = cleaned.str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False)
    return pd.to_numeric(cleaned, errors="coerce")


def find_input_file(user_path: str | None) -> Path:
    if user_path:
        path = Path(user_path).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path

    candidate_names = [
        "co2_capture_carbon.csv",
        "Data_CO2_adsorption_0417.csv",
        "data_CO2_adsorption_0417.csv",
    ]
    for search_dir in (PROJECT_DATA_DIR, DATA_DIR):
        for name in candidate_names:
            path = search_dir / name
            if path.exists():
                return path

    raise FileNotFoundError("No input CSV found in the data directory.")


def load_dataframe(input_path: str | None) -> pd.DataFrame:
    csv_path = find_input_file(input_path)
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df = df.rename(columns={col: canonicalize_column_name(col) for col in df.columns})
    for column in df.columns:
        df[column] = df[column].map(clean_text)
    return df


def apply_figure_style() -> None:
    sns.set_theme(style="white", context="notebook")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "savefig.bbox": "tight",
            "savefig.transparent": False,
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.tick_params(axis="both", direction="out", width=1.0, length=4, color="black")
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("Density")


def build_plot_columns(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    selected = []
    for column, title, color in PLOT_CONFIGS:
        if column not in df.columns:
            continue
        numeric = safe_to_numeric(df[column])
        if numeric.notna().sum() < 5:
            continue
        selected.append((column, title, color))
    if not selected:
        raise ValueError("No usable numeric columns were found for KDE plotting.")
    return selected


def get_kde_clip(series: pd.Series) -> tuple[float | None, float | None]:
    finite = series[np.isfinite(series)]
    if finite.empty:
        return (None, None)
    lower = float(finite.min())
    upper = float(finite.max())
    if lower >= 0:
        lower = 0.0
    return (lower, upper)


def plot_kde_panels(df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    columns = build_plot_columns(df)
    n_cols = 3
    n_rows = math.ceil(len(columns) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14.5, 4.0 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (column, title, color) in zip(axes, columns):
        series = safe_to_numeric(df[column]).dropna()
        clip = get_kde_clip(series)
        sns.kdeplot(
            x=series,
            ax=ax,
            fill=True,
            color=color,
            linewidth=1.8,
            alpha=0.25,
            bw_adjust=0.9,
            cut=0,
            clip=clip,
        )
        ax.set_title(title, pad=6)
        style_axis(ax)

    for ax in axes[len(columns) :]:
        ax.set_visible(False)

    fig.subplots_adjust(wspace=0.28, hspace=0.40)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{OUTPUT_BASENAME}.png"
    pdf_path = output_dir / f"{OUTPUT_BASENAME}.pdf"
    fig.savefig(png_path, dpi=dpi, facecolor="white")
    fig.savefig(pdf_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def main() -> None:
    args = parse_args()
    apply_figure_style()
    df = load_dataframe(args.input)
    plot_kde_panels(df, Path(args.output_dir), args.dpi)


if __name__ == "__main__":
    main()
