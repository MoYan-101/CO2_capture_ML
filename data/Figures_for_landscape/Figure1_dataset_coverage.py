#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import tempfile
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=RuntimeWarning)

DATA_DIR = Path(__file__).resolve().parent
OUTPUT_BASENAME = "Figure1_dataset_coverage"
METADATA_COLUMNS = ["Ref number", "Carbon precursors", "DOI", "Year", "Title"]
NUMERIC_COLUMNS = [
    "Sbet (m2g-1)",
    "Vtotal (cm3g-1)",
    "Vmicro (cm3g-1)",
    "V_ultra (cm3g-1)",
    "P (bar)",
    "T (K)",
    "C (%)",
    "O (%)",
    "N (%)",
    "Uptake (mmolg-1)",
]
BENCHMARKS = [
    (273.0, 0.15, "273 K, 0.15 bar"),
    (298.0, 0.15, "298 K, 0.15 bar"),
    (273.0, 1.0, "273 K, 1 bar"),
    (298.0, 1.0, "298 K, 1 bar"),
]
FAMILY_ORDER = [
    "wood/sawdust/bamboo",
    "lignin/lignosulfonate",
    "nut/shell/seed",
    "straw/husk/stalk/agro-residue",
    "sludge/biowaste",
    "polymer/resin",
    "coal/pitch/tar/coke",
    "other",
]
FAMILY_PALETTE = {
    "wood/sawdust/bamboo": "#5B8E7D",
    "lignin/lignosulfonate": "#A8704C",
    "nut/shell/seed": "#C28E2B",
    "straw/husk/stalk/agro-residue": "#7A9E3B",
    "sludge/biowaste": "#4F7CAC",
    "polymer/resin": "#8C5A9E",
    "coal/pitch/tar/coke": "#4D4D4D",
    "other": "#9E9E9E",
}
PRETTY_LABELS = {
    "Sbet (m2g-1)": "S$_\\mathrm{BET}$\n(m$^2$ g$^{-1}$)",
    "Vtotal (cm3g-1)": "V$_\\mathrm{total}$\n(cm$^3$ g$^{-1}$)",
    "Vmicro (cm3g-1)": "V$_\\mathrm{micro}$\n(cm$^3$ g$^{-1}$)",
    "V_ultra (cm3g-1)": "V$_\\mathrm{ultra}$\n(cm$^3$ g$^{-1}$)",
    "P (bar)": "Pressure\n(bar)",
    "T (K)": "Temperature\n(K)",
    "C (%)": "C\n(%)",
    "O (%)": "O\n(%)",
    "N (%)": "N\n(%)",
    "Uptake (mmolg-1)": "CO$_2$ uptake\n(mmol g$^{-1}$)",
}
CANONICAL_ALIASES = {
    "Ref number": ["refnumber", "refno", "reference", "referenceid", "ref"],
    "Carbon precursors": [
        "carbonprecursors",
        "carbonprecursor",
        "precursor",
        "precursors",
        "carbonsource",
        "feedstock",
        "rawmaterial",
    ],
    "Sample name": ["samplename", "sample", "sampleid", "samplecode", "specimenname"],
    "Sbet (m2g-1)": [
        "sbetm2g1",
        "sbet",
        "betsurfacearea",
        "betsurfaceaream2g1",
        "surfacearea",
        "surfaceaream2g1",
    ],
    "Vtotal (cm3g-1)": [
        "vtotalcm3g1",
        "vtotal",
        "totalporevolume",
        "totalporevolumecm3g1",
    ],
    "Vmicro (cm3g-1)": [
        "vmicrocm3g1",
        "vmicro",
        "microporevolume",
        "microporevolumecm3g1",
    ],
    "V_ultra (cm3g-1)": [
        "vultracm3g1",
        "vultra",
        "v_ultra",
        "v_ultracm3g1",
        "ultramicroporevolume",
        "ultramicroporevolumecm3g1",
        "ultraporevolume",
    ],
    "P (bar)": ["pbar", "pressurebar", "pressure", "adsorptionpressurebar"],
    "T (K)": ["tk", "temperaturek", "temperature", "tempk", "adsorptiontemperaturek"],
    "C (%)": ["cpct", "carbonpct", "catpct", "c"],
    "O (%)": ["opct", "oxygenpct", "oatpct", "o"],
    "N (%)": ["npct", "nitrogenpct", "natpct", "n"],
    "Uptake (mmolg-1)": [
        "uptakemmolg1",
        "co2uptakemmolg1",
        "uptake",
        "capacity",
        "capacitymmolg1",
    ],
    "DOI": ["doi"],
    "Year": ["year", "publicationyear"],
    "Title": ["title", "papertitle", "articletitle"],
}
ALIAS_LOOKUP = {
    alias: canonical
    for canonical, aliases in CANONICAL_ALIASES.items()
    for alias in aliases
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Figure 1 dataset coverage plots.")
    parser.add_argument("--input", type=str, default=None, help="Path to the input CSV.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DATA_DIR),
        help="Directory for figure outputs.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="Output DPI for PNG export.")
    return parser.parse_args()


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def simplify_key(value: object) -> str:
    text = clean_text(value).lower()
    text = text.replace("%", "pct")
    text = text.replace("at.pct", "pct").replace("atpct", "pct")
    text = text.replace("^", "")
    text = re.sub(r"[\s_/\\-]+", "", text)
    text = re.sub(r"[(){}\[\],.:;]", "", text)
    return text


def canonicalize_column_name(name: object) -> str:
    clean_name = clean_text(name)
    key = simplify_key(clean_name)
    if key in ALIAS_LOOKUP:
        return ALIAS_LOOKUP[key]
    if "vultra" in key or "ultramicroporevolume" in key:
        return "V_ultra (cm3g-1)"
    if "vmicro" in key or "microporevolume" in key:
        return "Vmicro (cm3g-1)"
    if "vtotal" in key or "totalporevolume" in key:
        return "Vtotal (cm3g-1)"
    if "sbet" in key or ("bet" in key and "surface" in key):
        return "Sbet (m2g-1)"
    return clean_name


def find_input_file(user_path: str | None) -> Path:
    if user_path:
        path = Path(user_path).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path

    search_dirs = [DATA_DIR, DATA_DIR.parent]
    candidate_names = [
        "Data_CO2_adsorption_0417.csv",
        "data_CO2_adsorption_0417.csv",
        "co2_capture_carbon.csv",
    ]
    for search_dir in search_dirs:
        for name in candidate_names:
            path = search_dir / name
            if path.exists():
                return path

    fallback = sorted(
        path
        for search_dir in search_dirs
        for path in search_dir.glob("*.csv")
        if not path.name.startswith("Figure")
        and "mapping" not in path.name.lower()
        and "correlation" not in path.name.lower()
    )
    if fallback:
        return fallback[0]
    raise FileNotFoundError(
        "No input CSV found in the data directory. Expected one of: "
        "Data_CO2_adsorption_0417.csv or co2_capture_carbon.csv."
    )


def read_input_dataframe(path: Path) -> pd.DataFrame:
    errors = []
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise RuntimeError("Failed to decode CSV. Tried encodings: " + "; ".join(errors))


def safe_to_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.map(clean_text)
    cleaned = cleaned.replace({"": np.nan, "-": np.nan, "–": np.nan, "—": np.nan})
    cleaned = cleaned.str.replace(",", "", regex=False)
    cleaned = cleaned.str.replace("−", "-", regex=False)
    while cleaned.fillna("").str.contains(r"\.\.", regex=True).any():
        cleaned = cleaned.str.replace("..", ".", regex=False)
    cleaned = cleaned.str.strip("<>~=≈")
    numeric = pd.to_numeric(cleaned, errors="coerce")
    unresolved = numeric.isna() & cleaned.notna()
    if unresolved.any():
        extracted = cleaned[unresolved].str.extract(
            r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False
        )
        numeric.loc[unresolved] = pd.to_numeric(extracted, errors="coerce")
    return numeric


def contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def map_precursor_family(value: object) -> str:
    text = clean_text(value).lower()
    if not text:
        return "other"

    if contains_any(text, ["lignin", "lignosulfonate"]):
        return "lignin/lignosulfonate"

    if contains_any(
        text,
        [
            "polymer",
            "resin",
            "phenolic",
            "pani",
            "polyacrylonitrile",
            "acrylonitrile",
            "melamine",
            "formaldehyde",
            "polystyrene",
            "polyamide",
            "polyethylene",
            "polypropylene",
            "pam",
            "pan ",
            "pan-",
            "pan@",
            "ion exchange",
        ],
    ):
        return "polymer/resin"

    if contains_any(
        text,
        [
            "coal",
            "pitch",
            "tar",
            "coke",
            "bitumen",
            "petroleum",
            "anthracite",
            "lignite",
            "peat",
            "yall",
            "aduun",
            "tugrug",
            "yang",
            "ovoo",
            "baganuur",
            "pendopo",
            "adaro",
            "mullier",
        ],
    ):
        return "coal/pitch/tar/coke"

    if contains_any(
        text,
        [
            "sludge",
            "waste",
            "biowaste",
            "sewage",
            "digestate",
            "manure",
            "coffee",
            "grounds",
            "spent",
            "peel",
            "fish",
            "scale",
            "tea",
            "pomace",
            "pulp",
            "food",
            "brewery",
            "distillery",
        ],
    ):
        return "sludge/biowaste"

    if contains_any(
        text,
        [
            "straw",
            "husk",
            "stalk",
            "agro",
            "residue",
            "bagasse",
            "cob",
            "pod",
            "bran",
            "leaf",
            "leaves",
            "grass",
            "reed",
            "frond",
            "fiber",
            "fibre",
            "potato",
            "starch",
            "rice",
            "bean",
            "soy",
            "castor",
            "palm",
        ],
    ):
        return "straw/husk/stalk/agro-residue"

    if contains_any(
        text,
        [
            "shell",
            "seed",
            "walnut",
            "coconut",
            "almond",
            "hazelnut",
            "apricot",
            "jujube",
            "pistachio",
            "kernel",
            "pit",
            "nut",
        ],
    ):
        return "nut/shell/seed"

    if contains_any(
        text,
        [
            "wood",
            "sawdust",
            "sawdust",
            "bamboo",
            "pine",
            "cedar",
            "poplar",
            "branch",
            "balsa",
            "timber",
            "oak",
            "fir",
            "spruce",
            "birch",
            "maple",
            "eucalyptus",
            "lumber",
            "bark",
        ],
    ):
        return "wood/sawdust/bamboo"

    return "other"


def apply_nature_style() -> None:
    sns.set_theme(style="white", context="notebook")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 15,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "axes.linewidth": 1.2,
            "axes.titlepad": 10,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "legend.fontsize": 13,
            "figure.titlesize": 18,
            "savefig.bbox": "tight",
            "savefig.transparent": False,
        }
    )


def style_axes(ax: plt.Axes) -> None:
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.2)
        ax.spines[side].set_color("black")
    ax.tick_params(axis="both", direction="in", width=1.1, length=5, color="black")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=19,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def save_figure(fig: plt.Figure, output_dir: Path, basename: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{basename}.png"
    fig.savefig(png_path, dpi=dpi)
    print(f"Saved {png_path}")


def load_and_prepare_data(input_path: str | None, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    csv_path = find_input_file(input_path)
    raw_df = read_input_dataframe(csv_path)
    raw_df = raw_df.rename(columns={col: canonicalize_column_name(col) for col in raw_df.columns})
    raw_df = raw_df.apply(lambda col: col.map(clean_text) if col.dtype == object else col)

    for column in METADATA_COLUMNS:
        if column in raw_df.columns:
            raw_df[column] = raw_df[column].replace("", np.nan).ffill()

    clean_df = raw_df.copy()
    for column in NUMERIC_COLUMNS:
        if column in clean_df.columns:
            clean_df[column] = safe_to_numeric(clean_df[column])

    if {"Vmicro (cm3g-1)", "Vtotal (cm3g-1)"} <= set(clean_df.columns):
        denom = clean_df["Vtotal (cm3g-1)"].replace(0, np.nan)
        clean_df["micropore_fraction"] = clean_df["Vmicro (cm3g-1)"] / denom
    else:
        clean_df["micropore_fraction"] = np.nan

    if {"V_ultra (cm3g-1)", "Vmicro (cm3g-1)"} <= set(clean_df.columns):
        denom = clean_df["Vmicro (cm3g-1)"].replace(0, np.nan)
        clean_df["ultramicropore_fraction"] = clean_df["V_ultra (cm3g-1)"] / denom

    if {"Uptake (mmolg-1)", "Sbet (m2g-1)"} <= set(clean_df.columns):
        denom = clean_df["Sbet (m2g-1)"].replace(0, np.nan)
        clean_df["uptake_per_surface_area"] = clean_df["Uptake (mmolg-1)"] / denom

    if {"Uptake (mmolg-1)", "Vtotal (cm3g-1)"} <= set(clean_df.columns):
        denom = clean_df["Vtotal (cm3g-1)"].replace(0, np.nan)
        clean_df["uptake_per_pore_volume"] = clean_df["Uptake (mmolg-1)"] / denom

    precursor_col = "Carbon precursors"
    if precursor_col in clean_df.columns:
        clean_df["Precursor family"] = clean_df[precursor_col].map(map_precursor_family)
        mapping_df = (
            clean_df[[precursor_col, "Precursor family"]]
            .dropna(subset=[precursor_col])
            .groupby([precursor_col, "Precursor family"], as_index=False)
            .size()
            .rename(columns={"size": "sample_count"})
            .sort_values(["Precursor family", "sample_count", precursor_col], ascending=[True, False, True])
        )
        mapping_path = output_dir / "precursor_family_mapping.csv"
        mapping_df.to_csv(mapping_path, index=False)
        print(f"Saved {mapping_path}")
    else:
        clean_df["Precursor family"] = "other"

    print(f"Loaded {clean_df.shape[0]} rows from {csv_path}")
    print("Benchmark subset counts:")
    for temperature, pressure, label in BENCHMARKS:
        mask = (
            np.isclose(clean_df.get("T (K)", np.nan), temperature, atol=0.25, equal_nan=False)
            & np.isclose(clean_df.get("P (bar)", np.nan), pressure, atol=1e-6, equal_nan=False)
        )
        print(f"  {label}: n={int(mask.sum())}")

    return raw_df, clean_df, csv_path


def make_figure(clean_df: pd.DataFrame) -> plt.Figure:
    year_df = clean_df[["Year"]].copy()
    if "Year" in year_df.columns:
        year_df["Year"] = safe_to_numeric(year_df["Year"])
        year_df = year_df.dropna(subset=["Year"])

    descriptor_columns = [col for col in PRETTY_LABELS if col in clean_df.columns]
    descriptor_counts = clean_df[descriptor_columns].notna().sum().sort_values(ascending=False)

    family_counts = (
        clean_df["Precursor family"]
        .fillna("other")
        .value_counts()
        .reindex(FAMILY_ORDER)
        .dropna()
        .astype(int)
    )

    tp_df = clean_df.dropna(subset=["T (K)", "P (bar)"]).copy()
    tp_counts = tp_df.groupby(["T (K)", "P (bar)"]).size().reset_index(name="count")
    heatmap_df = (
        tp_counts.pivot(index="T (K)", columns="P (bar)", values="count")
        .sort_index(ascending=True)
        .sort_index(axis=1, ascending=True)
    )

    fig, axes = plt.subplots(2, 2, figsize=(15.8, 11.2))
    ax_a, ax_b, ax_c, ax_d = axes.flat

    if not year_df.empty:
        years = year_df["Year"].astype(int)
        bins = np.arange(years.min() - 0.5, years.max() + 1.5, 1.0)
        ax_a.hist(years, bins=bins, color="#4C6A92", edgecolor="black", linewidth=0.8)
        ax_a.set_xlabel("Publication year")
        ax_a.set_ylabel("Count")
        ax_a.set_xticks(sorted(years.unique()))
        ax_a.tick_params(axis="x", rotation=45)
    else:
        ax_a.text(0.5, 0.5, "No year data", ha="center", va="center", transform=ax_a.transAxes)
    ax_a.set_title("Publication timeline")
    ax_a.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)

    if not heatmap_df.empty:
        sns.heatmap(
            heatmap_df,
            ax=ax_b,
            cmap="YlGnBu",
            annot=True,
            fmt=".0f",
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Sample count"},
        )
        ax_b.set_xlabel("Pressure (bar)")
        ax_b.set_ylabel("Temperature (K)")
    else:
        ax_b.text(0.5, 0.5, "No T/P data", ha="center", va="center", transform=ax_b.transAxes)
    ax_b.set_title("Experimental condition coverage")

    if not descriptor_counts.empty:
        labels = [PRETTY_LABELS.get(col, textwrap.fill(col, 12)) for col in descriptor_counts.index]
        ax_c.bar(labels, descriptor_counts.values, color="#B75D4B", edgecolor="black", linewidth=0.8)
        ax_c.set_ylabel("Non-missing count")
        ax_c.tick_params(axis="x", rotation=40)
        ax_c.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    else:
        ax_c.text(0.5, 0.5, "No descriptor data", ha="center", va="center", transform=ax_c.transAxes)
    ax_c.set_title("Descriptor completeness")

    if not family_counts.empty:
        y_positions = np.arange(len(family_counts))
        ax_d.barh(
            y_positions,
            family_counts.values,
            color=[FAMILY_PALETTE.get(name, "#9E9E9E") for name in family_counts.index],
            edgecolor="black",
            linewidth=0.8,
        )
        ax_d.set_yticks(y_positions)
        ax_d.set_yticklabels([textwrap.fill(name, 24) for name in family_counts.index])
        ax_d.invert_yaxis()
        ax_d.set_xlabel("Sample count")
        ax_d.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.7)
        for y_pos, value in zip(y_positions, family_counts.values):
            ax_d.text(value + 2, y_pos, f"{int(value)}", va="center", fontsize=10)
    else:
        ax_d.text(0.5, 0.5, "No precursor-family data", ha="center", va="center", transform=ax_d.transAxes)
    ax_d.set_title("Precursor-family coverage")

    for ax, label in zip(axes.flat, "ABCD"):
        style_axes(ax)
        add_panel_label(ax, label)

    fig.suptitle("Literature dataset coverage for CO$_2$ adsorption on carbon materials", y=1.02)
    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    apply_nature_style()
    output_dir = Path(args.output_dir).expanduser().resolve()
    _, clean_df, _ = load_and_prepare_data(args.input, output_dir)
    fig = make_figure(clean_df)
    save_figure(fig, output_dir, OUTPUT_BASENAME, dpi=args.dpi)
    plt.close(fig)


if __name__ == "__main__":
    main()
