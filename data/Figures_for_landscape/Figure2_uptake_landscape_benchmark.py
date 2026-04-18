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
from matplotlib.ticker import LogFormatterMathtext, LogLocator

warnings.filterwarnings("ignore", category=RuntimeWarning)

DATA_DIR = Path(__file__).resolve().parent
OUTPUT_BASENAME = "Figure2_uptake_landscape_benchmark"
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
    parser = argparse.ArgumentParser(description="Create Figure 2 uptake landscape plots.")
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


def apply_log_y_axis_style(ax: plt.Axes) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10, labelOnlyBase=True))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.tick_params(axis="y", which="minor", direction="out", width=0.9, length=4, color="black")


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


def benchmark_subsets(clean_df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    subsets = []
    for temperature, pressure, label in BENCHMARKS:
        mask = (
            np.isclose(clean_df.get("T (K)", np.nan), temperature, atol=0.25, equal_nan=False)
            & np.isclose(clean_df.get("P (bar)", np.nan), pressure, atol=1e-6, equal_nan=False)
        )
        subset = clean_df.loc[mask].copy()
        subset = subset.dropna(subset=["Uptake (mmolg-1)", "Precursor family"])
        subset = subset[subset["Uptake (mmolg-1)"] > 0].copy()
        subsets.append((label, subset))
    return subsets


def family_tick_label(family: str, count: int) -> str:
    return f"{textwrap.fill(family, 12)}\n(n={count})"


def make_figure(clean_df: pd.DataFrame) -> plt.Figure:
    subsets = benchmark_subsets(clean_df)
    y_max = 0.0
    y_min_positive = np.inf
    for _, subset in subsets:
        if not subset.empty:
            y_max = max(y_max, float(subset["Uptake (mmolg-1)"].max()))
            y_min_positive = min(y_min_positive, float(subset["Uptake (mmolg-1)"].min()))

    if not np.isfinite(y_min_positive):
        y_min_positive = 1e-2

    lower_decade = 10 ** np.floor(np.log10(y_min_positive))
    upper_decade = 10 ** np.ceil(np.log10(max(y_max, y_min_positive)))
    y_lower = lower_decade
    y_upper = upper_decade

    fig, axes = plt.subplots(2, 2, figsize=(18.2, 12.8), sharey=True)

    for ax, (panel_label, (condition_label, subset)) in zip(axes.flat, zip("ABCD", subsets)):
        if subset.empty:
            ax.text(0.5, 0.5, "No matched samples", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(condition_label)
            apply_log_y_axis_style(ax)
            ax.set_ylim(y_lower, y_upper)
            style_axes(ax)
            add_panel_label(ax, panel_label)
            continue

        grouped = subset.groupby("Precursor family")["Uptake (mmolg-1)"]
        order = grouped.median().sort_values(ascending=False).index.tolist()
        family_counts = grouped.size().reindex(order)
        palette = [FAMILY_PALETTE.get(name, "#9E9E9E") for name in order]

        use_violin = subset.shape[0] <= 180 and family_counts.max() <= 30
        if use_violin:
            sns.violinplot(
                data=subset,
                x="Precursor family",
                y="Uptake (mmolg-1)",
                order=order,
                palette=palette,
                inner=None,
                linewidth=0.0,
                cut=0,
                ax=ax,
            )

        sns.boxplot(
            data=subset,
            x="Precursor family",
            y="Uptake (mmolg-1)",
            order=order,
            width=0.28,
            showfliers=False,
            boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.0, "zorder": 3},
            whiskerprops={"color": "black", "linewidth": 1.0},
            capprops={"color": "black", "linewidth": 1.0},
            medianprops={"color": "black", "linewidth": 1.2},
            ax=ax,
        )
        sns.stripplot(
            data=subset,
            x="Precursor family",
            y="Uptake (mmolg-1)",
            order=order,
            hue="Precursor family",
            palette=FAMILY_PALETTE,
            dodge=False,
            jitter=0.24,
            size=4.3,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.35,
            ax=ax,
        )
        if ax.get_legend() is not None:
            ax.get_legend().remove()

        ax.set_title(f"{condition_label}  (n={len(subset)})")
        ax.set_xlabel("")
        ax.set_ylabel("CO$_2$ uptake (mmol g$^{-1}$)")
        apply_log_y_axis_style(ax)
        ax.set_ylim(y_lower, y_upper)
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(
            [family_tick_label(family, int(family_counts.loc[family])) for family in order],
            rotation=38,
            ha="right",
        )
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.65)
        style_axes(ax)
        add_panel_label(ax, panel_label)

    fig.suptitle("Benchmark CO$_2$ uptake landscape across precursor families", y=1.02)
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
