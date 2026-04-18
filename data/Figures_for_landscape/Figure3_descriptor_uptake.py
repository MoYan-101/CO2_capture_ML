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
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext, LogLocator

warnings.filterwarnings("ignore", category=RuntimeWarning)

DATA_DIR = Path(__file__).resolve().parent
OUTPUT_BASENAME = "Figure3_descriptor_uptake"
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
    parser = argparse.ArgumentParser(description="Create Figure 3 descriptor-uptake plots.")
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


def lowess_smooth(x: np.ndarray, y: np.ndarray, frac: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 5:
        order = np.argsort(x)
        return x[order], y[order]

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    n_points = x.size
    window = max(5, int(np.ceil(frac * n_points)))
    y_smooth = np.zeros(n_points, dtype=float)

    for i in range(n_points):
        distances = np.abs(x - x[i])
        nearest = np.argsort(distances)[:window]
        max_distance = distances[nearest][-1]
        if max_distance == 0:
            weights = np.ones_like(nearest, dtype=float)
        else:
            scaled = distances[nearest] / max_distance
            weights = (1 - scaled**3) ** 3
        design = np.column_stack([np.ones(nearest.size), x[nearest]])
        xtwx = design.T @ (weights[:, None] * design)
        xtwy = design.T @ (weights * y[nearest])
        beta = np.linalg.pinv(xtwx) @ xtwy
        y_smooth[i] = beta[0] + beta[1] * x[i]

    return x, y_smooth


def scatter_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_column: str,
    x_label: str,
    panel_label: str,
    norm: LogNorm,
    y_limits: tuple[float, float],
) -> None:
    subset = data[[x_column, "Uptake (mmolg-1)", "P (bar)"]].dropna()
    subset = subset[subset["Uptake (mmolg-1)"] > 0].copy()
    if subset.empty:
        ax.text(0.5, 0.5, "No complete cases", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel(x_label)
        ax.set_ylabel("CO$_2$ uptake (mmol g$^{-1}$)")
        apply_log_y_axis_style(ax)
        ax.set_ylim(*y_limits)
        style_axes(ax)
        add_panel_label(ax, panel_label)
        return

    scatter = ax.scatter(
        subset[x_column],
        subset["Uptake (mmolg-1)"],
        c=subset["P (bar)"],
        cmap="viridis",
        norm=norm,
        s=34,
        alpha=0.82,
        edgecolors="white",
        linewidths=0.35,
    )
    smooth_x, smooth_y = lowess_smooth(
        subset[x_column].to_numpy(dtype=float),
        subset["Uptake (mmolg-1)"].to_numpy(dtype=float),
        frac=0.33,
    )
    ax.plot(smooth_x, smooth_y, color="black", linewidth=1.8, zorder=4)
    ax.set_xlabel(x_label)
    ax.set_ylabel("CO$_2$ uptake (mmol g$^{-1}$)")
    ax.set_title(f"n={len(subset)}")
    apply_log_y_axis_style(ax)
    ax.set_ylim(*y_limits)
    ax.grid(color="#E0E0E0", linewidth=0.55, alpha=0.55)
    style_axes(ax)
    add_panel_label(ax, panel_label)
    return scatter


def make_figure(clean_df: pd.DataFrame) -> plt.Figure:
    descriptors = [
        ("Vmicro (cm3g-1)", "V$_\\mathrm{micro}$ (cm$^3$ g$^{-1}$)"),
        ("Sbet (m2g-1)", "S$_\\mathrm{BET}$ (m$^2$ g$^{-1}$)"),
        ("O (%)", "O content (%)"),
        ("N (%)", "N content (%)"),
    ]
    if "P (bar)" not in clean_df.columns or clean_df["P (bar)"].dropna().empty:
        raise ValueError("Pressure data are required for Figure 3.")

    positive_pressures = clean_df["P (bar)"].dropna()
    positive_pressures = positive_pressures[positive_pressures > 0]
    norm = LogNorm(vmin=float(positive_pressures.min()), vmax=float(positive_pressures.max()))
    positive_uptake = clean_df["Uptake (mmolg-1)"].dropna()
    positive_uptake = positive_uptake[positive_uptake > 0]
    y_min_positive = float(positive_uptake.min())
    y_max = float(positive_uptake.max())
    y_lower = 10 ** np.floor(np.log10(y_min_positive))
    y_upper = 10 ** np.ceil(np.log10(y_max))

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 11), constrained_layout=True)
    scatter_artist = None
    for ax, panel_label, (column, x_label) in zip(axes.flat, "ABCD", descriptors):
        if column not in clean_df.columns:
            ax.text(0.5, 0.5, f"Missing column:\n{column}", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel(x_label)
            ax.set_ylabel("CO$_2$ uptake (mmol g$^{-1}$)")
            apply_log_y_axis_style(ax)
            ax.set_ylim(y_lower, y_upper)
            style_axes(ax)
            add_panel_label(ax, panel_label)
            continue
        scatter_artist = scatter_panel(ax, clean_df, column, x_label, panel_label, norm, (y_lower, y_upper))

    if scatter_artist is not None:
        colorbar = fig.colorbar(scatter_artist, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
        colorbar.set_label("Pressure (bar)")
        colorbar.outline.set_linewidth(1.0)

    fig.suptitle("Descriptor-performance relationships for CO$_2$ adsorption", y=1.02)
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
