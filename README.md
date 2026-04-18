# CO2 Capture by Carbon

Suggested GitHub repository name: `carbon-capture-ml`

This repository is a clean CO2-capture project scaffold derived from
`MoYan-101/CommsEng_Ea` and retargeted to carbon-based adsorbents. The
current study framing is aligned with a data-driven article on uptake
profiles, structure-conditions-performance relationships, and
machine-learning-assisted interpretation for ACS Sustainable Chemistry &
Engineering.

## Current dataset

- Raw spreadsheet: `Data/Date-CO2 adsorption.xlsx`
- Working CSV: `data/co2_capture_carbon.csv`
- Default target: `Uptake (mmolg-1)`

The processed CSV was generated from the first worksheet in the Excel file
with normalized header names so the pipeline can consume it directly.

## Default model-ready feature setup

- Baseline numeric descriptors:
  - `Sbet (m2g-1)`
  - `Vtotal (cm3g-1)`
  - `Vmicro (cm3g-1)`
  - `P (bar)`
  - `T (K)`
- Chemistry-augmented numeric descriptors:
  - `C (%)`
  - `O (%)`
  - `N (%)`
- Metadata retained in CSV only:
  - `Carbon precursors`
  - `Sample name`
- Target:
  - `Uptake (mmolg-1)`

## Variable Notes

The core variables used in the current dataset are interpreted as follows:

- `Sbet (m2g-1)`: BET specific surface area
- `Vtotal (cm3g-1)`: total pore volume
- `Vultra (cm3g-1)`: ultramicropore volume
- `Vmicro (cm3g-1)`: micropore volume
- `T (K)`: adsorption temperature
- `P (bar)`: adsorption pressure
- `O (wt.%)`: oxygen content
- `N (wt.%)`: nitrogen content
- `Uptake (mmolg-1)`: CO2 uptake

## Dataset Curation Rationale

In the current literature, the CO2 adsorption mechanism of porous carbons is
usually discussed by considering pore structure and surface chemistry
together. However, the two contributions are rarely decoupled in an effective
way, which makes the intrinsic role of pore-size distribution difficult to
identify clearly.

This project aims to build a curated cross-literature carbon dataset and, as
far as possible, minimize interference from surface-chemistry effects so that
the dominant pore-structure variables governing CO2 adsorption can be
re-examined. Combined with process modeling and sensitivity analysis, this
framework is intended to clarify how the relative roles of ultramicropores and
micropores evolve with pressure, and to propose a minimally sufficient
pore-structure descriptor framework with broad applicability across the
literature.

### Inclusion Focus

- The main line of sample design should be activation, templating, or thermal
  treatment for pore tuning.
- The study should not intentionally center on heteroatom doping such as
  `N/S/P/B`.
- The study should not intentionally center on post-functionalization.
- The study should not intentionally center on loading metals or metal oxides.

### Exclusion Focus

Exclude studies in which the main scientific variable is clearly one of the
following:

- impurity-atom doping as the dominant design axis
- strengthened surface functional groups as the dominant design axis
- explicit oxidation, amination, or other post-synthetic surface modification

### Measurement Notes

- Pore-size distributions were derived from the `N2` adsorption isotherm at
  `77 K` using density functional theory (`DFT`).
- `Sbet` was determined using the Brunauer-Emmett-Teller (`BET`) method.
- Total pore volume was obtained from the adsorption data, and pore volume
  below `2 nm` was used to quantify the small-pore contribution.
- Elemental composition was measured by elemental analysis; reported values in
  the source notes are expressed as weight percent (`wt.%`).

## Quick start

1. Install dependencies into a virtual environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

2. Review the configs:

- `configs/config.baseline.yaml` for the structure-only main model
- `configs/config.surface_con.yaml` for the chemistry-augmented model
- `configs/config.surface_on.yaml` for the `O/N`-only chemistry comparison

3. Run the pipeline:

```bash
bash run.sh baseline
# or
bash run.sh surface_con
# or
bash run.sh surface_on
```

See `How to run` below for output paths, `RUN_ID`, and custom config usage.

## Reusable ML Environment

If you want one reusable ML environment for future projects, use the
provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate ml
```

If you later update dependencies:

```bash
conda env update -f environment.yml --prune
```

## Notes

- The model-ready configs intentionally exclude identifier-like categorical
  inputs such as `Carbon precursors` and `Sample name`; those columns remain
  in the CSVs as metadata only.
- Current training order for the model-ready configs is:
  `aggregate duplicates -> build X/Y -> grouped split -> standardize/train`.
- Raw source data remains in `Data/`; the pipeline reads the normalized CSV
  in `data/`.


## How to run

Run commands from the repository root.

Recommended model-ready runs:

```bash
bash run.sh baseline
bash run.sh surface_con
bash run.sh surface_on
```

Default behavior:

```bash
bash run.sh
```

This now defaults to `baseline`.

Custom run id:

```bash
RUN_ID=baseline_v1 bash run.sh baseline
```

Custom config path:

```bash
bash run.sh configs/config.baseline.yaml
```

What `run.sh` does:

- Runs `train.py`
- Then runs `inference.py`
- Then runs `visualization.py`

When prompted for `overfit_penalty_alpha`, press `Enter` to use the value from the config.

Output paths:

- Models and metadata:
  `models/<csv_name>/<run_id>/`
- Intermediate arrays, metrics, and inference outputs:
  `postprocessing/<csv_name>/<run_id>/`
- Final figures:
  `evaluation/figures/<csv_name>/<run_id>/`
