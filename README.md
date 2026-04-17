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

## Default feature setup

- Categorical descriptor: `Carbon precursors`
- Numeric descriptors:
  - `Sbet (m2g-1)`
  - `Vtotal (cm3g-1)`
  - `Vmicro (cm3g-1)`
  - `P (bar)`
  - `T (K)`
  - `C (%)`
  - `O (%)`
  - `N (%)`
- Target:
  - `Uptake (mmolg-1)`

## Quick start

1. Install dependencies into a virtual environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

2. Review the configs:

- `configs/config.test.yaml` for a fast smoke run
- `configs/config.full.yaml` for a longer search

3. Run the pipeline:

```bash
bash run.sh test
# or
bash run.sh full
```

## Notes

- The current configs are intentionally conservative and use only one
  categorical descriptor by default. Expand `text_cols`,
  `log_transform_cols`, and `heatmap_axes` after you finalize the paper's
  descriptor set.
- Raw source data remains in `Data/`; the pipeline reads the normalized CSV
  in `data/`.
