# CO2 Capture Carbon Preprocessed Datasets

This folder stores the layered preprocessing outputs for the literature-curated `CO2` adsorption dataset.

## Recommended ML Training Datasets

Use these two datasets for model training:

1. Primary baseline dataset:
   [03_model_ready/co2_capture_carbon_structure_baseline.csv](/media/herryao/81ca6f19-78c8-470d-b5a1-5f35b4678058/work_dir/Document/Yan/CO2_Capture/data_preprocessing/preprocessed/co2_capture_carbon/03_model_ready/co2_capture_carbon_structure_baseline.csv:1)

   Model inputs:
   `Sbet (m2g-1)`, `Vtotal (cm3g-1)`, `Vmicro (cm3g-1)`, `P (bar)`, `T (K)`

   Metadata retained in CSV only:
   `Carbon precursors`, `Sample name`

   Target:
   `Uptake (mmolg-1)`

   Use case:
   This is the main complete-case model for the paper. It keeps the core structure descriptors and experimental conditions without imputing `P`, `T`, `Vtotal`, or `Vmicro`.

2. Surface-chemistry refinement dataset:
   [03_model_ready/co2_capture_carbon_surface_chemistry_con.csv](/media/herryao/81ca6f19-78c8-470d-b5a1-5f35b4678058/work_dir/Document/Yan/CO2_Capture/data_preprocessing/preprocessed/co2_capture_carbon/03_model_ready/co2_capture_carbon_surface_chemistry_con.csv:1)

   Model inputs:
   `Sbet (m2g-1)`, `Vtotal (cm3g-1)`, `Vmicro (cm3g-1)`, `P (bar)`, `T (K)`, `C (%)`, `O (%)`, `N (%)`

   Metadata retained in CSV only:
   `Carbon precursors`, `Sample name`

   Target:
   `Uptake (mmolg-1)`

   Use case:
   This is the recommended chemistry-augmented model for testing whether adding elemental descriptors changes predictive performance or SHAP and PDP interpretation.

## Comparison Dataset

[03_model_ready/co2_capture_carbon_surface_chemistry_on.csv](/media/herryao/81ca6f19-78c8-470d-b5a1-5f35b4678058/work_dir/Document/Yan/CO2_Capture/data_preprocessing/preprocessed/co2_capture_carbon/03_model_ready/co2_capture_carbon_surface_chemistry_on.csv:1)

This file keeps only `O (%)` and `N (%)` as elemental descriptors. In the current dataset it has exactly the same `326` rows as the `C/O/N` dataset, so it is mainly useful as a controlled comparison for testing whether adding `C (%)` changes the model.

## Preprocessing Strategy

The preprocessing follows a conservative literature-data workflow:

1. Metadata forward-fill only:
   `Ref number`, `Carbon precursors`, `DOI`, `Year`, and `Title` are forward-filled because they are paper-level metadata recorded once per paper block, not true feature missingness.

2. Missing target removal:
   Rows with missing `Uptake (mmolg-1)` are removed and never used for supervised training.

3. Missing condition removal:
   Rows with missing `P (bar)` or `T (K)` are removed because these are core experimental conditions and should not be imputed.

4. Complete-case baseline construction:
   The main baseline dataset keeps only rows with complete `Sbet`, `Vtotal`, `Vmicro`, `P`, `T`, and `Uptake`.

5. Surface-chemistry subset construction:
   A second dataset is created by requiring complete `O (%)` and `N (%)`, and the recommended chemistry-training dataset further keeps complete `C (%)`.

6. No mainline imputation for `C/O/N`:
   `C (%)`, `O (%)`, and `N (%)` are not globally imputed for the main analysis because the missingness is paper-dependent rather than plausibly missing completely at random.

## Layer Outputs

- `01_metadata_ffill/`
  Metadata restored by forward-fill.
- `02_quality_filtered/`
  Intermediate filtered datasets after removing missing `Uptake`, then missing `P/T`.
- `03_model_ready/`
  Final complete-case datasets used for ML.
- `reports/`
  Row-count summary, per-stage missingness summary, and dataset manifest.

## Current Row Counts

- Raw dataset: `1293`
- After metadata forward-fill: `1293`
- After removing missing `Uptake`: `1179`
- After removing missing `P/T`: `1147`
- Main baseline dataset: `1000`
- `C/O/N` chemistry-training dataset: `326`
- `O/N` comparison dataset: `326`

## Why 1147 Became 1000

The step from `1147` to `1000` is the complete-case filter for the main baseline model.
Those `147` removed rows still had missing structure descriptors:

- missing only `Vmicro`: `84`
- missing only `Vtotal`: `42`
- missing both `Vtotal` and `Vmicro`: `20`
- missing `Sbet`, `Vtotal`, and `Vmicro`: `1`

So the drop is caused by incomplete structure descriptors, not by further deletion of `Uptake`, `P`, or `T`.

## Reproducibility

Regenerate this folder with:

```bash
.venv/bin/python -m data_preprocessing.prepare_co2_capture_datasets
```

The default output path is:

```text
data_preprocessing/preprocessed/co2_capture_carbon
```

## Training Configs

Use these config files for the model-ready training runs:

- [configs/config.baseline.yaml](/media/herryao/81ca6f19-78c8-470d-b5a1-5f35b4678058/work_dir/Document/Yan/CO2_Capture/configs/config.baseline.yaml:1)
  trains on `co2_capture_carbon_structure_baseline.csv`
- [configs/config.surface_con.yaml](/media/herryao/81ca6f19-78c8-470d-b5a1-5f35b4678058/work_dir/Document/Yan/CO2_Capture/configs/config.surface_con.yaml:1)
  trains on `co2_capture_carbon_surface_chemistry_con.csv`
- [configs/config.surface_on.yaml](/media/herryao/81ca6f19-78c8-470d-b5a1-5f35b4678058/work_dir/Document/Yan/CO2_Capture/configs/config.surface_on.yaml:1)
  trains on `co2_capture_carbon_surface_chemistry_on.csv`

Current training order:
`aggregate duplicates -> build X/Y -> grouped split -> standardize/train`

Convenient commands:

```bash
bash run.sh baseline
bash run.sh surface_con
bash run.sh surface_on
```
