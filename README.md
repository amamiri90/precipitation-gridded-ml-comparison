# Precipitation Gridded Dataset vs ML Models Comparison

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)

A reproducible pipeline that evaluates **10 gridded precipitation datasets** against daily station observations and fits three machine-learning models (Linear Regression, Random Forest, Gradient Boosting) to improve estimates.

Developed as part of a PhD study on precipitation dataset performance in hydrology.

---

## Gridded Datasets Evaluated

| Dataset | Source type |
|---------|-------------|
| CCS | Satellite |
| CDR | Satellite CDR |
| CHIRPS | Satellite + gauge blended |
| E-OBS | Gauge-based gridded |
| ERA5 | Reanalysis (ECMWF) |
| GPMIMERG | Satellite (GPM) |
| MSWEP | Multi-source blended |
| PDIR | Satellite |
| PERSIANN | Satellite (deep learning) |
| SM2RAIN | Soil-moisture inverted |

---

## Performance Metrics

| Metric | Full name |
|--------|-----------|
| **CC** | Pearson Correlation Coefficient |
| **RMSE** | Root Mean Square Error (mm) |
| **MAE** | Mean Absolute Error (mm) |
| **PBIAS** | Percent Bias (%) |
| **NSE** | Nash–Sutcliffe Efficiency |
| **KGE** | Kling–Gupta Efficiency |

---

## Pipeline Overview

```
Excel station files (.xlsx)
        │
        ▼
  Data loading & validation
  Date parsing & sorting
        │
        ▼
  Time feature extraction (YEAR, MONTH)
        │
        ▼
  Chronological 80/20 train–test split
        │
  ┌─────┬───────────────┬──────────────────┐
  │ LR  │ Random Forest │ Gradient Boosting │
  └─────┴───────────────┴──────────────────┘
        │  (predict on all rows)
        ▼
  Metrics (overall / by year / by month)
        │
  ┌─────────────┬─────────────────────────┐
  │ CSV outputs │ 300 dpi PNG scatter plots│
  └─────────────┴─────────────────────────┘
```

All ML pipelines include **median imputation** for missing gridded values
and **StandardScaler** for Linear Regression.

---

## Requirements

- Python 3.9 or higher
- See `requirements.txt`

```bash
pip install -r requirements.txt
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/precipitation-gridded-ml-comparison.git
cd precipitation-gridded-ml-comparison

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Data Format

Each station Excel file must contain the following columns (column names are case-sensitive):

| Column | Type | Description |
|--------|------|-------------|
| `Date` | date | Daily date |
| `OBSERVED` | float | Station-observed precipitation (mm) |
| `CCS` | float | CCS gridded value (mm) |
| `CDR` | float | CDR gridded value (mm) |
| `CHIRPS` | float | CHIRPS gridded value (mm) |
| `E-OBS` | float | E-OBS gridded value (mm) |
| `ERA5` | float | ERA5 gridded value (mm) |
| `GPMIMERG` | float | GPM IMERG gridded value (mm) |
| `MSWEP` | float | MSWEP gridded value (mm) |
| `PDIR` | float | PDIR gridded value (mm) |
| `PERSIANN` | float | PERSIANN gridded value (mm) |
| `SM2RAIN` | float | SM2RAIN gridded value (mm) |

Missing values (`NaN`) are handled automatically via median imputation.

See `data/README.md` for details on the expected file naming convention.

---

## Usage

### Option 1 — Environment variable (recommended)

```bash
# Linux / macOS
export PRECIP_DATA_DIR=/path/to/your/data/folder
python compare_precipitation_models.py

# Windows Command Prompt
set PRECIP_DATA_DIR=D:\PhD\karsilastirma-gunluk
python compare_precipitation_models.py

# Windows PowerShell
$env:PRECIP_DATA_DIR = "D:\PhD\karsilastirma-gunluk"
python compare_precipitation_models.py
```

### Option 2 — Run from the data folder

```bash
cd /path/to/your/data/folder
python /path/to/compare_precipitation_models.py
```

---

## Outputs

All outputs are saved inside your data folder:

```
<PRECIP_DATA_DIR>/
├── csv_outputs/
│   ├── {ID}_daily_with_predictions.csv        # obs + gridded + model predictions
│   ├── {ID}_baseline_overall_metrics.csv      # gridded dataset metrics (all data)
│   ├── {ID}_baseline_yearly_metrics.csv       # gridded metrics per year
│   ├── {ID}_baseline_monthly_metrics.csv      # gridded metrics per month
│   ├── {ID}_model_overall_metrics.csv         # LR/RF/GB metrics (all data)
│   ├── {ID}_model_yearly_metrics.csv          # LR/RF/GB metrics per year
│   ├── {ID}_model_monthly_metrics.csv         # LR/RF/GB metrics per month
│   ├── {ID}_best_model_overall.csv            # best model (overall)
│   ├── {ID}_best_model_by_year.csv            # best model per year
│   ├── {ID}_best_model_by_month.csv           # best model per month
│   └── ALL_*.csv                              # combined tables for all stations
└── figures/
    ├── {ID}_baseline_scatter_2x5.png          # obs vs 10 gridded datasets
    ├── {ID}_gridded_vs_best_model_2x5.png     # gridded vs best ML model
    ├── {ID}_model_year_chunk_01_7x3.png       # yearly scatter (LR/RF/GB)
    └── {ID}_best_model_month_4x3.png          # monthly best model panels
```

Best model selection rule: **lowest RMSE**, with **highest CC** as tiebreaker.

---

## Configuration

Key parameters are defined at the top of `compare_precipitation_models.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRAIN_RATIO` | `0.80` | Fraction of data used for training (chronological) |
| `RANDOM_STATE` | `42` | Seed for RF and GB reproducibility |
| `AX_MIN / AX_MAX` | `0 / 50` | Scatter plot axis limits (mm) |

---

## Reproducing Results

This pipeline is fully deterministic given the same input data and Python package versions. To pin exact versions:

```bash
pip freeze > requirements-lock.txt
```

---

## Citation

If you use this pipeline in your research, please cite:

```
@software{your_name_precip_ml,
  author  = {Your Name},
  title   = {Precipitation Gridded Dataset vs ML Models Comparison},
  year    = {2025},
  url     = {https://github.com/YOUR_USERNAME/precipitation-gridded-ml-comparison},
  doi     = {10.5281/zenodo.XXXXXXX}
}
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
