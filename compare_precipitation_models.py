# compare_precipitation_models.py
# ------------------------------------------------------------
# Station precipitation vs gridded datasets comparison
# Metrics: RMSE, MAE, CC, PBIAS, NSE, KGE
# Models:  Linear Regression, Random Forest, Gradient Boosting
#
# Gridded datasets: CCS, CDR, CHIRPS, E-OBS, ERA5,
#                   GPMIMERG, MSWEP, PDIR, PERSIANN, SM2RAIN
#
# Outputs:
#   - CSV summaries (overall / yearly / monthly)
#   - High-quality scatter plots (300 dpi PNG)
#
# Usage:
#   # Option 1 – set environment variable
#   export PRECIP_DATA_DIR=/path/to/your/data      # Linux / macOS
#   set  PRECIP_DATA_DIR=D:\PhD\karsilastirma-gunluk  # Windows
#   python compare_precipitation_models.py
#
#   # Option 2 – run from the folder that contains the .xlsx files
#   python compare_precipitation_models.py
#
# Required packages:
#   pip install -r requirements.txt
# ------------------------------------------------------------

import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

warnings.filterwarnings("ignore")

# ----------------------------- SETTINGS -----------------------------

# Portable base directory: reads PRECIP_DATA_DIR env-var, falls back to CWD
BASE_DIR = Path(os.getenv("PRECIP_DATA_DIR", "."))

# Station Excel files (filenames only – resolved against BASE_DIR)
STATION_FILES = [
    BASE_DIR / "17661_Daily.xlsx",
    BASE_DIR / "18387_Daily.xlsx",
]

DATE_COL   = "Date"
TARGET_COL = "OBSERVED"

GRID_COLS = [
    "CCS", "CDR", "CHIRPS", "E-OBS", "ERA5",
    "GPMIMERG", "MSWEP", "PDIR", "PERSIANN", "SM2RAIN",
]

MODEL_COLS = ["LR", "RF", "GB"]

CSV_DIR = BASE_DIR / "csv_outputs"
FIG_DIR = BASE_DIR / "figures"

TRAIN_RATIO  = 0.80   # chronological split – first 80 % for training
RANDOM_STATE = 42

AX_MIN = 0
AX_MAX = 50

# -------------------------------------------------------------------


def ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ #
#  DATA LOADING                                                        #
# ------------------------------------------------------------------ #

def load_station_file(file_path: Path) -> pd.DataFrame:
    """
    Read a station Excel file, validate required columns,
    parse dates, and coerce numeric columns.
    """
    df = pd.read_excel(file_path, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = [DATE_COL, TARGET_COL] + GRID_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{file_path.name} is missing columns: {missing}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).copy()
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    for c in [TARGET_COL] + GRID_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add YEAR, MONTH, and YEAR_MONTH helper columns."""
    df = df.copy()
    df["YEAR"]       = df[DATE_COL].dt.year
    df["MONTH"]      = df[DATE_COL].dt.month
    df["YEAR_MONTH"] = df[DATE_COL].dt.to_period("M").astype(str)
    return df


# ------------------------------------------------------------------ #
#  PERFORMANCE METRICS                                                 #
# ------------------------------------------------------------------ #

def safe_metrics(obs, sim) -> dict:
    """
    Compute RMSE, MAE, CC, PBIAS, NSE, and KGE between obs and sim.
    NaN / Inf pairs are dropped before calculation.
    Returns NaN for any metric that cannot be computed.
    """
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    mask = np.isfinite(obs) & np.isfinite(sim)
    obs  = obs[mask]
    sim  = sim[mask]
    n    = len(obs)

    if n == 0:
        return dict(n=0, rmse=np.nan, mae=np.nan, cc=np.nan,
                    pbias=np.nan, nse=np.nan, kge=np.nan)

    diff = sim - obs

    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae  = float(np.mean(np.abs(diff)))

    cc = (
        np.nan
        if (n < 2 or np.std(obs) == 0 or np.std(sim) == 0)
        else float(np.corrcoef(obs, sim)[0, 1])
    )

    obs_sum = np.sum(obs)
    pbias   = np.nan if obs_sum == 0 else float(100.0 * np.sum(diff) / obs_sum)

    denom = np.sum((obs - np.mean(obs)) ** 2)
    nse   = np.nan if denom == 0 else float(1.0 - np.sum(diff ** 2) / denom)

    obs_mean = np.mean(obs)
    obs_std  = np.std(obs)
    sim_std  = np.std(sim)

    if obs_mean == 0 or obs_std == 0:
        kge = np.nan
    else:
        alpha = sim_std / obs_std
        beta  = np.mean(sim) / obs_mean
        kge   = (
            np.nan
            if np.isnan(cc) or np.isnan(alpha) or np.isnan(beta)
            else float(1.0 - np.sqrt((cc - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
        )

    return dict(n=int(n), rmse=rmse, mae=mae, cc=cc,
                pbias=pbias, nse=nse, kge=kge)


def metrics_text(m: dict) -> str:
    """Format metric dict as a multi-line annotation string."""
    return (
        f"CC={m['cc']:.2f}\n"
        f"RMSE={m['rmse']:.2f}\n"
        f"MAE={m['mae']:.2f}\n"
        f"PBIAS={m['pbias']:.2f}\n"
        f"NSE={m['nse']:.2f}\n"
        f"KGE={m['kge']:.2f}"
    )


# ------------------------------------------------------------------ #
#  ML MODELS                                                           #
# ------------------------------------------------------------------ #

def fit_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chronological train/predict split (TRAIN_RATIO).
    Three sklearn Pipelines are trained on the first TRAIN_RATIO fraction
    and predictions are generated for ALL rows.

    Pipeline steps
    --------------
    LR : median imputation → standard scaling → LinearRegression
    RF : median imputation → RandomForestRegressor (300 trees)
    GB : median imputation → GradientBoostingRegressor (300 trees)
    """
    data = df.copy().dropna(subset=[TARGET_COL]).reset_index(drop=True)

    X = data[GRID_COLS]
    y = data[TARGET_COL]

    split_idx = max(1, min(int(len(data) * TRAIN_RATIO), len(data) - 1))
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]

    models = {
        "LR": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   LinearRegression()),
        ]),
        "RF": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model",   RandomForestRegressor(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                max_features="sqrt",
            )),
        ]),
        "GB": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model",   GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE,
            )),
        ]),
    }

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        data[name] = pipeline.predict(X)

    return data


# ------------------------------------------------------------------ #
#  METRICS TABLES                                                      #
# ------------------------------------------------------------------ #

def compute_metrics_table(
    df: pd.DataFrame,
    value_cols: list,
    station_id: str,
    result_type: str,
    group_col: str | None = None,
) -> pd.DataFrame:
    """
    Compute safe_metrics for each column in value_cols.

    Parameters
    ----------
    group_col : None → single aggregate row; str → one row per group value.
    """
    rows = []

    if group_col is None:
        obs = df[TARGET_COL].values
        for col in value_cols:
            m = safe_metrics(obs, df[col].values)
            rows.append(dict(Station=station_id, ResultType=result_type,
                             GroupType="ALL", GroupValue="ALL",
                             Dataset=col, **m))
    else:
        for group_value, g in df.groupby(group_col, dropna=True):
            obs = g[TARGET_COL].values
            for col in value_cols:
                m = safe_metrics(obs, g[col].values)
                rows.append(dict(Station=station_id, ResultType=result_type,
                                 GroupType=group_col, GroupValue=group_value,
                                 Dataset=col, **m))

    return pd.DataFrame(rows)


def select_best_model(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the best model per (Station, GroupType, GroupValue).
    Ranking rule: lowest RMSE first, then highest CC as tiebreaker.
    """
    out_rows = []
    for _, g in metrics_df.groupby(["Station", "GroupType", "GroupValue"], dropna=True):
        g2 = g.copy()
        g2["rmse_sort"] = g2["rmse"].fillna(np.inf)
        g2["cc_sort"]   = g2["cc"].fillna(-np.inf)
        best = g2.sort_values(["rmse_sort", "cc_sort"],
                               ascending=[True, False]).iloc[0]
        out_rows.append(dict(
            Station=best["Station"], GroupType=best["GroupType"],
            GroupValue=best["GroupValue"], BestModel=best["Dataset"],
            n=best["n"], rmse=best["rmse"], mae=best["mae"],
            cc=best["cc"], pbias=best["pbias"],
            nse=best["nse"], kge=best["kge"],
        ))
    return pd.DataFrame(out_rows)


# ------------------------------------------------------------------ #
#  CSV I/O                                                             #
# ------------------------------------------------------------------ #

def save_csv(df: pd.DataFrame, name: str) -> Path:
    path = CSV_DIR / name
    df.to_csv(path, index=False)
    return path


# ------------------------------------------------------------------ #
#  PLOTTING HELPERS                                                    #
# ------------------------------------------------------------------ #

def scatter_panel(
    ax,
    obs,
    pred,
    title: str,
    xlim: tuple = (0, 50),
    ylim: tuple = (0, 50),
) -> None:
    """Draw a single observed-vs-predicted scatter panel with metric annotation."""
    obs  = np.asarray(obs,  dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred)
    m    = safe_metrics(obs[mask], pred[mask])

    ax.scatter(obs[mask], pred[mask], s=10, alpha=0.60)
    ax.plot([xlim[0], xlim[1]], [ylim[0], ylim[1]],
            linestyle="--", linewidth=1, color="black")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Estimated")
    ax.text(
        0.04, 0.96, metrics_text(m),
        transform=ax.transAxes, va="top", fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
    )


# ------------------------------------------------------------------ #
#  PLOT FUNCTIONS                                                      #
# ------------------------------------------------------------------ #

def plot_baseline_grid(df: pd.DataFrame, station_id: str) -> None:
    """5 rows × 2 cols = 10 gridded datasets (obs vs each gridded)."""
    fig, axes = plt.subplots(5, 2, figsize=(14, 28))
    obs = df[TARGET_COL].values
    for i, col in enumerate(GRID_COLS):
        scatter_panel(axes.ravel()[i], obs, df[col].values,
                      title=col, xlim=(AX_MIN, AX_MAX), ylim=(AX_MIN, AX_MAX))
    fig.suptitle(f"Station {station_id} – Observed vs Gridded Datasets",
                 fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{station_id}_baseline_scatter_2x5.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_gridded_vs_best_model_grid(
    df: pd.DataFrame, station_id: str, best_model_name: str
) -> None:
    """5 rows × 2 cols: each gridded dataset vs best ML model."""
    fig, axes = plt.subplots(5, 2, figsize=(14, 28))
    for i, grid_col in enumerate(GRID_COLS):
        ax = axes.ravel()[i]
        scatter_panel(ax, df[grid_col].values, df[best_model_name].values,
                      title=f"{grid_col} vs {best_model_name}",
                      xlim=(AX_MIN, AX_MAX), ylim=(AX_MIN, AX_MAX))
        ax.set_xlabel(grid_col)
        ax.set_ylabel(best_model_name)
    fig.suptitle(
        f"Station {station_id} – Gridded Datasets vs Best Model ({best_model_name})",
        fontsize=14, y=0.995,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{station_id}_gridded_vs_best_model_2x5.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_year_model_grid(df: pd.DataFrame, station_id: str) -> None:
    """
    3 cols (LR, RF, GB) × up to 7 rows (years) per figure.
    Saves multiple figures in 7-year chunks if necessary.
    """
    years = sorted(df["YEAR"].dropna().unique().tolist())
    if not years:
        return

    for chunk_idx, start in enumerate(range(0, len(years), 7), start=1):
        chunk_years = years[start:start + 7]
        fig, axes   = plt.subplots(7, 3, figsize=(18, 28))

        # hide all panels, then re-enable only used ones
        for ax in axes.ravel():
            ax.axis("off")

        for r, year in enumerate(chunk_years):
            sub = df[df["YEAR"] == year].copy()
            obs = sub[TARGET_COL].values
            for c, model in enumerate(MODEL_COLS):
                axes[r, c].axis("on")
                scatter_panel(axes[r, c], obs, sub[model].values,
                              title=f"Year {int(year)} | {model}",
                              xlim=(AX_MIN, AX_MAX), ylim=(AX_MIN, AX_MAX))

        fig.suptitle(
            f"Station {station_id} – Model Results by Year (chunk {chunk_idx})",
            fontsize=14, y=0.995,
        )
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{station_id}_model_year_chunk_{chunk_idx:02d}_7x3.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_month_best_model_grid(
    df: pd.DataFrame, station_id: str, best_month_df: pd.DataFrame
) -> None:
    """
    3 rows × 4 cols = 12 months.
    Each panel shows the best ML model (LR / RF / GB) for that month.
    """
    month_name = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun",
                  7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}

    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    for i, month in enumerate(range(1, 13)):
        ax  = axes.ravel()[i]
        sub = df[df["MONTH"] == month].copy()
        if sub.empty:
            ax.axis("off")
            continue

        row = best_month_df[
            (best_month_df["Station"]    == station_id) &
            (best_month_df["GroupType"]  == "MONTH") &
            (best_month_df["GroupValue"] == month)
        ]
        if row.empty:
            ax.axis("off")
            continue

        best_model = row.iloc[0]["BestModel"]
        scatter_panel(ax, sub[TARGET_COL].values, sub[best_model].values,
                      title=f"{month_name[month]} | Best: {best_model}",
                      xlim=(AX_MIN, AX_MAX), ylim=(AX_MIN, AX_MAX))

    for j in range(12, len(axes.ravel())):
        axes.ravel()[j].axis("off")

    fig.suptitle(f"Station {station_id} – Best Model by Month",
                 fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{station_id}_best_model_month_4x3.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ #
#  PER-STATION ORCHESTRATION                                           #
# ------------------------------------------------------------------ #

def process_station(file_path: Path) -> dict:
    station_id = file_path.stem.split("_")[0]
    print(f"\n── Processing station {station_id} ──")

    df      = load_station_file(file_path)
    df      = add_time_columns(df)
    df_pred = fit_models(df)

    # Daily predictions CSV
    daily_out = df_pred[[DATE_COL, TARGET_COL] + GRID_COLS + MODEL_COLS].copy()
    save_csv(daily_out, f"{station_id}_daily_with_predictions.csv")

    # Baseline metrics (gridded datasets)
    baseline_all   = compute_metrics_table(df_pred, GRID_COLS, station_id, "Baseline")
    baseline_year  = compute_metrics_table(df_pred, GRID_COLS, station_id, "Baseline", "YEAR")
    baseline_month = compute_metrics_table(df_pred, GRID_COLS, station_id, "Baseline", "MONTH")

    # Model metrics (LR / RF / GB)
    model_all   = compute_metrics_table(df_pred, MODEL_COLS, station_id, "Model")
    model_year  = compute_metrics_table(df_pred, MODEL_COLS, station_id, "Model", "YEAR")
    model_month = compute_metrics_table(df_pred, MODEL_COLS, station_id, "Model", "MONTH")

    # Best model selection
    best_overall = select_best_model(model_all)
    best_year    = select_best_model(model_year)
    best_month   = select_best_model(model_month)

    # Save all CSV outputs
    for name, tbl in [
        (f"{station_id}_baseline_overall_metrics.csv",  baseline_all),
        (f"{station_id}_baseline_yearly_metrics.csv",   baseline_year),
        (f"{station_id}_baseline_monthly_metrics.csv",  baseline_month),
        (f"{station_id}_model_overall_metrics.csv",     model_all),
        (f"{station_id}_model_yearly_metrics.csv",      model_year),
        (f"{station_id}_model_monthly_metrics.csv",     model_month),
        (f"{station_id}_best_model_overall.csv",        best_overall),
        (f"{station_id}_best_model_by_year.csv",        best_year),
        (f"{station_id}_best_model_by_month.csv",       best_month),
    ]:
        save_csv(tbl, name)

    # Generate all plots
    plot_baseline_grid(df_pred, station_id)
    plot_gridded_vs_best_model_grid(df_pred, station_id,
                                    best_overall.iloc[0]["BestModel"])
    plot_year_model_grid(df_pred, station_id)
    plot_month_best_model_grid(df_pred, station_id, best_month)

    print(f"   Station {station_id} complete.")
    return dict(
        baseline_all=baseline_all, baseline_year=baseline_year,
        baseline_month=baseline_month, model_all=model_all,
        model_year=model_year, model_month=model_month,
        best_overall=best_overall, best_year=best_year,
        best_month=best_month,
    )


# ------------------------------------------------------------------ #
#  MAIN                                                                #
# ------------------------------------------------------------------ #

def main() -> None:
    ensure_dirs()

    all_tables: dict[str, list] = {k: [] for k in [
        "baseline_all", "baseline_year", "baseline_month",
        "model_all", "model_year", "model_month",
        "best_overall", "best_year", "best_month",
    ]}

    for fp in STATION_FILES:
        if not fp.exists():
            raise FileNotFoundError(
                f"Station file not found: {fp}\n"
                f"Set PRECIP_DATA_DIR to the folder containing the .xlsx files."
            )
        results = process_station(fp)
        for k in all_tables:
            all_tables[k].append(results[k])

    # Combined CSVs (all stations)
    for name, dfs in all_tables.items():
        save_csv(pd.concat(dfs, ignore_index=True), f"ALL_{name}.csv")

    print("\n✓ Done.")
    print(f"  CSV files  → {CSV_DIR}")
    print(f"  Figures    → {FIG_DIR}")


if __name__ == "__main__":
    main()
