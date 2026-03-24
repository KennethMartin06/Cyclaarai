"""
PET Bottle Environmental Impact Predictor (v2 - Tuned)
=======================================================
Trains three models:
  1. CO2 predictor       (kg CO2 per kg PET)
  2. Energy predictor    (MJ per kg PET)
  3. Env_Score predictor (composite 0-100 score)

Improvements over v1:
  - Cleans data before training (drops nulls, duplicates, out-of-range rows)
  - Adds engineered features (rPET x Energy_Source interaction, total recycled %)
  - Hyperparameter tuning via RandomizedSearchCV
  - Uses HistGradientBoosting for faster, often better performance
  - Clips Env_Score predictions to valid [0, 100] range
  - Uses LinearRegression for Energy (exact linear relationship in data)

Run:
    python train.py --data ../data/raw/pet_bottle_10000.xlsx
"""

import argparse
import json
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    KFold,
)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# -- column names ------------------------------------------------------------
RAW_FEATURES = [
    "rPET_Content (%)",
    "Bio_PET_Content (%)",
    "IV (dL/g)",
    "Processing_Temp (°C)",
    "Stretch_Ratio",
    "Energy_Source",
]
ENGINEERED_FEATURES = [
    "rPET_x_Energy",       # interaction: rPET * Energy_Source
    "Total_Recycled (%)",  # rPET + Bio_PET combined
]
FEATURES = RAW_FEATURES + ENGINEERED_FEATURES
TARGETS = ["CO2 (kg/kg)", "Energy (MJ/kg)", "Env_Score"]
ENERGY_LABELS = {0: "Coal", 1: "Grid mix", 2: "Renewable"}

# -- target bounds (for clipping predictions) --------------------------------
TARGET_BOUNDS = {
    "CO2 (kg/kg)":    (0, None),
    "Energy (MJ/kg)": (0, None),
    "Env_Score":       (0, 100),
}


# -- data loading & cleaning ------------------------------------------------
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    n_raw = len(df)
    print(f"  Loaded {n_raw:,} rows x {len(df.columns)} columns")

    # drop nulls and duplicates
    df.dropna(subset=RAW_FEATURES + TARGETS, inplace=True)
    df.drop_duplicates(inplace=True)

    # validate ranges
    df = df[df["IV (dL/g)"].between(0.60, 0.85)]
    df = df[df["Processing_Temp (°C)"].between(250, 290)]
    df = df[(df["rPET_Content (%)"] + df["Bio_PET_Content (%)"]) <= 100]
    df = df[df["Energy_Source"].isin([0, 1, 2])]

    n_clean = len(df)
    print(f"  After cleaning: {n_clean:,} rows ({n_raw - n_clean} removed)")
    return df.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rPET_x_Energy"] = df["rPET_Content (%)"] * df["Energy_Source"]
    df["Total_Recycled (%)"] = df["rPET_Content (%)"] + df["Bio_PET_Content (%)"]
    return df


# -- hyperparameter search spaces -------------------------------------------
def get_search_spaces():
    return {
        "GradientBoosting": {
            "model": GradientBoostingRegressor(random_state=42),
            "params": {
                "n_estimators": [200, 400, 600, 800],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "max_depth": [3, 4, 5, 6, 7],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "min_samples_leaf": [5, 10, 20],
            },
        },
        "RandomForest": {
            "model": RandomForestRegressor(random_state=42, n_jobs=-1),
            "params": {
                "n_estimators": [200, 300, 500, 700],
                "max_depth": [10, 15, 20, None],
                "min_samples_leaf": [2, 5, 10],
                "max_features": ["sqrt", "log2", 0.5, 0.8, 1.0],
            },
        },
        "HistGradientBoosting": {
            "model": HistGradientBoostingRegressor(random_state=42),
            "params": {
                "max_iter": [200, 400, 600, 800],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "max_depth": [3, 5, 7, 10, None],
                "min_samples_leaf": [5, 10, 20, 30],
                "max_leaf_nodes": [15, 31, 63, None],
                "l2_regularization": [0.0, 0.01, 0.1, 1.0],
            },
        },
    }


# -- evaluation --------------------------------------------------------------
def evaluate(model, X_test, y_test, target: str) -> dict:
    preds = model.predict(X_test)
    # clip to valid bounds
    lo, hi = TARGET_BOUNDS.get(target, (None, None))
    preds = np.clip(preds, lo, hi)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return {"r2": round(r2, 6), "mae": round(mae, 6),
            "rmse": round(rmse, 6), "preds": preds}


# -- training ----------------------------------------------------------------
N_SEARCH_ITER = 30   # random search iterations per candidate
CV_FOLDS = 5


def train_all(df: pd.DataFrame, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    df = add_features(df)
    X = df[FEATURES].copy()
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    results = {}

    for target in TARGETS:
        print(f"\n-- Training models for: {target} --")
        y = df[target].copy()
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Energy is an exact linear function of Temp + Energy_Source;
        # use LinearRegression directly (no tuning needed)
        if target == "Energy (MJ/kg)":
            print("  Energy target detected -- using LinearRegression (exact fit)")
            model = LinearRegression()
            model.fit(X_tr, y_tr)
            metrics = evaluate(model, X_te, y_te, target)
            best_name = "LinearRegression"
            print(f"  -> LinearRegression  |  Test R2={metrics['r2']}  "
                  f"MAE={metrics['mae']}  RMSE={metrics['rmse']}")
        else:
            # run hyperparameter search for each candidate
            search_spaces = get_search_spaces()
            best_name, best_model, best_cv = None, None, -np.inf

            for name, spec in search_spaces.items():
                print(f"  Tuning {name} ({N_SEARCH_ITER} random configs, "
                      f"{CV_FOLDS}-fold CV)...")
                search = RandomizedSearchCV(
                    spec["model"],
                    spec["params"],
                    n_iter=N_SEARCH_ITER,
                    cv=kf,
                    scoring="r2",
                    n_jobs=-1,
                    random_state=42,
                    refit=True,
                )
                search.fit(X_tr, y_tr)
                mean_cv = search.best_score_
                print(f"    Best CV R2 = {mean_cv:.6f}  "
                      f"params = {search.best_params_}")
                if mean_cv > best_cv:
                    best_cv = mean_cv
                    best_name = name
                    best_model = search.best_estimator_

            model = best_model
            metrics = evaluate(model, X_te, y_te, target)
            print(f"  -> Winner: {best_name}  |  Test R2={metrics['r2']}  "
                  f"MAE={metrics['mae']}  RMSE={metrics['rmse']}")

        # save model
        model_path = os.path.join(out_dir, f"model_{target.split()[0].lower()}.pkl")
        joblib.dump(model, model_path)

        results[target] = {
            "best_model_type": best_name,
            "model_path":      model_path,
            "metrics":         {k: v for k, v in metrics.items() if k != "preds"},
            "preds":           metrics["preds"],
            "y_test":          y_te.values,
            "X_test":          X_te,
            "model":           model,
        }

    return results


# -- feature importance -------------------------------------------------------
def get_importance(result: dict, feature_names: list) -> pd.DataFrame:
    model = result["model"]
    X_te = result["X_test"]
    y_te = result["y_test"]
    pi = permutation_importance(model, X_te, y_te,
                                n_repeats=10, random_state=42, n_jobs=-1)
    return pd.DataFrame({
        "feature":    feature_names,
        "importance": pi.importances_mean,
        "std":        pi.importances_std,
    }).sort_values("importance", ascending=False)


# -- plots --------------------------------------------------------------------
COLORS = {"CO2 (kg/kg)": "#D85A30",
          "Energy (MJ/kg)": "#378ADD",
          "Env_Score": "#1D9E75"}


def plot_results(results: dict, df: pd.DataFrame, out_dir: str):
    fig = plt.figure(figsize=(18, 14), facecolor="white")
    fig.suptitle("PET Bottle AI Model - Training Results (Tuned)",
                 fontsize=16, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    for row, target in enumerate(TARGETS):
        res = results[target]
        color = COLORS[target]
        preds = res["preds"]
        ytrue = res["y_test"]

        # col 0: actual vs predicted scatter
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.scatter(ytrue, preds, alpha=0.25, s=8, color=color, linewidths=0)
        mn, mx = min(ytrue.min(), preds.min()), max(ytrue.max(), preds.max())
        ax0.plot([mn, mx], [mn, mx], "k--", lw=1, alpha=0.5)
        ax0.set_xlabel("Actual", fontsize=10)
        ax0.set_ylabel("Predicted", fontsize=10)
        ax0.set_title(f"{target}\nR2={res['metrics']['r2']}  "
                      f"MAE={res['metrics']['mae']}", fontsize=10)
        ax0.tick_params(labelsize=8)

        # col 1: residuals
        ax1 = fig.add_subplot(gs[row, 1])
        residuals = preds - ytrue
        ax1.scatter(preds, residuals, alpha=0.2, s=8, color=color, linewidths=0)
        ax1.axhline(0, color="black", lw=1, ls="--", alpha=0.5)
        ax1.set_xlabel("Predicted", fontsize=10)
        ax1.set_ylabel("Residual", fontsize=10)
        ax1.set_title("Residuals", fontsize=10)
        ax1.tick_params(labelsize=8)

        # col 2: feature importance
        ax2 = fig.add_subplot(gs[row, 2])
        imp = get_importance(res, FEATURES)
        short = [f.replace(" (%)", "").replace(" (dL/g)", "")
                 .replace(" (C)", "").replace(" (kg/kg)", "")
                 .replace(" (MJ/kg)", "") for f in imp["feature"]]
        ax2.barh(short, imp["importance"], xerr=imp["std"],
                 color=color, alpha=0.75, height=0.6,
                 error_kw=dict(elinewidth=0.8, capsize=2))
        ax2.set_xlabel("Permutation importance", fontsize=10)
        ax2.set_title("Feature importance", fontsize=10)
        ax2.tick_params(labelsize=8)
        ax2.invert_yaxis()

    path = os.path.join(out_dir, "training_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved -> {path}")
    return path


def plot_what_if(results: dict, out_dir: str):
    """Show how CO2 and Env_Score change with rPET% and energy source."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")
    fig.suptitle("What-if: rPET content x Energy source impact",
                 fontsize=13, fontweight="bold")

    rPET_range = np.linspace(0, 100, 50)
    base_row = {"Bio_PET_Content (%)": 0, "IV (dL/g)": 0.76,
                "Processing_Temp (°C)": 270, "Stretch_Ratio": 3.2}

    es_styles = {0: ("Coal",      "#A32D2D", "-"),
                 1: ("Grid mix",  "#185FA5", "--"),
                 2: ("Renewable", "#0F6E56", "-.")}

    for ax, target in zip(axes, ["CO2 (kg/kg)", "Env_Score"]):
        model = results[target]["model"]
        lo, hi = TARGET_BOUNDS.get(target, (None, None))
        for es, (label, c, ls) in es_styles.items():
            rows = []
            for r in rPET_range:
                row = {**base_row, "rPET_Content (%)": r, "Energy_Source": es}
                row["rPET_x_Energy"] = r * es
                row["Total_Recycled (%)"] = r + base_row["Bio_PET_Content (%)"]
                rows.append(row)
            X_sim = pd.DataFrame(rows)[FEATURES]
            preds = np.clip(model.predict(X_sim), lo, hi)
            ax.plot(rPET_range, preds, ls=ls, color=c, lw=2, label=label)

        ax.set_xlabel("rPET content (%)", fontsize=11)
        ax.set_ylabel(target, fontsize=11)
        ax.set_title(target, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(labelsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "what_if_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved -> {path}")
    return path


# -- what-if simulator -------------------------------------------------------
def simulate(models: dict, rPET: float, bioPET: float, iv: float,
             temp: float, stretch: float, energy_source: int) -> dict:
    row = pd.DataFrame([{
        "rPET_Content (%)":     rPET,
        "Bio_PET_Content (%)":  bioPET,
        "IV (dL/g)":            iv,
        "Processing_Temp (°C)": temp,
        "Stretch_Ratio":        stretch,
        "Energy_Source":        energy_source,
        "rPET_x_Energy":        rPET * energy_source,
        "Total_Recycled (%)":   rPET + bioPET,
    }])[FEATURES]

    out = {}
    for t in TARGETS:
        pred = float(models[t].predict(row)[0])
        lo, hi = TARGET_BOUNDS.get(t, (None, None))
        if lo is not None:
            pred = max(pred, lo)
        if hi is not None:
            pred = min(pred, hi)
        out[t] = round(pred, 3 if t != "Env_Score" else 2)
    return out


# -- inverse optimiser -------------------------------------------------------
def optimise(models: dict, target_env_score: float,
             energy_source: int = 2) -> pd.DataFrame:
    rows = []
    for rPET in np.arange(0, 101, 5):
        for bioPET in np.arange(0, 51, 5):
            pred = simulate(models,
                            rPET=rPET, bioPET=bioPET, iv=0.76,
                            temp=270, stretch=3.2,
                            energy_source=energy_source)
            if pred["Env_Score"] >= target_env_score:
                rows.append({
                    "rPET (%)":       rPET,
                    "bioPET (%)":     bioPET,
                    "Energy source":  ENERGY_LABELS[energy_source],
                    "CO2 (kg/kg)":    pred["CO2 (kg/kg)"],
                    "Energy (MJ/kg)": pred["Energy (MJ/kg)"],
                    "Env_Score":      pred["Env_Score"],
                })
    if not rows:
        return pd.DataFrame(columns=["rPET (%)", "bioPET (%)", "Env_Score"])
    df = pd.DataFrame(rows).sort_values("rPET (%)").head(10)
    return df.reset_index(drop=True)


# -- save metadata -----------------------------------------------------------
def save_metrics(results: dict, out_dir: str):
    meta = {}
    for target, res in results.items():
        meta[target] = {
            "best_model_type": res["best_model_type"],
            "model_path":      res["model_path"],
            "metrics":         res["metrics"],
        }
    path = os.path.join(out_dir, "model_metrics.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metrics saved -> {path}")


# -- main --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(
        os.path.dirname(__file__), "..", "data", "raw", "pet_bottle_10000.xlsx"))
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()

    print("\n=== PET Bottle AI - Training Pipeline (v2 - Tuned) ===\n")

    print("1. Loading & cleaning data...")
    df = load_and_clean(args.data)

    print("\n2. Engineering features...")
    print(f"  Added: {ENGINEERED_FEATURES}")
    print(f"  Total features: {len(FEATURES)}")

    print("\n3. Training models (with hyperparameter tuning)...")
    results = train_all(df, args.out_dir)

    print("\n4. Generating training result plots...")
    plot_results(results, df, args.out_dir)

    print("\n5. Generating what-if analysis plots...")
    plot_what_if(results, args.out_dir)

    print("\n6. Saving metrics...")
    save_metrics(results, args.out_dir)

    # load saved models back for inference demo
    models = {
        t: joblib.load(results[t]["model_path"])
        for t in TARGETS
    }

    print("\n7. Example predictions:")
    examples = [
        dict(rPET=100, bioPET=0,  iv=0.76, temp=265, stretch=3.2, energy_source=2,
             label="100% rPET + renewable"),
        dict(rPET=50,  bioPET=0,  iv=0.76, temp=270, stretch=3.2, energy_source=1,
             label="50% rPET + grid mix"),
        dict(rPET=0,   bioPET=0,  iv=0.80, temp=275, stretch=3.2, energy_source=0,
             label="Virgin PET + coal"),
        dict(rPET=75,  bioPET=25, iv=0.76, temp=265, stretch=3.2, energy_source=2,
             label="75% rPET + 25% bioPET + renewable"),
    ]
    print(f"  {'Scenario':<35}  {'CO2':>8}  {'Energy':>10}  {'Env_Score':>10}")
    print("  " + "-" * 70)
    for ex in examples:
        label = ex.pop("label")
        pred = simulate(models, **ex)
        print(f"  {label:<35}  "
              f"{pred['CO2 (kg/kg)']:>8.3f}  "
              f"{pred['Energy (MJ/kg)']:>10.3f}  "
              f"{pred['Env_Score']:>10.2f}")

    print("\n8. Optimiser - min rPET% to reach Env_Score >= 85 with renewable energy:")
    opts = optimise(models, target_env_score=85, energy_source=2)
    print(opts.to_string(index=False))

    print("\n=== Done. All outputs in:", args.out_dir, "===\n")


if __name__ == "__main__":
    main()
