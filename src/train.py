"""
PET Bottle Environmental Impact Predictor
==========================================
Trains three models:
  1. CO2 predictor       (kg CO2 per kg PET)
  2. Energy predictor    (MJ per kg PET)
  3. Env_Score predictor (composite 0-100 score)

Uses GradientBoostingRegressor + RandomForestRegressor,
selects the better one per target via cross-validation.

Run:
    python train.py --data pet_bottle_10000.xlsx
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

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# ── column names ────────────────────────────────────────────────────────────
FEATURES = [
    "rPET_Content (%)",
    "Bio_PET_Content (%)",
    "IV (dL/g)",
    "Processing_Temp (°C)",
    "Stretch_Ratio",
    "Energy_Source",
]
TARGETS = ["CO2 (kg/kg)", "Energy (MJ/kg)", "Env_Score"]
ENERGY_LABELS = {0: "Coal", 1: "Grid mix", 2: "Renewable"}

# ── helpers ─────────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
    missing = df[FEATURES + TARGETS].isnull().sum().sum()
    print(f"  Missing values: {missing}")
    return df


def build_models() -> dict:
    """Return candidate models for each target."""
    gb_params = dict(n_estimators=400, learning_rate=0.05,
                     max_depth=5, subsample=0.8,
                     min_samples_leaf=10, random_state=42)
    rf_params = dict(n_estimators=300, max_depth=None,
                     min_samples_leaf=5, n_jobs=-1, random_state=42)
    return {
        "GradientBoosting": GradientBoostingRegressor(**gb_params),
        "RandomForest":     RandomForestRegressor(**rf_params),
    }


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    return {"r2": round(r2, 4), "mae": round(mae, 4),
            "rmse": round(rmse, 4), "preds": preds}


# ── training ────────────────────────────────────────────────────────────────
def train_all(df: pd.DataFrame, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    X = df[FEATURES].copy()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for target in TARGETS:
        print(f"\n── Training models for: {target} ──")
        y = df[target].copy()
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        candidates = build_models()
        best_name, best_model, best_cv = None, None, -np.inf

        for name, model in candidates.items():
            cv_scores = cross_val_score(
                model, X_tr, y_tr, cv=kf, scoring="r2", n_jobs=-1
            )
            mean_cv = cv_scores.mean()
            print(f"  {name:20s}  CV R² = {mean_cv:.4f} ± {cv_scores.std():.4f}")
            if mean_cv > best_cv:
                best_cv, best_name, best_model = mean_cv, name, model

        # final fit on full training set
        best_model.fit(X_tr, y_tr)
        metrics = evaluate(best_model, X_te, y_te)
        print(f"  → Best: {best_name}  |  Test R²={metrics['r2']}  "
              f"MAE={metrics['mae']}  RMSE={metrics['rmse']}")

        # save model
        model_path = os.path.join(out_dir, f"model_{target.split()[0].lower()}.pkl")
        joblib.dump(best_model, model_path)

        results[target] = {
            "best_model_type": best_name,
            "model_path":      model_path,
            "metrics":         {k: v for k, v in metrics.items() if k != "preds"},
            "preds":           metrics["preds"],
            "y_test":          y_te.values,
            "X_test":          X_te,
            "model":           best_model,
        }

    return results


# ── feature importance ───────────────────────────────────────────────────────
def get_importance(result: dict, feature_names: list) -> pd.DataFrame:
    model = result["model"]
    X_te  = result["X_test"]
    y_te  = result["y_test"]
    pi    = permutation_importance(model, X_te, y_te,
                                   n_repeats=10, random_state=42, n_jobs=-1)
    return pd.DataFrame({
        "feature":    feature_names,
        "importance": pi.importances_mean,
        "std":        pi.importances_std,
    }).sort_values("importance", ascending=False)


# ── plots ────────────────────────────────────────────────────────────────────
COLORS = {"CO2 (kg/kg)": "#D85A30",
          "Energy (MJ/kg)": "#378ADD",
          "Env_Score": "#1D9E75"}

def plot_results(results: dict, df: pd.DataFrame, out_dir: str):
    fig = plt.figure(figsize=(18, 14), facecolor="white")
    fig.suptitle("PET Bottle AI Model — Training Results",
                 fontsize=16, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    for row, target in enumerate(TARGETS):
        res   = results[target]
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
        ax0.set_title(f"{target}\nR²={res['metrics']['r2']}  "
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
                 .replace(" (°C)", "").replace(" (kg/kg)", "")
                 .replace(" (MJ/kg)", "") for f in imp["feature"]]
        bars = ax2.barh(short, imp["importance"], xerr=imp["std"],
                        color=color, alpha=0.75, height=0.6,
                        error_kw=dict(elinewidth=0.8, capsize=2))
        ax2.set_xlabel("Permutation importance", fontsize=10)
        ax2.set_title("Feature importance", fontsize=10)
        ax2.tick_params(labelsize=8)
        ax2.invert_yaxis()

    path = os.path.join(out_dir, "training_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved → {path}")
    return path


def plot_what_if(results: dict, out_dir: str):
    """Show how CO2 and Env_Score change with rPET% and energy source."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")
    fig.suptitle("What-if: rPET content × Energy source impact",
                 fontsize=13, fontweight="bold")

    rPET_range = np.linspace(0, 100, 50)
    base_row   = {"Bio_PET_Content (%)": 0, "IV (dL/g)": 0.76,
                  "Processing_Temp (°C)": 270, "Stretch_Ratio": 3.2}

    es_styles = {0: ("Coal",      "#A32D2D", "-"),
                 1: ("Grid mix",  "#185FA5", "--"),
                 2: ("Renewable", "#0F6E56", "-.")}

    for ax, target in zip(axes, ["CO2 (kg/kg)", "Env_Score"]):
        model = results[target]["model"]
        color = COLORS[target]
        for es, (label, c, ls) in es_styles.items():
            rows = []
            for r in rPET_range:
                row = {**base_row, "rPET_Content (%)": r, "Energy_Source": es}
                rows.append(row)
            X_sim = pd.DataFrame(rows)[FEATURES]
            preds = model.predict(X_sim)
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
    print(f"  Plot saved → {path}")
    return path


# ── what-if simulator ────────────────────────────────────────────────────────
def simulate(models: dict, rPET: float, bioPET: float, iv: float,
             temp: float, stretch: float, energy_source: int) -> dict:
    """
    Predict CO2, Energy, and Env_Score for a given formulation.

    Parameters
    ----------
    rPET          : recycled PET content (0–100 %)
    bioPET        : bio-based PET content (0–50 %)
    iv            : intrinsic viscosity (0.60–0.85 dL/g)
    temp          : processing temperature (250–290 °C)
    stretch       : stretch blow moulding ratio (2.5–4.0)
    energy_source : 0=coal, 1=grid mix, 2=renewable

    Returns
    -------
    dict with predicted CO2, Energy, Env_Score
    """
    row = pd.DataFrame([{
        "rPET_Content (%)":     rPET,
        "Bio_PET_Content (%)":  bioPET,
        "IV (dL/g)":            iv,
        "Processing_Temp (°C)": temp,
        "Stretch_Ratio":        stretch,
        "Energy_Source":        energy_source,
    }])[FEATURES]

    return {
        "CO2 (kg/kg)":    round(float(models["CO2 (kg/kg)"].predict(row)[0]), 3),
        "Energy (MJ/kg)": round(float(models["Energy (MJ/kg)"].predict(row)[0]), 3),
        "Env_Score":      round(float(models["Env_Score"].predict(row)[0]), 2),
    }


# ── inverse optimiser ────────────────────────────────────────────────────────
def optimise(models: dict, target_env_score: float,
             energy_source: int = 2) -> pd.DataFrame:
    """
    Find the minimum rPET% + bioPET% combinations that meet target_env_score,
    given a fixed energy_source.

    Returns top-10 candidates sorted by rPET% (ascending = cheapest).
    """
    rows = []
    for rPET in np.arange(0, 101, 5):
        for bioPET in np.arange(0, 51, 5):
            pred = simulate(models,
                            rPET=rPET, bioPET=bioPET, iv=0.76,
                            temp=270, stretch=3.2,
                            energy_source=energy_source)
            if pred["Env_Score"] >= target_env_score:
                rows.append({
                    "rPET (%)":      rPET,
                    "bioPET (%)":    bioPET,
                    "Energy source": ENERGY_LABELS[energy_source],
                    "CO2 (kg/kg)":   pred["CO2 (kg/kg)"],
                    "Energy (MJ/kg)":pred["Energy (MJ/kg)"],
                    "Env_Score":     pred["Env_Score"],
                })
    if not rows:
        return pd.DataFrame(columns=["rPET (%)","bioPET (%)","Env_Score"])
    df = pd.DataFrame(rows).sort_values("rPET (%)").head(10)
    return df.reset_index(drop=True)


# ── save metadata ─────────────────────────────────────────────────────────────
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
    print(f"  Metrics saved → {path}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default="C:/Users/kenne/OneDrive/Desktop/CyclaaraAI/data/raw/pet_bottle_10000.xlsx")
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()

    print("\n=== PET Bottle AI — Training Pipeline ===\n")

    print("1. Loading data...")
    df = load_data(args.data)

    print("\n2. Training models...")
    results = train_all(df, args.out_dir)

    print("\n3. Generating training result plots...")
    plot_results(results, df, args.out_dir)

    print("\n4. Generating what-if analysis plots...")
    plot_what_if(results, args.out_dir)

    print("\n5. Saving metrics...")
    save_metrics(results, args.out_dir)

    # load saved models back for inference demo
    models = {
        t: joblib.load(results[t]["model_path"])
        for t in TARGETS
    }

    print("\n6. Example predictions:")
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
    print("  " + "-"*70)
    for ex in examples:
        label = ex.pop("label")
        pred  = simulate(models, **ex)
        print(f"  {label:<35}  "
              f"{pred['CO2 (kg/kg)']:>8.3f}  "
              f"{pred['Energy (MJ/kg)']:>10.3f}  "
              f"{pred['Env_Score']:>10.2f}")

    print("\n7. Optimiser — min rPET% to reach Env_Score ≥ 85 with renewable energy:")
    opts = optimise(models, target_env_score=85, energy_source=2)
    print(opts.to_string(index=False))

    print("\n=== Done. All outputs in:", args.out_dir, "===\n")


if __name__ == "__main__":
    main()
