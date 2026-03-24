"""
PET Bottle AI — Inference & What-if Simulator
==============================================
Load the trained models and run interactive predictions.

Usage examples:
    # predict one formulation
    python inference.py predict \
        --rPET 80 --bioPET 10 --iv 0.76 \
        --temp 268 --stretch 3.2 --energy 2

    # find min rPET to hit Env_Score >= 85
    python inference.py optimise --target_score 85 --energy 2

    # compare energy source impact at fixed rPET=50%
    python inference.py compare_energy --rPET 50 --bioPET 0
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd

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
MODEL_DIR = "outputs"


def load_models(model_dir: str = MODEL_DIR) -> dict:
    models = {}
    for t in TARGETS:
        fname = f"model_{t.split()[0].lower()}.pkl"
        path  = os.path.join(model_dir, fname)
        models[t] = joblib.load(path)
    return models


def predict_one(models, rPET, bioPET, iv, temp, stretch, energy_source):
    row = pd.DataFrame([{
        "rPET_Content (%)":     rPET,
        "Bio_PET_Content (%)":  bioPET,
        "IV (dL/g)":            iv,
        "Processing_Temp (°C)": temp,
        "Stretch_Ratio":        stretch,
        "Energy_Source":        energy_source,
    }])[FEATURES]
    return {t: round(float(models[t].predict(row)[0]),
                     3 if t != "Env_Score" else 2)
            for t in TARGETS}


def cmd_predict(args, models):
    pred = predict_one(models,
                       rPET=args.rPET, bioPET=args.bioPET, iv=args.iv,
                       temp=args.temp, stretch=args.stretch,
                       energy_source=args.energy)
    print("\n── Prediction ──────────────────────────────")
    print(f"  rPET content     : {args.rPET}%")
    print(f"  Bio-PET content  : {args.bioPET}%")
    print(f"  IV               : {args.iv} dL/g")
    print(f"  Processing temp  : {args.temp}°C")
    print(f"  Stretch ratio    : {args.stretch}")
    print(f"  Energy source    : {ENERGY_LABELS[args.energy]}")
    print("────────────────────────────────────────────")
    print(f"  CO2 footprint    : {pred['CO2 (kg/kg)']} kg CO2/kg PET")
    print(f"  Energy use       : {pred['Energy (MJ/kg)']} MJ/kg PET")
    print(f"  Env_Score        : {pred['Env_Score']} / 100")
    print()


def cmd_optimise(args, models):
    print(f"\n── Optimiser: minimum rPET% to reach Env_Score ≥ {args.target_score} "
          f"with {ENERGY_LABELS[args.energy]} energy ──")
    rows = []
    for rPET in np.arange(0, 101, 5):
        for bioPET in np.arange(0, 51, 5):
            pred = predict_one(models, rPET=rPET, bioPET=bioPET,
                               iv=0.76, temp=270, stretch=3.2,
                               energy_source=args.energy)
            if pred["Env_Score"] >= args.target_score:
                rows.append({
                    "rPET (%)":       int(rPET),
                    "bioPET (%)":     int(bioPET),
                    "CO2 (kg/kg)":    pred["CO2 (kg/kg)"],
                    "Energy (MJ/kg)": pred["Energy (MJ/kg)"],
                    "Env_Score":      pred["Env_Score"],
                })
    if not rows:
        print(f"  No combination achieves Env_Score ≥ {args.target_score} "
              f"with {ENERGY_LABELS[args.energy]} energy.\n")
        return
    df = pd.DataFrame(rows).sort_values("rPET (%)").head(10).reset_index(drop=True)
    print(df.to_string(index=False))
    print()


def cmd_compare_energy(args, models):
    print(f"\n── Energy source comparison at rPET={args.rPET}%, "
          f"bioPET={args.bioPET}% ──")
    header = f"  {'Energy source':<12}  {'CO2 (kg/kg)':>12}  "
    header += f"{'Energy (MJ/kg)':>14}  {'Env_Score':>10}"
    print(header)
    print("  " + "─" * 54)
    for es, label in ENERGY_LABELS.items():
        pred = predict_one(models, rPET=args.rPET, bioPET=args.bioPET,
                           iv=0.76, temp=270, stretch=3.2,
                           energy_source=es)
        print(f"  {label:<12}  "
              f"{pred['CO2 (kg/kg)']:>12.3f}  "
              f"{pred['Energy (MJ/kg)']:>14.3f}  "
              f"{pred['Env_Score']:>10.2f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="PET Bottle AI — Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--model_dir", default=MODEL_DIR)
    sub = parser.add_subparsers(dest="command")

    # predict
    p = sub.add_parser("predict")
    p.add_argument("--rPET",    type=float, required=True)
    p.add_argument("--bioPET",  type=float, default=0)
    p.add_argument("--iv",      type=float, default=0.76)
    p.add_argument("--temp",    type=float, default=270)
    p.add_argument("--stretch", type=float, default=3.2)
    p.add_argument("--energy",  type=int,   default=1,
                   choices=[0, 1, 2], help="0=coal 1=grid 2=renewable")

    # optimise
    o = sub.add_parser("optimise")
    o.add_argument("--target_score", type=float, required=True)
    o.add_argument("--energy", type=int, default=2, choices=[0, 1, 2])

    # compare_energy
    c = sub.add_parser("compare_energy")
    c.add_argument("--rPET",   type=float, required=True)
    c.add_argument("--bioPET", type=float, default=0)

    args = parser.parse_args()
    models = load_models(args.model_dir)

    if args.command == "predict":
        cmd_predict(args, models)
    elif args.command == "optimise":
        cmd_optimise(args, models)
    elif args.command == "compare_energy":
        cmd_compare_energy(args, models)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
