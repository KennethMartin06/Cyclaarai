# CyclaaraAI

AI-powered environmental impact prediction system for PET (polyethylene terephthalate) bottle manufacturing. Predicts CO2 emissions, energy consumption, and composite environmental scores based on formulation parameters and processing conditions.

## Project Structure

```
CyclaaraAI/
├── App.jsx                  # React app entry point
├── Dashboard.jsx            # BlueWave AI dashboard UI
├── requirements.txt         # Python dependencies
├── src/
│   ├── train.py             # ML training pipeline
│   ├── inference.py         # Prediction & optimization CLI
│   └── clear.py             # Data cleaning utility
├── data/
│   └── raw/
│       ├── pet_bottle_10000.xlsx   # 10,000 training records
│       └── Parameter_list.xlsx     # Parameter reference
├── outputs/
│   ├── model_co2.pkl        # Trained CO2 prediction model
│   ├── model_energy.pkl     # Trained energy prediction model
│   ├── model_env_score.pkl  # Trained environmental score model
│   ├── model_metrics.json   # Model performance metrics
│   ├── training_results.png # Training visualization
│   └── what_if_analysis.png # Sensitivity analysis plot
└── notebooks/               # Future analysis notebooks
```

## Models

| Model | Target | Algorithm | R² Score | MAE |
|-------|--------|-----------|----------|-----|
| CO2 | kg CO2 per kg PET | GradientBoosting | 0.9993 | 0.0116 |
| Energy | MJ per kg PET | LinearRegression | 1.0 | 0.0 |
| Env_Score | Composite score (0-100) | GradientBoosting | 0.9995 | 0.2541 |

### Input Features

| Feature | Range | Description |
|---------|-------|-------------|
| rPET_Content (%) | 0-100 | Recycled PET percentage |
| Bio_PET_Content (%) | 0-50 | Bio-based PET percentage |
| IV (dL/g) | 0.60-0.85 | Intrinsic viscosity |
| Processing_Temp (C) | 250-290 | Manufacturing temperature |
| Stretch_Ratio | 2.5-4.0 | Blow-molding stretch factor |
| Energy_Source | 0, 1, 2 | Coal / Grid mix / Renewable |
| rPET_x_Energy | derived | Interaction: rPET * Energy_Source |
| Total_Recycled (%) | derived | rPET + Bio_PET combined |

## Usage

### Training

```bash
python src/train.py --data data/raw/pet_bottle_10000.xlsx --out_dir outputs
```

### Inference

```bash
# Predict a single formulation
python src/inference.py predict --rPET 80 --bioPET 10 --iv 0.76 --temp 268 --stretch 3.2 --energy 2

# Find minimum rPET% to achieve target Env_Score
python src/inference.py optimise --target_score 85 --energy 2

# Compare energy source impact
python src/inference.py compare_energy --rPET 50 --bioPET 0
```

## Tech Stack

- **Backend**: Python, scikit-learn, pandas, numpy, matplotlib
- **Frontend**: React, Tailwind CSS
- **Models**: GradientBoosting, RandomForest, HistGradientBoosting, LinearRegression

## Installation

```bash
pip install -r requirements.txt
```

---

## Changelog

### v2 — Model Optimization & Tuning (2026-03-24)

**What changed:**
- Integrated data cleaning into training pipeline (removed 2,494 invalid rows: rPET+BioPET > 100%, out-of-range IV/temp values)
- Added feature engineering: `rPET_x_Energy` interaction term and `Total_Recycled (%)` combined feature
- Replaced fixed hyperparameters with RandomizedSearchCV (30 configs x 3 algorithms x 5-fold CV)
- Added HistGradientBoostingRegressor as a third model candidate
- Switched Energy model from RandomForest to LinearRegression (Energy is an exact linear function of Temp + Energy_Source)
- Added prediction clipping for Env_Score to valid [0, 100] range
- Updated inference.py to use engineered features and prediction clipping
- Fixed hardcoded Windows path in train.py to use portable relative paths
- Added scipy to requirements.txt

**Performance improvements:**
| Model | v1 R² | v2 R² | v1 MAE | v2 MAE | MAE Reduction |
|-------|-------|-------|--------|--------|---------------|
| CO2 | 0.9984 | 0.9993 | 0.0191 | 0.0116 | -39% |
| Energy | 1.0 | 1.0 | 0.0 | 0.0 | (exact fit) |
| Env_Score | 0.9988 | 0.9995 | 0.4264 | 0.2541 | -40% |

### v1 — Initial Release (2026-03-24)

- Initial project setup with CO2, Energy, and Env_Score prediction models
- GradientBoosting and RandomForest with fixed hyperparameters
- React dashboard UI (BlueWave AI) with placeholder model cards
- Training pipeline, inference CLI, and data cleaning utility
- 10,000 synthetic PET bottle production training records
- What-if analysis and sensitivity plots
