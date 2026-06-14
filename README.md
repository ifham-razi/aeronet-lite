# AeroNet Lite

Autonomous Drone Delivery Simulation built for the BS Data Science AI semester project (SP2026).

A 10x10 grid simulator covering five AI modules:

1. **Layout Validator** — Constraint Satisfaction (CSP)
2. **Fleet Selector** — Heuristic / Genetic Algorithm
3. **Path Planner** — A* search
4. **Disruption Handler** — Real-time replanning
5. **ML Pipeline** — Demand forecasting (regression) + flight anomaly detection (classification)

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python -m src.main
```

## Project layout

```
aeronet_lite/
  data/                 # raw + processed datasets
  src/                  # all module code
    grid_model.py
    layout_validator.py
    fleet_selector.py
    astar_planner.py
    delivery_simulator.py
    ml_pipeline.py
    visualization.py
    main.py
  notebooks/            # ML experimentation
  report/figures/       # plots for the final report
```

## Datasets

Real data is auto-detected at startup. If a CSV is missing the pipeline falls
back to in-process synthetic data so the demo always runs.

| Purpose | Source | Path | Status |
|---|---|---|---|
| Demand forecasting | Kaggle Bike Sharing Demand | `data/raw/bike_sharing/train.csv` | auto-loaded |
| Grid density | Kaggle US City Population Densities | `data/raw/population_density/uscitypopdensity.csv` | auto-loaded |
| Anomaly classifier | synthetic (4 labeled regimes) | generated in-process | auto-loaded |
| Delivery time (extension) | Kaggle Amazon Delivery | `data/raw/amazon_delivery/amazon_delivery.csv` | available, not wired |
| UAV fault (extension) | CMU ALFA dataset | `data/raw/uav_fault/processed/` | available, not wired (per-flight ROS topics — too complex for 2-week scope) |
