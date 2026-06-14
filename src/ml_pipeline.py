"""ML Pipeline for AeroNet Lite (Module 5).

Two models live here:

1. Demand model      - regression on hourly delivery demand.
                       Trains on the Bike Sharing Demand CSV when present,
                       otherwise on synthetic in-process data so the demo
                       runs without any downloads.
2. Anomaly model     - multiclass classifier over drone telemetry that
                       labels each tick as Normal / BatteryAnomaly /
                       RouteAnomaly / SensorSpike.

Both models expose a thin dataclass wrapper (`DemandModel`, `AnomalyModel`)
with a single-row inference helper so the simulator and Streamlit UI can
call `.predict(...)` / `.classify(...)` without touching sklearn directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Path conventions used by the rest of the project.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
DEFAULT_BIKE_CSV = os.path.join(
    _PROJECT_ROOT, "data", "raw", "bike_sharing", "train.csv"
)
DEFAULT_AMAZON_CSV = os.path.join(
    _PROJECT_ROOT, "data", "raw", "amazon_delivery", "amazon_delivery.csv"
)

DEMAND_FEATURES: List[str] = [
    "hour",
    "day_of_week",
    "temperature",
    "weather",
]

DELIVERY_TIME_NUMERIC: List[str] = ["distance_km", "agent_age", "agent_rating"]
DELIVERY_TIME_CATEGORICAL: List[str] = ["weather", "traffic", "vehicle", "area"]
DELIVERY_TIME_FEATURES: List[str] = DELIVERY_TIME_NUMERIC + DELIVERY_TIME_CATEGORICAL

TELEMETRY_FEATURES: List[str] = [
    "battery_drop",
    "speed",
    "route_deviation",
    "altitude_change",
    "speed_change",
]

ANOMALY_CLASSES: List[str] = [
    "Normal",
    "BatteryAnomaly",
    "RouteAnomaly",
    "SensorSpike",
]


# --------------------------------------------------------------------------- #
# Dataclass wrappers
# --------------------------------------------------------------------------- #
@dataclass
class DemandModel:
    model: Any
    mae: float
    rmse: float
    feature_names: List[str] = field(default_factory=lambda: list(DEMAND_FEATURES))

    def predict(self, features: Dict[str, float]) -> float:
        """Predict expected demand for a single feature dict."""
        row = [[float(features.get(name, 0.0)) for name in self.feature_names]]
        x = pd.DataFrame(row, columns=self.feature_names)
        y = float(self.model.predict(x)[0])
        # Demand is non-negative.
        return max(0.0, y)


@dataclass
class DeliveryTimeModel:
    model: Any  # sklearn Pipeline (preprocessor + RandomForestRegressor)
    mae: float
    rmse: float
    feature_names: List[str] = field(default_factory=lambda: list(DELIVERY_TIME_FEATURES))

    def predict(self, features: Dict[str, Any]) -> float:
        """Predict delivery time in minutes for a single feature dict."""
        row = {name: features.get(name) for name in self.feature_names}
        x = pd.DataFrame([row])
        y = float(self.model.predict(x)[0])
        return max(0.0, y)


@dataclass
class AnomalyModel:
    model: Any
    accuracy: float
    confusion: List[List[int]]
    class_names: List[str] = field(default_factory=lambda: list(ANOMALY_CLASSES))
    feature_names: List[str] = field(default_factory=lambda: list(TELEMETRY_FEATURES))

    def classify(self, telemetry: Dict[str, float]) -> str:
        """Classify a single telemetry reading. Returns class name string."""
        row = [[float(telemetry.get(name, 0.0)) for name in self.feature_names]]
        x = pd.DataFrame(row, columns=self.feature_names)
        idx = int(self.model.predict(x)[0])
        if 0 <= idx < len(self.class_names):
            return self.class_names[idx]
        return "Normal"


# --------------------------------------------------------------------------- #
# Synthetic data generators
# --------------------------------------------------------------------------- #
def generate_synthetic_demand(n_days: int = 60, seed: int = 42) -> pd.DataFrame:
    """Synthetic hourly bike-share-style demand.

    Mimics the shape of the Bike Sharing Demand dataset:
      * morning + evening commute peaks
      * weekday/weekend split
      * temperature lift
      * weather penalty
    """
    rng = np.random.default_rng(seed)
    n_rows = n_days * 24
    hours = np.tile(np.arange(24), n_days)
    days = np.repeat(np.arange(n_days) % 7, 24)
    temperature = 15.0 + 10.0 * np.sin(np.arange(n_rows) / (24 * 7)) + rng.normal(0, 2, n_rows)
    weather = rng.choice([1, 2, 3, 4], size=n_rows, p=[0.65, 0.20, 0.12, 0.03])

    # Two commute peaks at hours 8 and 17 on weekdays.
    weekday = (days < 5).astype(float)
    morning = np.exp(-0.5 * ((hours - 8) / 1.6) ** 2)
    evening = np.exp(-0.5 * ((hours - 17) / 1.8) ** 2)
    midday = np.exp(-0.5 * ((hours - 13) / 3.0) ** 2)

    base = 30.0
    demand = (
        base
        + 180.0 * weekday * (morning + evening)
        + 110.0 * (1 - weekday) * midday
        + 4.0 * (temperature - 15.0)
        - 25.0 * (weather - 1)
        + rng.normal(0, 12, n_rows)
    )
    demand = np.clip(demand, 0, None)

    return pd.DataFrame(
        {
            "hour": hours.astype(int),
            "day_of_week": days.astype(int),
            "temperature": temperature.astype(float),
            "weather": weather.astype(int),
            "count": demand.astype(float),
        }
    )


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance between two lat/lng points in kilometres."""
    r = 6371.0
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlng = np.radians(lng2 - lng1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlng / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def _load_amazon_delivery(csv_path: str) -> pd.DataFrame:
    """Load and clean the Kaggle Amazon Delivery CSV.

    Computes haversine distance between store and drop, drops rows missing
    target or required columns, fills the small number of NaN traffic /
    weather / agent_rating values with sensible defaults.
    """
    df = pd.read_csv(csv_path)
    # Trim trailing whitespace from string columns (the raw CSV has e.g. "High ").
    for c in DELIVERY_TIME_CATEGORICAL:
        if c.title() in df.columns or c.capitalize() in df.columns:
            pass
    df = df.rename(
        columns={
            "Agent_Age": "agent_age",
            "Agent_Rating": "agent_rating",
            "Weather": "weather",
            "Traffic": "traffic",
            "Vehicle": "vehicle",
            "Area": "area",
            "Delivery_Time": "delivery_time",
        }
    )
    for c in DELIVERY_TIME_CATEGORICAL:
        df[c] = df[c].astype(str).str.strip()
    df["agent_rating"] = df["agent_rating"].fillna(df["agent_rating"].median())
    df["traffic"] = df["traffic"].replace({"nan": "Low"}).fillna("Low")
    df["weather"] = df["weather"].replace({"nan": "Sunny"}).fillna("Sunny")

    df["distance_km"] = df.apply(
        lambda r: _haversine_km(
            r["Store_Latitude"], r["Store_Longitude"],
            r["Drop_Latitude"], r["Drop_Longitude"],
        ),
        axis=1,
    )
    # A handful of rows have store==drop or absurd coords; clamp distance.
    df = df[(df["distance_km"] >= 0.05) & (df["distance_km"] <= 100.0)]
    df = df.dropna(subset=["delivery_time"])
    return df[DELIVERY_TIME_FEATURES + ["delivery_time"]].reset_index(drop=True)


def _load_bike_sharing(csv_path: str) -> pd.DataFrame:
    """Load the Kaggle Bike Sharing Demand CSV and reshape to our 4 features."""
    df = pd.read_csv(csv_path)
    # Kaggle file has: datetime, season, holiday, workingday, weather, temp,
    # atemp, humidity, windspeed, casual, registered, count.
    dt = pd.to_datetime(df["datetime"])
    out = pd.DataFrame(
        {
            "hour": dt.dt.hour.astype(int),
            "day_of_week": dt.dt.dayofweek.astype(int),
            "temperature": df["temp"].astype(float),
            "weather": df["weather"].astype(int),
            "count": df["count"].astype(float),
        }
    )
    return out


def generate_synthetic_telemetry(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Synthetic drone telemetry with four labeled regimes.

    Class generation rules (kept aligned with the project spec):
      * Normal          - low battery drop + low route deviation
      * BatteryAnomaly  - high battery drop
      * RouteAnomaly    - high route deviation
      * SensorSpike     - high altitude_change OR high speed_change
    """
    rng = np.random.default_rng(seed)
    # Roughly balanced four-way split.
    counts = {
        "Normal": int(n * 0.50),
        "BatteryAnomaly": int(n * 0.18),
        "RouteAnomaly": int(n * 0.18),
        "SensorSpike": n - int(n * 0.50) - int(n * 0.18) - int(n * 0.18),
    }

    rows: List[Dict[str, float]] = []

    # Normal: low battery drop, low deviation, mild sensor jitter.
    for _ in range(counts["Normal"]):
        rows.append(
            {
                "battery_drop": float(rng.uniform(0.1, 1.5)),
                "speed": float(rng.uniform(8.0, 14.0)),
                "route_deviation": float(rng.uniform(0.0, 1.5)),
                "altitude_change": float(rng.uniform(0.0, 1.0)),
                "speed_change": float(rng.uniform(0.0, 1.0)),
                "label": "Normal",
            }
        )

    # BatteryAnomaly: high battery drop dominates.
    for _ in range(counts["BatteryAnomaly"]):
        rows.append(
            {
                "battery_drop": float(rng.uniform(4.0, 9.0)),
                "speed": float(rng.uniform(6.0, 14.0)),
                "route_deviation": float(rng.uniform(0.0, 2.0)),
                "altitude_change": float(rng.uniform(0.0, 1.5)),
                "speed_change": float(rng.uniform(0.0, 1.5)),
                "label": "BatteryAnomaly",
            }
        )

    # RouteAnomaly: high route deviation dominates.
    for _ in range(counts["RouteAnomaly"]):
        rows.append(
            {
                "battery_drop": float(rng.uniform(0.2, 2.0)),
                "speed": float(rng.uniform(7.0, 15.0)),
                "route_deviation": float(rng.uniform(4.0, 9.0)),
                "altitude_change": float(rng.uniform(0.0, 1.5)),
                "speed_change": float(rng.uniform(0.0, 1.5)),
                "label": "RouteAnomaly",
            }
        )

    # SensorSpike: high altitude or speed change.
    for _ in range(counts["SensorSpike"]):
        spike_alt = rng.random() < 0.5
        rows.append(
            {
                "battery_drop": float(rng.uniform(0.2, 2.0)),
                "speed": float(rng.uniform(7.0, 15.0)),
                "route_deviation": float(rng.uniform(0.0, 2.0)),
                "altitude_change": float(rng.uniform(4.0, 9.0)) if spike_alt else float(rng.uniform(0.0, 1.5)),
                "speed_change": float(rng.uniform(0.0, 1.5)) if spike_alt else float(rng.uniform(4.0, 9.0)),
                "label": "SensorSpike",
            }
        )

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Training entry points
# --------------------------------------------------------------------------- #
def train_demand_model(csv_path: Optional[str] = None) -> DemandModel:
    """Train the demand regressor.

    Tries `csv_path` first, then `DEFAULT_BIKE_CSV`, then synthetic data.
    Picks whichever of LinearRegression / RandomForestRegressor scores the
    lower RMSE on the held-out split.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import train_test_split

    src_label = "synthetic"
    df: Optional[pd.DataFrame] = None

    candidate = csv_path or DEFAULT_BIKE_CSV
    if candidate and os.path.isfile(candidate):
        try:
            df = _load_bike_sharing(candidate)
            src_label = f"csv:{candidate}"
        except Exception as exc:  # noqa: BLE001
            print(f"[demand] failed to load CSV {candidate}: {exc}; using synthetic")
            df = None

    if df is None:
        df = generate_synthetic_demand()

    X = df[DEMAND_FEATURES]
    y = df["count"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_mae = float(mean_absolute_error(y_test, lr_pred))
    lr_rmse = float(np.sqrt(mean_squared_error(y_test, lr_pred)))

    rf = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_mae = float(mean_absolute_error(y_test, rf_pred))
    rf_rmse = float(np.sqrt(mean_squared_error(y_test, rf_pred)))

    print(f"[demand] data source: {src_label} (rows={len(df)})")
    print(f"[demand] LinearRegression  MAE={lr_mae:.3f}  RMSE={lr_rmse:.3f}")
    print(f"[demand] RandomForest      MAE={rf_mae:.3f}  RMSE={rf_rmse:.3f}")

    if rf_rmse <= lr_rmse:
        winner_name = "RandomForest"
        chosen = DemandModel(model=rf, mae=rf_mae, rmse=rf_rmse)
    else:
        winner_name = "LinearRegression"
        chosen = DemandModel(model=lr, mae=lr_mae, rmse=lr_rmse)

    print(f"[demand] selected: {winner_name}  MAE={chosen.mae:.3f}  RMSE={chosen.rmse:.3f}")
    return chosen


def train_delivery_time_model(
    csv_path: Optional[str] = None,
) -> Optional[DeliveryTimeModel]:
    """Train a delivery-time regressor on the Kaggle Amazon Delivery dataset.

    Returns None when the CSV is missing — the simulator treats this model
    as optional (unlike demand, which falls back to synthetic data).
    """
    candidate = csv_path or DEFAULT_AMAZON_CSV
    if not os.path.isfile(candidate):
        return None

    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    try:
        df = _load_amazon_delivery(candidate)
    except Exception as exc:  # noqa: BLE001
        print(f"[delivery_time] failed to load CSV {candidate}: {exc}")
        return None

    X = df[DELIVERY_TIME_FEATURES]
    y = df["delivery_time"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), DELIVERY_TIME_CATEGORICAL),
        ],
        remainder="passthrough",
    )
    pipe = Pipeline(
        [
            ("pre", pre),
            ("rf", RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)),
        ]
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    mae = float(mean_absolute_error(y_test, pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))

    print(f"[delivery_time] data source: csv:{candidate} (rows={len(df)})")
    print(f"[delivery_time] RandomForest      MAE={mae:.3f}  RMSE={rmse:.3f}  (target: minutes)")

    return DeliveryTimeModel(model=pipe, mae=mae, rmse=rmse)


def predict_delivery_eta(
    model: DeliveryTimeModel,
    distance_km: float,
    weather: str = "Sunny",
    traffic: str = "Low",
    vehicle: str = "motorcycle",
    area: str = "Urban",
    agent_age: int = 30,
    agent_rating: float = 4.7,
) -> float:
    """Predict delivery ETA (minutes) for given conditions.

    The simulator passes Manhattan-grid distance scaled to km; categorical
    defaults reflect typical clear-day conditions so the prediction can be
    treated as a baseline ETA per delivery.
    """
    return model.predict(
        {
            "distance_km": float(distance_km),
            "agent_age": int(agent_age),
            "agent_rating": float(agent_rating),
            "weather": weather,
            "traffic": traffic,
            "vehicle": vehicle,
            "area": area,
        }
    )


def train_anomaly_model() -> AnomalyModel:
    """Train the telemetry anomaly classifier on synthetic data."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.model_selection import train_test_split

    df = generate_synthetic_telemetry()
    label_to_idx = {name: i for i, name in enumerate(ANOMALY_CLASSES)}
    y = df["label"].map(label_to_idx).astype(int)
    X = df[TELEMETRY_FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    acc = float(accuracy_score(y_test, pred))
    cm = confusion_matrix(y_test, pred, labels=list(range(len(ANOMALY_CLASSES))))

    print(f"[anomaly] data source: synthetic (rows={len(df)})")
    print(f"[anomaly] RandomForest accuracy={acc:.4f}")
    print("[anomaly] confusion matrix (rows=true, cols=pred):")
    header = "             " + "  ".join(f"{n:>14s}" for n in ANOMALY_CLASSES)
    print(header)
    for i, name in enumerate(ANOMALY_CLASSES):
        row_vals = "  ".join(f"{int(v):>14d}" for v in cm[i])
        print(f"  {name:<10s}  {row_vals}")

    return AnomalyModel(
        model=clf,
        accuracy=acc,
        confusion=cm.astype(int).tolist(),
    )


# --------------------------------------------------------------------------- #
# Grid integration
# --------------------------------------------------------------------------- #
def predict_zone_demand(grid, demand_model: DemandModel) -> Dict[Tuple[int, int], float]:
    """Per-cell expected demand at hour=12, weekday=Monday, temp=20, weather=1.

    Multiplied by `density / 5000` so residential blocks (the densest zone)
    are anchored at 1.0x and emptier zones scale down proportionally.
    """
    base_features = {
        "hour": 12,
        "day_of_week": 0,
        "temperature": 20.0,
        "weather": 1,
    }
    base_pred = demand_model.predict(base_features)

    out: Dict[Tuple[int, int], float] = {}
    for row in grid:
        for cell in row:
            scale = float(cell.density) / 5000.0
            out[(cell.row, cell.col)] = float(base_pred * scale)
    return out


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=" * 60)
    print("AeroNet Lite - ML Pipeline smoke test")
    print("=" * 60)

    demand = train_demand_model()
    print()
    anomaly = train_anomaly_model()
    print()

    # Grid integration sanity check.
    from src.grid_model import build_sample_grid

    grid = build_sample_grid()
    zone_demand = predict_zone_demand(grid, demand)
    sample_cells = [(1, 1), (1, 3), (4, 0), (8, 0), (0, 4)]
    print("[integration] predicted demand for sample cells (hour=12, temp=20):")
    for pos in sample_cells:
        print(f"  cell {pos} zone={grid[pos[0]][pos[1]].zone.value:<11s} "
              f"density={grid[pos[0]][pos[1]].density:>5d}  "
              f"demand={zone_demand[pos]:.3f}")

    # One-shot inference samples for both models.
    print()
    sample_demand = demand.predict(
        {"hour": 8, "day_of_week": 1, "temperature": 22.0, "weather": 1}
    )
    print(f"[inference] demand at Tue 08:00, 22C, clear: {sample_demand:.2f}")

    sample_class = anomaly.classify(
        {
            "battery_drop": 6.0,
            "speed": 10.0,
            "route_deviation": 0.5,
            "altitude_change": 0.5,
            "speed_change": 0.3,
        }
    )
    print(f"[inference] high battery drop telemetry classified as: {sample_class}")

    sample_class2 = anomaly.classify(
        {
            "battery_drop": 0.5,
            "speed": 11.0,
            "route_deviation": 0.3,
            "altitude_change": 0.4,
            "speed_change": 6.5,
        }
    )
    print(f"[inference] high speed_change telemetry classified as:  {sample_class2}")

    print()
    print("Smoke test complete.")
