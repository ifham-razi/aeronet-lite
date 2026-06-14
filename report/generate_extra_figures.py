"""Generate additional figures for the project report.

Produces, into report/figures/:
  - confusion_matrix.png       seaborn heatmap of the anomaly classifier
  - grid_gallery.png           4 procedurally generated layouts side by side
  - demand_model_comparison.png  LR vs RF MAE/RMSE bar chart
  - feature_importance.png     RF feature importances for demand + anomaly
  - delivery_time_scatter.png  actual vs predicted on a held-out split
  - architecture_diagram.png   high-level module + data flow

Run from the project root:
    python -m report.generate_extra_figures
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from src.grid_model import (
    GRID_SIZE,
    Zone,
    build_random_grid,
    enrich_grid_with_population,
    load_population_buckets,
)
from src.ml_pipeline import (
    DEFAULT_AMAZON_CSV,
    DEFAULT_BIKE_CSV,
    DELIVERY_TIME_CATEGORICAL,
    DELIVERY_TIME_FEATURES,
    TELEMETRY_FEATURES,
    _load_amazon_delivery,
    _load_bike_sharing,
    generate_synthetic_demand,
    generate_synthetic_telemetry,
    train_anomaly_model,
    train_delivery_time_model,
    train_demand_model,
)
from src.visualization import plot_zone_map

OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANOMALY_CLASSES = ["Normal", "BatteryAnomaly", "RouteAnomaly", "SensorSpike"]


def fig_confusion_matrix() -> Path:
    """Heatmap version of the anomaly classifier confusion matrix."""
    model = train_anomaly_model()
    cm = np.array(model.confusion)
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=ANOMALY_CLASSES, yticklabels=ANOMALY_CLASSES,
        cbar_kws={"label": "count"}, ax=ax, square=True,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(
        f"Anomaly classifier confusion matrix\n"
        f"Random Forest, accuracy = {model.accuracy:.3f}"
    )
    plt.tight_layout()
    out = OUT_DIR / "confusion_matrix.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_grid_gallery() -> Path:
    """Four different procedurally generated layouts in a 2x2 grid."""
    seeds = [11, 42, 101, 777]
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    buckets = load_population_buckets()
    for ax, seed in zip(axes.flatten(), seeds):
        grid = build_random_grid(seed=seed)
        enrich_grid_with_population(grid, buckets, seed=seed)
        _draw_zone_map_on_axis(grid, ax, title=f"seed = {seed}")
    fig.suptitle(
        "Procedurally generated city layouts (4 different seeds)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    out = OUT_DIR / "grid_gallery.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def _draw_zone_map_on_axis(grid, ax, title: str = "") -> None:
    """Lightweight reimplementation of plot_zone_map onto a given axis."""
    zone_color = {
        Zone.RESIDENTIAL: "#A8D5E2",
        Zone.COMMERCIAL: "#F4C28F",
        Zone.HOSPITAL: "#F5A8A8",
        Zone.SCHOOL: "#B5E2B5",
        Zone.INDUSTRIAL: "#C9A77C",
        Zone.OPEN: "#F5F5F5",
    }
    # Render zones as colored cells
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cell = grid[r][c]
            ax.add_patch(plt.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                facecolor=zone_color[cell.zone],
                edgecolor="white", linewidth=0.8,
            ))
    # Overlays
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cell = grid[r][c]
            if cell.is_hub:
                ax.add_patch(plt.Rectangle(
                    (c - 0.36, r - 0.36), 0.72, 0.72,
                    facecolor="#1a3da8", edgecolor="white", linewidth=1.5,
                ))
                ax.text(c, r, "H", color="white", fontweight="bold",
                        ha="center", va="center", fontsize=10)
            elif cell.is_charging:
                ax.add_patch(plt.Rectangle(
                    (c - 0.32, r - 0.32), 0.64, 0.64,
                    facecolor="#0e7c66", edgecolor="white", linewidth=1.2,
                ))
                ax.text(c, r, "C", color="white", fontweight="bold",
                        ha="center", va="center", fontsize=9)
            elif cell.is_medical_pickup:
                ax.text(c, r, "+", color="#c8243a", fontweight="bold",
                        ha="center", va="center", fontsize=22)
            elif cell.no_fly:
                ax.add_patch(plt.Rectangle(
                    (c - 0.36, r - 0.36), 0.72, 0.72,
                    facecolor="#222", edgecolor="white", linewidth=1.5,
                ))
                ax.text(c, r, "X", color="white", fontweight="bold",
                        ha="center", va="center", fontsize=10)
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(GRID_SIZE - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=11)


def fig_demand_model_comparison() -> Path:
    """Bar chart comparing Linear Regression vs Random Forest on demand data."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import train_test_split

    if Path(DEFAULT_BIKE_CSV).exists():
        df = _load_bike_sharing(DEFAULT_BIKE_CSV)
        src_label = "Bike Sharing CSV"
    else:
        df = generate_synthetic_demand()
        src_label = "Synthetic"

    features = ["hour", "day_of_week", "temperature", "weather"]
    X, y = df[features], df["count"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr = LinearRegression().fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1).fit(X_train, y_train)

    lr_pred, rf_pred = lr.predict(X_test), rf.predict(X_test)
    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_rmse = float(np.sqrt(mean_squared_error(y_test, lr_pred)))
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = float(np.sqrt(mean_squared_error(y_test, rf_pred)))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    metrics = ["MAE", "RMSE"]
    lr_vals = [lr_mae, lr_rmse]
    rf_vals = [rf_mae, rf_rmse]
    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax.bar(x - width / 2, lr_vals, width, label="Linear Regression", color="#cccccc", edgecolor="#666")
    bars2 = ax.bar(x + width / 2, rf_vals, width, label="Random Forest", color="#1f77b4", edgecolor="#0a3a6e")
    for bar, val in zip(list(bars1) + list(bars2), lr_vals + rf_vals):
        ax.annotate(f"{val:.1f}", xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Error (lower is better)")
    ax.set_title(f"Demand Model Comparison on {src_label} ({len(df):,} rows)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = OUT_DIR / "demand_model_comparison.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_feature_importance() -> Path:
    """Two-panel feature importance: demand RF + anomaly RF."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split

    # Demand RF
    if Path(DEFAULT_BIKE_CSV).exists():
        df_demand = _load_bike_sharing(DEFAULT_BIKE_CSV)
    else:
        df_demand = generate_synthetic_demand()
    demand_features = ["hour", "day_of_week", "temperature", "weather"]
    Xd, yd = df_demand[demand_features], df_demand["count"]
    Xd_tr, _, yd_tr, _ = train_test_split(Xd, yd, test_size=0.2, random_state=42)
    rf_d = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1).fit(Xd_tr, yd_tr)
    demand_imp = sorted(zip(demand_features, rf_d.feature_importances_),
                        key=lambda kv: kv[1], reverse=True)

    # Anomaly RF
    df_an = generate_synthetic_telemetry()
    label_to_idx = {n: i for i, n in enumerate(ANOMALY_CLASSES)}
    Xa, ya = df_an[TELEMETRY_FEATURES], df_an["label"].map(label_to_idx).astype(int)
    Xa_tr, _, ya_tr, _ = train_test_split(Xa, ya, test_size=0.2, random_state=42, stratify=ya)
    rf_a = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1).fit(Xa_tr, ya_tr)
    anomaly_imp = sorted(zip(TELEMETRY_FEATURES, rf_a.feature_importances_),
                         key=lambda kv: kv[1], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    names_d, vals_d = zip(*demand_imp)
    axes[0].barh(list(names_d)[::-1], list(vals_d)[::-1], color="#1f77b4", edgecolor="#0a3a6e")
    axes[0].set_title("Demand RF — feature importance")
    axes[0].set_xlabel("Gini importance")
    axes[0].grid(axis="x", linestyle="--", alpha=0.4)

    names_a, vals_a = zip(*anomaly_imp)
    axes[1].barh(list(names_a)[::-1], list(vals_a)[::-1], color="#d62728", edgecolor="#7c1313")
    axes[1].set_title("Anomaly RF — feature importance")
    axes[1].set_xlabel("Gini importance")
    axes[1].grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out = OUT_DIR / "feature_importance.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_delivery_time_scatter() -> Path:
    """Actual vs predicted delivery time on a held-out Amazon split."""
    if not Path(DEFAULT_AMAZON_CSV).exists():
        return Path()  # silently skip

    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    df = _load_amazon_delivery(DEFAULT_AMAZON_CSV)
    X, y = df[DELIVERY_TIME_FEATURES], df["delivery_time"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ("pre", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), DELIVERY_TIME_CATEGORICAL),
        ], remainder="passthrough")),
        ("rf", RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)),
    ]).fit(X_train, y_train)

    pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))

    fig, ax = plt.subplots(figsize=(7, 6.5))
    rng = np.random.default_rng(42)
    sample = rng.choice(len(y_test), size=min(2500, len(y_test)), replace=False)
    ax.scatter(y_test.values[sample], pred[sample],
               alpha=0.18, s=12, color="#1f77b4", edgecolor="none")
    lim_max = max(y_test.max(), pred.max()) * 1.02
    ax.plot([0, lim_max], [0, lim_max], "--", color="#444", lw=1.3, label="ideal y = x")
    ax.set_xlabel("Actual delivery time (minutes)")
    ax.set_ylabel("Predicted delivery time (minutes)")
    ax.set_title(
        f"Amazon Delivery Time Regressor: actual vs predicted\n"
        f"Random Forest · MAE = {mae:.2f} min · RMSE = {rmse:.2f} min · n_test = {len(y_test):,}"
    )
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = OUT_DIR / "delivery_time_scatter.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_architecture_diagram() -> Path:
    """High-level module + data flow diagram drawn in matplotlib."""
    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    def block(x, y, w, h, label, sub, fc):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05,rounding_size=0.15",
            linewidth=1.4, edgecolor="#222", facecolor=fc,
        ))
        ax.text(x + w / 2, y + h * 0.62, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color="#111")
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                fontsize=8.5, color="#444", style="italic")

    def arrow(x1, y1, x2, y2, color="#666", style="-|>", lw=1.4, label=None):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                            connectionstyle="arc3,rad=0.0"),
        )
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, label,
                    ha="center", fontsize=8, color=color)

    # Center: shared grid
    block(3.6, 2.7, 2.8, 1.4, "Shared Grid Model",
          "10x10 cells · zones · hubs · no-fly", "#FFE7A8")

    # Modules around the grid
    block(0.2, 5.4, 2.6, 1.2, "Layout Validator", "CSP rules R1–R4", "#D5E8FF")
    block(3.7, 5.4, 2.6, 1.2, "Fleet Selector",   "Brute force / GA", "#D9F2D6")
    block(7.2, 5.4, 2.6, 1.2, "A* Path Planner",  "Cheapest 4-conn paths", "#FFD6D6")

    block(0.2, 0.4, 2.6, 1.2, "Disruption Handler", "Replan around no-fly", "#F5E0FF")
    block(3.7, 0.4, 2.6, 1.2, "ML Pipeline",
          "Demand · Anomaly · ETA", "#FFE0CC")
    block(7.2, 0.4, 2.6, 1.2, "Streamlit Dashboard",
          "Plotly · step-through · auto-play", "#E5F4F2")

    # Data sources at far left/right
    block(0.2, 3.1, 2.0, 0.8, "Bike Sharing\n(10,886)", "demand", "#FFFFFF")
    block(0.2, 1.9, 2.0, 0.8, "Amazon Delivery\n(43,551)", "ETA", "#FFFFFF")
    block(7.8, 3.1, 2.0, 0.8, "US Pop Density", "grid buckets", "#FFFFFF")
    block(7.8, 1.9, 2.0, 0.8, "Synthetic Telemetry\n(2,000)", "anomaly", "#FFFFFF")

    # Arrows: modules -> shared grid
    cx, cy = 5.0, 3.4
    arrow(1.5, 5.4, cx - 1.0, cy + 0.4, color="#1f4e79")
    arrow(5.0, 5.4, cx, cy + 0.7, color="#1f4e79")
    arrow(8.5, 5.4, cx + 1.0, cy + 0.4, color="#1f4e79")
    arrow(1.5, 1.6, cx - 1.0, cy - 0.6, color="#1f4e79", style="<|-")
    arrow(5.0, 1.6, cx, cy - 0.7, color="#1f4e79", style="<|-|>")
    arrow(8.5, 1.6, cx + 1.0, cy - 0.6, color="#1f4e79", style="<|-")

    # Datasets feed ML pipeline
    arrow(2.2, 3.5, 3.7, 1.3, color="#a04040", lw=1.0)
    arrow(2.2, 2.3, 3.7, 1.0, color="#a04040", lw=1.0)
    arrow(7.8, 2.3, 6.3, 1.0, color="#a04040", lw=1.0)
    arrow(7.8, 3.5, 6.3, 1.3, color="#a04040", lw=1.0)

    # Title
    ax.text(5.0, 6.85, "AeroNet Lite — System Architecture",
            ha="center", fontsize=14, fontweight="bold")

    # Legend
    legend_elems = [
        mpatches.Patch(facecolor="#FFE7A8", edgecolor="#222", label="Shared state"),
        mpatches.Patch(facecolor="#D5E8FF", edgecolor="#222", label="AI module"),
        mpatches.Patch(facecolor="#FFFFFF", edgecolor="#222", label="External dataset"),
        Line2D([0], [0], color="#1f4e79", lw=1.6, label="grid I/O"),
        Line2D([0], [0], color="#a04040", lw=1.0, label="dataset feed"),
    ]
    ax.legend(handles=legend_elems, loc="lower center", ncol=5,
              bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9)

    plt.tight_layout()
    out = OUT_DIR / "architecture_diagram.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("Generating extra figures into", OUT_DIR)
    for fn in (
        fig_architecture_diagram,
        fig_grid_gallery,
        fig_confusion_matrix,
        fig_demand_model_comparison,
        fig_feature_importance,
        fig_delivery_time_scatter,
    ):
        path = fn()
        if path and path.exists():
            print(f"  wrote {path.name}  ({path.stat().st_size // 1024} KB)")
        else:
            print(f"  skipped {fn.__name__}")
    print("Done.")
