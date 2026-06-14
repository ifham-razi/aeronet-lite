"""Matplotlib-based visualization for AeroNet Lite."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .grid_model import (
    GRID_SIZE,
    Grid,
    Zone,
    build_sample_grid,
)

ZONE_COLORS: Dict[Zone, str] = {
    Zone.RESIDENTIAL: "#A8D5E2",
    Zone.COMMERCIAL: "#F4C28F",
    Zone.HOSPITAL: "#F5A8A8",
    Zone.SCHOOL: "#B5E2B5",
    Zone.INDUSTRIAL: "#C9A77C",
    Zone.OPEN: "#F5F5F5",
}

HUB_COLOR = "#1F4E79"
CHG_COLOR = "#2CA6A4"
MED_COLOR = "#C0392B"
NOFLY_COLOR = "#000000"


def _finalize(fig: plt.Figure, save_path: Optional[str]) -> None:
    if save_path is None:
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _draw_zone_background(ax: plt.Axes, grid: Grid) -> None:
    size = len(grid)
    for r in range(size):
        for c in range(size):
            cell = grid[r][c]
            color = ZONE_COLORS[cell.zone]
            rect = mpatches.Rectangle(
                (c - 0.5, r - 0.5), 1.0, 1.0, facecolor=color, edgecolor="#CCCCCC", linewidth=0.5
            )
            ax.add_patch(rect)


def _draw_overlays(ax: plt.Axes, grid: Grid) -> None:
    size = len(grid)
    for r in range(size):
        for c in range(size):
            cell = grid[r][c]
            if cell.no_fly:
                ax.add_patch(
                    mpatches.Rectangle(
                        (c - 0.45, r - 0.45), 0.9, 0.9, facecolor=NOFLY_COLOR, edgecolor="black"
                    )
                )
                ax.text(c, r, "NO-FLY", ha="center", va="center", color="white", fontsize=7, fontweight="bold")
            elif cell.is_hub:
                ax.add_patch(
                    mpatches.Rectangle(
                        (c - 0.4, r - 0.4), 0.8, 0.8, facecolor=HUB_COLOR, edgecolor="black"
                    )
                )
                ax.text(c, r, "HUB", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
            elif cell.is_charging:
                ax.add_patch(
                    mpatches.Rectangle(
                        (c - 0.4, r - 0.4), 0.8, 0.8, facecolor=CHG_COLOR, edgecolor="black"
                    )
                )
                ax.text(c, r, "CHG", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
            elif cell.is_medical_pickup:
                ax.plot(c, r, marker="P", markersize=14, color=MED_COLOR, markeredgecolor="black")
                ax.text(c, r - 0.32, "MED", ha="center", va="center", color=MED_COLOR, fontsize=7, fontweight="bold")


def _format_grid_axes(ax: plt.Axes, size: int, title: str) -> None:
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels(range(size))
    ax.set_yticklabels(range(size))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(title)
    ax.grid(True, color="#DDDDDD", linewidth=0.5)
    ax.set_axisbelow(True)


def _zone_legend_handles() -> List[mpatches.Patch]:
    handles: List[mpatches.Patch] = [
        mpatches.Patch(facecolor=ZONE_COLORS[z], edgecolor="#888888", label=z.value)
        for z in Zone
    ]
    handles.append(mpatches.Patch(facecolor=HUB_COLOR, edgecolor="black", label="HUB"))
    handles.append(mpatches.Patch(facecolor=CHG_COLOR, edgecolor="black", label="Charging"))
    handles.append(
        Line2D([0], [0], marker="P", color="w", markerfacecolor=MED_COLOR,
               markeredgecolor="black", markersize=10, label="Medical pickup")
    )
    handles.append(mpatches.Patch(facecolor=NOFLY_COLOR, edgecolor="black", label="No-fly"))
    return handles


def plot_zone_map(
    grid: Grid,
    save_path: Optional[str] = None,
    title: str = "AeroNet Lite — Zone Map",
) -> None:
    size = len(grid)
    fig, ax = plt.subplots(figsize=(9, 8))
    _draw_zone_background(ax, grid)
    _draw_overlays(ax, grid)
    _format_grid_axes(ax, size, title)
    ax.legend(
        handles=_zone_legend_handles(),
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=9,
    )
    fig.tight_layout()
    _finalize(fig, save_path)


def plot_route_map(
    grid: Grid,
    routes: List[List[Tuple[int, int]]],
    drone_labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    size = len(grid)
    fig, ax = plt.subplots(figsize=(9, 8))
    _draw_zone_background(ax, grid)
    _draw_overlays(ax, grid)
    _format_grid_axes(ax, size, "AeroNet Lite — Routes")

    cmap = plt.get_cmap("tab10")
    legend_handles: List[Line2D] = []
    for i, route in enumerate(routes):
        if not route:
            continue
        color = cmap(i % 10)
        ys = [p[0] for p in route]
        xs = [p[1] for p in route]
        ax.plot(xs, ys, color=color, linewidth=2.5, alpha=0.9, zorder=3)
        ax.plot(xs[0], ys[0], marker="o", markersize=11, color=color,
                markeredgecolor="black", zorder=4)
        ax.plot(xs[-1], ys[-1], marker="*", markersize=16, color=color,
                markeredgecolor="black", zorder=4)
        label = drone_labels[i] if drone_labels and i < len(drone_labels) else f"Drone {i + 1}"
        legend_handles.append(Line2D([0], [0], color=color, linewidth=2.5, label=label))

    zone_handles = _zone_legend_handles()
    ax.legend(
        handles=legend_handles + zone_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=8,
    )
    fig.tight_layout()
    _finalize(fig, save_path)


def plot_demand_heatmap(
    grid: Grid,
    demand_dict: Optional[Dict[Tuple[int, int], float]] = None,
    save_path: Optional[str] = None,
) -> None:
    size = len(grid)
    matrix = np.zeros((size, size), dtype=float)
    if demand_dict is None:
        for r in range(size):
            for c in range(size):
                matrix[r, c] = grid[r][c].demand
    else:
        for (r, c), v in demand_dict.items():
            if 0 <= r < size and 0 <= c < size:
                matrix[r, c] = float(v)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(matrix, cmap="YlOrRd", origin="upper")
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title("AeroNet Lite — Demand Heatmap")
    ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
    ax.grid(which="minor", color="#CCCCCC", linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    for r in range(size):
        for c in range(size):
            val = matrix[r, c]
            if val > 0:
                ax.text(c, r, f"{val:.1f}", ha="center", va="center",
                        color="black", fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Demand")
    fig.tight_layout()
    _finalize(fig, save_path)


def plot_event_timeline(
    events: List[Tuple[int, str]],
    save_path: Optional[str] = None,
) -> None:
    n = max(len(events), 1)
    fig_height = max(3.0, 0.4 * n + 1.0)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    ax.axis("off")
    ax.set_title("AeroNet Lite — Event Timeline", loc="left", fontsize=13, fontweight="bold")

    if not events:
        ax.text(0.02, 0.5, "(no events)", fontsize=11, color="#666666", transform=ax.transAxes)
    else:
        sorted_events = sorted(events, key=lambda e: e[0])
        n_ev = len(sorted_events)
        for i, (step, msg) in enumerate(sorted_events):
            y = 1.0 - (i + 1) / (n_ev + 1)
            ax.text(0.02, y, f"  •  t={step:>3}", fontsize=10, fontweight="bold",
                    color="#1F4E79", transform=ax.transAxes, family="monospace")
            ax.text(0.18, y, msg, fontsize=10, color="#222222", transform=ax.transAxes)

    _finalize(fig, save_path)


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..", "report", "figures")
    base = os.path.abspath(base)
    os.makedirs(base, exist_ok=True)

    grid = build_sample_grid()

    plot_zone_map(grid, save_path=os.path.join(base, "zone_map.png"))

    fake_route = [(1, 3), (2, 3), (3, 3), (4, 3), (4, 2)]
    plot_route_map(
        grid,
        routes=[fake_route],
        drone_labels=["Drone A"],
        save_path=os.path.join(base, "route_map_demo.png"),
    )

    plot_demand_heatmap(grid, save_path=os.path.join(base, "demand_heatmap.png"))

    sample_events = [
        (0, "Drone A dispatched from HUB (1,3)"),
        (3, "Charging pad reached at (2,4)"),
        (7, "Medical pickup completed at (4,2)"),
        (12, "Drone A returned to HUB"),
    ]
    plot_event_timeline(sample_events, save_path=os.path.join(base, "event_timeline_demo.png"))

    print("Figures written to:", base)
    for name in ("zone_map.png", "route_map_demo.png", "demand_heatmap.png", "event_timeline_demo.png"):
        p = os.path.join(base, name)
        if os.path.exists(p):
            print(f"  {name}: {os.path.getsize(p)} bytes")
        else:
            print(f"  {name}: MISSING")
