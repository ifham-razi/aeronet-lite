"""Interactive Streamlit dashboard for AeroNet Lite.

Run with:  streamlit run src/streamlit_app.py

Features:
  * Step-through controls (Step / Run to end / Auto-play / Restart).
  * Animated city map (Plotly) with zone backgrounds, building icons,
    drone markers + trails, planned routes, no-fly overlays.
  * Live KPIs, event log, drone status table, ML metrics,
    demand heatmap, layout-validator report.
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.grid_model import GRID_SIZE, Zone
from src.ml_pipeline import (
    train_anomaly_model,
    train_delivery_time_model,
    train_demand_model,
)
from src.simulation import TOTAL_STEPS, Simulation


# --------------------------------------------------------------------------- #
# Visual config — chosen to read like an actual city block
# --------------------------------------------------------------------------- #
ZONE_FILL = {
    Zone.RESIDENTIAL: "#A8D5E2",
    Zone.COMMERCIAL: "#F4C28F",
    Zone.HOSPITAL: "#F5A8A8",
    Zone.SCHOOL: "#B5E2B5",
    Zone.INDUSTRIAL: "#C9A77C",
    Zone.OPEN: "#F2EFE6",
}
ZONE_ICON = {
    Zone.RESIDENTIAL: "\U0001F3E0",   # house
    Zone.COMMERCIAL: "\U0001F3EC",    # office building
    Zone.HOSPITAL: "\U0001F3E5",      # hospital
    Zone.SCHOOL: "\U0001F3EB",        # school
    Zone.INDUSTRIAL: "\U0001F3ED",    # factory
    Zone.OPEN: "\U0001F33F",          # herb / greenery
}
DRONE_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]
STATUS_COLOR = {
    "idle": "#888888",
    "to_pickup": "#1f77b4",
    "to_dropoff": "#2ca02c",
    "returning": "#9467bd",
    "delayed": "#d62728",
    "failed": "#7b1d1d",
    "anomaly": "#ff7f0e",
}

st.set_page_config(
    page_title="AeroNet Lite Dashboard",
    page_icon="\U0001F681",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Cached model loaders — train once per app session, reuse across restarts.
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Training demand model on Bike Sharing data...")
def get_demand_model():
    return train_demand_model()


@st.cache_resource(show_spinner="Training anomaly classifier...")
def get_anomaly_model():
    return train_anomaly_model()


@st.cache_resource(show_spinner="Training delivery-time model on Amazon data...")
def get_eta_model():
    return train_delivery_time_model()


# --------------------------------------------------------------------------- #
# Session state init
# --------------------------------------------------------------------------- #
def init_simulation(seed: int) -> Simulation:
    sim = Simulation(seed=seed)
    sim.preloaded_demand = get_demand_model()
    sim.preloaded_anomaly = get_anomaly_model()
    sim.preloaded_eta = get_eta_model()
    return sim


if "sim" not in st.session_state:
    default_seed = int(time.time()) % 1_000_000
    st.session_state.run_seed = default_seed
    st.session_state.sim = init_simulation(default_seed)
    st.session_state.autoplay = False
    st.session_state.autoplay_delay = 0.6


sim: Simulation = st.session_state.sim


# --------------------------------------------------------------------------- #
# Plotly renderer — the city map
# --------------------------------------------------------------------------- #
def render_city_map(sim: Simulation) -> go.Figure:
    fig = go.Figure()

    if sim.grid is None:
        fig.update_layout(
            height=560,
            annotations=[dict(
                x=0.5, y=0.5, xref="paper", yref="paper",
                text="City not initialized — click <b>Step</b> to build the grid.",
                showarrow=False, font=dict(size=18, color="#666"),
            )],
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        return fig

    grid = sim.grid

    # Zone background as colored rectangles + an icon per cell.
    icon_x: List[int] = []
    icon_y: List[int] = []
    icon_text: List[str] = []
    cell_hover: List[str] = []

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cell = grid[r][c]
            color = ZONE_FILL[cell.zone]
            fig.add_shape(
                type="rect",
                x0=c - 0.5, x1=c + 0.5,
                y0=r - 0.5, y1=r + 0.5,
                fillcolor=color,
                line=dict(color="#ffffff", width=1.5),
                layer="below",
            )
            icon_x.append(c)
            icon_y.append(r)
            icon_text.append(ZONE_ICON[cell.zone])
            tags = []
            if cell.is_hub: tags.append("HUB")
            if cell.is_charging: tags.append("CHG")
            if cell.is_medical_pickup: tags.append("MED")
            if cell.no_fly: tags.append("NO-FLY")
            tag_str = " | ".join(tags) if tags else "—"
            cell_hover.append(
                f"({r},{c}) {cell.zone.value}<br>"
                f"density={cell.density}  demand={cell.demand:.2f}<br>"
                f"flags: {tag_str}"
            )

    # Cell icons (transparent invisible scatter points carry the hover text).
    fig.add_trace(go.Scatter(
        x=icon_x, y=icon_y,
        mode="text",
        text=icon_text,
        textfont=dict(size=22),
        hovertext=cell_hover,
        hoverinfo="text",
        showlegend=False,
    ))

    # Hub markers — gold landing pad
    hub_xy = [(c.col, c.row) for row in grid for c in row if c.is_hub]
    if hub_xy:
        xs, ys = zip(*hub_xy)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            marker=dict(symbol="square", size=34, color="rgba(0,0,0,0)",
                        line=dict(color="#1a3da8", width=3)),
            text=["HUB"] * len(xs),
            textfont=dict(color="#1a3da8", size=11, family="Arial Black"),
            textposition="middle center",
            name="Drone hub",
            hoverinfo="text",
            hovertext=[f"Hub @ ({y},{x})" for x, y in zip(xs, ys)],
        ))

    # Charging pads
    chg_xy = [(c.col, c.row) for row in grid for c in row if c.is_charging]
    if chg_xy:
        xs, ys = zip(*chg_xy)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            marker=dict(symbol="diamond", size=24, color="rgba(0,0,0,0)",
                        line=dict(color="#0e7c66", width=3)),
            text=["⚡"] * len(xs),  # lightning bolt
            textfont=dict(size=14),
            textposition="middle center",
            name="Charging pad",
            hoverinfo="text",
            hovertext=[f"Charging pad @ ({y},{x})" for x, y in zip(xs, ys)],
        ))

    # Medical pickups
    med_xy = [(c.col, c.row) for row in grid for c in row if c.is_medical_pickup]
    if med_xy:
        xs, ys = zip(*med_xy)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            marker=dict(symbol="cross", size=24,
                        color="#c8243a", line=dict(color="white", width=2)),
            text=[""] * len(xs),
            name="Medical pickup",
            hoverinfo="text",
            hovertext=[f"Medical pickup @ ({y},{x})" for x, y in zip(xs, ys)],
        ))

    # No-fly cells (overlay an X)
    nofly_xy = [(c.col, c.row) for row in grid for c in row if c.no_fly]
    if nofly_xy:
        xs, ys = zip(*nofly_xy)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            marker=dict(symbol="x", size=36, color="#222",
                        line=dict(color="#222", width=4)),
            text=["NO-FLY"] * len(xs),
            textfont=dict(color="#222", size=10, family="Arial Black"),
            textposition="bottom center",
            name="No-fly cell",
            hoverinfo="text",
            hovertext=[f"No-fly cell @ ({y},{x})" for x, y in zip(xs, ys)],
        ))

    # Routes — full remaining path per active drone, color-coded.
    for idx, (drone_id, path) in enumerate(sim.active_routes()):
        if len(path) < 2:
            continue
        color = DRONE_PALETTE[idx % len(DRONE_PALETTE)]
        xs = [p[1] for p in path]
        ys = [p[0] for p in path]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=color, width=4, dash="dot"),
            opacity=0.55,
            name=f"Route {drone_id}",
            hoverinfo="skip",
        ))

    # Drone trails (where each drone has been).
    for idx, drone in enumerate(sim.drones):
        trail = sim.drone_trails.get(drone.drone_id, [])
        if len(trail) < 2:
            continue
        color = DRONE_PALETTE[idx % len(DRONE_PALETTE)]
        xs = [p[1] for p in trail]
        ys = [p[0] for p in trail]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=color, width=2),
            opacity=0.85,
            name=f"Trail {drone.drone_id}",
            hoverinfo="skip",
            showlegend=False,
        ))

    # Drone current positions — large triangle markers with status colors.
    # When several drones share a cell (typical at the home hub), spread them
    # in a small ring so each one is independently visible/hoverable.
    if sim.drones:
        from collections import defaultdict
        import math

        idx_of = {d.drone_id: i for i, d in enumerate(sim.drones)}
        groups: Dict[Tuple[int, int], list] = defaultdict(list)
        for d in sim.drones:
            groups[d.current_pos].append(d)

        for pos, group in groups.items():
            n = len(group)
            if n == 1:
                offsets = [(0.0, 0.0)]
            else:
                radius = 0.28
                offsets = [
                    (radius * math.cos(2 * math.pi * i / n - math.pi / 2),
                     radius * math.sin(2 * math.pi * i / n - math.pi / 2))
                    for i in range(n)
                ]
            for drone, (dx, dy) in zip(group, offsets):
                idx = idx_of[drone.drone_id]
                x = pos[1] + dx
                y = pos[0] + dy
                outline = STATUS_COLOR.get(drone.status, "#444")
                fill = DRONE_PALETTE[idx % len(DRONE_PALETTE)]
                delivery_id = drone.delivery.id if drone.delivery else "—"
                eta = (
                    f"{drone.delivery.predicted_eta_min:.0f} min"
                    if drone.delivery and drone.delivery.predicted_eta_min is not None
                    else "—"
                )
                anomaly_tag = ""
                if drone.anomaly_label and drone.anomaly_label != "Normal":
                    anomaly_tag = f"<br><b>{drone.anomaly_label}</b>"
                fig.add_trace(go.Scatter(
                    x=[x], y=[y], mode="markers+text",
                    marker=dict(
                        symbol="triangle-up",
                        size=22,
                        color=fill,
                        line=dict(color=outline, width=3),
                    ),
                    text=[drone.drone_id],
                    textposition="top center",
                    textfont=dict(color="#222", size=10, family="Arial Black"),
                    name=f"{drone.drone_id} ({drone.status})",
                    hoverinfo="text",
                    hovertext=(
                        f"<b>{drone.drone_id}</b> ({drone.spec.name})<br>"
                        f"Status: {drone.status}<br>"
                        f"Pos: ({pos[0]},{pos[1]})  Battery: {drone.battery:.0f}%<br>"
                        f"Delivery: {delivery_id}  ETA: {eta}"
                        f"{anomaly_tag}"
                    ),
                    showlegend=False,
                ))

    # Layout polish — match cell aspect, lock to integer grid, dark gridlines.
    fig.update_layout(
        height=620,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="#fdfcf7",
        showlegend=True,
        legend=dict(
            x=1.02, y=1.0, bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc", borderwidth=1,
        ),
    )
    fig.update_xaxes(
        range=[-0.5, GRID_SIZE - 0.5],
        showgrid=False, zeroline=False,
        tickmode="array", tickvals=list(range(GRID_SIZE)),
        side="top",
    )
    fig.update_yaxes(
        range=[GRID_SIZE - 0.5, -0.5],  # invert so row 0 is top
        showgrid=False, zeroline=False,
        scaleanchor="x", scaleratio=1,
        tickmode="array", tickvals=list(range(GRID_SIZE)),
    )
    return fig


def render_demand_heatmap(sim: Simulation) -> go.Figure:
    fig = go.Figure()
    if not sim.zone_demand:
        # Render cell.demand as a fallback before forecast runs.
        if sim.grid is None:
            return fig
        z = [[sim.grid[r][c].demand for c in range(GRID_SIZE)] for r in range(GRID_SIZE)]
    else:
        z = [
            [sim.zone_demand.get((r, c), 0.0) for c in range(GRID_SIZE)]
            for r in range(GRID_SIZE)
        ]
    fig.add_trace(go.Heatmap(
        z=z,
        x=list(range(GRID_SIZE)),
        y=list(range(GRID_SIZE)),
        colorscale="YlOrRd",
        colorbar=dict(title="demand"),
        hovertemplate="(%{y},%{x})  demand=%{z:.1f}<extra></extra>",
        zsmooth=False,
    ))
    # Constrain both axes to the data range; lock aspect so cells stay square
    # without ballooning the x-axis when the container is wide.
    fig.update_layout(
        height=560,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="#ffffff",
    )
    fig.update_xaxes(
        range=[-0.5, GRID_SIZE - 0.5],
        showgrid=False, zeroline=False,
        tickmode="array", tickvals=list(range(GRID_SIZE)),
        constrain="domain",
        side="top",
    )
    fig.update_yaxes(
        range=[GRID_SIZE - 0.5, -0.5],  # row 0 at top
        showgrid=False, zeroline=False,
        tickmode="array", tickvals=list(range(GRID_SIZE)),
        scaleanchor="x", scaleratio=1, constrain="domain",
    )
    return fig


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.title("\U0001F681  AeroNet Lite — Drone Delivery Simulation")
st.caption(
    f"Run seed: **{st.session_state.run_seed}** · "
    f"Step **{sim.current_step}** / {TOTAL_STEPS}"
)

# ---- Sidebar: controls ---- #
with st.sidebar:
    st.header("Controls")

    seed_input = st.number_input(
        "Seed", min_value=0, max_value=10_000_000,
        value=int(st.session_state.run_seed), step=1,
        help="Same seed → same grid + deliveries. Change for a different city.",
    )

    cols = st.columns(2)
    with cols[0]:
        if st.button("\U000021BB Restart", use_container_width=True):
            st.session_state.run_seed = int(seed_input)
            st.session_state.sim = init_simulation(int(seed_input))
            st.session_state.autoplay = False
            st.rerun()
    with cols[1]:
        if st.button("\U0001F3B2 New random", use_container_width=True):
            new_seed = int(time.time() * 1000) % 1_000_000
            st.session_state.run_seed = new_seed
            st.session_state.sim = init_simulation(new_seed)
            st.session_state.autoplay = False
            st.rerun()

    step_disabled = sim.finished
    cols2 = st.columns(2)
    with cols2[0]:
        if st.button("▶ Step", disabled=step_disabled, use_container_width=True):
            sim.step()
            st.rerun()
    with cols2[1]:
        if st.button("⏩ Run to end", disabled=step_disabled, use_container_width=True):
            while not sim.finished:
                sim.step()
            st.session_state.autoplay = False
            st.rerun()

    st.markdown("---")
    st.session_state.autoplay = st.toggle(
        "▶️ Auto-play",
        value=st.session_state.autoplay,
        disabled=sim.finished,
        help="Advance one step every N seconds.",
    )
    st.session_state.autoplay_delay = st.slider(
        "Step delay (s)", min_value=0.2, max_value=2.5,
        value=float(st.session_state.autoplay_delay), step=0.1,
    )

    st.markdown("---")
    st.subheader("Status")
    if sim.finished:
        st.success("Simulation complete")
    elif sim.current_step == 0:
        st.info("Click **Step** or **Auto-play** to begin")
    else:
        st.info(f"Running · step {sim.current_step}/{TOTAL_STEPS}")


# ---- KPI bar ---- #
completed, delayed, failed, in_progress = sim.delivery_summary()
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Step", f"{sim.current_step}/{TOTAL_STEPS}")
k2.metric("Completed", completed)
k3.metric("In progress", in_progress)
k4.metric("Delayed", delayed)
k5.metric("Failed", failed)
fleet_label = "—"
if sim.selection:
    fleet_label = f"{sim.selection.light_count}L · {sim.selection.heavy_count}H"
k6.metric("Fleet", fleet_label)

# ---- Main map ---- #
st.subheader("City map")
st.plotly_chart(render_city_map(sim), use_container_width=True, config={"displayModeBar": False})

# ---- Tabs ---- #
tab_events, tab_drones, tab_deliveries, tab_demand, tab_models, tab_layout = st.tabs(
    ["\U0001F4E2 Event log", "\U0001F681 Drones", "\U0001F4E6 Deliveries",
     "\U0001F525 Demand heatmap", "\U0001F4CA Model metrics", "✅ Layout"]
)

with tab_events:
    if not sim.events:
        st.info("Events will appear here as the simulation runs.")
    else:
        # Show most recent first.
        df = pd.DataFrame(sim.events, columns=["Step", "Event"])
        st.dataframe(
            df.iloc[::-1].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
            height=420,
        )

with tab_drones:
    if not sim.drones:
        st.info("Drones appear after step 2 (fleet selection).")
    else:
        rows = []
        for d in sim.drones:
            delivery_id = d.delivery.id if d.delivery else "—"
            eta = "—"
            if d.delivery and d.delivery.predicted_eta_min is not None:
                eta = f"{d.delivery.predicted_eta_min:.0f} min"
            rows.append({
                "Drone": d.drone_id,
                "Type": d.spec.name,
                "Hub": str(d.home_hub),
                "Position": str(d.current_pos),
                "Status": d.status,
                "Battery": f"{d.battery:.0f}%",
                "Delivery": delivery_id,
                "ETA": eta,
                "Anomaly": d.anomaly_label or "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab_deliveries:
    if not sim.deliveries:
        st.info("Deliveries appear after step 4.")
    else:
        rows = []
        for d in sim.deliveries:
            rows.append({
                "ID": d.id,
                "Pickup": str(d.pickup),
                "Drop-off": str(d.dropoff),
                "Drone": d.assigned_drone_id or "—",
                "Status": d.status,
                "ETA (min)": (
                    f"{d.predicted_eta_min:.0f}"
                    if d.predicted_eta_min is not None else "—"
                ),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab_demand:
    if not sim.zone_demand:
        st.info("Forecast runs at step 15. Showing static cell demand for now.")
    st.plotly_chart(render_demand_heatmap(sim), use_container_width=True)

with tab_models:
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**Demand model** (Bike Sharing)")
        if sim.demand_model:
            st.metric("MAE",  f"{sim.demand_model.mae:.2f}")
            st.metric("RMSE", f"{sim.demand_model.rmse:.2f}")
        else:
            st.write("Not trained yet.")
    with cols[1]:
        st.markdown("**Anomaly model** (synthetic)")
        if sim.anomaly_model:
            st.metric("Accuracy", f"{sim.anomaly_model.accuracy:.3f}")
            st.write("Confusion matrix:")
            cm = pd.DataFrame(
                sim.anomaly_model.confusion,
                index=sim.anomaly_model.class_names,
                columns=sim.anomaly_model.class_names,
            )
            st.dataframe(cm)
        else:
            st.write("Not trained yet.")
    with cols[2]:
        st.markdown("**Delivery-time** (Amazon)")
        if sim.eta_model:
            st.metric("MAE",  f"{sim.eta_model.mae:.2f} min")
            st.metric("RMSE", f"{sim.eta_model.rmse:.2f} min")
        else:
            st.write("Optional dataset missing — model not trained.")

with tab_layout:
    if sim.layout_report is None:
        st.info("Layout validation runs at step 1.")
    else:
        report = sim.layout_report
        if report.valid:
            st.success(f"All {len(report.passed)} CSP rules pass: " + ", ".join(report.passed))
        else:
            st.error(f"{len(report.failed)} violation(s) detected")
            for v in report.failed:
                st.write(f"- **{v.rule_id}** at {v.cell}: {v.message}  \n  *Suggested fix:* {v.suggestion}")
        if report.passed:
            st.write("Rules passed: " + ", ".join(report.passed))


# --------------------------------------------------------------------------- #
# Auto-play tick
# --------------------------------------------------------------------------- #
if st.session_state.autoplay and not sim.finished:
    time.sleep(st.session_state.autoplay_delay)
    sim.step()
    st.rerun()
