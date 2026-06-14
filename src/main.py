"""AeroNet Lite — 20-step simulation orchestrator.

Run with:  python -m src.main           (random seed each run)
           python -m src.main --seed 11 (reproduce a specific run)
"""
from __future__ import annotations

import argparse
import os
import time
from typing import List, Tuple

from .astar_planner import Delivery
from .delivery_simulator import (
    DroneState,
    activate_no_fly,
    assign_delivery,
    detect_anomaly,
    drone_states_from_fleet,
    force_return_to_hub,
    generate_deliveries,
    step_drone,
    summarize,
    synthesize_telemetry,
)
from .fleet_selector import (
    assign_drones_to_hubs,
    print_fleet,
    select_fleet_brute_force,
)
from .grid_model import (
    build_random_grid,
    build_sample_grid,
    enrich_grid_with_population,
    load_population_buckets,
)
from .layout_validator import print_report, validate_layout
from .ml_pipeline import (
    predict_zone_demand,
    train_anomaly_model,
    train_delivery_time_model,
    train_demand_model,
)
from .visualization import (
    plot_demand_heatmap,
    plot_event_timeline,
    plot_route_map,
    plot_zone_map,
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "report", "figures")


def banner(text: str) -> None:
    print()
    print("=" * 70)
    print(f"  {text}")
    print("=" * 70)


def log_step(step: int, message: str, events: List[Tuple[int, str]]) -> None:
    line = f"Step {step:>2}: {message}"
    print(line)
    events.append((step, message))


def active_drones(drones: List[DroneState]) -> List[DroneState]:
    return [
        d
        for d in drones
        if d.status in ("to_pickup", "to_dropoff", "returning")
    ]


def run_movement(
    step: int,
    drones: List[DroneState],
    events: List[Tuple[int, str]],
    ticks: int = 2,
) -> None:
    """Advance every active drone by `ticks` cells. Log meaningful events;
    movement positions are summarized once at the end of the step.
    """
    for _ in range(ticks):
        for d in drones:
            msg = step_drone(d, grid=None)
            if msg:
                log_step(step, msg, events)
    moved = [
        f"{d.drone_id}@{d.current_pos}"
        for d in drones
        if d.status in ("to_pickup", "to_dropoff", "returning")
    ]
    if moved:
        log_step(step, "Drones moved: " + ", ".join(moved), events)


def main(run_seed: int) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    events: List[Tuple[int, str]] = []

    banner("AeroNet Lite — 20-Step Simulation")
    print(f"[run] seed = {run_seed}  (use --seed {run_seed} to reproduce this run)")

    # --- Steps 1-3: init, validate, select fleet -----------------------------
    grid = build_random_grid(seed=run_seed)
    buckets = load_population_buckets()
    if enrich_grid_with_population(grid, buckets, seed=run_seed):
        print("[grid] enriched cell densities from real US city population data")
    plot_zone_map(grid, save_path=os.path.join(FIG_DIR, "zone_map.png"))

    report = validate_layout(grid)
    if report.valid:
        log_step(1, "Layout validation passed.", events)
    else:
        log_step(1, f"Layout validation failed ({len(report.failed)} violations).", events)
        print_report(report)

    log_step(2, "Selecting fleet under budget 10000...", events)
    selection = select_fleet_brute_force(grid, budget=10000)
    assign_drones_to_hubs(selection, grid)
    log_step(
        3,
        f"Fleet selected: {selection.light_count} light drones, "
        f"{selection.heavy_count} heavy drones (cost {selection.total_cost}, "
        f"score {selection.score:.2f}).",
        events,
    )
    print_fleet(selection)

    drones = drone_states_from_fleet(selection.drones)

    # --- Steps 4-6: generate deliveries, plan routes -------------------------
    log_step(4, "Training ML models and generating deliveries...", events)
    demand_model = train_demand_model()
    anomaly_model = train_anomaly_model()
    eta_model = train_delivery_time_model()

    deliveries = generate_deliveries(grid, n=4, seed=run_seed)
    log_step(4, f"Generated {len(deliveries)} deliveries.", events)

    for delivery in deliveries:
        msg = assign_delivery(delivery, drones, grid, eta_model=eta_model)
        log_step(5, msg or f"Delivery {delivery.id} unassigned", events)

    # Snapshot initial routes before any disruption (each drone's first segment).
    initial_routes = [d.segments[0] for d in drones if d.segments]
    plot_route_map(
        grid,
        initial_routes,
        drone_labels=[d.drone_id for d in drones if d.segments],
        save_path=os.path.join(FIG_DIR, "routes_initial.png"),
    )
    log_step(6, "Initial routes plotted to report/figures/routes_initial.png", events)

    # --- Steps 7-10: move drones --------------------------------------------
    for step in range(7, 11):
        run_movement(step, drones, events)

    # --- Step 11: activate a no-fly cell ------------------------------------
    # Pick a cell that's actually a few steps ahead on some drone's route so
    # the disruption forces a real A* reroute rather than being a no-op.
    no_fly_pos = (3, 4)
    for d in drones:
        if d.status not in ("to_pickup", "to_dropoff", "returning") or not d.segments:
            continue
        seg = d.segments[d.seg_idx]
        ahead = seg[d.pos_in_seg + 2 : d.pos_in_seg + 6]
        for r, c in ahead:
            cell = grid[r][c]
            if not cell.no_fly and not cell.is_hub and not cell.is_charging:
                no_fly_pos = (r, c)
                break
        if no_fly_pos != (3, 4):
            break

    nf_events = activate_no_fly(grid, no_fly_pos, drones)
    for msg in nf_events:
        log_step(11, msg, events)

    # --- Steps 12-14: continue moving, reroutes already applied -------------
    for step in range(12, 15):
        run_movement(step, drones, events)

    # --- Steps 15-17: demand forecast, optionally add a delivery ------------
    log_step(15, "Running demand forecast across grid...", events)
    zone_demand = predict_zone_demand(grid, demand_model)
    plot_demand_heatmap(
        grid,
        demand_dict=zone_demand,
        save_path=os.path.join(FIG_DIR, "demand_heatmap.png"),
    )
    top_cells = sorted(zone_demand.items(), key=lambda kv: kv[1], reverse=True)[:3]
    log_step(
        16,
        "Top forecast cells: " + ", ".join(f"{p}={v:.1f}" for p, v in top_cells),
        events,
    )

    # If any drone idle, add a delivery to a high-demand cell.
    if any(d.status == "idle" for d in drones):
        new_pickup = (4, 2)  # medical pickup
        new_dropoff = top_cells[0][0]
        if new_dropoff != new_pickup:
            extra = Delivery(
                id=len(deliveries) + 1,
                pickup=new_pickup,
                dropoff=new_dropoff,
            )
            deliveries.append(extra)
            msg = assign_delivery(extra, drones, grid, eta_model=eta_model)
            log_step(17, msg or f"Extra delivery {extra.id} unassigned", events)
        else:
            log_step(17, "No new delivery added (top-demand cell is the pickup).", events)
    else:
        log_step(17, "No idle drones available for additional delivery.", events)

    # --- Step 18: inject + detect an anomaly --------------------------------
    target = next((d for d in drones if d.status != "idle"), drones[0])
    telemetry = synthesize_telemetry(target, force_anomaly=True, seed=run_seed)
    label, anomaly_msg = detect_anomaly(target, anomaly_model, telemetry)
    log_step(18, anomaly_msg, events)

    # --- Step 19: handle anomaly --------------------------------------------
    if label != "Normal":
        msg = force_return_to_hub(target, grid)
        log_step(19, msg, events)
    else:
        log_step(19, "No anomaly action required.", events)

    # Final movement ticks to let drones resolve (steps 19-20).
    run_movement(19, drones, events, ticks=2)
    run_movement(20, drones, events, ticks=2)

    # --- Step 20: summary ---------------------------------------------------
    completed, delayed, failed, in_progress = summarize(deliveries)
    log_step(
        20,
        f"Simulation complete. {completed} completed, {delayed} delayed, "
        f"{failed} failed, {in_progress} still in progress.",
        events,
    )

    # Final figures.
    final_routes = [d.segments[d.seg_idx] for d in drones if d.segments]
    if final_routes:
        plot_route_map(
            grid,
            final_routes,
            drone_labels=[d.drone_id for d in drones if d.segments],
            save_path=os.path.join(FIG_DIR, "routes_final.png"),
        )
    plot_event_timeline(
        events,
        save_path=os.path.join(FIG_DIR, "event_timeline.png"),
    )

    banner("Event log saved to report/figures/event_timeline.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AeroNet Lite simulator")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run seed (default: derived from current time, so each run differs)",
    )
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else int(time.time()) % 1_000_000
    main(seed)
