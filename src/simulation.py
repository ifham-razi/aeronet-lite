"""Stepwise simulation orchestrator.

Wraps the 20-step scenario from `main.py` so the Streamlit dashboard can
advance one step at a time, restart, or replay a seeded run.

Each public `step()` call advances the simulation by one logical step
(matching the spec's 20-step plan), returning the events emitted on that
step. State is exposed as plain attributes so the UI layer can read it
freely without reaching into private internals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .astar_planner import Delivery
from .delivery_simulator import (
    activate_no_fly,
    assign_delivery,
    detect_anomaly,
    drone_states_from_fleet,
    DroneState,
    force_return_to_hub,
    generate_deliveries,
    step_drone,
    summarize,
    synthesize_telemetry,
)
from .fleet_selector import (
    FleetSelection,
    assign_drones_to_hubs,
    select_fleet_brute_force,
)
from .grid_model import (
    Grid,
    build_random_grid,
    enrich_grid_with_population,
    load_population_buckets,
)
from .layout_validator import ValidationReport, validate_layout
from .ml_pipeline import (
    AnomalyModel,
    DemandModel,
    DeliveryTimeModel,
    predict_zone_demand,
    train_anomaly_model,
    train_delivery_time_model,
    train_demand_model,
)


TOTAL_STEPS = 20
MOVEMENT_TICKS_PER_STEP = 2


@dataclass
class Simulation:
    seed: int
    movement_ticks: int = MOVEMENT_TICKS_PER_STEP

    # Populated as steps run.
    grid: Optional[Grid] = None
    layout_report: Optional[ValidationReport] = None
    selection: Optional[FleetSelection] = None
    drones: List[DroneState] = field(default_factory=list)
    deliveries: List[Delivery] = field(default_factory=list)
    demand_model: Optional[DemandModel] = None
    anomaly_model: Optional[AnomalyModel] = None
    eta_model: Optional[DeliveryTimeModel] = None
    zone_demand: Dict[Tuple[int, int], float] = field(default_factory=dict)
    no_fly_history: List[Tuple[int, int]] = field(default_factory=list)
    anomaly_target_id: Optional[str] = None
    anomaly_label: Optional[str] = None
    events: List[Tuple[int, str]] = field(default_factory=list)
    drone_trails: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)

    current_step: int = 0
    finished: bool = False

    # Optional injection: pass pre-trained models so the UI can warm-start
    # via Streamlit's @cache_resource and avoid retraining on every restart.
    preloaded_demand: Optional[DemandModel] = None
    preloaded_anomaly: Optional[AnomalyModel] = None
    preloaded_eta: Optional[DeliveryTimeModel] = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def step(self) -> List[str]:
        """Advance one logical step. Returns the new event strings."""
        if self.finished:
            return []
        self.current_step += 1
        s = self.current_step
        before = len(self.events)

        if s == 1:
            self._step_init_and_validate()
        elif s == 2:
            self._step_select_fleet()
        elif s == 3:
            self._step_announce_fleet()
        elif s == 4:
            self._step_train_models_and_generate()
        elif s == 5:
            self._step_assign_deliveries()
        elif s == 6:
            self._step_initial_routes_snapshot()
        elif 7 <= s <= 10:
            self._step_movement(s)
        elif s == 11:
            self._step_activate_no_fly()
        elif 12 <= s <= 14:
            self._step_movement(s)
        elif s == 15:
            self._step_demand_forecast()
        elif s == 16:
            self._step_top_demand()
        elif s == 17:
            self._step_extra_delivery()
        elif s == 18:
            self._step_inject_anomaly()
        elif s == 19:
            self._step_handle_anomaly_and_move()
        elif s == 20:
            self._step_final_movement_and_summary()

        self._record_drone_positions()

        if s >= TOTAL_STEPS:
            self.finished = True
        return [msg for step, msg in self.events[before:]]

    def reset(self, seed: Optional[int] = None) -> None:
        keep_demand = self.preloaded_demand
        keep_anomaly = self.preloaded_anomaly
        keep_eta = self.preloaded_eta
        new_seed = seed if seed is not None else self.seed
        self.__init__(seed=new_seed)  # type: ignore[misc]
        self.preloaded_demand = keep_demand
        self.preloaded_anomaly = keep_anomaly
        self.preloaded_eta = keep_eta

    # ------------------------------------------------------------------ #
    # Per-step implementations
    # ------------------------------------------------------------------ #
    def _step_init_and_validate(self) -> None:
        self.grid = build_random_grid(seed=self.seed)
        buckets = load_population_buckets()
        if enrich_grid_with_population(self.grid, buckets, seed=self.seed):
            self._log(1, "Cell densities enriched from real US city population data.")
        report = validate_layout(self.grid)
        self.layout_report = report
        if report.valid:
            self._log(1, "Layout validation passed.")
        else:
            self._log(1, f"Layout validation failed ({len(report.failed)} violations).")

    def _step_select_fleet(self) -> None:
        self._log(2, "Selecting fleet under budget 10000...")
        selection = select_fleet_brute_force(self.grid, budget=10000)
        assign_drones_to_hubs(selection, self.grid)
        self.selection = selection
        self.drones = drone_states_from_fleet(selection.drones)
        for d in self.drones:
            self.drone_trails[d.drone_id] = [d.current_pos]

    def _step_announce_fleet(self) -> None:
        s = self.selection
        self._log(
            3,
            f"Fleet selected: {s.light_count} light + {s.heavy_count} heavy drones "
            f"(cost {s.total_cost}, score {s.score:.2f}).",
        )

    def _step_train_models_and_generate(self) -> None:
        self._log(4, "Training ML models and generating deliveries...")
        self.demand_model = self.preloaded_demand or train_demand_model()
        self.anomaly_model = self.preloaded_anomaly or train_anomaly_model()
        self.eta_model = self.preloaded_eta or train_delivery_time_model()
        self.deliveries = generate_deliveries(self.grid, n=4, seed=self.seed)
        self._log(4, f"Generated {len(self.deliveries)} deliveries.")

    def _step_assign_deliveries(self) -> None:
        for delivery in self.deliveries:
            msg = assign_delivery(delivery, self.drones, self.grid, eta_model=self.eta_model)
            self._log(5, msg or f"Delivery {delivery.id} unassigned")

    def _step_initial_routes_snapshot(self) -> None:
        self._log(6, "Initial routes locked in for the active fleet.")

    def _step_movement(self, step: int) -> None:
        self._tick_drones(step, self.movement_ticks)

    def _step_activate_no_fly(self) -> None:
        no_fly_pos = self._pick_no_fly_target()
        self.no_fly_history.append(no_fly_pos)
        nf_events = activate_no_fly(self.grid, no_fly_pos, self.drones)
        for msg in nf_events:
            self._log(11, msg)

    def _step_demand_forecast(self) -> None:
        self._log(15, "Running demand forecast across grid...")
        self.zone_demand = predict_zone_demand(self.grid, self.demand_model)

    def _step_top_demand(self) -> None:
        top = sorted(self.zone_demand.items(), key=lambda kv: kv[1], reverse=True)[:3]
        msg = "Top forecast cells: " + ", ".join(f"{p}={v:.0f}" for p, v in top)
        self._log(16, msg)

    def _step_extra_delivery(self) -> None:
        idle = [d for d in self.drones if d.status == "idle"]
        if not idle:
            self._log(17, "No idle drones available for an additional delivery.")
            return
        # Pick top-demand cell as the new dropoff; medical pickup as source.
        top = max(self.zone_demand.items(), key=lambda kv: kv[1])[0]
        from .grid_model import Zone
        # Find a medical pickup cell or fall back to a residential one.
        pickup = None
        for row in self.grid:
            for cell in row:
                if cell.is_medical_pickup:
                    pickup = (cell.row, cell.col)
                    break
            if pickup:
                break
        if pickup is None or pickup == top:
            self._log(17, "No suitable extra delivery (pickup unavailable).")
            return
        extra = Delivery(id=len(self.deliveries) + 1, pickup=pickup, dropoff=top)
        self.deliveries.append(extra)
        msg = assign_delivery(extra, self.drones, self.grid, eta_model=self.eta_model)
        self._log(17, msg or f"Extra delivery {extra.id} unassigned")

    def _step_inject_anomaly(self) -> None:
        target = next((d for d in self.drones if d.status != "idle"), self.drones[0] if self.drones else None)
        if target is None:
            self._log(18, "No drones in fleet — anomaly step skipped.")
            return
        telemetry = synthesize_telemetry(target, force_anomaly=True, seed=self.seed)
        label, msg = detect_anomaly(target, self.anomaly_model, telemetry)
        self.anomaly_target_id = target.drone_id
        self.anomaly_label = label
        self._log(18, msg)

    def _step_handle_anomaly_and_move(self) -> None:
        target = next(
            (d for d in self.drones if d.drone_id == self.anomaly_target_id),
            None,
        )
        if target and self.anomaly_label and self.anomaly_label != "Normal":
            msg = force_return_to_hub(target, self.grid)
            self._log(19, msg)
        else:
            self._log(19, "No anomaly action required.")
        self._tick_drones(19, self.movement_ticks)

    def _step_final_movement_and_summary(self) -> None:
        self._tick_drones(20, self.movement_ticks)
        completed, delayed, failed, in_progress = summarize(self.deliveries)
        self._log(
            20,
            f"Simulation complete. {completed} completed, {delayed} delayed, "
            f"{failed} failed, {in_progress} still in progress.",
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _log(self, step: int, message: str) -> None:
        self.events.append((step, message))

    def _tick_drones(self, step: int, ticks: int) -> None:
        for _ in range(ticks):
            for d in self.drones:
                msg = step_drone(d, self.grid)
                if msg:
                    self._log(step, msg)

    def _pick_no_fly_target(self) -> Tuple[int, int]:
        """Pick a cell a few steps ahead on some active drone's path."""
        for d in self.drones:
            if d.status not in ("to_pickup", "to_dropoff", "returning") or not d.segments:
                continue
            seg = d.segments[d.seg_idx]
            ahead = seg[d.pos_in_seg + 2 : d.pos_in_seg + 6]
            for r, c in ahead:
                cell = self.grid[r][c]
                if not cell.no_fly and not cell.is_hub and not cell.is_charging:
                    return (r, c)
        # Fallback: any non-special open cell.
        for r in range(len(self.grid)):
            for c in range(len(self.grid[0])):
                cell = self.grid[r][c]
                if not cell.no_fly and not cell.is_hub and not cell.is_charging:
                    return (r, c)
        return (0, 0)

    def _record_drone_positions(self) -> None:
        for d in self.drones:
            trail = self.drone_trails.setdefault(d.drone_id, [])
            if not trail or trail[-1] != d.current_pos:
                trail.append(d.current_pos)

    # ------------------------------------------------------------------ #
    # Read-only views useful to the UI
    # ------------------------------------------------------------------ #
    def delivery_summary(self) -> Tuple[int, int, int, int]:
        return summarize(self.deliveries)

    def active_routes(self) -> List[Tuple[str, List[Tuple[int, int]]]]:
        out: List[Tuple[str, List[Tuple[int, int]]]] = []
        for d in self.drones:
            if not d.segments:
                continue
            segs = d.segments[d.seg_idx :]
            path = [d.current_pos]
            for seg in segs:
                path.extend([p for p in seg if not path or p != path[-1]])
            out.append((d.drone_id, path))
        return out
