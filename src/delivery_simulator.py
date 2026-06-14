"""Module 4 + integration: drone state machine, delivery generation,
no-fly activation, rerouting, anomaly handling.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .astar_planner import Delivery, PathResult, astar, plan_delivery_route
from .fleet_selector import Drone, DroneSpec
from .grid_model import GRID_SIZE, Cell, Grid, Zone, manhattan
from .ml_pipeline import AnomalyModel, DeliveryTimeModel, predict_delivery_eta

# One Manhattan grid cell ~ this many real-world kilometres. Used to feed the
# Amazon Delivery model with a distance value comparable to its training set.
KM_PER_CELL = 0.5

Pos = Tuple[int, int]

PHASE_TO_PICKUP = 0
PHASE_TO_DROPOFF = 1
PHASE_RETURNING = 2

PHASE_LABELS = {
    PHASE_TO_PICKUP: "to_pickup",
    PHASE_TO_DROPOFF: "to_dropoff",
    PHASE_RETURNING: "returning",
}


@dataclass
class DroneState:
    drone_id: str
    spec: DroneSpec
    home_hub: Pos
    current_pos: Pos
    battery: float = 100.0
    status: str = "idle"  # idle | to_pickup | to_dropoff | returning | delayed | failed | anomaly | done
    delivery: Optional[Delivery] = None
    segments: List[List[Pos]] = field(default_factory=list)  # 3 segments per delivery
    seg_idx: int = 0
    pos_in_seg: int = 0
    anomaly_label: Optional[str] = None
    telemetry: dict = field(default_factory=dict)


def drone_states_from_fleet(drones: List[Drone]) -> List[DroneState]:
    return [
        DroneState(
            drone_id=d.id,
            spec=d.spec,
            home_hub=d.home_hub,
            current_pos=d.home_hub,
        )
        for d in drones
    ]


def generate_deliveries(grid: Grid, n: int = 5, seed: int = 7) -> List[Delivery]:
    """Generate plausible deliveries: pickups in residential / medical cells,
    drop-offs in residential / commercial / hospital / industrial cells.
    """
    rng = random.Random(seed)
    pickup_cells = [
        (c.row, c.col)
        for row in grid
        for c in row
        if c.zone == Zone.RESIDENTIAL or c.is_medical_pickup
    ]
    dropoff_cells = [
        (c.row, c.col)
        for row in grid
        for c in row
        if c.zone in (Zone.RESIDENTIAL, Zone.COMMERCIAL, Zone.HOSPITAL, Zone.INDUSTRIAL)
        and not c.no_fly
    ]
    deliveries: List[Delivery] = []
    for i in range(1, n + 1):
        pickup = rng.choice(pickup_cells)
        dropoff = rng.choice([d for d in dropoff_cells if d != pickup])
        deliveries.append(Delivery(id=i, pickup=pickup, dropoff=dropoff))
    return deliveries


def _nearest_idle_drone(delivery: Delivery, drones: List[DroneState]) -> Optional[DroneState]:
    idle = [d for d in drones if d.status == "idle"]
    if not idle:
        return None
    return min(idle, key=lambda d: manhattan(d.current_pos, delivery.pickup))


def assign_delivery(
    delivery: Delivery,
    drones: List[DroneState],
    grid: Grid,
    eta_model: Optional[DeliveryTimeModel] = None,
) -> Optional[str]:
    """Assign delivery to nearest idle drone, plan all 3 segments. Returns event message.

    If `eta_model` is provided, also annotates the delivery with a predicted
    ETA (minutes) based on the Amazon Delivery regressor.
    """
    drone = _nearest_idle_drone(delivery, drones)
    if drone is None:
        delivery.status = "unassigned"
        return f"Delivery {delivery.id} could not be assigned (no idle drones)"

    seg1, seg2, seg3 = plan_delivery_route(drone.home_hub, delivery, grid)
    if not (seg1.success and seg2.success and seg3.success):
        delivery.status = "unroutable"
        return f"Delivery {delivery.id} could not be routed from hub {drone.home_hub}"

    drone.delivery = delivery
    drone.segments = [seg1.path, seg2.path, seg3.path]
    drone.seg_idx = PHASE_TO_PICKUP
    drone.pos_in_seg = 0
    drone.status = PHASE_LABELS[PHASE_TO_PICKUP]
    delivery.assigned_drone_id = drone.drone_id
    delivery.status = "in_progress"

    if eta_model is not None:
        # Total trip = hub -> pickup -> dropoff -> hub. Use that distance for ETA.
        total_cells = (
            manhattan(drone.home_hub, delivery.pickup)
            + manhattan(delivery.pickup, delivery.dropoff)
            + manhattan(delivery.dropoff, drone.home_hub)
        )
        delivery.predicted_eta_min = predict_delivery_eta(
            eta_model,
            distance_km=total_cells * KM_PER_CELL,
        )
        return (
            f"Delivery {delivery.id} assigned to Drone {drone.drone_id} "
            f"(ETA {delivery.predicted_eta_min:.0f} min)"
        )
    return f"Delivery {delivery.id} assigned to Drone {drone.drone_id}"


def _finish_delivery(drone: DroneState) -> str:
    delivery = drone.delivery
    delivery_id = delivery.id if delivery else None
    terminal = delivery is not None and delivery.status in ("aborted", "delayed", "failed")
    if delivery is not None and not terminal:
        delivery.status = "completed"
    drone.status = "idle"
    drone.delivery = None
    drone.segments = []
    drone.seg_idx = 0
    if delivery_id is None:
        return f"Drone {drone.drone_id} returned to hub"
    if terminal:
        return f"Drone {drone.drone_id} returned to hub (delivery {delivery_id} {delivery.status})"
    return f"Drone {drone.drone_id} completed delivery {delivery_id}"


def step_drone(drone: DroneState, grid: Grid) -> Optional[str]:
    """Advance a drone one cell along its current segment. Returns event message or None."""
    if drone.status in ("idle", "delayed", "failed", "done", "anomaly"):
        return None

    seg = drone.segments[drone.seg_idx]
    if drone.pos_in_seg + 1 < len(seg):
        drone.pos_in_seg += 1
        drone.current_pos = seg[drone.pos_in_seg]
        drone.battery = max(0.0, drone.battery - 1.0)
        # If we just landed on the final cell of the final segment, finish now
        # rather than waiting for the next tick to detect end-of-segment.
        if drone.pos_in_seg == len(seg) - 1 and drone.seg_idx == len(drone.segments) - 1:
            return _finish_delivery(drone)
        return None

    # End of intermediate segment — advance phase
    drone.seg_idx += 1
    drone.pos_in_seg = 0
    if drone.seg_idx >= len(drone.segments):
        return _finish_delivery(drone)

    drone.status = PHASE_LABELS.get(drone.seg_idx, "returning")
    drone.current_pos = drone.segments[drone.seg_idx][0]
    return f"Drone {drone.drone_id} entered phase {drone.status}"


def activate_no_fly(grid: Grid, pos: Pos, drones: List[DroneState]) -> List[str]:
    """Flip a cell to no-fly. Reroute any drone whose remaining path crosses it
    in the current OR any future segment (a delivery has up to 3 legs).
    """
    r, c = pos
    grid[r][c].no_fly = True
    events: List[str] = [f"No-fly cell activated at ({r}, {c})"]

    active = (
        PHASE_LABELS[PHASE_TO_PICKUP],
        PHASE_LABELS[PHASE_TO_DROPOFF],
        PHASE_LABELS[PHASE_RETURNING],
    )
    for d in drones:
        if d.status not in active or not d.segments:
            continue
        affected = pos in d.segments[d.seg_idx][d.pos_in_seg:]
        if not affected:
            for i in range(d.seg_idx + 1, len(d.segments)):
                if pos in d.segments[i]:
                    affected = True
                    break
        if affected:
            events.append(reroute_drone(d, grid))
    return events


def reroute_drone(drone: DroneState, grid: Grid) -> str:
    """Re-run A* from current position through all remaining goals.

    Each leg's GOAL (pickup, dropoff, hub) is preserved; only the path between
    them is replanned against the updated no-fly state.
    """
    if drone.seg_idx >= len(drone.segments) or not drone.segments:
        return f"Drone {drone.drone_id} has nothing to reroute"

    goals = [drone.segments[i][-1] for i in range(drone.seg_idx, len(drone.segments))]
    new_segments = list(drone.segments[: drone.seg_idx])
    start = drone.current_pos
    for goal in goals:
        result: PathResult = astar(start, goal, grid)
        if not result.success:
            drone.status = "delayed"
            if drone.delivery:
                drone.delivery.status = "delayed"
            return f"Drone {drone.drone_id} cannot reach destination safely (delivery delayed)"
        new_segments.append(result.path)
        start = goal
    drone.segments = new_segments
    drone.pos_in_seg = 0
    return f"Drone {drone.drone_id} rerouted using A*"


def synthesize_telemetry(drone: DroneState, force_anomaly: bool = False, seed: int = 1) -> dict:
    """Build a telemetry sample for the anomaly classifier. Optionally inject an anomaly."""
    rng = random.Random(seed + hash(drone.drone_id) % 1000)
    if force_anomaly:
        kind = rng.choice(["battery", "route", "sensor"])
        if kind == "battery":
            return {
                "battery_drop": rng.uniform(8, 12),
                "speed": rng.uniform(8, 12),
                "route_deviation": rng.uniform(0, 1),
                "altitude_change": rng.uniform(0, 2),
                "speed_change": rng.uniform(0, 2),
            }
        if kind == "route":
            return {
                "battery_drop": rng.uniform(0.5, 2),
                "speed": rng.uniform(8, 12),
                "route_deviation": rng.uniform(8, 12),
                "altitude_change": rng.uniform(0, 2),
                "speed_change": rng.uniform(0, 2),
            }
        return {
            "battery_drop": rng.uniform(0.5, 2),
            "speed": rng.uniform(8, 12),
            "route_deviation": rng.uniform(0, 1),
            "altitude_change": rng.uniform(8, 12),
            "speed_change": rng.uniform(8, 12),
        }
    return {
        "battery_drop": rng.uniform(0.3, 1.5),
        "speed": rng.uniform(8, 12),
        "route_deviation": rng.uniform(0, 1),
        "altitude_change": rng.uniform(0, 1.5),
        "speed_change": rng.uniform(0, 1.5),
    }


def detect_anomaly(drone: DroneState, anomaly_model: AnomalyModel, telemetry: dict) -> Tuple[str, str]:
    """Run telemetry through the classifier. Update drone state if anomalous.
    Returns (label, event_message).
    """
    label = anomaly_model.classify(telemetry)
    drone.telemetry = telemetry
    drone.anomaly_label = label
    if label == "Normal":
        return label, f"Drone {drone.drone_id} telemetry normal"
    return label, f"{label} detected for Drone {drone.drone_id}"


def force_return_to_hub(drone: DroneState, grid: Grid) -> str:
    """Abort current delivery, reroute drone to home hub."""
    result = astar(drone.current_pos, drone.home_hub, grid)
    if not result.success:
        drone.status = "failed"
        if drone.delivery:
            drone.delivery.status = "failed"
        return f"Drone {drone.drone_id} stranded — cannot reach hub"
    drone.segments = [result.path]
    drone.seg_idx = 0
    drone.pos_in_seg = 0
    drone.status = PHASE_LABELS[PHASE_RETURNING]
    if drone.delivery:
        drone.delivery.status = "aborted"
    return f"Drone {drone.drone_id} forced to return to hub {drone.home_hub}"


def summarize(deliveries: List[Delivery]) -> Tuple[int, int, int, int]:
    completed = sum(1 for d in deliveries if d.status == "completed")
    delayed = sum(1 for d in deliveries if d.status in ("delayed", "aborted"))
    failed = sum(1 for d in deliveries if d.status in ("failed", "unroutable", "unassigned"))
    in_progress = sum(1 for d in deliveries if d.status == "in_progress")
    return completed, delayed, failed, in_progress
