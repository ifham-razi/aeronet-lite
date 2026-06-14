"""A* path planner for AeroNet Lite (Module 3).

Plans cheapest 4-connected paths over the shared Grid contract, with a
discount for entering Commercial cells and hard avoidance of no-fly cells.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .grid_model import (
    Grid,
    Zone,
    build_sample_grid,
    get_neighbors,
    manhattan,
)

Pos = Tuple[int, int]


@dataclass
class PathResult:
    path: List[Pos]
    cost: float
    success: bool
    message: str


@dataclass
class Delivery:
    id: int
    pickup: Pos
    dropoff: Pos
    assigned_drone_id: Optional[str] = None
    status: str = "pending"
    predicted_eta_min: Optional[float] = None


def move_cost(grid: Grid, to_row: int, to_col: int) -> float:
    """Cost of stepping into (to_row, to_col)."""
    cell = grid[to_row][to_col]
    if cell.zone == Zone.COMMERCIAL:
        return 0.8
    return 1.0


def _reconstruct(parents: Dict[Pos, Pos], end: Pos) -> List[Pos]:
    path = [end]
    while end in parents:
        end = parents[end]
        path.append(end)
    path.reverse()
    return path


def astar(start: Pos, goal: Pos, grid: Grid) -> PathResult:
    if not (0 <= start[0] < len(grid) and 0 <= start[1] < len(grid[0])):
        return PathResult([], 0.0, False, f"Start {start} out of bounds")
    if not (0 <= goal[0] < len(grid) and 0 <= goal[1] < len(grid[0])):
        return PathResult([], 0.0, False, f"Goal {goal} out of bounds")
    if grid[start[0]][start[1]].no_fly:
        return PathResult([], 0.0, False, f"Start {start} is no-fly")
    if grid[goal[0]][goal[1]].no_fly:
        return PathResult([], 0.0, False, f"Goal {goal} is no-fly")

    if start == goal:
        return PathResult([start], 0.0, True, "Start equals goal")

    open_heap: List[Tuple[float, int, Pos]] = []
    counter = 0
    g_cost: Dict[Pos, float] = {start: 0.0}
    parents: Dict[Pos, Pos] = {}
    closed: set[Pos] = set()

    heapq.heappush(open_heap, (manhattan(start, goal), counter, start))

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            path = _reconstruct(parents, goal)
            return PathResult(path, g_cost[goal], True, "ok")
        closed.add(current)

        for nr, nc in get_neighbors(current[0], current[1], len(grid)):
            if grid[nr][nc].no_fly:
                continue
            neighbor = (nr, nc)
            if neighbor in closed:
                continue
            tentative = g_cost[current] + move_cost(grid, nr, nc)
            if tentative < g_cost.get(neighbor, float("inf")):
                g_cost[neighbor] = tentative
                parents[neighbor] = current
                f = tentative + manhattan(neighbor, goal)
                counter += 1
                heapq.heappush(open_heap, (f, counter, neighbor))

    return PathResult([], 0.0, False, f"No path from {start} to {goal}")


def plan_delivery_route(
    hub: Pos, delivery: Delivery, grid: Grid
) -> Tuple[PathResult, PathResult, PathResult]:
    leg1 = astar(hub, delivery.pickup, grid)
    leg2 = astar(delivery.pickup, delivery.dropoff, grid)
    leg3 = astar(delivery.dropoff, hub, grid)
    return leg1, leg2, leg3


def total_route(segments: Tuple[PathResult, ...]) -> PathResult:
    all_success = all(seg.success for seg in segments)
    combined: List[Pos] = []
    total_cost = 0.0
    for seg in segments:
        if not seg.path:
            continue
        if combined and combined[-1] == seg.path[0]:
            combined.extend(seg.path[1:])
        else:
            combined.extend(seg.path)
        total_cost += seg.cost
    msg = "ok" if all_success else "one or more segments failed"
    return PathResult(combined, total_cost, all_success, msg)


if __name__ == "__main__":
    grid = build_sample_grid()

    start: Pos = (1, 3)
    goal: Pos = (5, 6)
    result = astar(start, goal, grid)
    print(f"A* {start} -> {goal}")
    print(f"  success: {result.success}")
    print(f"  message: {result.message}")
    print(f"  cost:    {result.cost:.2f}")
    print(f"  path:    {result.path}")
    print()

    hub: Pos = (1, 3)
    delivery = Delivery(id=1, pickup=(4, 2), dropoff=(7, 8))
    legs = plan_delivery_route(hub, delivery, grid)
    leg_names = ("hub->pickup", "pickup->dropoff", "dropoff->hub")
    for name, leg in zip(leg_names, legs):
        print(f"{name}: success={leg.success} cost={leg.cost:.2f} len={len(leg.path)}")
        print(f"  path: {leg.path}")
    total = total_route(legs)
    print()
    print(f"Total route: success={total.success} cost={total.cost:.2f} cells={len(total.path)}")
    print(f"  path: {total.path}")
