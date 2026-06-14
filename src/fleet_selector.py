"""Fleet Selector for AeroNet Lite (Module 2).

Picks how many LIGHT vs HEAVY drones to buy under a budget by maximising a
fitness score that trades off coverage against cost.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple

from src.grid_model import Grid, build_sample_grid, hubs


# ---------------------------------------------------------------------------
# Drone type catalogue
# ---------------------------------------------------------------------------

@dataclass
class DroneSpec:
    name: str
    cost: int
    payload_kg: float
    range_cells: int


LIGHT = DroneSpec(name="LIGHT", cost=1000, payload_kg=2, range_cells=12)
HEAVY = DroneSpec(name="HEAVY", cost=1800, payload_kg=5, range_cells=20)


@dataclass
class Drone:
    id: str
    spec: DroneSpec
    home_hub: Tuple[int, int]


@dataclass
class FleetSelection:
    light_count: int
    heavy_count: int
    total_cost: int
    coverage_pct: float
    score: float
    drones: List[Drone] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Demand / coverage / fitness
# ---------------------------------------------------------------------------

def estimate_total_demand(grid: Grid) -> float:
    return sum(cell.demand for row in grid for cell in row)


def coverage(light_count: int, heavy_count: int, grid: Grid) -> float:
    total_demand = estimate_total_demand(grid)
    if total_demand <= 0:
        return 0.0
    # Approximate per-drone coverage using its range in cells, weighted by the
    # average demand per cell.
    cells = sum(1 for row in grid for _ in row)
    avg_demand = total_demand / cells
    serviceable = (
        light_count * LIGHT.range_cells * avg_demand
        + heavy_count * HEAVY.range_cells * avg_demand
    )
    pct = (serviceable / total_demand) * 100.0
    return min(pct, 100.0)


def _total_cost(light_count: int, heavy_count: int) -> int:
    return light_count * LIGHT.cost + heavy_count * HEAVY.cost


def fitness(light_count: int, heavy_count: int, grid: Grid, budget: int) -> float:
    cost = _total_cost(light_count, heavy_count)
    if cost > budget:
        return -math.inf
    cov_pct = coverage(light_count, heavy_count, grid)
    budget_used_pct = (cost / budget) * 100.0 if budget > 0 else 0.0
    return 0.75 * cov_pct - 0.25 * budget_used_pct


def _build_selection(light_count: int, heavy_count: int, grid: Grid, budget: int) -> FleetSelection:
    cost = _total_cost(light_count, heavy_count)
    cov_pct = coverage(light_count, heavy_count, grid)
    score = fitness(light_count, heavy_count, grid, budget)
    return FleetSelection(
        light_count=light_count,
        heavy_count=heavy_count,
        total_cost=cost,
        coverage_pct=cov_pct,
        score=score,
    )


# ---------------------------------------------------------------------------
# Brute-force selector
# ---------------------------------------------------------------------------

def select_fleet_brute_force(grid: Grid, budget: int = 10000) -> FleetSelection:
    best: FleetSelection | None = None
    for light in range(11):
        for heavy in range(11):
            score = fitness(light, heavy, grid, budget)
            if best is None or score > best.score:
                best = _build_selection(light, heavy, grid, budget)
    assert best is not None
    assign_drones_to_hubs(best, grid)
    return best


# ---------------------------------------------------------------------------
# Genetic Algorithm selector
# ---------------------------------------------------------------------------

def _random_chromosome() -> Tuple[int, int]:
    return (random.randint(0, 10), random.randint(0, 10))


def _tournament(pop: List[Tuple[int, int]], scores: List[float], k: int = 3) -> Tuple[int, int]:
    contenders = random.sample(range(len(pop)), k)
    winner = max(contenders, key=lambda i: scores[i])
    return pop[winner]


def _crossover(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
    # Single-point crossover on a 2-gene chromosome: take gene 0 from a, gene 1 from b.
    return (a[0], b[1])


def _mutate(chrom: Tuple[int, int], rate: float = 0.2) -> Tuple[int, int]:
    light, heavy = chrom
    if random.random() < rate:
        light = max(0, min(10, light + random.choice((-1, 1))))
    if random.random() < rate:
        heavy = max(0, min(10, heavy + random.choice((-1, 1))))
    return (light, heavy)


def select_fleet_ga(
    grid: Grid,
    budget: int = 10000,
    generations: int = 30,
    population: int = 20,
) -> FleetSelection:
    random.seed(42)

    pop: List[Tuple[int, int]] = [_random_chromosome() for _ in range(population)]
    scores = [fitness(l, h, grid, budget) for (l, h) in pop]

    best_chrom = max(pop, key=lambda c: fitness(c[0], c[1], grid, budget))
    best_score = fitness(best_chrom[0], best_chrom[1], grid, budget)

    for _ in range(generations):
        new_pop: List[Tuple[int, int]] = []
        # Elitism: keep current best.
        new_pop.append(best_chrom)
        while len(new_pop) < population:
            p1 = _tournament(pop, scores)
            p2 = _tournament(pop, scores)
            child = _crossover(p1, p2)
            child = _mutate(child)
            new_pop.append(child)

        pop = new_pop
        scores = [fitness(l, h, grid, budget) for (l, h) in pop]

        gen_best = max(pop, key=lambda c: fitness(c[0], c[1], grid, budget))
        gen_best_score = fitness(gen_best[0], gen_best[1], grid, budget)
        if gen_best_score > best_score:
            best_chrom = gen_best
            best_score = gen_best_score

    selection = _build_selection(best_chrom[0], best_chrom[1], grid, budget)
    assign_drones_to_hubs(selection, grid)
    return selection


# ---------------------------------------------------------------------------
# Drone-to-hub assignment
# ---------------------------------------------------------------------------

def assign_drones_to_hubs(selection: FleetSelection, grid: Grid) -> List[Drone]:
    hub_cells = hubs(grid)
    drones: List[Drone] = []

    if not hub_cells:
        selection.drones = drones
        return drones

    # Build a flat list of drone specs to place: lights first, then heavies.
    specs = [LIGHT] * selection.light_count + [HEAVY] * selection.heavy_count

    for idx, spec in enumerate(specs):
        hub = hub_cells[idx % len(hub_cells)]
        drones.append(Drone(id=f"D{idx + 1}", spec=spec, home_hub=hub.pos))

    selection.drones = drones
    return drones


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_fleet(selection: FleetSelection) -> None:
    print("Fleet Selection")
    print(f"  Light drones:  {selection.light_count}")
    print(f"  Heavy drones:  {selection.heavy_count}")
    print(f"  Total cost:    {selection.total_cost}")
    print(f"  Coverage:      {selection.coverage_pct:.2f}%")
    print(f"  Fitness score: {selection.score:.4f}")
    if selection.drones:
        print("  Drones:")
        for d in selection.drones:
            print(f"    {d.id:>4}  {d.spec.name:<5}  home_hub={d.home_hub}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    g = build_sample_grid()

    print("=== Brute-force selector ===")
    bf = select_fleet_brute_force(g, budget=10000)
    print_fleet(bf)

    print()
    print("=== Genetic Algorithm selector ===")
    ga = select_fleet_ga(g, budget=10000, generations=30, population=20)
    print_fleet(ga)
