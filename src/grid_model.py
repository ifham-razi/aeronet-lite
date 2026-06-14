"""Shared 10x10 grid model for AeroNet Lite.

Every module reads from this contract: the grid is a 2D list of Cell objects
indexed by [row][col] with row/col in [0, GRID_SIZE).
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

GRID_SIZE = 10

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
DEFAULT_POPULATION_CSV = os.path.join(
    _PROJECT_ROOT, "data", "raw", "population_density", "uscitypopdensity.csv"
)


class Zone(str, Enum):
    RESIDENTIAL = "Residential"
    COMMERCIAL = "Commercial"
    HOSPITAL = "Hospital"
    SCHOOL = "School"
    INDUSTRIAL = "Industrial"
    OPEN = "Open"


@dataclass
class Cell:
    row: int
    col: int
    zone: Zone = Zone.OPEN
    density: int = 0
    is_hub: bool = False
    is_charging: bool = False
    is_medical_pickup: bool = False
    no_fly: bool = False
    demand: float = 0.0

    @property
    def pos(self) -> Tuple[int, int]:
        return (self.row, self.col)


Grid = List[List[Cell]]


def empty_grid(size: int = GRID_SIZE) -> Grid:
    return [[Cell(r, c) for c in range(size)] for r in range(size)]


def in_bounds(row: int, col: int, size: int = GRID_SIZE) -> bool:
    return 0 <= row < size and 0 <= col < size


def get_neighbors(row: int, col: int, size: int = GRID_SIZE) -> List[Tuple[int, int]]:
    """4-directional neighbors, bounds-checked."""
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = row + dr, col + dc
        if in_bounds(nr, nc, size):
            out.append((nr, nc))
    return out


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def cells_of(grid: Grid, predicate) -> List[Cell]:
    return [c for row in grid for c in row if predicate(c)]


def hubs(grid: Grid) -> List[Cell]:
    return cells_of(grid, lambda c: c.is_hub)


def charging_pads(grid: Grid) -> List[Cell]:
    return cells_of(grid, lambda c: c.is_charging)


def medical_pickups(grid: Grid) -> List[Cell]:
    return cells_of(grid, lambda c: c.is_medical_pickup)


# Density values per zone (used for demand estimation + visualization).
ZONE_DENSITY = {
    Zone.RESIDENTIAL: 5000,
    Zone.COMMERCIAL: 3000,
    Zone.HOSPITAL: 1500,
    Zone.SCHOOL: 2000,
    Zone.INDUSTRIAL: 800,
    Zone.OPEN: 100,
}


def build_sample_grid() -> Grid:
    """Sample 10x10 layout matching the reference image in the project spec.

    HUBs at (1,3) and (5,6); charging pads at (2,4) and (6,1);
    a no-fly cell at (3,6); medical pickup near a hospital cluster.
    """
    grid = empty_grid()

    layout = [
        ["Res", "Res", "Com", "Com", "Opn", "Opn", "Sch", "Sch", "Opn", "Opn"],
        ["Res", "Hub", "Com", "Hub", "Opn", "Opn", "Sch", "Opn", "Opn", "Opn"],
        ["Res", "Com", "Chg", "Opn", "Opn", "Opn", "Opn", "Opn", "Ind", "Ind"],
        ["Opn", "Opn", "Opn", "Opn", "Opn", "Opn", "NoF", "Opn", "Ind", "Ind"],
        ["Hos", "Hos", "Opn", "Res", "Res", "Com", "Com", "Chg", "Opn", "Opn"],
        ["Hos", "Opn", "Opn", "Res", "Res", "Com", "Hub", "Com", "Opn", "Opn"],
        ["Opn", "Chg", "Opn", "Opn", "Res", "Res", "Com", "Com", "Opn", "Sch"],
        ["Opn", "Opn", "Opn", "Opn", "Opn", "Res", "Res", "Opn", "Opn", "Sch"],
        ["Ind", "Ind", "Opn", "Com", "Com", "Opn", "Opn", "Opn", "Opn", "Opn"],
        ["Ind", "Ind", "Opn", "Com", "Com", "Opn", "Opn", "Hos", "Hos", "Opn"],
    ]

    code_to_zone = {
        "Res": Zone.RESIDENTIAL,
        "Com": Zone.COMMERCIAL,
        "Hos": Zone.HOSPITAL,
        "Sch": Zone.SCHOOL,
        "Ind": Zone.INDUSTRIAL,
        "Opn": Zone.OPEN,
        "Hub": Zone.OPEN,
        "Chg": Zone.OPEN,
        "NoF": Zone.OPEN,
    }

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            code = layout[r][c]
            cell = grid[r][c]
            cell.zone = code_to_zone[code]
            cell.density = ZONE_DENSITY[cell.zone]
            cell.demand = float(cell.density) / 1000.0
            if code == "Hub":
                cell.is_hub = True
            elif code == "Chg":
                cell.is_charging = True
            elif code == "NoF":
                cell.no_fly = True

    # Mark a medical pickup adjacent to a hospital.
    grid[4][2].is_medical_pickup = True

    return grid


def grid_to_zone_matrix(grid: Grid) -> List[List[str]]:
    return [[c.zone.value for c in row] for row in grid]


# --------------------------------------------------------------------------- #
# Procedural grid generation
# --------------------------------------------------------------------------- #
def _try_build_random_grid(rng: random.Random) -> Optional[Grid]:
    """Single attempt to construct a valid random layout.

    Constraints are honored by construction:
      R1: industrial placed only when no neighbor is hospital/school
      R2: residential placed only inside Manhattan-3 of some hub
      R3: charging placed at Manhattan-1 from each hub
      R4: medical pickup placed adjacent to a hospital cell
    """
    n_hubs = rng.choice([2, 3])
    hub_set: List[Tuple[int, int]] = []
    for _ in range(60):
        if len(hub_set) >= n_hubs:
            break
        r = rng.randrange(1, GRID_SIZE - 1)
        c = rng.randrange(1, GRID_SIZE - 1)
        if all(manhattan((r, c), h) >= 4 for h in hub_set):
            hub_set.append((r, c))
    if len(hub_set) < 2:
        return None

    charging_set: List[Tuple[int, int]] = []
    for hr, hc in hub_set:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        rng.shuffle(offsets)
        placed = False
        for dr, dc in offsets:
            r, c = hr + dr, hc + dc
            if not in_bounds(r, c):
                continue
            if (r, c) in hub_set or (r, c) in charging_set:
                continue
            charging_set.append((r, c))
            placed = True
            break
        if not placed:
            return None

    reserved = set(hub_set) | set(charging_set)
    zones = [[Zone.OPEN for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

    # Residential: place near each hub, only within Manhattan-3.
    for hr, hc in hub_set:
        candidates: List[Tuple[int, int]] = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if (r, c) in reserved:
                    continue
                if manhattan((r, c), (hr, hc)) <= 3:
                    candidates.append((r, c))
        rng.shuffle(candidates)
        for r, c in candidates[: rng.randint(4, 7)]:
            zones[r][c] = Zone.RESIDENTIAL

    # Hospital cluster (2-3 cells, contiguous).
    hospital_cells = _grow_cluster(zones, reserved, rng, Zone.HOSPITAL, size_range=(2, 3))
    if not hospital_cells:
        return None

    # School cluster.
    school_cells = _grow_cluster(zones, reserved, rng, Zone.SCHOOL, size_range=(2, 3))
    if not school_cells:
        return None

    # Industrial: prefer corners, never adjacent to hospital/school.
    forbidden_neighbors = {Zone.HOSPITAL, Zone.SCHOOL}
    industrial_cells = _grow_industrial(
        zones, reserved, rng, forbidden_neighbors=forbidden_neighbors,
    )
    if not industrial_cells:
        return None

    # Commercial: scatter through remaining open space.
    open_cells = [
        (r, c)
        for r in range(GRID_SIZE)
        for c in range(GRID_SIZE)
        if zones[r][c] == Zone.OPEN and (r, c) not in reserved
    ]
    rng.shuffle(open_cells)
    n_comm = min(len(open_cells), rng.randint(8, 14))
    for r, c in open_cells[:n_comm]:
        zones[r][c] = Zone.COMMERCIAL

    # Medical pickup: pick an Open cell adjacent to a hospital.
    medical_pos: Optional[Tuple[int, int]] = None
    rng.shuffle(hospital_cells)
    for hr, hc in hospital_cells:
        adj = list(get_neighbors(hr, hc))
        rng.shuffle(adj)
        for nr, nc in adj:
            if (nr, nc) in reserved:
                continue
            if zones[nr][nc] not in (Zone.HOSPITAL, Zone.INDUSTRIAL):
                medical_pos = (nr, nc)
                break
        if medical_pos:
            break
    if medical_pos is None:
        return None

    # Build the actual Grid.
    grid = empty_grid()
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cell = grid[r][c]
            cell.zone = zones[r][c]
            cell.density = ZONE_DENSITY[cell.zone]
            cell.demand = float(cell.density) / 1000.0
    for hr, hc in hub_set:
        grid[hr][hc].is_hub = True
    for cr, cc in charging_set:
        grid[cr][cc].is_charging = True
    grid[medical_pos[0]][medical_pos[1]].is_medical_pickup = True
    return grid


def _grow_cluster(
    zones: List[List[Zone]],
    reserved: set,
    rng: random.Random,
    zone: Zone,
    size_range: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    target = rng.randint(*size_range)
    for _ in range(40):
        r0 = rng.randrange(GRID_SIZE)
        c0 = rng.randrange(GRID_SIZE)
        if (r0, c0) in reserved or zones[r0][c0] != Zone.OPEN:
            continue
        cluster = [(r0, c0)]
        frontier = list(get_neighbors(r0, c0))
        rng.shuffle(frontier)
        for nr, nc in frontier:
            if len(cluster) >= target:
                break
            if (nr, nc) in reserved:
                continue
            if zones[nr][nc] != Zone.OPEN:
                continue
            cluster.append((nr, nc))
        if len(cluster) >= 2:
            for r, c in cluster:
                zones[r][c] = zone
            return cluster
    return None


def _grow_industrial(
    zones: List[List[Zone]],
    reserved: set,
    rng: random.Random,
    forbidden_neighbors: set,
) -> Optional[List[Tuple[int, int]]]:
    """Place 2-4 industrial cells preferentially in a corner, never adjacent
    to hospital or school cells.
    """
    corners = [(0, 0), (0, GRID_SIZE - 1), (GRID_SIZE - 1, 0), (GRID_SIZE - 1, GRID_SIZE - 1)]
    rng.shuffle(corners)
    for cr, cc in corners:
        candidate_cells = []
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r, c = cr + dr, cc + dc
                if not in_bounds(r, c):
                    continue
                if (r, c) in reserved or zones[r][c] != Zone.OPEN:
                    continue
                # Forbid adjacency to hospital/school.
                bad = False
                for nr, nc in get_neighbors(r, c):
                    if zones[nr][nc] in forbidden_neighbors:
                        bad = True
                        break
                if not bad:
                    candidate_cells.append((r, c))
        if len(candidate_cells) >= 2:
            rng.shuffle(candidate_cells)
            chosen = candidate_cells[: rng.randint(2, min(4, len(candidate_cells)))]
            for r, c in chosen:
                zones[r][c] = Zone.INDUSTRIAL
            return chosen
    return None


def build_random_grid(seed: int) -> Grid:
    """Procedurally generate a valid 10x10 layout from `seed`.

    Tries multiple times with deterministic sub-seeds; if none of the attempts
    yields a valid layout, falls back to the hardcoded sample grid so the
    simulator always has something to run on.
    """
    for attempt in range(40):
        rng = random.Random(seed * 1000 + attempt)
        grid = _try_build_random_grid(rng)
        if grid is None:
            continue
        # Inline validation — avoid import cycle with layout_validator.
        if _layout_passes(grid):
            return grid
    return build_sample_grid()


def _layout_passes(grid: Grid) -> bool:
    """Lightweight in-module validation matching layout_validator's R1-R4.

    Kept here to avoid a circular import; layout_validator depends on this
    module, not the other way around.
    """
    hub_list = hubs(grid)
    charging_list = charging_pads(grid)
    medical_list = medical_pickups(grid)
    if not hub_list or not charging_list or not medical_list:
        return False

    for row in grid:
        for cell in row:
            # R1
            if cell.zone == Zone.INDUSTRIAL:
                for nr, nc in get_neighbors(cell.row, cell.col):
                    if grid[nr][nc].zone in (Zone.SCHOOL, Zone.HOSPITAL):
                        return False
            # R2
            if cell.zone == Zone.RESIDENTIAL:
                if not any(manhattan(cell.pos, h.pos) <= 3 for h in hub_list):
                    return False
    # R3
    for h in hub_list:
        if not any(manhattan(h.pos, p.pos) <= 2 for p in charging_list):
            return False
    # R4
    if not any(
        manhattan(h.pos, m.pos) <= 1
        for h in cells_of(grid, lambda c: c.zone == Zone.HOSPITAL)
        for m in medical_list
    ):
        return False
    return True


def load_population_buckets(
    csv_path: Optional[str] = None,
) -> Optional[Dict[str, List[int]]]:
    """Load real US city population densities and bucket into low/medium/high.

    Returns None if the CSV is missing or pandas is unavailable, so the caller
    can fall back to the static `ZONE_DENSITY` defaults.
    """
    path = csv_path or DEFAULT_POPULATION_CSV
    if not os.path.exists(path):
        return None
    try:
        import pandas as pd
    except ImportError:
        return None

    df = pd.read_csv(path, encoding="utf-8-sig")
    col = next(
        (c for c in df.columns if "density" in c.lower()),
        None,
    )
    if col is None:
        return None
    values = sorted(int(v) for v in df[col].dropna() if v > 0)
    if len(values) < 3:
        return None
    third = len(values) // 3
    return {
        "low": values[:third],
        "medium": values[third : 2 * third],
        "high": values[2 * third :],
    }


# Mapping zone -> bucket name. Hospitals/schools have more variable density;
# we map them to 'medium' so the grid still feels diverse.
_ZONE_BUCKET = {
    Zone.RESIDENTIAL: "high",
    Zone.COMMERCIAL: "medium",
    Zone.HOSPITAL: "medium",
    Zone.SCHOOL: "medium",
    Zone.INDUSTRIAL: "low",
    Zone.OPEN: "low",
}


def enrich_grid_with_population(
    grid: Grid,
    buckets: Optional[Dict[str, List[int]]] = None,
    seed: int = 0,
) -> bool:
    """Replace each cell's density with a sampled real-world density value
    drawn from the bucket associated with its zone. Demand is recomputed.

    Returns True if real data was applied; False if the buckets argument was
    None (caller should keep the static `ZONE_DENSITY` defaults).
    """
    if buckets is None:
        return False
    rng = random.Random(seed)
    for row in grid:
        for cell in row:
            bucket = _ZONE_BUCKET.get(cell.zone, "low")
            sample = rng.choice(buckets[bucket])
            cell.density = int(sample)
            cell.demand = float(cell.density) / 1000.0
    return True


if __name__ == "__main__":
    g = build_sample_grid()
    print(f"Grid {GRID_SIZE}x{GRID_SIZE} built.")
    print(f"  Hubs:           {[c.pos for c in hubs(g)]}")
    print(f"  Charging pads:  {[c.pos for c in charging_pads(g)]}")
    print(f"  Medical pickup: {[c.pos for c in medical_pickups(g)]}")
    print(f"  No-fly cells:   {[c.pos for c in cells_of(g, lambda x: x.no_fly)]}")
