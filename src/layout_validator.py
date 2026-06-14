"""CSP-style layout validator for AeroNet Lite (Module 1).

Implements four hard constraints over the shared Grid contract:
    R1 - Industrial cells must not be 4-adjacent to a School/Hospital.
    R2 - Every Residential cell must be within Manhattan distance 3 of a Hub.
    R3 - Every Hub must have a Charging Pad within Manhattan distance 2.
    R4 - At least one Hospital cell must have a Medical Pickup within Manhattan distance 1.

All checkers gather *every* violation; they never short-circuit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from .grid_model import (
    Cell,
    Grid,
    Zone,
    build_sample_grid,
    cells_of,
    charging_pads,
    get_neighbors,
    hubs,
    manhattan,
    medical_pickups,
)


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------

@dataclass
class Violation:
    rule_id: str               # "R1", "R2", "R3", "R4"
    cell: Tuple[int, int]      # (row, col) of the offending cell
    message: str
    suggestion: str


@dataclass
class ValidationReport:
    passed: List[str] = field(default_factory=list)
    failed: List[Violation] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return len(self.failed) == 0


# ---------------------------------------------------------------------------
# Rule checkers
# ---------------------------------------------------------------------------

# Sensitive zones that may not sit next to Industrial.
_SENSITIVE = {Zone.SCHOOL, Zone.HOSPITAL}


def check_industrial_safety(grid: Grid) -> List[Violation]:
    """R1: no Industrial cell may be 4-adjacent to a School or Hospital."""
    violations: List[Violation] = []
    for row in grid:
        for cell in row:
            if cell.zone is not Zone.INDUSTRIAL:
                continue
            for nr, nc in get_neighbors(cell.row, cell.col):
                neighbor = grid[nr][nc]
                if neighbor.zone in _SENSITIVE:
                    violations.append(Violation(
                        rule_id="R1",
                        cell=cell.pos,
                        message=(
                            f"Industrial cell {cell.pos} is adjacent to "
                            f"{neighbor.zone.value} cell {neighbor.pos}."
                        ),
                        suggestion=(
                            f"Relocate the Industrial cell at {cell.pos} or "
                            f"introduce an Open buffer between it and the "
                            f"{neighbor.zone.value} at {neighbor.pos}."
                        ),
                    ))
    return violations


def check_residential_coverage(grid: Grid) -> List[Violation]:
    """R2: every Residential cell within Manhattan distance 3 of some Hub."""
    violations: List[Violation] = []
    hub_positions = [h.pos for h in hubs(grid)]

    residentials = cells_of(grid, lambda c: c.zone is Zone.RESIDENTIAL)

    if not hub_positions:
        # Without any hub, every Residential cell trivially fails the rule.
        for cell in residentials:
            violations.append(Violation(
                rule_id="R2",
                cell=cell.pos,
                message=f"Residential cell {cell.pos} has no Hub on the grid.",
                suggestion="Place at least one Drone Hub within distance 3.",
            ))
        return violations

    for cell in residentials:
        nearest = min(manhattan(cell.pos, h) for h in hub_positions)
        if nearest > 3:
            violations.append(Violation(
                rule_id="R2",
                cell=cell.pos,
                message=(
                    f"Residential cell {cell.pos} is {nearest} away from the "
                    f"nearest Hub (limit 3)."
                ),
                suggestion=(
                    f"Add a Drone Hub within Manhattan distance 3 of {cell.pos} "
                    f"(e.g. closer than the current nearest hub)."
                ),
            ))
    return violations


def check_hub_charging(grid: Grid) -> List[Violation]:
    """R3: every Hub must have a Charging Pad within Manhattan distance 2."""
    violations: List[Violation] = []
    pad_positions = [p.pos for p in charging_pads(grid)]

    for hub in hubs(grid):
        if not pad_positions:
            violations.append(Violation(
                rule_id="R3",
                cell=hub.pos,
                message=f"Hub {hub.pos} has no Charging Pad on the grid.",
                suggestion=f"Place a Charging Pad within distance 2 of {hub.pos}.",
            ))
            continue
        nearest = min(manhattan(hub.pos, p) for p in pad_positions)
        if nearest > 2:
            violations.append(Violation(
                rule_id="R3",
                cell=hub.pos,
                message=(
                    f"Hub {hub.pos} is {nearest} away from the nearest "
                    f"Charging Pad (limit 2)."
                ),
                suggestion=(
                    f"Add a Charging Pad within Manhattan distance 2 of {hub.pos}."
                ),
            ))
    return violations


def check_medical_access(grid: Grid) -> List[Violation]:
    """R4: at least one Hospital must have a Medical Pickup within distance 1."""
    hospitals = cells_of(grid, lambda c: c.zone is Zone.HOSPITAL)
    pickup_positions = [m.pos for m in medical_pickups(grid)]

    # No hospitals at all: vacuously satisfied (rule speaks about hospitals).
    if not hospitals:
        return []

    if not pickup_positions:
        # Single aggregate violation anchored at the first hospital.
        anchor = hospitals[0].pos
        return [Violation(
            rule_id="R4",
            cell=anchor,
            message="No Medical Pickup point exists anywhere on the grid.",
            suggestion=(
                f"Place a Medical Pickup within Manhattan distance 1 of a "
                f"Hospital (e.g. adjacent to {anchor})."
            ),
        )]

    for hosp in hospitals:
        if any(manhattan(hosp.pos, p) <= 1 for p in pickup_positions):
            return []  # Rule satisfied by at least one hospital.

    # No hospital has a pickup within distance 1: report each hospital so the
    # planner sees every candidate site for the fix.
    return [
        Violation(
            rule_id="R4",
            cell=h.pos,
            message=(
                f"Hospital {h.pos} has no Medical Pickup within Manhattan "
                f"distance 1 (and no other hospital does either)."
            ),
            suggestion=(
                f"Add a Medical Pickup adjacent to (or on) {h.pos}."
            ),
        )
        for h in hospitals
    ]


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------

_RULE_TITLES = {
    "R1": "R1 - Industrial safety buffer",
    "R2": "R2 - Residential hub coverage",
    "R3": "R3 - Hub charging access",
    "R4": "R4 - Hospital medical pickup",
}


def validate_layout(grid: Grid) -> ValidationReport:
    """Run all four rule checkers and aggregate the results."""
    report = ValidationReport()

    rule_runs = (
        ("R1", check_industrial_safety),
        ("R2", check_residential_coverage),
        ("R3", check_hub_charging),
        ("R4", check_medical_access),
    )

    for rule_id, checker in rule_runs:
        rule_violations = checker(grid)
        if rule_violations:
            report.failed.extend(rule_violations)
        else:
            report.passed.append(_RULE_TITLES[rule_id])

    return report


def print_report(report: ValidationReport) -> None:
    """Print a clean, human-readable validation report."""
    bar = "=" * 64
    print(bar)
    print("AeroNet Lite - CSP Layout Validation Report")
    print(bar)

    status = "VALID" if report.valid else "INVALID"
    total_violations = len(report.failed)
    print(f"Overall status:    {status}")
    print(f"Rules passed:      {len(report.passed)} / 4")
    print(f"Total violations:  {total_violations}")
    print()

    print("-- Passed rules --")
    if report.passed:
        for title in report.passed:
            print(f"  [OK] {title}")
    else:
        print("  (none)")
    print()

    print("-- Failed rules --")
    if not report.failed:
        print("  (none)")
        print(bar)
        return

    # Group violations by rule_id for tidy output.
    by_rule: dict[str, List[Violation]] = {}
    for v in report.failed:
        by_rule.setdefault(v.rule_id, []).append(v)

    for rule_id in sorted(by_rule):
        title = _RULE_TITLES.get(rule_id, rule_id)
        violations = by_rule[rule_id]
        print(f"  [FAIL] {title}  ({len(violations)} violation(s))")
        for v in violations:
            print(f"     - cell {v.cell}: {v.message}")
            print(f"       fix: {v.suggestion}")
        print()

    print(bar)


# ---------------------------------------------------------------------------
# Manual run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    grid = build_sample_grid()
    report = validate_layout(grid)
    print_report(report)
