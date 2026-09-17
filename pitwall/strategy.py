"""Simulate total race time for a given tyre strategy using the fitted
degradation model and pit loss estimate, and compare candidate strategies.
"""

from __future__ import annotations

from dataclasses import dataclass

from pitwall.degradation import DegradationModel
from pitwall.pit_loss import PitLossModel

Stint = tuple[str, int]  # (compound, number_of_laps)


@dataclass
class StrategyResult:
    plan: list[Stint]
    total_race_time: float
    num_stops: int


def simulate_strategy(
    model: DegradationModel,
    pit_loss: PitLossModel,
    plan: list[Stint],
    driver_pace: float,
    race_laps: int,
) -> StrategyResult:
    """Total time to complete `race_laps` laps on the given stint plan.

    `driver_pace` is the flat per-driver offset from the degradation model
    (use a specific driver's fitted pace, or an average, to represent "a
    car of this speed"). Each stop costs `pit_loss.typical_seconds` on top
    of the laps it takes to run the plan.
    """
    plan_laps = sum(laps for _, laps in plan)
    if plan_laps != race_laps:
        raise ValueError(f"plan covers {plan_laps} laps, race is {race_laps} laps")

    total_time = 0.0
    lap_number = 0
    for compound, stint_laps in plan:
        for tyre_age in range(stint_laps):
            lap_number += 1
            total_time += driver_pace + model.fuel_slope * lap_number + model.deg_rate[compound] * tyre_age

    num_stops = len(plan) - 1
    total_time += num_stops * pit_loss.typical_seconds

    return StrategyResult(plan=plan, total_race_time=total_time, num_stops=num_stops)


def best_one_stop(
    model: DegradationModel, pit_loss: PitLossModel, compounds: tuple[str, str], driver_pace: float, race_laps: int
) -> StrategyResult:
    """Sweep every possible single pit lap and return the fastest split."""
    best = None
    for stop_lap in range(1, race_laps):
        plan = [(compounds[0], stop_lap), (compounds[1], race_laps - stop_lap)]
        result = simulate_strategy(model, pit_loss, plan, driver_pace, race_laps)
        if best is None or result.total_race_time < best.total_race_time:
            best = result
    return best
