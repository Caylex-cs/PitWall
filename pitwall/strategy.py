"""Simulate total race time for a given tyre strategy using the fitted
degradation model and pit loss estimate, and compare candidate strategies.

Optionally folds in a TrafficModel (pitwall/traffic.py) so a strategy that
rejoins into traffic is charged for the pace lost following another car,
not just the pit-lane time -- this is what actually drives real undercut/
overcut decisions: pitting can be "free" in clear air but costly if it
drops you into a train of cars you can't pass.
"""

from __future__ import annotations

from dataclasses import dataclass

from pitwall.degradation import DegradationModel
from pitwall.pit_loss import PitLossModel
from pitwall.traffic import TrafficModel

Stint = tuple[str, int]  # (compound, number_of_laps)
GapProfile = dict[int, float]  # lap_number -> assumed gap (seconds) to the car ahead


@dataclass
class StrategyResult:
    plan: list[Stint]
    total_race_time: float
    num_stops: int
    traffic_cost: float = 0.0  # seconds of the total attributable to the gap_profile, for reporting


def traffic_profile_after_stops(
    plan: list[Stint], gap_after_stop: float, laps_in_traffic: int
) -> GapProfile:
    """A gap-to-car-ahead profile assuming a driver rejoins `laps_in_traffic`
    laps of running at `gap_after_stop` seconds behind another car after
    every stop in `plan` (e.g. released into a train they can't clear),
    and clear air (no entry -> penalty 0) everywhere else."""
    profile: GapProfile = {}
    lap_number = 0
    stop_laps = []
    for compound, stint_laps in plan[:-1]:
        lap_number += stint_laps
        stop_laps.append(lap_number)  # lap on which the driver rejoins after that stop

    for stop_lap in stop_laps:
        for offset in range(laps_in_traffic):
            profile[stop_lap + 1 + offset] = gap_after_stop
    return profile


def simulate_strategy(
    model: DegradationModel,
    pit_loss: PitLossModel,
    plan: list[Stint],
    driver_pace: float,
    race_laps: int,
    traffic: TrafficModel | None = None,
    gap_profile: GapProfile | None = None,
) -> StrategyResult:
    """Total time to complete `race_laps` laps on the given stint plan.

    `driver_pace` is the flat per-driver offset from the degradation model
    (use a specific driver's fitted pace, or an average, to represent "a
    car of this speed"). Each stop costs `pit_loss.typical_seconds` on top
    of the laps it takes to run the plan.

    If `traffic` and `gap_profile` are given, each lap's assumed gap to the
    car ahead (missing laps are treated as clear air) adds `traffic.penalty`
    on top of the clear-air pace -- see traffic_profile_after_stops for a
    ready-made "rejoins in a train after every stop" profile.
    """
    plan_laps = sum(laps for _, laps in plan)
    if plan_laps != race_laps:
        raise ValueError(f"plan covers {plan_laps} laps, race is {race_laps} laps")

    total_time = 0.0
    traffic_cost = 0.0
    lap_number = 0
    for compound, stint_laps in plan:
        for tyre_age in range(stint_laps):
            lap_number += 1
            total_time += driver_pace + model.fuel_slope * lap_number + model.deg_rate[compound] * tyre_age
            if traffic is not None and gap_profile is not None and lap_number in gap_profile:
                cost = traffic.penalty(gap_profile[lap_number])
                total_time += cost
                traffic_cost += cost

    num_stops = len(plan) - 1
    total_time += num_stops * pit_loss.typical_seconds

    return StrategyResult(plan=plan, total_race_time=total_time, num_stops=num_stops, traffic_cost=traffic_cost)


def best_one_stop(
    model: DegradationModel,
    pit_loss: PitLossModel,
    compounds: tuple[str, str],
    driver_pace: float,
    race_laps: int,
    traffic: TrafficModel | None = None,
    gap_after_stop: float | None = None,
    laps_in_traffic: int = 0,
) -> StrategyResult:
    """Sweep every possible single pit lap and return the fastest split.

    If `gap_after_stop` is given (with `traffic`), every candidate stop lap
    is charged the traffic cost of rejoining at that gap for
    `laps_in_traffic` laps -- so a strategy that's fastest in clear air can
    lose to one that pits into a different (emptier) part of the track.
    """
    best = None
    for stop_lap in range(1, race_laps):
        plan = [(compounds[0], stop_lap), (compounds[1], race_laps - stop_lap)]
        gap_profile = (
            traffic_profile_after_stops(plan, gap_after_stop, laps_in_traffic)
            if gap_after_stop is not None
            else None
        )
        result = simulate_strategy(model, pit_loss, plan, driver_pace, race_laps, traffic, gap_profile)
        if best is None or result.total_race_time < best.total_race_time:
            best = result
    return best
