"""A real multi-car race simulator: the whole grid, lap by lap, with
emergent traffic -- not one isolated car charged a hand-picked gap profile
(that simpler version lives in strategy.py's traffic_profile_after_stops).

Every lap, each car's clear-air pace comes from the fitted degradation
model (pitwall.degradation). Each car is then charged a traffic penalty
(pitwall.traffic) from *last* lap's gap to whoever was directly ahead of
it, and the field is re-sorted by cumulative time to get the next lap's
running order and gaps. An overtake isn't a scripted event: if a faster
car's pace advantage outweighs the penalty from being stuck behind, the
gap keeps closing lap after lap until it crosses zero and the sort simply
puts that car ahead next lap.

Stated simplifications:
- Lap 1 gaps come from the real starting grid (OpenF1 `position`
  endpoint), seeded at zero time separation -- a "perfect start"
  assumption. Reaction times and first-corner incidents aren't modelled.
- The traffic penalty is a function of gap only. There's no explicit
  DRS/overtake-success mechanic or "this car defends and blocks" logic --
  enough of a pace advantage always eventually gets through.
- A driver who retired mid-race in the source data drops out of the
  simulation after their last completed lap (they stop accumulating time
  and stop blocking anyone), rather than being forced to "finish" a race
  they didn't.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pitwall.degradation import DegradationModel
from pitwall.pit_loss import PitLossModel
from pitwall.traffic import TrafficModel

Stint = tuple[str, int]  # (compound, number_of_laps)
LapSchedule = dict[int, tuple[str, int, bool]]  # lap_number -> (compound, tyre_age, pits_this_lap)


def build_lap_schedule(plan: list[Stint]) -> LapSchedule:
    """Expand a stint plan into a per-lap (compound, tyre_age, pits_this_lap)
    schedule. A car pits on the last lap of every stint except its last."""
    schedule: LapSchedule = {}
    lap_number = 0
    for stint_index, (compound, stint_laps) in enumerate(plan):
        is_last_stint = stint_index == len(plan) - 1
        for tyre_age in range(stint_laps):
            lap_number += 1
            pits_this_lap = tyre_age == stint_laps - 1 and not is_last_stint
            schedule[lap_number] = (compound, tyre_age, pits_this_lap)
    return schedule


@dataclass
class LapSnapshot:
    lap_number: int
    order: list[int]  # driver numbers, running order, leader first
    cumulative_time: dict[int, float]
    gap_to_ahead: dict[int, float | None]


@dataclass
class RaceSimResult:
    finishing_order: list[int]  # driver numbers, race winner first
    total_time: dict[int, float]
    laps_completed: dict[int, int]
    history: list[LapSnapshot] = field(default_factory=list)

    def gap_to_winner(self, driver_number: int) -> float:
        winner = self.finishing_order[0]
        return self.total_time[driver_number] - self.total_time[winner]


def simulate_race(
    model: DegradationModel,
    pit_loss: PitLossModel,
    traffic: TrafficModel,
    plans: dict[int, list[Stint]],
    starting_order: list[int],
) -> RaceSimResult:
    """Simulate every driver in `plans` together for as many laps as their
    own plan covers. `starting_order` (driver numbers, grid order) seeds
    lap 1's running order; every car starts at cumulative_time 0.
    """
    schedules = {driver: build_lap_schedule(plan) for driver, plan in plans.items()}
    race_laps = {driver: max(schedule, default=0) for driver, schedule in schedules.items()}

    cumulative = {driver: 0.0 for driver in plans}
    order = [driver for driver in starting_order if driver in plans]
    history: list[LapSnapshot] = []

    max_laps = max(race_laps.values(), default=0)
    for lap in range(1, max_laps + 1):
        active_order = [driver for driver in order if lap <= race_laps[driver]]

        gap_to_ahead: dict[int, float | None] = {}
        for i, driver in enumerate(active_order):
            gap_to_ahead[driver] = None if i == 0 else cumulative[driver] - cumulative[active_order[i - 1]]

        for driver in active_order:
            compound, tyre_age, pits_this_lap = schedules[driver][lap]
            raw_lap_time = model.predict_lap(driver, lap, compound, tyre_age)
            penalty = traffic.penalty(gap_to_ahead[driver])
            pit_cost = pit_loss.typical_seconds if pits_this_lap else 0.0
            cumulative[driver] += raw_lap_time + penalty + pit_cost

        # Drivers who've retired by this lap are simply dropped from `order`:
        # `active_order` re-derives from race_laps each lap, so they can
        # never re-enter, and nothing downstream reads `order` for them.
        order = sorted(active_order, key=lambda driver: cumulative[driver])

        history.append(
            LapSnapshot(lap_number=lap, order=list(order), cumulative_time=dict(cumulative), gap_to_ahead=gap_to_ahead)
        )

    finishing_order = sorted(plans, key=lambda driver: (-race_laps[driver], cumulative[driver]))
    return RaceSimResult(
        finishing_order=finishing_order,
        total_time=cumulative,
        laps_completed=race_laps,
        history=history,
    )
