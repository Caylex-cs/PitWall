"""Pairwise undercut/overcut calculator.

The actual pit-wall question isn't "what's the optimal race strategy" in
the abstract, it's "we're 1.0s behind them right now, on these tyres -- if
we pit this lap and they respond in N laps, do we come out ahead, and by
when?" This answers exactly that, using each driver's own fitted pace,
degradation and pit-loss numbers, in isolation from the other 18 cars on
track (the same simplification a strategist sketches on a whiteboard).

For a specific scenario, cross-check it against pitwall.race_simulator
(re-run the whole grid with just these two drivers' pit laps changed,
everyone else's real strategy held fixed) to see whether traffic from the
rest of the field changes the answer -- see scripts/undercut_calculator.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from pitwall.degradation import DegradationModel
from pitwall.pit_loss import PitLossModel
from pitwall.traffic import TrafficModel

GapProfile = dict[int, float]


@dataclass
class TyreState:
    compound: str
    tyre_age: int  # laps on this tyre as of the end of the current lap


@dataclass
class UndercutResult:
    current_lap: int
    horizon_lap: int
    current_gap: float  # attacker_cumulative_time - defender_cumulative_time; positive = attacker behind
    gap_by_lap: dict[int, float]
    final_gap: float
    swing: float  # current_gap - final_gap; positive = attacker gained ground
    attacker_now_ahead: bool


def _advance(
    model: DegradationModel,
    pit_loss: PitLossModel,
    driver: int,
    start_lap: int,
    horizon_lap: int,
    state: TyreState,
    pit_lap: int | None,
    new_compound: str | None,
    traffic: TrafficModel | None,
    gap_profile: GapProfile | None,
) -> dict[int, float]:
    """Cumulative extra race time this driver accrues from `start_lap` to
    each lap up to `horizon_lap`, given when (if at all) they pit."""
    compound, tyre_age = state.compound, state.tyre_age
    cumulative = 0.0
    deltas: dict[int, float] = {}
    for lap in range(start_lap, horizon_lap + 1):
        cumulative += model.predict_lap(driver, lap, compound, tyre_age)
        if traffic is not None and gap_profile is not None and lap in gap_profile:
            cumulative += traffic.penalty(gap_profile[lap])
        if pit_lap is not None and lap == pit_lap:
            cumulative += pit_loss.typical_seconds
            compound, tyre_age = new_compound, 0
        else:
            tyre_age += 1
        deltas[lap] = cumulative
    return deltas


def project_gap(
    model: DegradationModel,
    pit_loss: PitLossModel,
    attacker: int,
    defender: int,
    current_lap: int,
    current_gap: float,
    attacker_state: TyreState,
    defender_state: TyreState,
    horizon_lap: int,
    attacker_pit_lap: int | None = None,
    defender_pit_lap: int | None = None,
    new_compound: str | None = None,
    traffic: TrafficModel | None = None,
    attacker_gap_profile: GapProfile | None = None,
    defender_gap_profile: GapProfile | None = None,
) -> UndercutResult:
    """Project how `current_gap` (attacker's cumulative time minus
    defender's; positive means the attacker is behind) evolves from
    `current_lap` to `horizon_lap`, given each driver's pit-lap choice
    (None = no stop in this window) and the compound they switch to.
    """
    attacker_deltas = _advance(
        model, pit_loss, attacker, current_lap + 1, horizon_lap, attacker_state,
        attacker_pit_lap, new_compound, traffic, attacker_gap_profile,
    )
    defender_deltas = _advance(
        model, pit_loss, defender, current_lap + 1, horizon_lap, defender_state,
        defender_pit_lap, new_compound, traffic, defender_gap_profile,
    )

    gap_by_lap = {
        lap: current_gap + attacker_deltas[lap] - defender_deltas[lap] for lap in attacker_deltas
    }
    final_gap = gap_by_lap[horizon_lap]

    return UndercutResult(
        current_lap=current_lap,
        horizon_lap=horizon_lap,
        current_gap=current_gap,
        gap_by_lap=gap_by_lap,
        final_gap=final_gap,
        swing=current_gap - final_gap,
        attacker_now_ahead=final_gap < 0,
    )


def latest_safe_response_lap(
    model: DegradationModel,
    pit_loss: PitLossModel,
    attacker: int,
    defender: int,
    current_lap: int,
    current_gap: float,
    attacker_state: TyreState,
    defender_state: TyreState,
    attacker_pit_lap: int,
    new_compound: str,
    max_response_lap: int,
) -> int | None:
    """Evaluated at `max_response_lap`, find the latest lap the defender
    can wait before pitting (onto `new_compound`) and still be ahead of
    the attacker. Each extra lap the defender waits is one more lap spent
    losing time to the attacker's fresh tyres, so there's a single
    crossover: safe up to some lap, then not. Returns None if even
    responding the lap right after the attacker isn't enough.
    """
    last_safe_lap = None
    for defender_pit_lap in range(attacker_pit_lap + 1, max_response_lap + 1):
        result = project_gap(
            model, pit_loss, attacker, defender, current_lap, current_gap,
            attacker_state, defender_state, max_response_lap,
            attacker_pit_lap=attacker_pit_lap, defender_pit_lap=defender_pit_lap, new_compound=new_compound,
        )
        if result.attacker_now_ahead:
            break
        last_safe_lap = defender_pit_lap
    return last_safe_lap
