"""Estimate the time cost of a pit stop from this session's actual stops.

OpenF1's `pit.lane_duration` is the time from crossing the pit entry line to
crossing the pit exit line (deceleration, lane transit at the speed limit,
the stationary tyre change, and acceleration back up). That's a *segment*
of a lap, not strictly "extra time versus staying out" -- a car that stayed
on track also spends time covering that same stretch of circuit, just
faster. Publicly quoted "pit loss" numbers are usually a bit lower than raw
lane_duration for exactly that reason.

We don't have a clean way to measure that residual from this endpoint
alone, so this module reports lane_duration itself as the stop cost. That
slightly overstates the true strategic cost of a stop, but it's the most
directly measurable number in the data and keeps the model's assumptions
visible rather than hidden behind an unverified correction factor.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pitwall.race_data import RaceData


@dataclass
class PitLossModel:
    typical_seconds: float  # robust (outlier-trimmed) central estimate to use in simulation
    mean_seconds: float
    median_seconds: float
    n_stops: int
    outliers: pd.DataFrame  # stops excluded from typical_seconds (slow stops, traffic, etc.)


def fit_pit_loss(race: RaceData) -> PitLossModel:
    pit = race.pit.dropna(subset=["lane_duration"])

    q1, q3 = pit["lane_duration"].quantile([0.25, 0.75])
    iqr = q3 - q1
    fence = q3 + 1.5 * iqr

    clean = pit[pit["lane_duration"] <= fence]
    outliers = pit[pit["lane_duration"] > fence]

    return PitLossModel(
        typical_seconds=clean["lane_duration"].median(),
        mean_seconds=pit["lane_duration"].mean(),
        median_seconds=pit["lane_duration"].median(),
        n_stops=len(pit),
        outliers=outliers[["driver_number", "lap_number", "lane_duration"]],
    )
