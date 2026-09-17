"""Estimate how much pace a driver loses running close behind another car,
and let the strategy simulator apply that cost to specific laps.

The degradation model (driver_pace + fuel_slope*lap + deg_rate*tyre_age)
assumes clear air. Whenever a lap is slower than that prediction, part of
the gap is noise, but part of it is systematic: cars running within a
second or two of the car ahead lose lap time to "dirty air" (aero wake)
and being unable to use every inch of track while defending/attacking,
and this gets worse the harder a circuit is to follow through. We isolate
that effect by regressing each lap's leftover residual (actual time minus
model-predicted clear-air time) against `interval`, the timed gap to the
car ahead at the end of that lap.

This is exactly the effect that makes overtaking-difficulty circuit-
specific: Monza's long straights and heavy braking zones make following
(and passing) cheap, so its penalty curve should be flat and small; a
technical, narrow circuit like the Hungaroring should show a much steeper
one. Comparing the fitted curves across circuits is the point, not just
fitting one in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pitwall.degradation import DegradationModel
from pitwall.race_data import RaceData, laps_with_gap_to_car_ahead, laps_with_stint_info

# Gap-to-car-ahead bucket edges (seconds). Bucketing (rather than a smooth
# fit) keeps the estimate transparent and lets it be non-monotonic if the
# data says so (e.g. a slipstream benefit at a mid-range gap on a power track).
GAP_BUCKETS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, float("inf")]
MAX_RESIDUAL_MAGNITUDE = 5.0  # seconds; drop laps this far off model prediction (lock-ups, near-misses, not traffic)


@dataclass
class TrafficModel:
    bucket_edges: list[float]
    penalty_by_bucket: list[float]  # median seconds lost, one per bucket, aligned to bucket_edges[i]..[i+1]
    n_laps_by_bucket: list[int]
    clear_air_baseline: float  # penalty at the largest gap bucket, subtracted so clear air reads ~0

    def penalty(self, gap_seconds: float | None) -> float:
        """Extra seconds this lap costs versus clear air, for a given gap to the car ahead."""
        if gap_seconds is None or np.isnan(gap_seconds):
            return 0.0
        for i, edge in enumerate(self.bucket_edges[1:]):
            if gap_seconds < edge:
                return self.penalty_by_bucket[i] - self.clear_air_baseline
        return self.penalty_by_bucket[-1] - self.clear_air_baseline


def fit_traffic_penalty(race: RaceData, model: DegradationModel) -> TrafficModel:
    laps = laps_with_stint_info(race)
    laps = laps[(laps["lap_number"] > 1) & ~laps["is_pit_out_lap"] & ~laps["is_pit_in_lap"]]
    laps = laps.dropna(subset=["lap_duration"])
    laps = laps[laps["compound"].isin(model.compounds)]  # drop tyres the degradation model couldn't fit (e.g. n=1)

    gaps = laps_with_gap_to_car_ahead(race)[["driver_number", "lap_number", "interval"]]
    laps = laps.merge(gaps, on=["driver_number", "lap_number"], how="inner").dropna(subset=["interval"])

    predicted = laps.apply(
        lambda r: model.predict_lap(r["driver_number"], r["lap_number"], r["compound"], r["tyre_age"]), axis=1
    )
    laps = laps.assign(residual=laps["lap_duration"] - predicted)
    laps = laps[laps["residual"].abs() <= MAX_RESIDUAL_MAGNITUDE]

    bucket_index = pd.cut(laps["interval"], GAP_BUCKETS, right=False, labels=False)
    penalties = []
    counts = []
    for i in range(len(GAP_BUCKETS) - 1):
        bucket_laps = laps[bucket_index == i]
        counts.append(len(bucket_laps))
        penalties.append(bucket_laps["residual"].median() if len(bucket_laps) else float("nan"))

    clear_air_baseline = next((p for p in reversed(penalties) if not np.isnan(p)), 0.0)

    return TrafficModel(
        bucket_edges=GAP_BUCKETS,
        penalty_by_bucket=penalties,
        n_laps_by_bucket=counts,
        clear_air_baseline=clear_air_baseline,
    )
