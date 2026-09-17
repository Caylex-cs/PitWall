"""Fit a tyre degradation + fuel-load model from a race's lap times.

Lap time is modelled as:

    lap_time = driver_pace[driver] + fuel_slope * lap_number
               + deg_rate[compound] * tyre_age

`driver_pace` absorbs each car/driver's baseline pace (setup, engine, driver
skill) so it doesn't get mistaken for tyre effects. `fuel_slope` is shared
across the field and captures the car getting quicker as fuel burns off.
`deg_rate` is fit per compound and is the number we actually want: seconds
per lap of pace lost for each additional lap on a given tyre.

Within one stint, lap_number and tyre_age move in lockstep for a single
driver, so they can't be told apart from that driver's data alone. The
model only works because different drivers pit on different laps: at any
given lap_number, tyre_age varies across the field, which is what lets the
regression separate "the race has gone on this long" from "this tyre is
this old."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pitwall.race_data import RaceData, laps_with_stint_info

MAX_FUEL_CORRECTED_OUTLIER = 1.5  # seconds slower than a driver's stint median -> dropped (traffic, lock-ups, etc.)
MIN_LAPS_PER_COMPOUND = 5  # below this a compound's slope is noise, not a degradation estimate


def clean_laps(race: RaceData) -> pd.DataFrame:
    """Laps usable for pace fitting: no lap 1, no in/out laps, no traffic outliers."""
    laps = laps_with_stint_info(race)
    laps = laps[(laps["lap_number"] > 1) & ~laps["is_pit_out_lap"] & ~laps["is_pit_in_lap"]]
    laps = laps.dropna(subset=["lap_duration"])

    stint_median = laps.groupby(["driver_number", "stint_number"])["lap_duration"].transform("median")
    return laps[laps["lap_duration"] <= stint_median + MAX_FUEL_CORRECTED_OUTLIER]


@dataclass
class DegradationModel:
    driver_pace: dict[int, float]
    fuel_slope: float
    deg_rate: dict[str, float]
    compounds: list[str]
    excluded_compounds: dict[str, int]  # compound -> lap count, too few to fit

    def predict_lap(self, driver_number: int, lap_number: int, compound: str, tyre_age: int) -> float:
        return (
            self.driver_pace[driver_number]
            + self.fuel_slope * lap_number
            + self.deg_rate[compound] * tyre_age
        )


def fit_degradation_model(race: RaceData) -> DegradationModel:
    laps = clean_laps(race)

    lap_counts = laps["compound"].value_counts()
    excluded_compounds = lap_counts[lap_counts < MIN_LAPS_PER_COMPOUND].to_dict()
    laps = laps[~laps["compound"].isin(excluded_compounds)]

    drivers = sorted(laps["driver_number"].unique())
    compounds = sorted(laps["compound"].unique())

    driver_index = {d: i for i, d in enumerate(drivers)}
    compound_index = {c: i for i, c in enumerate(compounds)}

    n = len(laps)
    n_cols = len(drivers) + 1 + len(compounds)  # driver dummies + fuel slope + per-compound tyre_age
    X = np.zeros((n, n_cols))
    y = laps["lap_duration"].to_numpy()

    rows = laps.reset_index(drop=True)
    for i, row in rows.iterrows():
        X[i, driver_index[row["driver_number"]]] = 1.0
        X[i, len(drivers)] = row["lap_number"]
        X[i, len(drivers) + 1 + compound_index[row["compound"]]] = row["tyre_age"]

    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)

    driver_pace = {d: coeffs[driver_index[d]] for d in drivers}
    fuel_slope = coeffs[len(drivers)]
    deg_rate = {c: coeffs[len(drivers) + 1 + compound_index[c]] for c in compounds}

    return DegradationModel(
        driver_pace=driver_pace,
        fuel_slope=fuel_slope,
        deg_rate=deg_rate,
        compounds=compounds,
        excluded_compounds=excluded_compounds,
    )
