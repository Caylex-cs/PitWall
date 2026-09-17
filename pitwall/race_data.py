"""Load a single race session into tidy pandas tables, with a JSON cache
on disk so re-running an analysis doesn't re-hit the OpenF1 API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pitwall import openf1

CACHE_DIR = Path(__file__).resolve().parent / "data" / "races"

_ENDPOINTS = {
    "laps": openf1.get_laps,
    "stints": openf1.get_stints,
    "pit": openf1.get_pit,
    "drivers": openf1.get_drivers,
    "race_control": openf1.get_race_control,
    "weather": openf1.get_weather,
}


def _load_raw(session_key: int, name: str) -> list[dict]:
    cache_path = CACHE_DIR / str(session_key) / f"{name}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    data = _ENDPOINTS[name](session_key)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data))
    return data


@dataclass
class RaceData:
    session_key: int
    laps: pd.DataFrame
    stints: pd.DataFrame
    pit: pd.DataFrame
    drivers: pd.DataFrame
    race_control: pd.DataFrame
    weather: pd.DataFrame


def load_race(session_key: int) -> RaceData:
    """Fetch (or read from cache) every table needed for strategy analysis."""
    tables = {name: pd.DataFrame(_load_raw(session_key, name)) for name in _ENDPOINTS}
    return RaceData(session_key=session_key, **tables)


def laps_with_stint_info(race: RaceData) -> pd.DataFrame:
    """Laps joined with the compound/tyre-age of the stint each lap was run in."""
    laps = race.laps.copy()
    stints = race.stints.copy()

    rows = []
    for driver, driver_stints in stints.groupby("driver_number"):
        for _, stint in driver_stints.iterrows():
            lap_range = range(stint["lap_start"], stint["lap_end"] + 1)
            for lap_number in lap_range:
                rows.append(
                    {
                        "driver_number": driver,
                        "lap_number": lap_number,
                        "compound": stint["compound"],
                        "stint_number": stint["stint_number"],
                        "tyre_age": stint["tyre_age_at_start"] + (lap_number - stint["lap_start"]),
                    }
                )
    stint_lookup = pd.DataFrame(rows)

    merged = laps.merge(stint_lookup, on=["driver_number", "lap_number"], how="inner")

    # The lap after a stop is a pit-out lap (slow exit); the lap a driver pits
    # on on is itself run mostly at speed but ends in the pit lane, so OpenF1
    # records its duration as inflated too. Flag both so callers can exclude
    # them from pace/degradation fitting.
    pit_in_laps = set(zip(race.pit["driver_number"], race.pit["lap_number"]))
    merged["is_pit_in_lap"] = merged.apply(
        lambda r: (r["driver_number"], r["lap_number"]) in pit_in_laps, axis=1
    )
    return merged
