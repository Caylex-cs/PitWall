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
    "intervals": openf1.get_intervals,
    "position": openf1.get_position,
}

# Interval samples land every ~4s; a lap end matched more than this many
# seconds from the nearest sample is treated as unmatched (no gap data).
_INTERVAL_MATCH_TOLERANCE_SECONDS = 8


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
    intervals: pd.DataFrame
    position: pd.DataFrame


def load_race(session_key: int) -> RaceData:
    """Fetch (or read from cache) every table needed for strategy analysis."""
    tables = {name: pd.DataFrame(_load_raw(session_key, name)) for name in _ENDPOINTS}
    return RaceData(session_key=session_key, **tables)


def _stints_with_resolved_ends(race: RaceData) -> pd.DataFrame:
    """OpenF1 leaves `lap_end` null for a driver's final stint when it was
    never explicitly closed by a later pit stop (a retirement, or simply the
    last stint of the race for some drivers). Fill it with the last lap that
    driver actually ran. A stint with no `lap_start` at all (e.g. a car that
    retired before completing a lap) has no usable range and is dropped."""
    stints = race.stints.dropna(subset=["lap_start"]).copy()
    last_lap = race.laps.groupby("driver_number")["lap_number"].max()
    stints["lap_end"] = stints["lap_end"].fillna(stints["driver_number"].map(last_lap))
    stints = stints.dropna(subset=["lap_end"])
    stints["lap_start"] = stints["lap_start"].astype(int)
    stints["lap_end"] = stints["lap_end"].astype(int)
    return stints


def driver_plan(race: RaceData, driver_number: int) -> list[tuple[str, int]]:
    """A driver's actual stint plan as [(compound, stint_length_in_laps), ...]."""
    stints = _stints_with_resolved_ends(race)
    stints = stints[stints["driver_number"] == driver_number].sort_values("stint_number")
    return [(row["compound"], row["lap_end"] - row["lap_start"] + 1) for _, row in stints.iterrows()]


def num_stops(race: RaceData, driver_number: int) -> int:
    return len(race.stints[race.stints["driver_number"] == driver_number]) - 1


def all_driver_numbers(race: RaceData) -> list[int]:
    """Every driver who appears in the session, grid-position order isn't
    implied -- use starting_grid_order for that."""
    return sorted(race.drivers["driver_number"].unique())


def driver_last_lap(race: RaceData, driver_number: int) -> int:
    """The last lap a driver actually completed (their retirement lap, or
    the race's final lap if they finished)."""
    laps = race.laps[race.laps["driver_number"] == driver_number]["lap_number"]
    return int(laps.max()) if len(laps) else 0


def actual_total_time(race: RaceData, driver_number: int) -> float:
    """Sum of a driver's actual recorded lap times (their real result, for
    validating a simulation against)."""
    return race.laps[race.laps["driver_number"] == driver_number]["lap_duration"].sum()


def starting_grid_order(race: RaceData) -> list[int]:
    """Driver numbers in starting-grid order, from each driver's earliest
    `position` sample (OpenF1 doesn't expose a dedicated grid endpoint, but
    position tracking starts before the race and reflects grid order)."""
    position = race.position.sort_values("date")
    first_position = position.groupby("driver_number").first()
    return list(first_position.sort_values("position").index)


def laps_with_stint_info(race: RaceData) -> pd.DataFrame:
    """Laps joined with the compound/tyre-age of the stint each lap was run in."""
    laps = race.laps.copy()
    stints = _stints_with_resolved_ends(race)

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


def laps_with_gap_to_car_ahead(race: RaceData) -> pd.DataFrame:
    """Laps joined with `interval` (gap to the car ahead) and `gap_to_leader`
    as of each lap's end, for measuring traffic effects and for plotting a
    real gap-to-leader chart.

    OpenF1's `intervals` endpoint is a time series (~1 sample every 4s per
    driver), not one row per lap, so each lap is matched to its nearest
    interval sample within `_INTERVAL_MATCH_TOLERANCE_SECONDS`. The race
    leader has no car ahead and always has `interval` 0 in the source data.
    """
    laps = race.laps.copy()
    laps["lap_end_time"] = (
        pd.to_datetime(laps["date_start"], format="ISO8601") + pd.to_timedelta(laps["lap_duration"], unit="s")
    ).astype("datetime64[ns, UTC]")

    intervals = race.intervals.copy()
    intervals["date"] = pd.to_datetime(intervals["date"], format="ISO8601").astype("datetime64[ns, UTC]")

    matched = []
    for driver, driver_laps in laps.dropna(subset=["lap_end_time"]).groupby("driver_number"):
        driver_intervals = intervals[intervals["driver_number"] == driver].sort_values("date")
        if driver_intervals.empty:
            continue
        merged = pd.merge_asof(
            driver_laps.sort_values("lap_end_time"),
            driver_intervals[["date", "interval", "gap_to_leader"]],
            left_on="lap_end_time",
            right_on="date",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=_INTERVAL_MATCH_TOLERANCE_SECONDS),
        )
        matched.append(merged)

    return pd.concat(matched, ignore_index=True) if matched else laps.assign(interval=pd.NA)
