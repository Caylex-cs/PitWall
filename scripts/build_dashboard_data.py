#!/usr/bin/env python3
"""Run the full pipeline for the two validated races and export one
consolidated JSON the dashboard artifact reads. Nothing here is a new
model -- it just calls the existing modules and shapes their output for
charts instead of stdout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pitwall.degradation import fit_degradation_model
from pitwall.pit_loss import fit_pit_loss
from pitwall.race_data import (
    actual_total_time,
    all_driver_numbers,
    driver_last_lap,
    driver_plan,
    laps_with_gap_to_car_ahead,
    laps_with_stint_info,
    load_race,
    num_stops,
    starting_grid_order,
)
from pitwall.race_simulator import simulate_race
from pitwall.strategy import best_one_stop, simulate_strategy
from pitwall.traffic import GAP_BUCKETS, fit_traffic_penalty
from pitwall.undercut import TyreState, project_gap

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "pitwall" / "data" / "dashboard.json"

TOP_N_FOR_GAP_CHART = 8

RACES = [
    {
        "session_key": 9590,
        "label": "2024 Italian Grand Prix (Monza)",
        "swap_driver": 4,  # NOR
        "undercut": {"attacker": 4, "defender": 16, "current_lap": 13, "attacker_pit_lap": 14, "new_compound": "HARD", "horizon_lap": 25},
    },
    {
        "session_key": 9928,
        "label": "2025 Hungarian Grand Prix (Hungaroring)",
        "swap_driver": 81,  # PIA
        "undercut": {"attacker": 81, "defender": 16, "current_lap": 17, "attacker_pit_lap": 18, "new_compound": "HARD", "horizon_lap": 30},
    },
]


def actual_classification(race, drivers: list[int]) -> list[int]:
    return sorted(drivers, key=lambda d: (-driver_last_lap(race, d), actual_total_time(race, d)))


def build_race(config: dict) -> dict:
    session_key = config["session_key"]
    race = load_race(session_key)
    drivers_df = race.drivers.drop_duplicates("driver_number").set_index("driver_number")
    acronym = drivers_df["name_acronym"]
    team = drivers_df["team_name"]
    color = drivers_df["team_colour"]

    model = fit_degradation_model(race)
    pit_loss = fit_pit_loss(race)
    traffic = fit_traffic_penalty(race, model)
    race_laps = int(race.laps["lap_number"].max())

    driver_info = {
        int(d): {"acronym": acronym[d], "team": team[d], "color": f"#{color[d]}"} for d in acronym.index
    }

    # --- degradation + lap scatter -----------------------------------
    clean = laps_with_stint_info(race)
    clean = clean[(clean["lap_number"] > 1) & ~clean["is_pit_out_lap"] & ~clean["is_pit_in_lap"]]
    clean = clean.dropna(subset=["lap_duration"])
    lap_scatter = [
        {
            "driver": int(r["driver_number"]),
            "lap": int(r["lap_number"]),
            "compound": r["compound"],
            "tyre_age": int(r["tyre_age"]),
            "lap_duration": round(float(r["lap_duration"]), 3),
        }
        for _, r in clean.iterrows()
    ]

    degradation = {
        "fuel_slope": round(model.fuel_slope, 4),
        "deg_rate": {c: round(v, 4) for c, v in model.deg_rate.items()},
        "excluded_compounds": model.excluded_compounds,
        "driver_pace": {int(d): round(v, 3) for d, v in model.driver_pace.items()},
    }

    # --- pit loss ------------------------------------------------------
    pit_rows = race.pit.dropna(subset=["lane_duration"])
    outlier_keys = set(zip(pit_loss.outliers["driver_number"], pit_loss.outliers["lap_number"]))
    pit_stops = [
        {
            "driver": int(r["driver_number"]),
            "lap": int(r["lap_number"]),
            "lane_duration": round(float(r["lane_duration"]), 2),
            "outlier": (int(r["driver_number"]), int(r["lap_number"])) in outlier_keys,
        }
        for _, r in pit_rows.iterrows()
    ]

    # --- strategy sweep --------------------------------------------------
    compounds = sorted(model.deg_rate, key=lambda c: model.deg_rate[c])
    low_deg, high_deg = compounds[0], compounds[-1]
    avg_pace = sum(model.driver_pace.values()) / len(model.driver_pace)
    one_stop_curve = []
    for stop_lap in range(1, race_laps):
        plan = [(high_deg, stop_lap), (low_deg, race_laps - stop_lap)]
        r = simulate_strategy(model, pit_loss, plan, avg_pace, race_laps)
        one_stop_curve.append({"stop_lap": stop_lap, "total_time": round(r.total_race_time, 1)})
    best_one = best_one_stop(model, pit_loss, (high_deg, low_deg), avg_pace, race_laps)
    best_two = None
    for first in range(1, race_laps - 1):
        for second in range(first + 1, race_laps):
            plan = [(high_deg, first), (low_deg, second - first), (low_deg, race_laps - second)]
            r = simulate_strategy(model, pit_loss, plan, avg_pace, race_laps)
            if best_two is None or r.total_race_time < best_two.total_race_time:
                best_two = r

    strategy = {
        "low_deg_compound": low_deg,
        "high_deg_compound": high_deg,
        "one_stop_curve": one_stop_curve,
        "best_one_stop": {"stop_lap": best_one.plan[0][1], "total_time": round(best_one.total_race_time, 1)},
        "best_two_stop": {
            "stop_laps": [best_two.plan[0][1], best_two.plan[0][1] + best_two.plan[1][1]],
            "total_time": round(best_two.total_race_time, 1),
        },
    }

    # --- traffic curve ---------------------------------------------------
    traffic_curve = [
        {
            "gap_lo": GAP_BUCKETS[i],
            "gap_hi": GAP_BUCKETS[i + 1] if GAP_BUCKETS[i + 1] != float("inf") else None,
            "penalty": round(traffic.penalty((GAP_BUCKETS[i] + min(GAP_BUCKETS[i + 1], GAP_BUCKETS[i] + 1)) / 2), 3),
            "n": traffic.n_laps_by_bucket[i],
        }
        for i in range(len(GAP_BUCKETS) - 1)
    ]

    # --- full-grid simulation vs actual -----------------------------------
    all_drivers = [d for d in all_driver_numbers(race) if d in model.driver_pace]
    plans = {d: driver_plan(race, d) for d in all_drivers}
    plans = {d: p for d, p in plans.items() if p}
    grid = starting_grid_order(race)
    sim = simulate_race(model, pit_loss, traffic, plans, grid)
    actual_order = actual_classification(race, list(plans))

    winner_laps = sim.laps_completed[sim.finishing_order[0]]
    winner_actual_time = actual_total_time(race, actual_order[0])
    actual_pos = {d: i + 1 for i, d in enumerate(actual_order)}
    sim_pos = {d: i + 1 for i, d in enumerate(sim.finishing_order)}

    standings = []
    for i, d in enumerate(sim.finishing_order):
        dnf = sim.laps_completed[d] < winner_laps
        standings.append(
            {
                "driver": int(d),
                "sim_pos": i + 1,
                "actual_pos": actual_pos[d],
                "dnf": dnf,
                "sim_gap": None if dnf else round(sim.gap_to_winner(d), 1),
                "actual_gap": None if dnf else round(actual_total_time(race, d) - winner_actual_time, 1),
                "laps_completed": sim.laps_completed[d],
            }
        )
    mean_abs_error = sum(abs(s["sim_pos"] - s["actual_pos"]) for s in standings) / len(standings)

    # --- gap-to-leader chart (top finishers) ------------------------------
    top_drivers = [d for d in actual_order if not next(s["dnf"] for s in standings if s["driver"] == d)][
        :TOP_N_FOR_GAP_CHART
    ]
    gaps = laps_with_gap_to_car_ahead(race)
    leader = actual_order[0]
    actual_gap_by_driver_lap: dict[int, dict[int, float]] = {}
    for d in top_drivers:
        rows = gaps[gaps["driver_number"] == d][["lap_number", "gap_to_leader"]].dropna()
        actual_gap_by_driver_lap[d] = {int(r["lap_number"]): float(r["gap_to_leader"]) for _, r in rows.iterrows()}

    sim_leader_cumulative = {snap.lap_number: snap.cumulative_time[leader] for snap in sim.history}
    gap_chart = {}
    for d in top_drivers:
        series = []
        for snap in sim.history:
            if d not in snap.cumulative_time:
                continue
            sim_gap = snap.cumulative_time[d] - sim_leader_cumulative[snap.lap_number]
            actual_gap = actual_gap_by_driver_lap.get(d, {}).get(snap.lap_number)
            series.append({"lap": snap.lap_number, "sim_gap": round(sim_gap, 2), "actual_gap": actual_gap})
        gap_chart[int(d)] = series

    # --- swap-driver scenario ---------------------------------------------
    swap_driver = config["swap_driver"]
    real_stops = num_stops(race, swap_driver)
    swap_race_laps = sum(l for _, l in plans[swap_driver])
    if real_stops == 1:
        best_alt = None
        for first in range(1, swap_race_laps - 1):
            for second in range(first + 1, swap_race_laps):
                plan = [(high_deg, first), (low_deg, second - first), (low_deg, swap_race_laps - second)]
                r = simulate_strategy(model, pit_loss, plan, model.driver_pace[swap_driver], swap_race_laps)
                if best_alt is None or r.total_race_time < best_alt.total_race_time:
                    best_alt = r
        alt_plan, alt_label = best_alt.plan, "best 2-stop"
    else:
        alt = best_one_stop(model, pit_loss, (high_deg, low_deg), model.driver_pace[swap_driver], swap_race_laps)
        alt_plan, alt_label = alt.plan, "best 1-stop"

    swapped_plans = dict(plans)
    swapped_plans[swap_driver] = alt_plan
    swapped_sim = simulate_race(model, pit_loss, traffic, swapped_plans, grid)
    swapped_pos = {d: i + 1 for i, d in enumerate(swapped_sim.finishing_order)}[swap_driver]

    swap_scenario = {
        "driver": int(swap_driver),
        "real_plan": plans[swap_driver],
        "real_stops": real_stops,
        "alt_plan": alt_plan,
        "alt_label": alt_label,
        "real_pos": sim_pos[swap_driver],
        "real_gap": round(sim.gap_to_winner(swap_driver), 1),
        "alt_pos": swapped_pos,
        "alt_gap": round(swapped_sim.gap_to_winner(swap_driver), 1),
    }

    # --- undercut scenario --------------------------------------------------
    uc = config["undercut"]
    laps_stint = laps_with_stint_info(race)

    def state_at(driver, lap):
        row = laps_stint[(laps_stint["driver_number"] == driver) & (laps_stint["lap_number"] == lap)].iloc[0]
        return TyreState(compound=row["compound"], tyre_age=int(row["tyre_age"]))

    attacker_state = state_at(uc["attacker"], uc["current_lap"])
    defender_state = state_at(uc["defender"], uc["current_lap"])
    gap_now = (
        race.laps[(race.laps["driver_number"] == uc["attacker"]) & (race.laps["lap_number"] <= uc["current_lap"])]["lap_duration"].sum()
        - race.laps[(race.laps["driver_number"] == uc["defender"]) & (race.laps["lap_number"] <= uc["current_lap"])]["lap_duration"].sum()
    )
    # Defender's real next pit lap after current_lap, walked from their
    # actual stint plan (a stint boundary is the lap before the next
    # stint's lap_start).
    defender_pit_lap = uc["attacker_pit_lap"] + 1
    lap_cursor = 0
    for _, stint_laps in plans[uc["defender"]]:
        lap_cursor += stint_laps
        if lap_cursor > uc["current_lap"]:
            defender_pit_lap = lap_cursor
            break

    undercut_result = project_gap(
        model, pit_loss, uc["attacker"], uc["defender"], uc["current_lap"], gap_now,
        attacker_state, defender_state, uc["horizon_lap"],
        attacker_pit_lap=uc["attacker_pit_lap"], defender_pit_lap=defender_pit_lap, new_compound=uc["new_compound"],
    )
    undercut_scenario = {
        "attacker": int(uc["attacker"]),
        "defender": int(uc["defender"]),
        "current_lap": uc["current_lap"],
        "current_gap": round(gap_now, 2),
        "attacker_pit_lap": uc["attacker_pit_lap"],
        "defender_pit_lap": defender_pit_lap,
        "new_compound": uc["new_compound"],
        "gap_by_lap": {lap: round(g, 2) for lap, g in undercut_result.gap_by_lap.items()},
        "swing": round(undercut_result.swing, 2),
    }

    return {
        "session_key": session_key,
        "label": config["label"],
        "race_laps": race_laps,
        "drivers": driver_info,
        "degradation": degradation,
        "lap_scatter": lap_scatter,
        "pit_loss": {"typical_seconds": round(pit_loss.typical_seconds, 2), "stops": pit_stops},
        "strategy": strategy,
        "traffic_curve": traffic_curve,
        "simulation": {"mean_abs_position_error": round(mean_abs_error, 2), "standings": standings},
        "gap_chart": gap_chart,
        "leader": int(leader),
        "swap_scenario": swap_scenario,
        "undercut_scenario": undercut_scenario,
    }


def main() -> None:
    data = {"races": [build_race(config) for config in RACES]}
    OUTPUT_PATH.write_text(json.dumps(data))
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"Wrote {OUTPUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
