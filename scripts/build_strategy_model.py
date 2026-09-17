#!/usr/bin/env python3
"""Fit the degradation + pit-loss model for the chosen race (2024 Italian GP,
Monza, session_key=9590) and validate it: does it correctly predict that the
1-stop and 2-stop strategies actually used were close to a toss-up, matching
what really happened (a near-50/50 split across the field)?
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pitwall.degradation import fit_degradation_model
from pitwall.pit_loss import fit_pit_loss
from pitwall.race_data import load_race
from pitwall.strategy import best_one_stop, simulate_strategy

SESSION_KEY = 9590
RACE_LAPS = 53
LECLERC, NORRIS = 16, 4  # driver numbers


def main() -> None:
    race = load_race(SESSION_KEY)
    drivers = race.drivers.drop_duplicates("driver_number").set_index("driver_number")["name_acronym"]

    model = fit_degradation_model(race)
    pit_loss = fit_pit_loss(race)

    print("=== Degradation model ===")
    print(f"fuel_slope: {model.fuel_slope:+.4f} s/lap")
    for c, r in sorted(model.deg_rate.items()):
        print(f"  {c:8s} deg_rate: {r:+.4f} s/lap of tyre age")
    if model.excluded_compounds:
        print(f"  (excluded from fit, too few clean laps: {model.excluded_compounds})")

    print("\n=== Pit loss model ===")
    print(f"typical stop: {pit_loss.typical_seconds:.1f}s (median of {pit_loss.n_stops} stops, outliers excluded)")
    if len(pit_loss.outliers):
        print(f"outlier stops excluded from that estimate:\n{pit_loss.outliers.to_string(index=False)}")

    print("\n=== Actual strategies: Leclerc (1-stop) vs Norris (2-stop) ===")
    leclerc_plan = [("MEDIUM", 15), ("HARD", 38)]
    norris_plan = [("MEDIUM", 14), ("HARD", 18), ("HARD", 21)]

    leclerc_result = simulate_strategy(model, pit_loss, leclerc_plan, model.driver_pace[LECLERC], RACE_LAPS)
    norris_result = simulate_strategy(model, pit_loss, norris_plan, model.driver_pace[NORRIS], RACE_LAPS)

    print(f"Model-predicted LEC total race time: {leclerc_result.total_race_time:.1f}s ({leclerc_result.num_stops} stop)")
    print(f"Model-predicted NOR total race time: {norris_result.total_race_time:.1f}s ({norris_result.num_stops} stops)")
    print(f"Model-predicted gap: {norris_result.total_race_time - leclerc_result.total_race_time:+.1f}s (NOR - LEC)")

    actual_leclerc_time = race.laps[race.laps["driver_number"] == LECLERC]["lap_duration"].sum()
    actual_norris_time = race.laps[race.laps["driver_number"] == NORRIS]["lap_duration"].sum()
    print(f"\nActual summed lap time  LEC: {actual_leclerc_time:.1f}s")
    print(f"Actual summed lap time  NOR: {actual_norris_time:.1f}s")
    print(f"Actual gap: {actual_norris_time - actual_leclerc_time:+.1f}s (NOR - LEC)")

    print("\n=== Field-average car: best 1-stop vs best 2-stop (Medium -> Hard[, Hard]) ===")
    avg_pace = sum(model.driver_pace.values()) / len(model.driver_pace)
    one_stop = best_one_stop(model, pit_loss, ("MEDIUM", "HARD"), avg_pace, RACE_LAPS)
    print(f"Best 1-stop: pit after lap {one_stop.plan[0][1]} -> total {one_stop.total_race_time:.1f}s")

    best_two_stop = None
    for first_stop in range(1, RACE_LAPS - 1):
        for second_stop in range(first_stop + 1, RACE_LAPS):
            plan = [
                ("MEDIUM", first_stop),
                ("HARD", second_stop - first_stop),
                ("HARD", RACE_LAPS - second_stop),
            ]
            result = simulate_strategy(model, pit_loss, plan, avg_pace, RACE_LAPS)
            if best_two_stop is None or result.total_race_time < best_two_stop.total_race_time:
                best_two_stop = result
    print(
        f"Best 2-stop: pit after laps {best_two_stop.plan[0][1]} and "
        f"{best_two_stop.plan[0][1] + best_two_stop.plan[1][1]} -> total {best_two_stop.total_race_time:.1f}s"
    )
    print(f"Model gap between best strategies: {best_two_stop.total_race_time - one_stop.total_race_time:+.1f}s (2-stop - 1-stop)")


if __name__ == "__main__":
    main()
