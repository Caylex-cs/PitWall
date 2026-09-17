#!/usr/bin/env python3
"""Fit the degradation + pit-loss model for a race and validate it: does it
correctly predict the real gap between a 1-stop and a 2-stop strategy, and
does the best-vs-best 1-stop/2-stop margin match how the field actually
split between the two plans?

Usage:
    python scripts/build_strategy_model.py --session-key 9590   # 2024 Monza
    python scripts/build_strategy_model.py --session-key 9928   # 2025 Hungary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pitwall.degradation import fit_degradation_model
from pitwall.pit_loss import fit_pit_loss
from pitwall.race_data import driver_plan, load_race, num_stops
from pitwall.strategy import best_one_stop, simulate_strategy


def pick_representative(race, model, target_stops: int) -> int | None:
    """Among drivers who made exactly `target_stops` stops, the one with the
    most laps run (so their plan spans close to the full race, not a driver
    who retired early with a coincidentally matching stop count)."""
    candidates = [d for d in model.driver_pace if num_stops(race, d) == target_stops]
    if not candidates:
        return None
    laps_run = race.laps.groupby("driver_number")["lap_number"].max()
    return max(candidates, key=lambda d: laps_run.get(d, 0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-key", type=int, required=True)
    args = parser.parse_args()

    race = load_race(args.session_key)
    drivers = race.drivers.drop_duplicates("driver_number").set_index("driver_number")["name_acronym"]
    race_laps = int(race.laps["lap_number"].max())

    print(f"=== session_key={args.session_key}, race_laps={race_laps} ===\n")

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

    one_stop_driver = pick_representative(race, model, target_stops=1)
    two_stop_driver = pick_representative(race, model, target_stops=2)

    if one_stop_driver is None or two_stop_driver is None:
        print("\nNo clean 1-stop vs 2-stop pair to compare in this race; skipping validation section.")
        return

    one_plan = driver_plan(race, one_stop_driver)
    two_plan = driver_plan(race, two_stop_driver)

    print(f"\n=== Actual strategies: {drivers[one_stop_driver]} ({len(one_plan) - 1}-stop) "
          f"vs {drivers[two_stop_driver]} ({len(two_plan) - 1}-stop) ===")
    print(f"{drivers[one_stop_driver]} plan: {one_plan}")
    print(f"{drivers[two_stop_driver]} plan: {two_plan}")

    one_result = simulate_strategy(model, pit_loss, one_plan, model.driver_pace[one_stop_driver], race_laps)
    two_result = simulate_strategy(model, pit_loss, two_plan, model.driver_pace[two_stop_driver], race_laps)

    print(f"Model-predicted {drivers[one_stop_driver]} total race time: {one_result.total_race_time:.1f}s")
    print(f"Model-predicted {drivers[two_stop_driver]} total race time: {two_result.total_race_time:.1f}s")
    print(f"Model-predicted gap: {two_result.total_race_time - one_result.total_race_time:+.1f}s "
          f"({drivers[two_stop_driver]} - {drivers[one_stop_driver]})")

    actual_one = race.laps[race.laps["driver_number"] == one_stop_driver]["lap_duration"].sum()
    actual_two = race.laps[race.laps["driver_number"] == two_stop_driver]["lap_duration"].sum()
    print(f"\nActual summed lap time  {drivers[one_stop_driver]}: {actual_one:.1f}s")
    print(f"Actual summed lap time  {drivers[two_stop_driver]}: {actual_two:.1f}s")
    print(f"Actual gap: {actual_two - actual_one:+.1f}s ({drivers[two_stop_driver]} - {drivers[one_stop_driver]})")

    compounds = sorted(model.deg_rate, key=lambda c: model.deg_rate[c])  # least -> most degradation
    if len(compounds) < 2:
        print("\nNot enough distinct fitted compounds to run the field-average sweep.")
        return
    hard, medium = compounds[0], compounds[-1]

    print(f"\n=== Field-average car: best 1-stop vs best 2-stop ({medium} -> {hard}[, {hard}]) ===")
    avg_pace = sum(model.driver_pace.values()) / len(model.driver_pace)
    one_stop = best_one_stop(model, pit_loss, (medium, hard), avg_pace, race_laps)
    print(f"Best 1-stop: pit after lap {one_stop.plan[0][1]} -> total {one_stop.total_race_time:.1f}s")

    best_two_stop = None
    for first_stop in range(1, race_laps - 1):
        for second_stop in range(first_stop + 1, race_laps):
            plan = [(medium, first_stop), (hard, second_stop - first_stop), (hard, race_laps - second_stop)]
            result = simulate_strategy(model, pit_loss, plan, avg_pace, race_laps)
            if best_two_stop is None or result.total_race_time < best_two_stop.total_race_time:
                best_two_stop = result
    print(
        f"Best 2-stop: pit after laps {best_two_stop.plan[0][1]} and "
        f"{best_two_stop.plan[0][1] + best_two_stop.plan[1][1]} -> total {best_two_stop.total_race_time:.1f}s"
    )
    print(f"Model gap between best strategies: {best_two_stop.total_race_time - one_stop.total_race_time:+.1f}s "
          f"(2-stop - 1-stop)")


if __name__ == "__main__":
    main()
