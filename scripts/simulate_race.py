#!/usr/bin/env python3
"""Simulate the whole grid together (not one isolated car) and check it
against what actually happened, then re-run one driver on a different
strategy to see how the *emergent* traffic from 19 other real strategies
changes their result -- not a hand-picked gap assumption.

Usage:
    python scripts/simulate_race.py --session-key 9590   # 2024 Monza
    python scripts/simulate_race.py --session-key 9928   # 2025 Hungary
"""

from __future__ import annotations

import argparse
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
    load_race,
    num_stops,
    starting_grid_order,
)
from pitwall.race_simulator import simulate_race
from pitwall.strategy import best_one_stop, simulate_strategy
from pitwall.traffic import fit_traffic_penalty


def actual_classification(race, drivers: list[int]) -> list[int]:
    """Real finishing order: most laps completed, then lowest total time --
    the standard motorsport rule, applied the same way the simulator's
    finishing_order is computed so the two are comparable."""
    return sorted(drivers, key=lambda d: (-driver_last_lap(race, d), actual_total_time(race, d)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-key", type=int, required=True)
    parser.add_argument("--swap-driver", type=int, help="Driver number to re-run on the opposite stop count")
    args = parser.parse_args()

    race = load_race(args.session_key)
    acronym = race.drivers.drop_duplicates("driver_number").set_index("driver_number")["name_acronym"]

    model = fit_degradation_model(race)
    pit_loss = fit_pit_loss(race)
    traffic = fit_traffic_penalty(race, model)

    all_drivers = [d for d in all_driver_numbers(race) if d in model.driver_pace]
    plans = {d: driver_plan(race, d) for d in all_drivers}
    plans = {d: plan for d, plan in plans.items() if plan}  # drop drivers with no usable stint at all
    grid = starting_grid_order(race)

    print(f"=== session_key={args.session_key}: simulating {len(plans)}/{len(all_drivers)} drivers on the grid ===\n")

    result = simulate_race(model, pit_loss, traffic, plans, grid)
    actual_order = actual_classification(race, list(plans))

    # A "gap to winner" from raw total-time subtraction is only meaningful
    # between cars that ran the same number of laps -- a retiree naturally
    # accumulates less total time just by running fewer laps, which is not
    # the same thing as being "ahead." Flag those rather than print a
    # misleadingly small (or negative) number.
    winner_laps = result.laps_completed[result.finishing_order[0]]
    winner_actual_time = actual_total_time(race, actual_order[0])

    print(f"{'pos':>3}  {'driver':6}  {'sim gap':>10}  {'actual gap':>10}  {'sim pos':>7}  {'actual pos':>10}")
    actual_pos = {d: i + 1 for i, d in enumerate(actual_order)}
    sim_pos = {d: i + 1 for i, d in enumerate(result.finishing_order)}
    for i, driver in enumerate(result.finishing_order):
        if result.laps_completed[driver] < winner_laps:
            sim_gap_str = f"-{winner_laps - result.laps_completed[driver]}lap"
            actual_gap_str = sim_gap_str
        else:
            sim_gap_str = f"{result.gap_to_winner(driver):.1f}s"
            actual_gap_str = f"{actual_total_time(race, driver) - winner_actual_time:.1f}s"
        print(
            f"{i + 1:>3}  {acronym.get(driver, driver):6}  {sim_gap_str:>10}  {actual_gap_str:>10}  "
            f"{sim_pos[driver]:>7}  {actual_pos[driver]:>10}"
        )

    position_errors = [abs(sim_pos[d] - actual_pos[d]) for d in plans]
    print(f"\nMean |simulated position - actual position|: {sum(position_errors) / len(position_errors):.2f}")

    if args.swap_driver:
        driver = args.swap_driver
        if driver not in plans:
            print(f"\ndriver {driver} has no usable plan in this race, skipping swap scenario.")
            return

        real_stops = num_stops(race, driver)
        compounds = sorted(model.deg_rate, key=lambda c: model.deg_rate[c])
        high_deg, low_deg = compounds[-1], compounds[0]
        race_laps = sum(stint_laps for _, stint_laps in plans[driver])

        if real_stops == 1:
            # Build the model's best 2-stop plan of the same total distance.
            best_two_stop = None
            for first_stop in range(1, race_laps - 1):
                for second_stop in range(first_stop + 1, race_laps):
                    candidate = [
                        (high_deg, first_stop),
                        (low_deg, second_stop - first_stop),
                        (low_deg, race_laps - second_stop),
                    ]
                    r = simulate_strategy(model, pit_loss, candidate, model.driver_pace[driver], race_laps)
                    if best_two_stop is None or r.total_race_time < best_two_stop.total_race_time:
                        best_two_stop = r
            alt_plan = best_two_stop.plan
            alt_label = "best 2-stop"
        else:
            alt_plan = best_one_stop(model, pit_loss, (high_deg, low_deg), model.driver_pace[driver], race_laps).plan
            alt_label = "best 1-stop"

        print(f"\n=== Swap {acronym.get(driver, driver)} from their real {real_stops}-stop to the model's {alt_label} ===")
        print(f"real plan: {plans[driver]}")
        print(f"alt plan:  {alt_plan}")

        swapped_plans = dict(plans)
        swapped_plans[driver] = alt_plan
        swapped_result = simulate_race(model, pit_loss, traffic, swapped_plans, grid)

        real_pos = sim_pos[driver]
        swapped_pos = {d: i + 1 for i, d in enumerate(swapped_result.finishing_order)}[driver]
        print(
            f"Finishing position with real strategy (all 19 others real too): P{real_pos}, "
            f"gap to winner {result.gap_to_winner(driver):.1f}s"
        )
        print(
            f"Finishing position with {alt_label} (everyone else unchanged): P{swapped_pos}, "
            f"gap to winner {swapped_result.gap_to_winner(driver):.1f}s"
        )


if __name__ == "__main__":
    main()
