#!/usr/bin/env python3
"""For every driver, sweep every possible single pit lap (and every fitted
tyre compound to switch onto), holding the other 19 drivers' *real*
strategies fixed, and re-simulate the full grid each time.

This is the data behind the dashboard's "play with a driver's strategy"
panel: "if everything else stays the same, how does changing when (and
onto what) this driver pits change their result?" Each point is an exact
full-grid simulation, not an approximation -- there are only a few
thousand combinations per race and each simulate_race call takes a few
milliseconds, so brute force is fast enough to precompute everything the
UI could ask for and ship it as data (a static artifact has no backend to
call back into for a fresh simulation).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pitwall.degradation import fit_degradation_model
from pitwall.pit_loss import fit_pit_loss
from pitwall.race_data import all_driver_numbers, driver_plan, load_race, starting_grid_order
from pitwall.race_simulator import simulate_race
from pitwall.traffic import fit_traffic_penalty

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "pitwall" / "data" / "pitstop_sweep.json"
SESSION_KEYS = [9590, 9928]


def classify(sim, real_total_time, actual_order):
    """Position (laps-completed first, then time) and gap to the winner
    for every driver in a simulated race, using the same rule as the rest
    of the pipeline so results line up with the dashboard's other panels."""
    winner = sim.finishing_order[0]
    winner_time = sim.total_time[winner]
    winner_laps = sim.laps_completed[winner]
    result = {}
    for i, d in enumerate(sim.finishing_order):
        dnf = sim.laps_completed[d] < winner_laps
        result[d] = {
            "position": i + 1,
            "gap": None if dnf else round(sim.total_time[d] - winner_time, 2),
            "laps_down": (winner_laps - sim.laps_completed[d]) if dnf else 0,
        }
    return result


def build_race(session_key: int) -> dict:
    race = load_race(session_key)
    acronym = race.drivers.drop_duplicates("driver_number").set_index("driver_number")["name_acronym"]

    model = fit_degradation_model(race)
    pit_loss = fit_pit_loss(race)
    traffic = fit_traffic_penalty(race, model)

    plans = {d: driver_plan(race, d) for d in all_driver_numbers(race) if d in model.driver_pace}
    plans = {d: p for d, p in plans.items() if p}
    grid = starting_grid_order(race)
    race_laps = max(sum(l for _, l in p) for p in plans.values())

    compounds = [c for c in model.deg_rate]  # excludes anything too thin to fit (see excluded_compounds)

    baseline_sim = simulate_race(model, pit_loss, traffic, plans, grid)
    real_result = classify(baseline_sim, None, None)

    drivers_meta = {}
    sweep = {}
    t0 = time.time()
    total_sims = 0
    for driver, plan in plans.items():
        driver = int(driver)
        first_compound = plan[0][0]
        race_laps_this_driver = sum(l for _, l in plan)
        drivers_meta[driver] = {
            "acronym": acronym[driver],
            "first_compound": first_compound,
            "real_stops": len(plan) - 1,
            "real_position": real_result[driver]["position"],
            "real_gap": real_result[driver]["gap"],
            "real_laps_down": real_result[driver]["laps_down"],
        }
        sweep[driver] = {}
        for compound in compounds:
            points = []
            for lap in range(2, race_laps_this_driver - 1):
                trial_plans = dict(plans)
                trial_plans[driver] = [(first_compound, lap), (compound, race_laps_this_driver - lap)]
                sim = simulate_race(model, pit_loss, traffic, trial_plans, grid)
                total_sims += 1
                result = classify(sim, None, None)[driver]
                points.append({"lap": lap, "position": result["position"], "gap": result["gap"]})
            sweep[driver][compound] = points

    print(f"  {total_sims} simulations in {time.time() - t0:.1f}s")

    return {
        "session_key": session_key,
        "race_laps": race_laps,
        "compounds": compounds,
        "drivers": drivers_meta,
        "sweep": {str(d): v for d, v in sweep.items()},
    }


def main() -> None:
    out = {}
    for key in SESSION_KEYS:
        print(f"session_key={key}")
        out[str(key)] = build_race(key)
    OUTPUT_PATH.write_text(json.dumps(out))
    print(f"Wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
