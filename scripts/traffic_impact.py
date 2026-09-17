#!/usr/bin/env python3
"""Show what the traffic model adds: the same "pit stop drops you into a
train of cars" scenario, evaluated with and without the fitted traffic
penalty, and how much that shifts the optimal strategy.

Usage:
    python scripts/traffic_impact.py --session-key 9590   # 2024 Monza (easy to follow)
    python scripts/traffic_impact.py --session-key 9928   # 2025 Hungary (hard to follow)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pitwall.degradation import fit_degradation_model
from pitwall.pit_loss import fit_pit_loss
from pitwall.race_data import load_race
from pitwall.strategy import best_one_stop
from pitwall.traffic import GAP_BUCKETS, fit_traffic_penalty

GAP_AFTER_STOP = 0.8  # seconds behind the car ahead, assumed for a driver who rejoins in a train
LAPS_IN_TRAFFIC = 4  # how many laps it takes to clear that car


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-key", type=int, required=True)
    args = parser.parse_args()

    race = load_race(args.session_key)
    race_laps = int(race.laps["lap_number"].max())

    model = fit_degradation_model(race)
    pit_loss = fit_pit_loss(race)
    traffic = fit_traffic_penalty(race, model)

    print(f"=== session_key={args.session_key}, race_laps={race_laps} ===\n")
    print("=== Traffic penalty vs gap to car ahead (relative to clear air) ===")
    for i in range(len(GAP_BUCKETS) - 1):
        lo, hi = GAP_BUCKETS[i], GAP_BUCKETS[i + 1]
        n = traffic.n_laps_by_bucket[i]
        penalty = traffic.penalty((lo + min(hi, lo + 1)) / 2)  # representative point in the bucket
        print(f"  gap [{lo:>4}, {hi:>5}) s : n={n:3d}  penalty={penalty:+.3f}s/lap")

    compounds = sorted(model.deg_rate, key=lambda c: model.deg_rate[c])
    high_deg, low_deg = compounds[-1], compounds[0]
    avg_pace = sum(model.driver_pace.values()) / len(model.driver_pace)

    print(f"\n=== Best 1-stop ({high_deg} -> {low_deg}), average car ===")
    clear_air = best_one_stop(model, pit_loss, (high_deg, low_deg), avg_pace, race_laps)
    print(f"Clear air:  pit after lap {clear_air.plan[0][1]:2d} -> total {clear_air.total_race_time:.1f}s")

    in_traffic = best_one_stop(
        model,
        pit_loss,
        (high_deg, low_deg),
        avg_pace,
        race_laps,
        traffic=traffic,
        gap_after_stop=GAP_AFTER_STOP,
        laps_in_traffic=LAPS_IN_TRAFFIC,
    )
    print(
        f"Rejoins {GAP_AFTER_STOP}s behind a car for {LAPS_IN_TRAFFIC} laps after stopping: "
        f"pit after lap {in_traffic.plan[0][1]:2d} -> total {in_traffic.total_race_time:.1f}s "
        f"(+{in_traffic.traffic_cost:.2f}s from traffic)"
    )
    print(
        f"\nOptimal pit lap shifts by {in_traffic.plan[0][1] - clear_air.plan[0][1]:+d} lap(s); "
        f"total time cost of the same traffic scenario: {in_traffic.total_race_time - clear_air.total_race_time:+.2f}s "
        f"(most of it, {in_traffic.traffic_cost:.2f}s, is the traffic penalty itself -- the rest is from "
        f"the model re-optimising the stop lap around it)"
    )


if __name__ == "__main__":
    main()
