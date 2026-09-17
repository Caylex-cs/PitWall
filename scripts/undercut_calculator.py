#!/usr/bin/env python3
"""The pit-wall question: "we're X seconds behind them right now -- if we
pit this lap, do we come out ahead, and how long can they wait before
responding?" Reads the two drivers' real current gap and tyre state from
the race, projects the undercut/overcut with pitwall.undercut, and then
cross-checks the specific scenario against the full-grid simulator (the
other 18 real strategies included) to see if traffic changes the answer.

Usage (the real Monza 2024 case this race actually turned on -- Norris
pits lap 14 to try to jump Leclerc, who stays out to lap 15):
    python scripts/undercut_calculator.py --session-key 9590 \\
        --attacker 4 --defender 16 --current-lap 13 \\
        --attacker-pit-lap 14 --new-compound HARD --horizon-lap 25
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
    driver_plan,
    laps_with_stint_info,
    load_race,
    starting_grid_order,
)
from pitwall.race_simulator import simulate_race
from pitwall.traffic import fit_traffic_penalty
from pitwall.undercut import TyreState, latest_safe_response_lap, project_gap


def current_state(race, driver: int, lap: int) -> TyreState:
    laps = laps_with_stint_info(race)
    row = laps[(laps["driver_number"] == driver) & (laps["lap_number"] == lap)].iloc[0]
    return TyreState(compound=row["compound"], tyre_age=int(row["tyre_age"]))


def cumulative_time_through(race, driver: int, lap: int) -> float:
    laps = race.laps
    return laps[(laps["driver_number"] == driver) & (laps["lap_number"] <= lap)]["lap_duration"].sum()


def rebuild_plan_with_new_stop(plan: list[tuple[str, int]], pit_lap: int, new_compound: str) -> list[tuple[str, int]]:
    """Truncate `plan` at `pit_lap` and continue on `new_compound` for the
    same total distance -- used to graft a hypothetical pit lap onto a
    driver's real plan for the full-grid cross-check."""
    total_laps = sum(laps for _, laps in plan)
    return [(plan[0][0], pit_lap), (new_compound, total_laps - pit_lap)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session-key", type=int, required=True)
    parser.add_argument("--attacker", type=int, required=True, help="driver trying to gain the position")
    parser.add_argument("--defender", type=int, required=True, help="driver currently ahead")
    parser.add_argument("--current-lap", type=int, required=True)
    parser.add_argument("--attacker-pit-lap", type=int, required=True)
    parser.add_argument("--defender-pit-lap", type=int, help="if omitted, searches for the latest safe response")
    parser.add_argument("--new-compound", required=True)
    parser.add_argument("--horizon-lap", type=int, required=True)
    args = parser.parse_args()

    race = load_race(args.session_key)
    acronym = race.drivers.drop_duplicates("driver_number").set_index("driver_number")["name_acronym"]

    model = fit_degradation_model(race)
    pit_loss = fit_pit_loss(race)

    attacker_state = current_state(race, args.attacker, args.current_lap)
    defender_state = current_state(race, args.defender, args.current_lap)
    current_gap = cumulative_time_through(race, args.attacker, args.current_lap) - cumulative_time_through(
        race, args.defender, args.current_lap
    )

    print(f"=== Lap {args.current_lap}: {acronym[args.attacker]} vs {acronym[args.defender]} ===")
    print(f"Current gap: {current_gap:+.2f}s ({'attacker behind' if current_gap > 0 else 'attacker ahead'})")
    print(f"{acronym[args.attacker]} on {attacker_state.compound} (age {attacker_state.tyre_age})")
    print(f"{acronym[args.defender]} on {defender_state.compound} (age {defender_state.tyre_age})")

    if args.defender_pit_lap:
        result = project_gap(
            model, pit_loss, args.attacker, args.defender, args.current_lap, current_gap,
            attacker_state, defender_state, args.horizon_lap,
            attacker_pit_lap=args.attacker_pit_lap, defender_pit_lap=args.defender_pit_lap,
            new_compound=args.new_compound,
        )
        print(f"\n{acronym[args.attacker]} pits lap {args.attacker_pit_lap}, "
              f"{acronym[args.defender]} responds lap {args.defender_pit_lap}")
        for lap in sorted(result.gap_by_lap):
            print(f"  lap {lap:3d}: gap {result.gap_by_lap[lap]:+7.2f}s")
        print(f"\nSwing by lap {args.horizon_lap}: {result.swing:+.2f}s "
              f"({'undercut works' if result.attacker_now_ahead else 'undercut falls short'})")
    else:
        safe_lap = latest_safe_response_lap(
            model, pit_loss, args.attacker, args.defender, args.current_lap, current_gap,
            attacker_state, defender_state, args.attacker_pit_lap, args.new_compound, args.horizon_lap,
        )
        print(f"\n{acronym[args.attacker]} pits lap {args.attacker_pit_lap}; "
              f"searching {acronym[args.defender]}'s latest safe response by lap {args.horizon_lap}...")
        if safe_lap is None:
            print(f"No response lap keeps {acronym[args.defender]} ahead -- the undercut wins outright.")
        else:
            print(f"{acronym[args.defender]} can wait until lap {safe_lap} and still be ahead at lap {args.horizon_lap}.")

    # Cross-check against the full grid: graft these exact pit laps onto
    # both drivers' real plans, hold everyone else's real plan fixed, and
    # re-simulate the whole race to see if traffic from the other 18 cars
    # changes the answer.
    defender_pit_lap = args.defender_pit_lap or (safe_lap + 1 if safe_lap else args.attacker_pit_lap + 1)
    real_plans = {d: driver_plan(race, d) for d in race.drivers["driver_number"].unique() if d in model.driver_pace}
    real_plans = {d: p for d, p in real_plans.items() if p}

    # Hypothetical: both drivers locked onto one more stint of `new_compound`
    # for the rest of the race after their (real or hypothetical) pit lap --
    # the classic two-compound undercut picture, evaluated with the other
    # 18 real strategies around them.
    hypothetical_plans = dict(real_plans)
    hypothetical_plans[args.attacker] = rebuild_plan_with_new_stop(
        real_plans[args.attacker], args.attacker_pit_lap, args.new_compound
    )
    hypothetical_plans[args.defender] = rebuild_plan_with_new_stop(
        real_plans[args.defender], defender_pit_lap, args.new_compound
    )

    traffic = fit_traffic_penalty(race, model)
    grid = starting_grid_order(race)

    def gap_at(sim, lap):
        for snapshot in sim.history:
            if snapshot.lap_number == lap:
                return snapshot.cumulative_time[args.attacker] - snapshot.cumulative_time[args.defender]
        return None

    hypothetical_sim = simulate_race(model, pit_loss, traffic, hypothetical_plans, grid)
    print(f"\nFull-grid cross-check, one-more-stint-of-{args.new_compound} hypothetical "
          f"(other {len(hypothetical_plans) - 2} real strategies included): "
          f"gap at lap {args.horizon_lap} = {gap_at(hypothetical_sim, args.horizon_lap):+.2f}s")

    real_sim = simulate_race(model, pit_loss, traffic, real_plans, grid)
    real_horizon = min(args.horizon_lap, real_sim.laps_completed[args.attacker], real_sim.laps_completed[args.defender])
    final_lap = min(real_sim.laps_completed[args.attacker], real_sim.laps_completed[args.defender])
    print(f"For comparison, with each driver's actual complete strategy (includes any further real stops): "
          f"gap at lap {real_horizon} = {gap_at(real_sim, real_horizon):+.2f}s, "
          f"gap at race end (lap {final_lap}) = {gap_at(real_sim, final_lap):+.2f}s")

    print(
        "\nNote: the lap-by-lap projection above ignores the other 18 cars (clear air); "
        "the full-grid numbers include real traffic, which is usually why they differ."
    )


if __name__ == "__main__":
    main()
