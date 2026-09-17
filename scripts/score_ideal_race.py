#!/usr/bin/env python3
"""Rank candidate dry races against an "ideal first race" profile.

Ideal profile for the race we build/validate the pit wall tool against:
  1. Dry throughout            -> no rain-driven strategy calls to model.
  2. A points race, not Sprint -> full grid, standard distance, normal stops.
  3. Zero red flags            -> no restarts / neutralised sectors to special-case.
  4. Few/no Safety Car or VSC  -> "free" pit stops under SC distort stint math;
                                   a green-flag race isolates pure strategy first.
  5. Permanent circuit         -> street tracks correlate with SC/red-flag risk;
                                   a permanent track is the more typical case.
  6. Full race distance run    -> no early termination / partial classification.
  7. Real pit-stop data exists -> OpenF1's pit endpoint has gaps early in 2023.
  8. Meaningful stop activity  -> enough pit stops to compare strategies, but not
                                   a chaotic high-degradation race.

Each candidate is hard-filtered on (2), (3), (6), (7), then scored on the rest.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pitwall.openf1 import get_laps, get_pit, get_race_control, get_race_sessions

CANDIDATES_PATH = Path(__file__).resolve().parent.parent / "pitwall" / "data" / "candidate_dry_races.json"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "pitwall" / "data" / "ideal_race_ranking.json"

# Street / temporary circuits: historically far more likely to bring out red
# flags and safety cars than permanent road courses, so they're penalised
# (not hard-excluded) as less "typical" of a baseline race.
STREET_CIRCUITS = {"Jeddah", "Baku", "Monaco", "Miami", "Las Vegas", "Singapore", "Montreal", "Melbourne"}


def analyze_race_control(messages: list[dict]) -> dict:
    sc = vsc = red = 0
    for m in messages:
        message = (m.get("message") or "").upper()
        if m.get("category") == "SafetyCar" and "DEPLOYED" in message:
            if "VIRTUAL" in message:
                vsc += 1
            else:
                sc += 1
        elif m.get("flag") == "RED" and message == "RED FLAG":
            red += 1
    return {"safety_cars": sc, "vsc": vsc, "red_flags": red}


def main() -> None:
    candidates = json.load(CANDIDATES_PATH.open())
    dry_keys = {c["session_key"] for c in candidates if c["status"] == "dry"}

    sessions = {s["session_key"]: s for s in get_race_sessions()}

    rows = []
    for key in sorted(dry_keys):
        session = sessions.get(key)
        if session is None or session["session_name"] != "Race":
            continue  # drop Sprint sessions, criterion (2)

        race_control = get_race_control(key)
        time.sleep(0.15)
        pit = get_pit(key)
        time.sleep(0.15)
        laps = get_laps(key)
        time.sleep(0.15)

        flags = analyze_race_control(race_control)
        pit_available = len(pit) > 0
        driver_laps = {}
        for lap in laps:
            driver_laps.setdefault(lap["driver_number"], 0)
            driver_laps[lap["driver_number"]] = max(driver_laps[lap["driver_number"]], lap["lap_number"] or 0)
        max_lap = max(driver_laps.values()) if driver_laps else 0
        # A driver classified far short of the race winner's lap count usually means
        # a red-flag-shortened race or a DNF-heavy session, not full green-flag running.
        finishers_near_full_distance = sum(1 for v in driver_laps.values() if v >= max_lap - 1)

        rows.append(
            {
                "session_key": key,
                "meeting_key": session["meeting_key"],
                "year": session["year"],
                "country_name": session["country_name"],
                "circuit_short_name": session["circuit_short_name"],
                "date_start": session["date_start"],
                "is_street_circuit": session["circuit_short_name"] in STREET_CIRCUITS,
                "safety_cars": flags["safety_cars"],
                "vsc": flags["vsc"],
                "red_flags": flags["red_flags"],
                "pit_data_available": pit_available,
                "total_pit_stops": len(pit),
                "max_lap": max_lap,
                "finishers_near_full_distance": finishers_near_full_distance,
            }
        )
        print(f"  {session['year']} {session['country_name']:<20} sc={flags['safety_cars']} vsc={flags['vsc']} red={flags['red_flags']} pit_stops={len(pit)}")

    # Hard filters: criteria (3), (6), (7).
    eligible = [
        r
        for r in rows
        if r["red_flags"] == 0
        and r["pit_data_available"]
        and r["finishers_near_full_distance"] >= 15  # full distance run for (most of) the field
    ]

    def score(r: dict) -> float:
        s = 0.0
        s -= 3 * r["safety_cars"]
        s -= 1 * r["vsc"]
        s -= 2 if r["is_street_circuit"] else 0
        # Reward a "normal" one/two-stop race: enough stops to compare strategies
        # (>=1 per car) without a tyre-degradation fiasco (>2.2 per car).
        stops_per_car = r["total_pit_stops"] / 20
        if 1.0 <= stops_per_car <= 2.2:
            s += 3
        s += 1 if r["year"] >= 2024 else 0  # slightly fresher data schema
        return s

    for r in eligible:
        r["score"] = score(r)
    eligible.sort(key=lambda r: r["score"], reverse=True)

    OUTPUT_PATH.write_text(json.dumps({"eligible": eligible, "all_checked": rows}, indent=2))

    print(f"\nChecked {len(rows)} dry, non-sprint races; {len(eligible)} passed the hard filters.")
    print(f"Ranking written to {OUTPUT_PATH}\n")
    print("Top 5:")
    for r in eligible[:5]:
        print(
            f"  score={r['score']:+.1f}  {r['year']} {r['country_name']:<20} "
            f"sc={r['safety_cars']} vsc={r['vsc']} street={r['is_street_circuit']} stops={r['total_pit_stops']}"
        )


if __name__ == "__main__":
    main()
