#!/usr/bin/env python3
"""Break ties among the top-scoring races from score_ideal_race.py.

score_ideal_race.py filters on cleanliness (no red flags, usable pit data,
full distance run) and scores on disruption (SC/VSC) and circuit type, but
several races tie once a session is clean and permanent. The tiebreaker
here is criterion (8) from that script's docstring, made concrete:

  strategy_balance -> how evenly the field split between a 1-stop and a
      2-stop race (min/max of the two counts). A race where the split is
      lopsided (e.g. 19 cars on the same plan) gives a strategy tool little
      to compare; a race split closer to 50/50 means both plans were
      genuinely competitive -- exactly the case a pit wall tool needs to
      reason about first.
  compounds_used -> how many of the three dry compounds (soft/medium/hard)
      actually appeared in a race stint. More compounds means more tyre
      degradation data points to validate a model against.

Both come from the /stints endpoint, which score_ideal_race.py doesn't
call (it only needed pit-stop counts, not per-driver strategy shape).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pitwall.openf1 import get_stints

RANKING_PATH = Path(__file__).resolve().parent.parent / "pitwall" / "data" / "ideal_race_ranking.json"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "pitwall" / "data" / "ideal_race_pick.json"


def strategy_shape(stints: list[dict]) -> dict:
    stops_by_driver: dict[int, int] = {}
    compounds: set[str] = set()
    for s in stints:
        driver = s["driver_number"]
        stops_by_driver[driver] = max(stops_by_driver.get(driver, 0), s["stint_number"] - 1)
        compounds.add(s["compound"])

    one_stop = sum(1 for v in stops_by_driver.values() if v == 1)
    two_stop = sum(1 for v in stops_by_driver.values() if v == 2)
    balance = min(one_stop, two_stop) / max(one_stop, two_stop) if max(one_stop, two_stop) else 0.0

    return {
        "one_stop_drivers": one_stop,
        "two_stop_drivers": two_stop,
        "compounds_used": sorted(compounds),
        "strategy_balance": round(balance, 3),
    }


def main() -> None:
    ranking = json.load(RANKING_PATH.open())
    eligible = ranking["eligible"]
    top_score = max(r["score"] for r in eligible)
    tied = [r for r in eligible if r["score"] == top_score]

    print(f"{len(tied)} races tied at score={top_score:+.1f}; breaking the tie on strategy shape.\n")

    for r in tied:
        stints = get_stints(r["session_key"])
        time.sleep(0.15)
        shape = strategy_shape(stints)
        r.update(shape)
        r["diversity_score"] = len(shape["compounds_used"]) + 3 * shape["strategy_balance"]
        print(
            f"  {r['year']} {r['country_name']:<16} {r['circuit_short_name']:<18} "
            f"1-stop={shape['one_stop_drivers']:2d} 2-stop={shape['two_stop_drivers']:2d} "
            f"balance={shape['strategy_balance']:.2f} compounds={shape['compounds_used']} "
            f"diversity={r['diversity_score']:.2f}"
        )

    tied.sort(key=lambda r: r["diversity_score"], reverse=True)
    pick = tied[0]

    OUTPUT_PATH.write_text(json.dumps({"tied_candidates": tied, "pick": pick}, indent=2))

    print(f"\nPicked: {pick['year']} {pick['country_name']} ({pick['circuit_short_name']}), "
          f"session_key={pick['session_key']}")
    print(f"Written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
