#!/usr/bin/env python3
"""Pull Race sessions from OpenF1 and shortlist the ones run in dry weather.

A race is a "candidate dry race" when every weather sample recorded during
the session has rainfall == 0. Sessions with no weather data are reported
separately (unknown) rather than assumed dry.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pitwall.openf1 import get_race_sessions, get_weather

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "pitwall" / "data" / "candidate_dry_races.json"


def classify(session: dict, weather: list[dict]) -> dict:
    result = {
        "session_key": session["session_key"],
        "meeting_key": session["meeting_key"],
        "year": session["year"],
        "country_name": session["country_name"],
        "location": session["location"],
        "circuit_short_name": session["circuit_short_name"],
        "date_start": session["date_start"],
    }
    if not weather:
        result["status"] = "unknown"
        return result

    rainfall_samples = [w["rainfall"] for w in weather if w.get("rainfall") is not None]
    wet_samples = sum(1 for r in rainfall_samples if r > 0)
    result["weather_samples"] = len(rainfall_samples)
    result["wet_samples"] = wet_samples
    result["status"] = "dry" if wet_samples == 0 else "wet"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--year", type=int, help="Restrict to a single season")
    parser.add_argument("--sleep", type=float, default=0.2, help="Delay between requests (s)")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    sessions = [
        s
        for s in get_race_sessions()
        if not s.get("is_cancelled") and datetime.fromisoformat(s["date_end"]) <= now
    ]
    if args.year:
        sessions = [s for s in sessions if s["year"] == args.year]
    sessions.sort(key=lambda s: s["date_start"])

    results = []
    for session in sessions:
        weather = get_weather(session["session_key"])
        results.append(classify(session, weather))
        time.sleep(args.sleep)

    dry = [r for r in results if r["status"] == "dry"]
    wet = [r for r in results if r["status"] == "wet"]
    unknown = [r for r in results if r["status"] == "unknown"]

    print(f"Race sessions checked: {len(results)}")
    print(f"  dry:     {len(dry)}")
    print(f"  wet:     {len(wet)}")
    print(f"  unknown: {len(unknown)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} sessions to {args.output}")


if __name__ == "__main__":
    main()
