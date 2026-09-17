"""Thin client for the OpenF1 REST API (https://openf1.org)."""

from __future__ import annotations

import requests

BASE_URL = "https://api.openf1.org/v1"
TIMEOUT = 30


def _get(path: str, **params) -> list[dict]:
    response = requests.get(f"{BASE_URL}/{path}", params=params, timeout=TIMEOUT)
    if response.status_code == 404:
        # OpenF1 returns 404 (rather than an empty list) when a session has no data yet.
        return []
    response.raise_for_status()
    return response.json()


def get_race_sessions() -> list[dict]:
    """All Race sessions across every season OpenF1 has data for."""
    return _get("sessions", session_type="Race")


def get_weather(session_key: int) -> list[dict]:
    """Weather samples (roughly one per minute) for a session."""
    return _get("weather", session_key=session_key)


def get_race_control(session_key: int) -> list[dict]:
    """Flags, safety car and red flag messages for a session."""
    return _get("race_control", session_key=session_key)


def get_pit(session_key: int) -> list[dict]:
    """Individual pit stop records for a session."""
    return _get("pit", session_key=session_key)


def get_laps(session_key: int) -> list[dict]:
    """Lap-by-lap timing records for a session."""
    return _get("laps", session_key=session_key)


def get_stints(session_key: int) -> list[dict]:
    """Per-driver tyre stint records (compound, stint length) for a session."""
    return _get("stints", session_key=session_key)


def get_drivers(session_key: int) -> list[dict]:
    """Driver/team info for a session."""
    return _get("drivers", session_key=session_key)


def get_intervals(session_key: int) -> list[dict]:
    """Time-series gap-to-car-ahead (`interval`) and gap-to-leader for a session."""
    return _get("intervals", session_key=session_key)


def get_position(session_key: int) -> list[dict]:
    """Time-series classification position per driver (one row per change)."""
    return _get("position", session_key=session_key)
