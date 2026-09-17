# Pit Wall

## OpenF1 data pulls

`pitwall/openf1.py` is a small client for the [OpenF1](https://openf1.org) API.

`scripts/fetch_candidate_dry_races.py` pulls every completed Race session and
classifies it as `dry`, `wet`, or `unknown` based on the session's weather
samples (`wet` = at least one sample with `rainfall > 0`). Results are written
to `pitwall/data/candidate_dry_races.json`.

```
pip install -r requirements.txt
python scripts/fetch_candidate_dry_races.py [--year 2024] [--output path.json]
```

`scripts/score_ideal_race.py` and `scripts/pick_ideal_race.py` rank those dry
candidates against an "ideal first race" profile (no red flags, minimal
Safety Car/VSC, permanent circuit, full distance, usable pit data, a real
strategic split in the field) and picked the **2024 Italian GP at Monza**
(`session_key=9590`) to build the strategy model against.

## Strategy model

- `pitwall/race_data.py` — loads and caches (under `pitwall/data/races/<key>/`)
  a session's laps, stints, pit stops, drivers, race control and weather.
- `pitwall/degradation.py` — fits lap time as
  `driver_pace[driver] + fuel_slope * lap_number + deg_rate[compound] * tyre_age`
  via OLS across the whole field, so tyre degradation is separated from fuel
  burn-off (which move together for any one driver, but not across drivers
  who pit on different laps).
- `pitwall/pit_loss.py` — the typical pit stop time cost, from this race's
  own pit stops (outlier stops excluded via IQR).
- `pitwall/strategy.py` — simulates total race time for a given stint plan,
  and sweeps for the best 1-stop / 2-stop split.

```
python scripts/build_strategy_model.py --session-key 9590   # 2024 Monza
python scripts/build_strategy_model.py --session-key 9928   # 2025 Hungary
```

fits the model for a race and validates it against what actually happened:
for both Monza 2024 and Hungary 2025 it correctly predicts the real 1-stop
vs 2-stop gap to within a few seconds over the full race distance, and its
best-1-stop-vs-best-2-stop sweep for an average car comes out close to a
toss-up — consistent with both races' fields actually splitting close to
50/50 between the two plans (which is why they were picked).

Two known limits, both deliberately surfaced by stress-testing on more
races rather than papered over:
- **No track-evolution term.** Fit against a mixed wet/dry race
  (`--session-key 9558`, 2024 British GP) and the degradation rates come
  out negative — the drying track swamps real tyre wear. This model should
  only be trusted on stable, dry-condition races.
- **No traffic model, on its own.** See below.

### Traffic / overtaking difficulty

`pitwall/traffic.py` fits how much pace a lap loses running close behind
another car: it takes each lap's leftover residual against the
degradation model (actual time minus predicted clear-air time) and bins it
by `interval` (OpenF1's timed gap to the car ahead), so it captures
circuit-specific overtaking difficulty rather than assuming one universal
"dirty air" cost.

`pitwall/strategy.py`'s `simulate_strategy`/`best_one_stop` now take an
optional `traffic` model and a per-lap gap profile (`traffic_profile_after_stops`
builds one for "rejoins behind another car for N laps after every stop").

```
python scripts/traffic_impact.py --session-key 9590   # Monza: fast, easy to follow
python scripts/traffic_impact.py --session-key 9928   # Hungary: tight, hard to follow
```

fits the traffic penalty and prices the same scenario (rejoining 0.8s
behind a car for 4 laps after a stop) at both circuits: it costs ~0.8s at
Monza but ~1.9s at Hungary — the model correctly prices Hungary as harder
to pass through, matching its real-world reputation, using nothing but
each race's own timing data.
