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
python scripts/build_strategy_model.py
```

fits the model for Monza 2024 and validates it against what actually
happened: it correctly predicts Leclerc's 1-stop beating Norris's 2-stop
(matching the real result), and finds the best 1-stop and 2-stop strategies
for an average car are within ~10 seconds of each other over 53 laps —
consistent with the field actually splitting close to 50/50 between the two
plans.
