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
