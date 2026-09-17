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

That version still only simulates one car against a gap profile you pick
by hand. `pitwall/race_simulator.py` is the real thing: every driver on
the grid, together, lap by lap. Each car's clear-air pace comes from the
degradation model; each lap it's charged the traffic model's penalty for
*last* lap's gap to whoever was ahead of it; the field is re-sorted by
cumulative time to get the next lap's gaps. Overtakes are never scripted
— if a car's pace advantage outweighs the penalty from being stuck behind,
the gap keeps closing lap after lap until it crosses zero and the sort
puts that car ahead next lap. Retirees drop out of the field (and stop
blocking anyone) after their last completed lap instead of being forced
to "finish."

```
python scripts/simulate_race.py --session-key 9590 --swap-driver 4    # 2024 Monza
python scripts/simulate_race.py --session-key 9928 --swap-driver 81   # 2025 Hungary
```

Feed it every driver's *real* strategy and it reconstructs the actual
race: mean |simulated finishing position − actual finishing position|
across the full 20-car grid was **0.70 at Monza** and **0.90 at Hungary**
— most positions exact, the rest off by one or two. `--swap-driver`
re-runs one driver on the model's alternate best strategy with the other
19 drivers' *real* strategies held fixed, and re-simulates the whole
field: at Monza, swapping Norris from his real 2-stop to the model's best
1-stop flips him from P2 to the win — a data-grounded reconstruction of
why Monza 2024 actually came down to a 1-stop-vs-2-stop strategy call.
At Hungary, swapping Piastri from his real 2-stop to a 1-stop makes him
*slower* (P2 either way, but a bigger gap) — the model isn't just biased
toward "fewer stops is always better," it can and does tell you a real
strategy was already the right call.

Simplifications worth knowing about: lap-1 gaps are seeded from the real
starting grid (OpenF1's `position` endpoint) but at zero time separation
("perfect start," no reaction times or first-corner incidents); the
traffic penalty is a function of gap only, with no explicit DRS/pass
success-or-fail mechanic, so a big enough pace advantage always
eventually gets through given enough laps.

### Undercut / overcut calculator

The actual pit-wall question isn't "what's optimal in the abstract," it's
"we're 1.0s behind them right now, on these tyres — if we pit this lap and
they respond in N laps, do we come out ahead?" `pitwall/undercut.py`
answers exactly that: given two drivers' current gap and tyre state, it
projects the gap lap by lap for a chosen pit-lap pair, or searches for the
defender's latest safe response lap. This runs in isolation (just the two
cars' fitted pace/degradation/pit-loss, clear air) — the same simplified
picture a strategist sketches on a whiteboard.

```
python scripts/undercut_calculator.py --session-key 9590 \
    --attacker 4 --defender 16 --current-lap 13 \
    --attacker-pit-lap 14 --new-compound HARD --horizon-lap 25
```

reads the two drivers' *real* gap and tyre age from the race, projects
the undercut, and then cross-checks the same scenario against the full
20-car `race_simulator` (the other 18 drivers' real strategies included)
to see whether real traffic changes the answer. Run against the actual
Monza 2024 Norris-undercuts-Leclerc lap: both the isolated calculator and
the full-grid check agree the undercut works in the short term — but
grafting in Norris's real, further second stop shows the gap flipping
back in Leclerc's favor by the finish (+2.9s), matching the real result.
The same calculator on Hungary 2025 (Piastri undercutting Leclerc at lap
18) shows the advantage holding and growing to a real 35.6s by the end —
also matching what actually happened. The two cross-checks (isolated vs.
full-grid) routinely disagree in *magnitude*, which is the point: it's
usually the other 18 cars' traffic making up the difference.
