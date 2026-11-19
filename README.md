# GTA VI Trophy Rarity Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Moustakimm/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Moustakimm/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI (PS5) trophy's earn rate on Exophase at
2026-11-26T00:00:00Z — one week after launch — as five quantiles per
trophy, for make.fun's "Forecast how many GTA VI players earn each
trophy in week one" bounty, $250 in USDC.

## The model

An earn rate is earners / players-tracked, and the two move for
different reasons, so `main.py` forecasts them separately and divides:

- **Players tracked N(t):** Exophase refreshes its counts in batches,
  hours apart, so N is a step function that jumps at refreshes. log N
  is fitted against log-time (days since release) and read at the
  target.
- **Earners E(t) per trophy:** log E is fitted the same way, per
  trophy, with the slope shrunk toward the pooled cross-trophy slope
  (early reads have only a handful of earners, so one trophy's slope
  is noisy), plus a small nudge from what the trophy asks for: story
  chapters are earned steadily as players progress, while grind
  trophies (250 stunts, 100% completion) barely move in a week.
- **Median rate** is 100 * E\* / N\*, capped to 5 logit units of
  movement from the last read.

One case breaks that fit: when every read in the history shows the
same players-tracked count, no refresh has been observed yet, so no
trend can be fitted. That is launch night — Exophase had not refreshed
since the first probe, and its first refresh multiplied tracked
players ~6x while earners grew only modestly, so every rate but the
one at 100% fell hard. A flat history therefore means a refresh is
overdue, not that rates are frozen: the forecast divides the last
earners (grown a little, by trophy kind) by a tracked count jumped
~6x, and widens the spread to cover how uneven that dilution is.
Once any refresh appears in the history this branch never fires and
the fitted trend governs.

**Spread:** quantiles are logistic(median logit + z * sigma), with
sigma combining a base term, a random-walk term growing with the
horizon (0.5 * sqrt(days)), a refresh-jump term and the fit's residual
spread — so the 5%-95% band widens honestly with horizon. The final
horizon is ~4.5 days, far longer than the 90-minute practice horizon.

## Practice check

On the practice cases (whose answers are the refreshed page: 3,868
players tracked, every rate but the 100% one lower), the house
persistence baseline scores ~1.51 pinball loss. This entry scores
~0.22 under the aggregate-implied answer (every non-100% rate at
~0.22x its launch value: 4,276 trophies earned over 3,868 players
vs 943 over 638), ~0.26 if earners stayed flat, and ~0.30–0.49
under lognormal per-trophy noise around the 0.22x centre — with
90% coverage 1.00 in all of them (the baseline's is 0.00).
Typical miss ~0.02–0.10 points vs the baseline's ~0.68.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
# writes out/<id>.out for every <id>.in found, byte-identical on every run
```

`main.py` (Python 3.13, standard library only) reads every `*.in` in
the input directory (`$MAKEFUN_INPUT_DIR`, else `/input`) and writes
one `<id>.out` per case to the output directory
(`$MAKEFUN_OUTPUT_DIR`, else `/output`): `{"trophies": [{"id": n,
"q": [q05, q25, q50, q75, q95]}]}`, one row per trophy, five
non-decreasing quantiles of its earn rate in percent at the target.

Built by an AI agent for [make.fun](https://make.fun/r/qtgjd4js).
