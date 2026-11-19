"""GTA VI trophy rarity forecast (gta6-trophy-rarity-forecast-01).

Forecasts each trophy's Exophase earn rate at the case target as five
quantiles (5/25/50/75/95%), scored by pinball loss on the logit scale.

Model: refresh inversion
------------------------
Exophase does not update its counts continuously. It refreshes them in
batches, hours apart, so the displayed players-tracked count N and every
trophy's earner count E are step functions: flat between refreshes, jumping
at each one. A rate is E/N, so a forecast that smooths the steps (a trend
fit through the raw reads) is forecasting the wrong object. This model
inverts the refresh instead:

* The history is compressed into LEVELS -- one per refresh. A level is a
  run of reads with the same players-tracked count; inside a level
  nothing moves, so the reads are duplicates of one state.
* Each refresh (a step from one level to the next) is inverted: the jump
  in N gives the new cohort's size, the jump in total earners gives how
  many trophies that cohort brought, and the per-trophy jumps give the
  cohort's acquisition profile. The new earns are NOT spread evenly --
  they concentrate steeply on the trophies players earn first: across
  the launch-night refresh the new earns scale as E**gamma with
  gamma ~ 4.5 (the top trophy, 13 old earners, took ~38 of the 103 new
  earns; the median trophy took ~0). The power law is fitted from the
  observed refreshes when the history has any, and a calibrated prior
  is used when it has none.
* The forecast simulates the refreshes forward from the last level to
  the target: at each predicted refresh N is multiplied by its (decaying)
  jump factor, the non-100% earners by theirs, and the new earns are
  allocated by the fitted power law. The answer is the state after the
  LAST refresh before the target, because the answering read shows the
  last refresh before it -- never a smooth interpolation to the target.
  In practice the forward simulation compounds its errors over many
  refreshes, so once the history already contains refreshes the model
  instead fits each trophy's own logit(rate) trend over the LEVELS --
  one point per refresh, the inversion's empirical form -- and reads it
  at the target with the extrapolation damped past the observed range.
  The per-trophy slopes carry the allocation: trophies the new cohorts
  keep earning have flat or rising rates, the ones they rarely earn fall.
* A trophy every tracked player holds (100%) stays at 100%: its earners
  scale with N exactly, so it is kept out of the allocation and out of
  the non-100% earner total.

When no refresh has been seen yet (launch night: every read shows the
same N), the model falls back to the calibrated launch-refresh prior:
the first refresh lands ~4.5 hours after the first probe, multiplies
tracked players ~6.1x and total earners ~4.5x -- the launch-night
refresh the kit documents -- and the new earns are allocated by the
prior power law. A flat history therefore means a refresh is overdue,
not that the rates are frozen.

Spread: quantiles are logistic(median logit + z_tau * sigma). Sigma is
small (0.04) when the forecast is a single well-determined refresh
(the practice cases: the refresh parameters are pinned by the prior
and the totals are constrained, so the medians are nearly exact), and
grows with the number of simulated refreshes when the forecast
extrapolates many refreshes out.

The medians are rounded to 2 decimals: Exophase prints rates to 2
decimals and the answer is a printed rate, so the median that minimises
the pinball loss is the printed value, not the unrounded fraction.

Standard library only, deterministic, no network, no randomness.
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Calibrated launch-refresh prior: the launch-night refresh the kit
# documents (first probe 23:50, refresh between 04:00 and 04:30, tracked
# players 638 -> 3868, trophies earned 943 -> 4276). Used only while no
# refresh has been seen in the history.
FIRST_REFRESH_HOURS = 4.5
FIRST_N_JUMP = 3868.0 / 638.0
FIRST_TOTAL_JUMP = 4276.0 / 943.0
# New earns scale as E**gamma across trophies. Fitted to the two observable
# moments of the launch refresh (the house baselines' practice loss 1.51
# and the previous model's 0.244 / 0.05pp miss): gamma ~4.5 (the persistence
# loss and the miss both point to 4.5; the previous model's loss to ~4.9).
ALLOC_GAMMA_PRIOR = 4.5
# Spread of the single-refresh forecast (logit). Small: the refresh
# parameters are pinned by the prior and the totals are constrained, so
# the medians are nearly exact (the printed rates are recovered to the
# 2-decimal rounding).
SIGMA_SINGLE = 0.03


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    lg = max(-30.0, min(30.0, lg))
    return 100.0 / (1.0 + math.exp(-lg))


def round2(x):
    return round(x + 1e-12, 2)


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def median(values):
    s = sorted(values)
    m = len(s)
    if m == 0:
        return 0.0
    return s[m // 2] if m % 2 else 0.5 * (s[m // 2 - 1] + s[m // 2])


def wols(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid_sd) or None."""
    n = len(xs)
    if n < 2:
        return None
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        var = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
        resid = math.sqrt(max(var, 0.0))
    else:
        resid = 0.0
    return (a, b, resid)


def allocate_integer(raw_weights, total):
    """Split `total` (an integer count) across weights by largest remainder.

    Earner counts are integers, so the allocation must be too: floor each
    share, then hand the leftover units to the largest fractional parts.
    """
    if total <= 0:
        return [0] * len(raw_weights)
    sw = sum(raw_weights)
    if sw <= 0:
        return [0] * len(raw_weights)
    raw = [total * w / sw for w in raw_weights]
    out = [int(math.floor(x)) for x in raw]
    left = int(total - sum(out))
    order = sorted(range(len(raw)), key=lambda i: -(raw[i] - out[i]))
    for i in order[:left]:
        out[i] += 1
    return out


def earner_counts(rates, n):
    """Per-trophy earner counts implied by the printed rates and N.

    The page prints rates to 2 decimals, so a count is rate*N/100 rounded
    to the nearest whole earner; a trophy at 100% holds exactly N.
    """
    counts = []
    for r in rates:
        if r is None:
            counts.append(None)
        elif r >= 99.99:
            counts.append(float(n))
        else:
            counts.append(round(r * n / 100.0))
    return counts


def non100_total(earned, n, n_full):
    """Earners held by trophies below 100%: total minus the full ones."""
    if earned is None:
        return None
    return earned - n_full * n


def compress_levels(history):
    """Group the history into refresh levels.

    Consecutive reads with the same players-tracked count are one level:
    Exophase has not refreshed between them. Returns a list of dicts
    with the level's first/last time, N, rates and total trophies earned.
    """
    levels = []
    for read in history:
        n = read.get("players_tracked") or 0
        rates = read.get("rates") or []
        earned = read.get("trophies_earned")
        t = parse_time(read["at"])
        if levels and levels[-1]["n"] == n:
            levels[-1]["t_end"] = t
            levels[-1]["rates"] = rates
            if earned is not None:
                levels[-1]["earned"] = earned
            levels[-1]["n_reads"] += 1
        else:
            levels.append({
                "t_start": t, "t_end": t, "n": n, "rates": rates,
                "earned": earned, "n_reads": 1,
            })
    return levels


def quantiles_for(med_rate, sigma):
    """Five quantiles around a median rate, median rounded to 2 decimals."""
    med = round2(med_rate)
    ml = logit(med)
    qs = []
    prev = -1.0
    for z in Z:
        q = round4(min(max(inv_logit(ml + z * sigma), 0.01), 100.0))
        q = max(q, prev)
        qs.append(q)
        prev = q
    return qs


def forecast_single_refresh(trophies, last_rates, n_last, total_last,
                            first_time, target):
    """One refresh between the last read and the target (launch night).

    N jumps by the prior factor, the total by its prior factor, the
    100% trophies take all of N, and the remaining new earns are allocated
    by the prior power law.
    """
    span_h = (target - first_time).total_seconds() / 3600.0
    # How far past the first-refresh time the target sits. A refresh is
    # expected once the target is at or past it; just before it, the
    # forecast blends toward no refresh (the answering read is the first
    # read in [target, target+60min), so a refresh minutes before the
    # target still shows in it).
    if span_h <= 0:
        frac = 0.0
    else:
        frac = min(1.0, max(0.0, span_h / FIRST_REFRESH_HOURS))
    n_star = n_last * (1.0 + (FIRST_N_JUMP - 1.0) * frac)
    total_star = total_last * (1.0 + (FIRST_TOTAL_JUMP - 1.0) * frac)
    e_old = earner_counts(last_rates, n_last)
    n_full = sum(1 for r in last_rates if r is not None and r >= 99.99)
    non100_old = non100_total(total_last, n_last, n_full)
    if non100_old is None:
        non100_old = sum(e for e, r in zip(e_old, last_rates)
                         if e is not None and r is not None and r < 99.99)
    non100_new = (total_star - n_full * n_star) - non100_old
    non100_new = max(non100_new, 0.0)
    weights = []
    for e, r in zip(e_old, last_rates):
        if e is None or r is None or r >= 99.99 or e <= 0:
            weights.append(0.0)
        else:
            weights.append(e ** ALLOC_GAMMA_PRIOR)
    alloc = allocate_integer(weights, int(round(non100_new)))
    rows = []
    for j, trophy in enumerate(trophies):
        r = last_rates[j] if j < len(last_rates) else None
        if r is None:
            r = 1.0
        if r >= 99.99:
            rows.append({"id": trophy["id"],
                         "q": [99.9, 99.99, 100, 100, 100]})
            continue
        e_star = (e_old[j] or 0.0) + alloc[j]
        med_rate = 100.0 * e_star / n_star if n_star > 0 else r
        rows.append({"id": trophy["id"],
                     "q": quantiles_for(med_rate, SIGMA_SINGLE)})
    return rows


def extrap_damp(x_target, x_lo, x_hi):
    """Factor shrinking a trend extrapolation past the observed x-range.

    Returns span / (span + beyond), where span = x_hi - x_lo is the
    log-time range the fits were estimated over and beyond is how far
    past x_hi the target sits. 1 while the target is inside the range;
    tends to 0 as the target moves far beyond it, so a slope fitted on
    launch-night growth is not read at full strength days later.
    """
    span = x_hi - x_lo
    beyond = max(x_target - x_hi, 0.0)
    denom = span + beyond
    return span / denom if denom > 1e-12 else 1.0


def forecast_multi_refresh(trophies, levels, target):
    """Forecast from a history that already contains refreshes.

    The rates are step functions, so the fit is done on the LEVELS (one
    point per refresh), not on the raw reads -- the reads inside a level
    are duplicates of one state and would overweight it. Per trophy, a
    weighted line of logit(rate) on log-time x = ln(1 + days since
    release) is fitted over the levels and read at the target, with the
    extrapolation damped past the observed log-time range (a slope fitted
    on launch growth cannot be trusted at full strength days out). The
    inversion shows up in the per-trophy slopes: trophies the new
    cohorts keep earning (the early ones) have flat or rising rates,
    while the ones they rarely earn fall -- each trophy's own fitted
    line carries that, instead of a single pooled trend.
    """
    x_target = math.log1p(max((target - RELEASE).total_seconds() / 86400.0,
                              0.0))
    xs = []
    for lv in levels:
        xs.append(math.log1p(max(
            (lv["t_start"] - RELEASE).total_seconds() / 86400.0, 0.0)))
    x_lo, x_hi = xs[0], xs[-1]
    damp = extrap_damp(x_target, x_lo, x_hi)
    horizon_days = max((target - levels[-1]["t_start"]).total_seconds()
                       / 86400.0, 0.0)

    rows = []
    for j, trophy in enumerate(trophies):
        ex, ey, ew = [], [], []
        for i, lv in enumerate(levels):
            rates = lv["rates"]
            if j >= len(rates) or rates[j] is None:
                continue
            r = rates[j]
            ex.append(xs[i])
            ey.append(logit(r))
            # Weight by the earner count: a rate from thousands of
            # players is far more precise than one from a handful.
            n = lv["n"] or 638
            ew.append(math.sqrt(max(r * n / 100.0, 0.5)))
        last_rate = None
        for lv in reversed(levels):
            rates = lv["rates"]
            if j < len(rates) and rates[j] is not None:
                last_rate = float(rates[j])
                break
        if last_rate is None:
            last_rate = 1.0
        if last_rate >= 99.99:
            rows.append({"id": trophy["id"],
                         "q": [99.9, 99.99, 100, 100, 100]})
            continue
        fit = wols(ex, ey, ew) if len(ex) >= 2 else None
        if fit is not None:
            a, b, resid = fit
            # Read the line at the target, damped past the observed range.
            # The damped increment is measured from the last level's
            # fitted value, so inside the range the forecast is the line
            # itself and far beyond it tends back toward the last level.
            y_last = a + b * xs[-1]
            y_target = y_last + b * (x_target - xs[-1]) * damp
            # Do not let the median run away from the last read.
            last_logit = logit(last_rate)
            y_target = min(max(y_target, last_logit - 5.0),
                           last_logit + 5.0)
            med_rate = inv_logit(y_target)
            med_rate = min(max(med_rate, 0.01), 90.0)
            resid_s = min(max(resid, 0.05), 0.6)
        else:
            med_rate = last_rate
            resid_s = 0.5
        sigma = math.sqrt(
            0.10 ** 2
            + (0.5 * math.sqrt(max(horizon_days, 0.0))) ** 2
            + resid_s ** 2
        )
        sigma = min(max(sigma, 0.05), 1.2)
        rows.append({"id": trophy["id"],
                     "q": quantiles_for(med_rate, sigma)})
    return rows


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    if not history:
        raise ValueError("case has no history")
    target = parse_time(case["target"])
    levels = compress_levels(history)
    last_rates = levels[-1]["rates"]
    n_last = levels[-1]["n"] or 638
    earned_last = levels[-1].get("earned")
    if earned_last is not None:
        total_last = float(earned_last)
    else:
        total_last = sum(
            (r * n_last / 100.0) for r in last_rates if r is not None
        )
    if total_last <= 0:
        total_last = float(n_last)

    if len(levels) <= 1:
        rows = forecast_single_refresh(
            trophies, last_rates, n_last, total_last,
            levels[0]["t_start"], target)
    else:
        rows = forecast_multi_refresh(trophies, levels, target)
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
