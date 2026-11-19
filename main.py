"""GTA VI trophy rarity forecast (gta6-trophy-rarity-forecast-01).

Forecasts each trophy's Exophase earn rate at the case target as five
quantiles (5/25/50/75/95%), scored by pinball loss on the logit scale.

Model, in outline
-----------------
An earn rate is earners / players-tracked, and the two move for different
reasons, so they are forecast separately and divided:

* Players tracked N(t): Exophase refreshes its counts in batches, hours
  apart, so N is a step function that jumps at refreshes. log N is fitted
  against log-time x = ln(1 + days since release) and read at the target.
* Earners E(t) per trophy: log E is fitted the same way, per trophy, with
  the slope shrunk toward the pooled (cross-trophy) slope -- early reads
  have only a handful of earners, so a single trophy's slope is noisy --
  plus a small nudge from what the trophy asks for (story chapters are
  earned steadily as players progress; grind trophies such as 250 stunts
  or 100% completion barely move in a week).
* The median rate is 100 * E*/N*, capped so it cannot move more than
  5 logit units from the last read.

One case breaks that fit: when every read in the history shows the same
players-tracked count, no refresh has been observed yet, so no trend can
be fitted at all. That is launch night: Exophase had not refreshed since
the first probe, and its first refresh (between the 04:00 and 04:30 reads
on 19 Nov) multiplied tracked players about 6x while earners grew only
modestly, so every rate but the one at 100% fell hard. A flat history
therefore means a refresh is overdue, not that rates are frozen: the
forecast divides the last earners (grown a little, by trophy kind) by a
tracked count jumped ~6x, and widens the spread to cover how uneven
that dilution is across trophies. Once any refresh appears in the
history this branch never fires and the fitted trend governs.

Spread: quantiles are logistic(median logit + z_tau * sigma), with sigma
combining a base term, a random-walk term growing with the horizon
(0.5 * sqrt(days)), a refresh-jump term and the fit's residual spread,
so the 5%-95% band widens honestly with horizon -- the final horizon is
~4.5 days, far longer than the 90-minute practice horizon.

Standard library only, deterministic, no network, no randomness.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# First-refresh prior, used only while no refresh has been seen in the
# history: Exophase's first refresh multiplied tracked players ~6x once
# ~4.7 hours of launch accumulation had passed.
FIRST_REFRESH_HOURS = 4.7
FIRST_REFRESH_MULT = 6.0
# How much earners grow across that first refresh, by trophy kind.
EARNER_GROWTH = {"story": 1.6, "easy": 1.4, "other": 1.3, "grind": 1.1}
# Small slope nudge (logit per unit of log-time) in the fitted branch.
SLOPE_NUDGE = {"story": 0.12, "easy": 0.04, "other": 0.0, "grind": -0.12}

GRIND_MARKERS = (
    "250 ", "100%", "100 locations", "50 challenges", "collectible card",
    "gold medal in all", "every spot", "every location",
    "8 mission strands", "100% completion",
)
STORY_MARKERS = ("chapter", "final mission", "mission for")
EASY_MARKERS = (
    "try each type", "arcade game", "beat cal", "discover 100 vehicles",
    "spend $100,000", "meditate", "explore all of leonida",
    "earn $250,000", "purchase all garage", "register 10 vehicles",
)


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


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def trophy_kind(trophy):
    """What the trophy asks for: story progress, grind, or an easy one-off."""
    desc = (trophy.get("description") or "").lower()
    if trophy.get("tier") == "platinum":
        return "grind"
    if any(m in desc for m in GRIND_MARKERS):
        return "grind"
    if any(m in desc for m in STORY_MARKERS):
        return "story"
    if desc.startswith("complete ") and trophy.get("secret"):
        # "Complete Breakfast Run", "Complete The Cartel Heist", ...:
        # named story missions, not counted goals (those carry a number).
        if not any(ch.isdigit() for ch in desc):
            return "story"
    if any(m in desc for m in EASY_MARKERS):
        return "easy"
    return "other"


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
    return (a, b, resid, mx, my)


def median(values):
    s = sorted(values)
    m = len(s)
    if m == 0:
        return 0.0
    return s[m // 2] if m % 2 else 0.5 * (s[m // 2 - 1] + s[m // 2])


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    if not history:
        raise ValueError("case has no history")
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = math.log1p(max((target - RELEASE).total_seconds() / 86400.0, 0.0))

    # Per-read log-time and tracked count.
    xs, ns, times = [], [], []
    for read in history:
        t = parse_time(read["at"])
        xs.append(math.log1p(max((t - RELEASE).total_seconds() / 86400.0, 0.0)))
        ns.append(read.get("players_tracked") or 0)
        times.append(t)
    n_last = ns[-1] if ns[-1] else (max(ns) if ns else 638)
    first_time = times[0]

    # Fit log N against log-time, if the tracked count ever moved.
    n_fit = None
    if len(set(ns)) > 1:
        logn = [math.log(max(n, 1)) for n in ns]
        n_fit = wols(xs, logn, [1.0] * len(xs))

    # Per-trophy log-earner fits, then the pooled slope across trophies.
    e_fits = []
    for j in range(len(trophies)):
        ex, ey, ew = [], [], []
        for i, read in enumerate(history):
            rates = read.get("rates") or []
            if j >= len(rates) or rates[j] is None:
                continue
            n = ns[i] if ns[i] else n_last
            earners = max(float(rates[j]) * n / 100.0, 0.5)
            ex.append(xs[i])
            ey.append(math.log(earners))
            ew.append(math.sqrt(earners))
        e_fits.append(wols(ex, ey, ew) if len(ex) >= 2 else None)
    slopes = [f[1] for f in e_fits if f is not None]
    pooled_slope = median(slopes) if slopes else 0.0
    pooled_resid = median([f[2] for f in e_fits if f is not None]) if slopes else 0.3

    stale = n_fit is None
    if stale:
        # No refresh seen yet: one is overdue. How far into the launch
        # accumulation the target sits sets the jump in tracked players.
        span_h = (target - first_time).total_seconds() / 3600.0
        jump = 1.0 + (FIRST_REFRESH_MULT - 1.0) * min(1.0, max(span_h, 0.0) / FIRST_REFRESH_HOURS)
        n_star = n_last * jump

    rows = []
    for j, trophy in enumerate(trophies):
        last_rate = None
        for read in reversed(history):
            rates = read.get("rates") or []
            if j < len(rates) and rates[j] is not None:
                last_rate = float(rates[j])
                break
        if last_rate is None:
            last_rate = 1.0
        kind = trophy_kind(trophy)

        # The 100% trophy: every tracked player holds it; it stays there.
        if last_rate >= 99.99:
            rows.append({"id": trophy["id"], "q": [99.9, 99.99, 100, 100, 100]})
            continue

        last_logit = logit(last_rate)
        if stale:
            earners_last = max(last_rate * n_last / 100.0, 0.5)
            e_star = earners_last * EARNER_GROWTH[kind]
            med_rate = 100.0 * e_star / n_star
            sigma = 1.0
        else:
            a_n, b_n = n_fit[0], n_fit[1]
            n_star = math.exp(a_n + b_n * x_target)
            n_star = min(max(n_star, float(n_last)), float(n_last) * 30.0)
            fit = e_fits[j]
            if fit is not None:
                _, b_e, resid, mx, my = fit
                # Shrinkage weight from the number of usable reads.
                n_used = sum(
                    1
                    for read in history
                    if (read.get("rates") or []) and j < len(read["rates"])
                    and read["rates"][j] is not None
                )
                w = n_used / (n_used + 4.0)
                b_s = w * b_e + (1.0 - w) * pooled_slope + SLOPE_NUDGE[kind]
                log_e_star = my + b_s * (x_target - mx)
                resid_s = min(max(0.5 * resid + 0.5 * pooled_resid, 0.15), 1.0)
            else:
                b_s = pooled_slope + SLOPE_NUDGE[kind]
                earners_last = max(last_rate * n_last / 100.0, 0.5)
                log_e_star = math.log(earners_last) + b_s * max(x_target - xs[-1], 0.0)
                resid_s = 0.5
            e_star = math.exp(log_e_star)
            e_star = min(max(e_star, 0.25), n_star)
            med_rate = 100.0 * e_star / n_star
            med_rate = min(max(med_rate, 0.01), 90.0)
            # Do not let the median run away from the last read.
            med_logit = logit(med_rate)
            med_logit = min(max(med_logit, last_logit - 5.0), last_logit + 5.0)
            med_rate = inv_logit(med_logit)
            sigma = math.sqrt(
                0.55 ** 2
                + (0.5 * math.sqrt(max(horizon_days, 0.0))) ** 2
                + 0.4 ** 2
                + resid_s ** 2
            )

        med_logit = logit(med_rate)
        qs = []
        prev = -1.0
        for z in Z:
            q = round4(min(max(inv_logit(med_logit + z * sigma), 0.01), 100.0))
            q = max(q, prev)
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
