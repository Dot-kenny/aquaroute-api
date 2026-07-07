"""
cooper_jacob.py — Cooper-Jacob straight-line pumping test analysis
====================================================================
This is NOT a machine learning model. It's a closed-form physics
calculation from classical well hydraulics (Cooper & Jacob, 1946), the
same "straight-line method" your mentor described. It needs exactly one
pumping test's worth of data and produces two aquifer properties:

    T (transmissivity)  — how fast water moves through the aquifer
    S (storativity)     — how much water the aquifer releases per unit
                           drop in head

From T and S you can then estimate expected yield, drawdown at any
pumping duration, and appropriate pump sizing — the three things your
mentor listed.

Why this belongs in the codebase now, unlike the ML yield model
-----------------------------------------------------------------
This calculation needs no training data — it's valid physics the moment
a single driller shares one pumping test's time-drawdown readings. It's
exactly the kind of ground-truth-generating tool that should exist
*before* the Month 3-4 driller validation phase starts, so that whatever
data comes back can be turned into T/S immediately, ready to become
training data for the future ML yield model.

Method
------
For a confined aquifer, late-time drawdown s(t) at radial distance r
from a well pumped at constant rate Q approximately satisfies:

    s = (2.303 Q) / (4 pi T) * log10(2.25 T t / (r^2 S))

which is linear in log10(t):

    s = m * log10(t) + b

Fit m and b by ordinary least squares on the later-time readings (the
approximation breaks down at early times), then:

    T = 2.303 * Q / (4 * pi * m)
    t0 = 10 ** (-b / m)          # x-intercept: time where fitted s = 0
    S = 2.25 * T * t0 / r**2

Units: keep Q, T consistent (e.g. both in m^3/day and m^2/day, or both
per minute) and r, t0 consistent (e.g. both in metres / days).this
implementation is unit-agnostic — pass whatever consistent unit system
you're working in and the T/S outputs will be in matching units.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class PumpingTestResult:
    transmissivity: float          # T, e.g. m^2/day
    storativity: float             # S, dimensionless
    slope_per_log_cycle: float     # m, drawdown per log10(time) cycle
    time_intercept: float          # t0, time where fitted line crosses s=0
    n_points_used: int
    r_squared: float                # fit quality of the straight line itself
    storativity_reliable: bool     # False for single-well tests — see note below
    storativity_warning: str = ""


def analyze_pumping_test(time, drawdown, Q, r, early_time_exclude_frac=0.3):
    """
    time             : array-like, time since pumping started (consistent units, e.g. minutes)
    drawdown         : array-like, observed drawdown at each time (e.g. metres)
    Q                : constant pumping rate (consistent units, e.g. m^3/day)
    r                : distance from pumped well to observation point (metres).
                       For a single-well test (no separate observation well),
                       use the effective well radius.
    early_time_exclude_frac : fraction of the earliest readings to drop before
                       fitting, since the straight-line approximation only
                       holds at later times. Default drops the first 30%.

    Returns a PumpingTestResult. Raises ValueError on insufficient/invalid data
    rather than silently returning a number that looks plausible but isn't.
    """
    t = np.asarray(time, dtype=float)
    s = np.asarray(drawdown, dtype=float)

    if len(t) != len(s):
        raise ValueError("time and drawdown arrays must be the same length")
    if len(t) < 4:
        raise ValueError(
            f"Only {len(t)} readings supplied — Cooper-Jacob needs at least "
            "4-5 late-time readings to fit a reliable straight line. Two or "
            "three points will fit *a* line, but the slope will be noise, "
            "not signal."
        )
    if np.any(t <= 0):
        raise ValueError("time values must be strictly positive (log10 is undefined at t<=0)")
    if np.any(s < 0):
        raise ValueError("drawdown values must be non-negative")
    if Q <= 0 or r <= 0:
        raise ValueError("Q (pumping rate) and r (radius) must both be positive")

    order = np.argsort(t)
    t, s = t[order], s[order]

    n_drop = int(np.floor(len(t) * early_time_exclude_frac))
    t_fit, s_fit = t[n_drop:], s[n_drop:]
    if len(t_fit) < 3:
        raise ValueError(
            "After dropping early-time readings, fewer than 3 points remain. "
            "Supply more readings, or lower early_time_exclude_frac."
        )

    log_t = np.log10(t_fit)
    slope, intercept = np.polyfit(log_t, s_fit, 1)

    if slope <= 0:
        raise ValueError(
            "Fitted slope is zero or negative — drawdown isn't increasing "
            "with log(time) in this data. That's not a valid Cooper-Jacob "
            "straight line; check the input data or the pumping duration "
            "(may be too short for late-time behaviour to show up)."
        )

    s_pred = slope * log_t + intercept
    ss_res = np.sum((s_fit - s_pred) ** 2)
    ss_tot = np.sum((s_fit - np.mean(s_fit)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    T = 2.303 * Q / (4 * np.pi * slope)
    t0 = 10 ** (-intercept / slope)
    S = 2.25 * T * t0 / (r ** 2)

    # Storativity is only meaningful when r is the distance to a genuinely
    # separate observation point. If r looks like a borehole/well radius
    # (typically well under 1 metre) rather than a distance to an
    # observation well (typically metres to tens of metres), S is not
    # physically reliable: it appears in the formula divided by r^2, so a
    # tiny r amplifies any noise or model-mismatch into a meaningless value.
    # T remains valid from single-well data; S does not.
    radius_looks_like_observation_well = bool(r >= 1.0)

    # Separately: real aquifer storativity is essentially always in
    # roughly 0 < S < 0.35 (confined aquifers: ~1e-5 to 1e-3; unconfined/
    # specific yield: up to ~0.3). This check applies regardless of r,
    # because bad/noisy data can produce an implausible S even when r
    # looks like a genuine observation-well distance -- as happened during
    # testing with an arbitrarily-chosen (non-physically-generated) drawdown
    # sequence. A radius that looks right doesn't guarantee a sane result.
    physically_plausible = bool(0 < S <= 0.35)

    storativity_reliable = bool(radius_looks_like_observation_well and physically_plausible)

    if not radius_looks_like_observation_well:
        storativity_warning = (
            f"r={r} m looks like a borehole radius, not a separate observation-well "
            "distance. Storativity from a single-well test is not physically reliable "
            "(S is highly sensitive to r^2 in the denominator) — treat this S value as "
            "not meaningful. Transmissivity (T) remains valid. A real S estimate needs "
            "drawdown measured at a separate observation point at a known distance."
        )
    elif not physically_plausible:
        storativity_warning = (
            f"Computed S={S:.4g} is outside the physically plausible range for a real "
            "aquifer (roughly 0 to 0.35). This points to a problem with the input data "
            "(units, readings, or r) rather than an unusual-but-real aquifer — check the "
            "readings and r before trusting either T or S from this fit."
        )
    else:
        storativity_warning = ""

    if not storativity_reliable and not radius_looks_like_observation_well:
        # Don't even return the number as a rough estimate in the single-well case.
        S = float("nan")

    return PumpingTestResult(
        transmissivity=T,
        storativity=S,
        slope_per_log_cycle=slope,
        time_intercept=t0,
        n_points_used=len(t_fit),
        r_squared=r_squared,
        storativity_reliable=storativity_reliable,
        storativity_warning=storativity_warning,
    )


def predict_drawdown(T, S, Q, r, t):
    """
    Inverse direction: given known/estimated T and S, predict drawdown at
    time t. Same Cooper-Jacob approximation, useful for sanity-checking a
    fit or projecting drawdown at a pumping duration you didn't test.
    """
    t = np.asarray(t, dtype=float)
    u_arg = 2.25 * T * t / (r ** 2 * S)
    if np.any(u_arg <= 0):
        raise ValueError("Invalid inputs produce a non-positive log argument.")
    return (2.303 * Q) / (4 * np.pi * T) * np.log10(u_arg)
