"""
monte_carlo.py — Monte Carlo uncertainty for AquaRoute's /predict endpoint.

Replaces the current heuristic confidence interval in api.py:

    uncertainty_frac = pred_std.mean() / (abs(pred_log_profile).mean() + 1e-6)
    ci_low  = aquifer_prob - uncertainty_frac * 0.35
    ci_high = aquifer_prob + uncertainty_frac * 0.35

...with an actual empirical distribution. gp.estimators_ (one
GaussianProcessRegressor per depth, wrapped in MultiOutputRegressor)
already returns a predictive mean AND std at every depth for a queried
(lat, lon) -- real per-depth uncertainty, currently collapsed into one
averaged scalar and used to widen a probability by an arbitrary 0.35
constant. This module draws repeated samples from those per-depth
predictive distributions, pushes each simulated resistivity profile
through the SAME trained RandomForestClassifier and classify_zone()
logic already in api.py, and reports empirical percentiles instead.

No external noise model needed (no VES measurement-error assumption, no
Winresist fit quality, no ADMT tolerance) -- the uncertainty already
lives inside the trained GP. This only makes proper use of it.

Caveat worth knowing: samples are drawn independently per depth
(mean_i, std_i from each per-depth estimator). The MultiOutputRegressor
wrapper doesn't expose cross-depth covariance, so this treats each depth
as conditionally independent given (lat, lon). Adjacent depths are almost
certainly correlated in reality (a profile that's resistive at 20m is
more likely resistive at 25m too), so this probably UNDER-estimates the
true spread slightly. Not worth solving for v1 -- fixing it properly
would mean refitting as a single multi-output-kernel GP instead of
independent per-depth ones, which is a bigger change than this ticket.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Optional


def monte_carlo_predict(
    lat: float,
    lon: float,
    gp,
    rf,
    log_scaler,
    coord_scaler,
    deep_depth_count: int,
    classify_zone_fn: Callable[[float], str],
    iso=None,
    pca=None,
    n_sims: int = 150,
    seed: Optional[int] = None,
) -> dict:
    """
    Run n_sims draws from the GP's own per-depth predictive distribution
    at (lat, lon), push each through the trained RF classifier and zone
    logic, and return empirical percentiles.

    Pass the SAME objects api.py already loads at module scope -- gp, rf,
    log_scaler, coord_scaler, DEEP_DEPTH_COUNT, classify_zone (and
    optionally iso + pca to also get a simulated anomaly rate). Nothing
    here is retrained; this only re-runs inference n_sims times with
    resampled inputs.

    Returns
    -------
    dict with:
      - aquifer_probability: {median, p05, p95}       -- replaces the
        old confidence_interval heuristic
      - deep_resistivity_ohm: {median, p05, p95}
      - zone_stability: {zone_name: fraction_of_sims}  -- e.g.
        {"weathered_fractured": 0.87, "transition": 0.13} -- shows
        whether the zone call itself is robust, not just the probability
      - majority_zone: the most common zone across sims
      - anomaly_rate: fraction of sims flagged anomalous by Isolation
        Forest, or None if iso/pca weren't passed
      - confidence_label: plain-language read of the probability spread
    """
    rng = np.random.default_rng(seed)
    coord_scaled = coord_scaler.transform([[lat, lon]])

    means = np.array([est.predict(coord_scaled)[0] for est in gp.estimators_])
    stds = np.array(
        [est.predict(coord_scaled, return_std=True)[1][0] for est in gp.estimators_]
    )

    probs = np.empty(n_sims)
    deep_resistivities = np.empty(n_sims)
    zones = []
    anomaly_flags = np.zeros(n_sims, dtype=bool)

    for i in range(n_sims):
        sampled_log_profile = rng.normal(loc=means, scale=stds)

        scaled = log_scaler.transform(sampled_log_profile.reshape(1, -1))
        probs[i] = rf.predict_proba(scaled)[0, 1]

        deep_log = sampled_log_profile[-deep_depth_count:]
        deep_resistivity = float(10 ** deep_log.mean())
        deep_resistivities[i] = deep_resistivity
        zones.append(classify_zone_fn(deep_resistivity))

        if iso is not None and pca is not None:
            pca_profile = pca.transform(scaled)
            anomaly_flags[i] = iso.predict(pca_profile)[0] == -1

    zone_counts = {z: zones.count(z) for z in set(zones)}
    zone_stability = {z: round(c / n_sims, 3) for z, c in zone_counts.items()}
    majority_zone = max(zone_counts, key=zone_counts.get)

    return {
        "n_sims": n_sims,
        "aquifer_probability": {
            "median": round(float(np.median(probs)), 3),
            "p05": round(float(np.percentile(probs, 5)), 3),
            "p95": round(float(np.percentile(probs, 95)), 3),
        },
        "deep_resistivity_ohm": {
            "median": round(float(np.median(deep_resistivities)), 1),
            "p05": round(float(np.percentile(deep_resistivities, 5)), 1),
            "p95": round(float(np.percentile(deep_resistivities, 95)), 1),
        },
        "zone_stability": zone_stability,
        "majority_zone": majority_zone,
        "anomaly_rate": round(float(anomaly_flags.mean()), 3) if iso is not None else None,
        "confidence_label": _confidence_label(probs),
    }


def _confidence_label(probs: np.ndarray) -> str:
    spread = np.percentile(probs, 95) - np.percentile(probs, 5)
    if spread < 0.15:
        return "high confidence"
    elif spread < 0.30:
        return "moderate confidence"
    return "low confidence -- wide simulated spread"
