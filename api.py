"""
api.py — AquaRoute siting API
==============================
FastAPI wrapper around the models trained in train_models.py.

    POST /predict   {"latitude": 8.49, "longitude": 4.55}
    ->  {
          "aquifer_probability": 0.82,
          "confidence_interval": [0.71, 0.90],
          "recommended_depth_range_m": [8, 15],
          "geological_zone": "weathered_fractured",
          "risk_flag": false,
          "risk_reason": null,
          "nearest_validated_station": {
            "ves_id": "VES42", "distance_km": 1.3,
            "aquifer_proxy": 1, "deep_resistivity_ohm": 87.4
          },
          "model_confidence_note": "..."
        }

Run locally:
    pip install fastapi uvicorn
    python train_models.py      # only needed once, or whenever data changes
    uvicorn api:app --reload --port 8000

Then:
    curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"latitude": 8.49, "longitude": 4.55}'

Deploy target per the roadmap: Google Cloud Run free tier or Render.com.
Both accept this file + requirements.txt unchanged (uvicorn as the ASGI
entrypoint, PORT env var already respected by --port ${PORT:-8000} in the
Dockerfile/Procfile you add at deploy time).
"""

import os
import json
import tempfile
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from report_generator import build_report

STORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_store")

app = FastAPI(
    title="AquaRoute Siting API",
    description="ML-powered borehole siting for basement-complex terrain (Ilorin, Kwara State pilot zone).",
    version="0.1.0",
)

# Allow the web frontend (served from any origin, including file:// during
# local testing) to call this API directly from the browser. Tighten
# allow_origins to your actual frontend domain before production launch.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load artifacts once at startup ──────────────────────────────────────────
gp = joblib.load(os.path.join(STORE_DIR, "gp_spatial.joblib"))
rf = joblib.load(os.path.join(STORE_DIR, "rf_classifier.joblib"))
iso = joblib.load(os.path.join(STORE_DIR, "iso_forest.joblib"))
log_scaler = joblib.load(os.path.join(STORE_DIR, "log_scaler.joblib"))
pca = joblib.load(os.path.join(STORE_DIR, "pca.joblib"))
coord_scaler = joblib.load(os.path.join(STORE_DIR, "coord_scaler.joblib"))
station_lookup = pd.read_csv(os.path.join(STORE_DIR, "station_lookup.csv"), index_col=0)

with open(os.path.join(STORE_DIR, "metadata.json")) as f:
    META = json.load(f)

AQUIFER_THRESHOLD_OHM = META["aquifer_threshold_ohm"]
ZONE_HIGH_OHM = META["zone_high_ohm"]
DEEP_DEPTH_COUNT = META["deep_depth_count"]
DEPTH_RANGE_M = META["empirical_depth_range_m"]
LAT_BOUNDS = META["lat_bounds"]
LON_BOUNDS = META["lon_bounds"]

# GP was fit with GaussianProcessRegressor(normalize_y=True) wrapped in
# MultiOutputRegressor -> gp.estimators_[i] gives per-depth predictive std.


class SiteRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="WGS84 latitude, decimal degrees")
    longitude: float = Field(..., ge=-180, le=180, description="WGS84 longitude, decimal degrees")


class NearestStation(BaseModel):
    ves_id: str
    distance_km: float
    aquifer_proxy: int
    deep_resistivity_ohm: float


class SiteResponse(BaseModel):
    aquifer_probability: float
    confidence_interval: list
    recommended_depth_range_m: list
    geological_zone: str
    risk_flag: bool
    risk_reason: str | None
    nearest_validated_station: NearestStation
    in_pilot_coverage_area: bool
    model_confidence_note: str


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def classify_zone(deep_resistivity_ohm: float) -> str:
    if deep_resistivity_ohm < AQUIFER_THRESHOLD_OHM:
        return "weathered_fractured"
    elif deep_resistivity_ohm < ZONE_HIGH_OHM:
        return "transition"
    else:
        return "fresh_basement"


@app.get("/")
def root():
    return {
        "service": "AquaRoute Siting API",
        "status": "ok",
        "pilot_zone": "Ilorin, Kwara State (basement complex)",
        "n_training_stations": META["n_stations"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=SiteResponse)
def predict(req: SiteRequest):
    lat, lon = req.latitude, req.longitude

    in_coverage = (LAT_BOUNDS[0] - 0.2 <= lat <= LAT_BOUNDS[1] + 0.2) and \
                  (LON_BOUNDS[0] - 0.2 <= lon <= LON_BOUNDS[1] + 0.2)

    # 1. GP spatial interpolation -> predicted log10-resistivity profile
    coord_scaled = coord_scaler.transform([[lat, lon]])
    pred_log_profile = np.array([
        est.predict(coord_scaled)[0] for est in gp.estimators_
    ])
    pred_std = np.array([
        est.predict(coord_scaled, return_std=True)[1][0] for est in gp.estimators_
    ])

    # 2. Deep-depth resistivity (drives aquifer signal + zone + depth logic)
    deep_log = pred_log_profile[-DEEP_DEPTH_COUNT:]
    deep_resistivity_ohm = float(10 ** deep_log.mean())
    deep_std_ohm = float(10 ** (deep_log.mean() + pred_std[-DEEP_DEPTH_COUNT:].mean()) - deep_resistivity_ohm)

    # 3. Random Forest aquifer probability, fed the GP's interpolated profile
    scaled_profile = log_scaler.transform(pred_log_profile.reshape(1, -1))
    aquifer_prob = float(rf.predict_proba(scaled_profile)[0, 1])

    # Rough confidence band: widen probability by normalized GP uncertainty
    uncertainty_frac = float(np.clip(pred_std.mean() / (np.abs(pred_log_profile).mean() + 1e-6), 0, 1))
    ci_low = max(0.0, aquifer_prob - uncertainty_frac * 0.35)
    ci_high = min(1.0, aquifer_prob + uncertainty_frac * 0.35)

    # 4. Isolation Forest anomaly flag on PCA-projected interpolated profile
    pca_profile = pca.transform(scaled_profile)
    anomaly_score = iso.decision_function(pca_profile)[0]
    is_anomaly = iso.predict(pca_profile)[0] == -1

    risk_flag = bool(is_anomaly) or (not in_coverage)
    if not in_coverage:
        risk_reason = "Location falls outside the pilot survey area (Ilorin basement complex); GP is extrapolating far beyond training data."
    elif is_anomaly:
        risk_reason = f"Interpolated resistivity profile is an outlier vs. the 85 training stations (isolation score {anomaly_score:.2f}); treat as low-confidence."
    else:
        risk_reason = None

    # 5. Nearest validated station (real ground truth, not model output)
    station_lookup["dist_km"] = haversine_km(lat, lon, station_lookup["lat"], station_lookup["lon"])
    nearest = station_lookup.loc[station_lookup["dist_km"].idxmin()]

    zone = classify_zone(deep_resistivity_ohm)

    note = (
        f"GP spatial model 5-fold CV R^2 = {META['gp_cv_r2_deepest_depth']:.2f} on held-out stations "
        f"({META['n_stations']} stations total). This is a sparse-data interpolation, not a validated "
        "ground-truth model yet — treat probability as a prioritization score, not a guarantee, until "
        "the driller-validation phase (Month 3-4) supplies real drilling outcomes."
    )

    return SiteResponse(
        aquifer_probability=round(aquifer_prob, 3),
        confidence_interval=[round(ci_low, 3), round(ci_high, 3)],
        recommended_depth_range_m=[round(DEPTH_RANGE_M[0], 1), round(DEPTH_RANGE_M[1], 1)],
        geological_zone=zone,
        risk_flag=risk_flag,
        risk_reason=risk_reason,
        nearest_validated_station=NearestStation(
            ves_id=str(nearest.name),
            distance_km=round(float(nearest["dist_km"]), 2),
            aquifer_proxy=int(nearest["aquifer_proxy"]),
            deep_resistivity_ohm=round(float(nearest["deep_resistivity_ohm"]), 1),
        ),
        in_pilot_coverage_area=in_coverage,
        model_confidence_note=note,
    )


@app.post("/report")
def report(req: SiteRequest):
    """
    Runs the same prediction as /predict, then renders it into the 3-page
    PDF siting report and returns the file directly. This is the endpoint
    the pay-per-report flow (Paystack -> GPS coords -> PDF by email) calls.
    """
    prediction = predict(req).dict()
    tmp_path = os.path.join(tempfile.gettempdir(), f"aquaroute_report_{req.latitude}_{req.longitude}.pdf")
    build_report(prediction, req.dict(), tmp_path)
    return FileResponse(tmp_path, media_type="application/pdf",
                         filename="aquaroute_siting_report.pdf")
