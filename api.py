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
import psycopg2
from pywebpush import webpush, WebPushException
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from report_generator import build_report
from cooper_jacob import analyze_pumping_test
from monte_carlo import monte_carlo_predict          # NEW

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

DATABASE_URL = os.environ.get("DATABASE_URL")
VAPID_PUBLIC_KEY = os.environ.get("VAPID_PUBLIC_KEY")
VAPID_PRIVATE_KEY = os.environ.get("VAPID_PRIVATE_KEY")
VAPID_CLAIM_EMAIL = os.environ.get("VAPID_CLAIM_EMAIL")


def get_db_connection():
    return psycopg2.connect(DATABASE_URL)


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS siting_readings (
            id SERIAL PRIMARY KEY,
            device_id TEXT,
            latitude DOUBLE PRECISION NOT NULL,
            longitude DOUBLE PRECISION NOT NULL,
            aquifer_probability DOUBLE PRECISION,
            geological_zone TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS push_subscriptions (
            id SERIAL PRIMARY KEY,
            endpoint TEXT UNIQUE NOT NULL,
            p256dh TEXT NOT NULL,
            auth TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    conn.commit()
    cur.close()
    conn.close()


init_db()


class SiteRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="WGS84 latitude, decimal degrees")
    longitude: float = Field(..., ge=-180, le=180, description="WGS84 longitude, decimal degrees")


class PushSubscriptionIn(BaseModel):
    endpoint: str
    keys: dict


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
    zone_stability: dict            # NEW -- e.g. {"weathered_fractured": 0.87, "transition": 0.13}
    mc_confidence_label: str        # NEW -- "high confidence" / "moderate confidence" / "low confidence..."
    mc_n_sims: int                  # NEW -- how many draws it ran, for transparency

class PumpingTestReadingIn(BaseModel):
    time_minutes: float = Field(..., gt=0)
    drawdown_m: float = Field(..., ge=0)


class PumpingTestRequest(BaseModel):
    pumping_rate_m3_per_day: float = Field(..., gt=0, description="Constant discharge rate, Q")
    effective_radius_m: float = Field(
        ..., gt=0,
        description="Distance from pumped well to where drawdown was measured. Use a separate "
                    "observation well's distance if you have one (needed for a reliable storativity "
                    "estimate); otherwise use the borehole radius — transmissivity will still be valid, "
                    "storativity will be flagged as unreliable."
    )
    readings: list[PumpingTestReadingIn] = Field(..., min_length=4)
    early_time_exclude_frac: float = Field(0.3, ge=0, le=0.9)


class PumpingTestResponse(BaseModel):
    transmissivity_m2_per_day: float
    storativity: float | None
    storativity_reliable: bool
    storativity_warning: str
    fit_r_squared: float
    n_readings_used: int
    note: str


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


@app.get("/stats")
def stats():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*), COUNT(DISTINCT device_id) FROM siting_readings")
        total, unique_users = cur.fetchone()
        cur.close()
        conn.close()
        return {"sites_screened": total, "unique_users": unique_users}
    except Exception as e:
        return {"sites_screened": None, "unique_users": None, "error": str(e)}


@app.post("/subscribe")
def subscribe(sub: PushSubscriptionIn):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO push_subscriptions (endpoint, p256dh, auth)
               VALUES (%s, %s, %s)
               ON CONFLICT (endpoint) DO NOTHING""",
            (sub.endpoint, sub.keys.get("p256dh"), sub.keys.get("auth"))
        )
        conn.commit()
        cur.close()
        conn.close()
        return {"status": "subscribed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=SiteResponse)
def predict(req: SiteRequest, request: Request):
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

   
   # Real confidence band: sample from the GP's own per-depth predictive
    # distribution, run each sample through the same RF classifier + zone
    # logic, and take empirical percentiles -- not a hand-tuned constant.
    mc = monte_carlo_predict(
        lat=lat, lon=lon,
        gp=gp, rf=rf, log_scaler=log_scaler, coord_scaler=coord_scaler,
        deep_depth_count=DEEP_DEPTH_COUNT, classify_zone_fn=classify_zone,
        iso=iso, pca=pca,
        n_sims=150,
    )
    ci_low, ci_high = mc["aquifer_probability"]["p05"], mc["aquifer_probability"]["p95"]
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

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        device_id = request.headers.get("X-Device-Id")
        cur.execute(
            """INSERT INTO siting_readings (device_id, latitude, longitude, aquifer_probability, geological_zone)
               VALUES (%s, %s, %s, %s, %s)""",
            (device_id, lat, lon, aquifer_prob, zone)
        )
        conn.commit()
        cur.close()
        conn.close()
   except Exception as e:
        print(f"Failed to log siting reading: {e}")

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
        zone_stability=mc["zone_stability"],              # NEW
        mc_confidence_label=mc["confidence_label"],        # NEW
        mc_n_sims=mc["n_sims"],                            # NEW
    )

@app.post("/report")
def report(req: SiteRequest, request: Request):
    """
    Runs the same prediction as /predict, then renders it into the 3-page
    PDF siting report and returns the file directly. This is the endpoint
    the pay-per-report flow (Paystack -> GPS coords -> PDF by email) calls.
    """
    prediction = predict(req, request).dict()
    tmp_path = os.path.join(tempfile.gettempdir(), f"aquaroute_report_{req.latitude}_{req.longitude}.pdf")
    build_report(prediction, req.dict(), tmp_path)
    return FileResponse(tmp_path, media_type="application/pdf",
                         filename="aquaroute_siting_report.pdf")


@app.post("/analyze-pumping-test", response_model=PumpingTestResponse)
def analyze_pumping_test_endpoint(req: PumpingTestRequest):
    """
    Cooper-Jacob straight-line analysis of a pumping test — pure physics,
    not a trained model. Independent of the siting prediction: this can be
    run for any well, whether or not AquaRoute was used to site it.

    This is real and usable today. It is NOT the future ML yield model
    (predicting expected yield for a new site with no pumping test) —
    that model doesn't exist yet because the training data (15+ real
    pumping tests) doesn't exist yet. See yield_module/train_yield_model.py.
    """
    try:
        # analyze_pumping_test needs t and Q in matching time units (both
        # "per day" here, since pumping_rate_m3_per_day is per day) — t0,
        # and therefore storativity, silently comes out wrong by a factor
        # of 1440 if time is left in minutes while Q is per day. T is
        # unaffected (it only depends on the slope of s vs log10(t), which
        # is invariant to a constant log-shift), which is exactly why this
        # class of bug is easy to miss: T looks right even when S is wrong.
        time_days = [r.time_minutes / 1440.0 for r in req.readings]
        result = analyze_pumping_test(
            time=time_days,
            drawdown=[r.drawdown_m for r in req.readings],
            Q=req.pumping_rate_m3_per_day,
            r=req.effective_radius_m,
            early_time_exclude_frac=req.early_time_exclude_frac,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    note = (
        "Transmissivity is derived directly from this pumping test via the Cooper-Jacob "
        "straight-line method and is not a machine-learning prediction."
        if result.storativity_reliable else
        "Transmissivity is reliable. Storativity is not — see storativity_warning."
    )

    return PumpingTestResponse(
        transmissivity_m2_per_day=result.transmissivity,
        storativity=None if (result.storativity != result.storativity) else result.storativity,  # NaN -> None
        storativity_reliable=result.storativity_reliable,
        storativity_warning=result.storativity_warning,
        fit_r_squared=result.r_squared,
        n_readings_used=result.n_points_used,
        note=note,
    )
