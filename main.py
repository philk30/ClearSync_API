from fastapi import FastAPI, Header, HTTPException
import lightgbm as lgb
import pandas as pd
from supabase import create_client
import joblib
import numpy as np
import os
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# ── Load artifacts at startup ──────────────────────────────────────────────────
model = lgb.Booster(model_file="los_lgbm_model.txt")

# Encoder pickle contains: encoder, cat_cols, feature_order
# Saved by retrain_pruned() in your training script
artifacts = joblib.load("los_encoder.pkl")
enc = artifacts["encoder"]
CAT_COLS = artifacts["cat_cols"]        # e.g. ['age_group', 'admission_type']
MODEL_FEATURES = artifacts["feature_order"]   # exact order model expects

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
API_SECRET = os.getenv("API_SECRET")

# All 13 features stored in Supabase — superset of what the model needs
ALL_COLS = [
    "age_group", "gender", "race", "ethnicity",
    "admission_type", "med_surg", "health_service_area", "zip3",
    "ccs_dx", "ccs_proc", "apr_drg", "apr_severity", "apr_rom",
]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(x_api_key: str = Header(None)):
    if x_api_key != API_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # 1. Pull all columns from Supabase
    rows = supabase.table("patients").select("*").execute()
    if not rows.data:
        return {"status": "ok", "count": 0, "message": "No patients found"}

    df = pd.DataFrame(rows.data)

    # 2. Encode categoricals
    df[CAT_COLS] = enc.transform(df[CAT_COLS].astype(str))

    # 3. Slice exactly the features the model needs, in the right order
    X = df[MODEL_FEATURES].astype(float)

    # 4. Convert to numpy — bypasses LightGBM pandas dtype inspection
    log_preds = model.predict(X.to_numpy())

    # 5. Back-transform from log scale
    preds_days = np.exp(log_preds).clip(min=1)

    # 6. Write predictions back to Supabase
    records = [
        {
            "patient_id":    row["id"],
            "prediction":    round(float(pred), 2),
            "model_version": "v1.0",
        }
        for row, pred in zip(rows.data, preds_days)
    ]
    supabase.table("predictions").upsert(
        records, on_conflict="patient_id").execute()

    return {"status": "ok", "count": len(records)}


@app.post("/predict-single/{patient_id}")
async def predict_single(patient_id: str, x_api_key: str = Header(None)):
    if x_api_key != API_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # 1. Fetch just this patient
    row = supabase.table("patients").select("*").eq("id", patient_id).execute()
    if not row.data:
        raise HTTPException(status_code=404, detail="Patient not found")

    df = pd.DataFrame(row.data)

    # 2. Encode, slice, predict
    df[CAT_COLS] = enc.transform(df[CAT_COLS].astype(str))
    X = df[MODEL_FEATURES].astype(float)
    log_preds = model.predict(X.to_numpy())
    preds_days = np.exp(log_preds).clip(min=1)

    # 3. Upsert single prediction
    record = {
        "patient_id":    row.data[0]["id"],
        "prediction":    round(float(preds_days[0]), 2),
        "model_version": "v1.0",
    }
    supabase.table("predictions").upsert(
        record, on_conflict="patient_id").execute()

    return {"status": "ok", "patient_id": patient_id, "prediction": record["prediction"]}

# From Kudzani's Claude Code


class PredictPreviewRequest(BaseModel):
    # Accept all 13 columns; only the 7 in MODEL_FEATURES are used.
    # Extra keys are ignored. Missing keys for non-model features are fine.
    age_group: Optional[str] = None
    gender: Optional[str] = None
    race: Optional[str] = None
    ethnicity: Optional[str] = None
    admission_type: Optional[str] = None
    med_surg: Optional[str] = None
    health_service_area: Optional[str] = None
    zip3: Optional[str] = None
    ccs_dx: Optional[str] = None
    ccs_proc: Optional[str] = None
    apr_drg: Optional[str] = None
    apr_severity: Optional[float] = None
    apr_rom: Optional[float] = None


@app.post("/predict-preview")
async def predict_preview(
    payload: PredictPreviewRequest,
    x_api_key: str = Header(None),
):
    """
    Stateless prediction. Accepts a feature dict, runs inference,
    returns LOS in days. Does NOT read from or write to Supabase.
    Used by the frontend Digital Twin for instant what-if previews.
    """
    if x_api_key != API_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # 1. Build a single-row DataFrame from the payload
    df = pd.DataFrame([payload.dict()])

    # 2. Validate that all required model features are present
    missing = [
        f for f in MODEL_FEATURES if f not in df.columns or df[f].isna().all()]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {missing}",
        )

    # 3. Encode categoricals (same path as /predict)
    df[CAT_COLS] = enc.transform(df[CAT_COLS].astype(str))

    # 4. Slice to model features in the exact order
    X = df[MODEL_FEATURES].astype(float)

    # 5. Predict, back-transform from log scale, floor at 1 day
    log_pred = model.predict(X.to_numpy())
    pred_days = float(np.exp(log_pred[0]).clip(min=1))

    return {
        "status": "ok",
        "prediction": round(pred_days, 2),
        "model_version": "v1.0",
    }
