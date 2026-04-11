from fastapi import FastAPI, Header, HTTPException
import lightgbm as lgb
import pandas as pd
from supabase import create_client
import joblib
import numpy as np
import os

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

    # 2. Encode categoricals using the same encoder fitted at training time
    df[CAT_COLS] = enc.transform(df[CAT_COLS].astype(str))

    # 3. Slice exactly the features the model needs, in the right order
    X = df[MODEL_FEATURES]

    # 4. Predict — model outputs log(LOS), so back-transform with exp()
    log_preds = model.predict(X)
    # floor at 1 day, matches training
    preds_days = np.exp(log_preds).clip(min=1)

    # 5. Write predictions back to Supabase
    records = [
        {
            "patient_id":   row["id"],
            "prediction":   round(float(pred), 2),   # predicted LOS in days
            "model_version": "v1.0",
        }
        for row, pred in zip(rows.data, preds_days)
    ]
    supabase.table("predictions").upsert(records).execute()

    return {"status": "ok", "count": len(records)}
