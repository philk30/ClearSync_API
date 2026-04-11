from fastapi import FastAPI, Header, HTTPException
import lightgbm as lgb
import pandas as pd
from supabase import create_client
import os

app = FastAPI()

model = lgb.Booster(model_file="los_lgbm_model.txt")
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
API_SECRET = os.getenv("API_SECRET")  # your custom auth key


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(x_api_key: str = Header(None)):
    if x_api_key != API_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Pull data from Supabase
    rows = supabase.table("your_input_table").select("*").execute()
    df = pd.DataFrame(rows.data)

    # Predict
    predictions = model.predict(df[["feature1", "feature2", "feature3"]])

    # Write back
    records = [
        {"id": row["id"], "prediction": float(pred)}
        for row, pred in zip(rows.data, predictions)
    ]
    supabase.table("predictions").upsert(records).execute()

    return {"status": "ok", "count": len(records)}
