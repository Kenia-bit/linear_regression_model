from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import io

app = FastAPI(title="Rwanda Basic Needs Prediction PI")

# EXplicit origns instead of "*" to restrict cross-origin requests to trusted clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://10.0.2.2:8000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

MODEL_PATH = "best_model.pkl"
DATA_PATH = "rwanda_basic_needs.csv"
model = joblib.load(MODEL_PATH)


class HouseholdInput(BaseModel):
    province: int = Field(..., ge=1, le=5)
    district: int = Field(..., ge=11, le=57)
    urban_rural: int = Field(..., ge=1, le=2)
    poverty_status: int = Field(..., ge=1, le=3)
    wealth_quintile: int = Field(..., ge=1, le=5)
    dwelling_type: int = Field(..., ge=1, le=10)
    occupancy_status: int = Field(..., ge=1, le=10)
    dwelling_year: float = Field(..., ge=0, le=100)
    water_source: int = Field(..., ge=1, le=20)
    water_maintenance_cost: float = Field(..., ge=0, le=50000)
    lighting_source: int = Field(..., ge=1, le=20)


@app.get("/")
def root():
    return {"message": "Rwanda Basic Needs API is running"}


@app.post("/predict")
def predict(data: HouseholdInput):
    input_df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(input_df)
    return {"predicted_basic_survive_index": round(float(prediction[0]), 2)}


@app.post("/retrain")
async def retrain(file: UploadFile = File(...)):
    global model
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a CSV file.")

    contents = await file.read()
    new_df = pd.read_csv(io.BytesIO(contents))

    base_df = pd.read_csv(DATA_PATH)
    if list(new_df.columns) != list(base_df.columns):
        raise HTTPException(status_code=400, detail="CSV columns do not match the expected format.")  

    updated_df = pd.concat([base_df, new_df], ignore_index=True).dropna()
    updated_df.to_csv(DATA_PATH, index=False)

    target = updated_df.columns[-1]
    X = updated_df.drop(columns=[target])
    Y = updated_df[target]

    model.fit(X, Y)
    joblib.dump(model, MODEL_PATH)

    return {
        "message": "Model retrained successfully",
        "total_rows_used": len(updated_df),
        }