import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="SDSS Space Object Classifier",
    description="API для классификации звезд, галактик и квазаров на основе данных SDSS DR14"
)

MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise RuntimeError("Модель или скалер не найдены. Сначала запустите src/modeling.py")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
CLASS_MAPPING = {0: "GALAXY", 1: "QSO", 2: "STAR"}

class AstronomicalObject(BaseModel):
    ra: float
    dec: float
    u: float
    g: float
    r: float
    i: float
    z: float
    redshift: float

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "SDSS Classifier API is running"}

@app.post("/predict")
def predict_object(data: AstronomicalObject):
    try:
        input_dict = data.model_dump()
        df = pd.DataFrame([input_dict])
        df['u-g'] = df['u'] - df['g']
        df['g-r'] = df['g'] - df['r']
        df['r-i'] = df['r'] - df['i']
        df['i-z'] = df['i'] - df['z']
        
        feature_cols = ['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'redshift', 'u-g', 'g-r', 'r-i', 'i-z']
        df_features = df[feature_cols]
        
        scaled_features = scaler.transform(df_features)
        
        pred_code = int(model.predict(scaled_features)[0])
        pred_class = CLASS_MAPPING.get(pred_code, "UNKNOWN")
        probabilities = model.predict_proba(scaled_features)[0].tolist()
        
        return {
            "prediction_code": pred_code,
            "prediction_class": pred_class,
            "probabilities": {
                "GALAXY": probabilities[0],
                "QSO": probabilities[1],
                "STAR": probabilities[2]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)