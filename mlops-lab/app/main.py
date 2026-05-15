from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import joblib
import io
import os
import time

app = FastAPI(title="MLOps API")

MODEL_PATH = "/app/modelos/model.pkl"

# Función para intentar cargar el modelo con reintentos
def get_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

@app.get("/")
def health_check():
    model = get_model()
    return {"status": "online", "model_loaded": model is not None}

@app.post("/predict-csv")
async def predict(file: UploadFile = File(...)):
    model = get_model()
    if not model:
        raise HTTPException(status_code=503, detail="Modelo no encontrado en el volumen.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        preds = model.predict(df)
        df["prediction"] = preds
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))