from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import joblib
import io
import os

app = FastAPI(title="MLOps API")

MODEL_PATH = "/app/modelos/model.pkl"


def get_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


@app.get("/")
def health_check():
    model = get_model()

    return {
        "status": "online",
        "model_loaded": model is not None
    }


@app.post("/predict-csv")
async def predict(file: UploadFile = File(...)):

    model = get_model()

    if not model:
        raise HTTPException(
            status_code=503,
            detail="Modelo no encontrado."
        )

    try:

        contents = await file.read()

        df = pd.read_csv(io.BytesIO(contents))

        # PREPROCESSING
        df["Age"] = df["Age"].fillna(
            df["Age"].median()
        )

        df["Embarked"] = df["Embarked"].fillna(
            df["Embarked"].mode()[0]
        )

        df = pd.get_dummies(
            df,
            columns=["Sex", "Embarked"],
            drop_first=True
        )

        columnas_eliminar = [
            "Name",
            "Ticket",
            "Cabin"
        ]

        for col in columnas_eliminar:
            if col in df.columns:
                df = df.drop(col, axis=1)

        if "Survived" in df.columns:
            df = df.drop("Survived", axis=1)

        columnas_modelo = [
            "PassengerId",
            "Pclass",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Sex_male",
            "Embarked_Q",
            "Embarked_S"
        ]

        for col in columnas_modelo:
            if col not in df.columns:
                df[col] = 0

        df = df[columnas_modelo]

        preds = model.predict(df)

        resultado = df.copy()

        resultado["prediccion"] = preds

        return resultado.to_dict(orient="records")

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )