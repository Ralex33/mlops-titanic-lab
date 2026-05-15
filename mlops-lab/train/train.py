import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train():
    # 1. Cargar datos (Asegúrate de que dataset.csv esté en la carpeta train/)
    df = pd.read_csv("dataset.csv")
    df.info() 
    df.isnull().sum()
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df = pd.get_dummies(df,columns=["Sex", "Embarked"],drop_first=True)
    df = df.drop(["Name", "Ticket", "Cabin"], axis=1)
    X = df.drop("Survived", axis=1)
    y = df["Survived"]  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 2. Entrenar
    model = RandomForestClassifier( n_estimators=100,random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Precisión del modelo: {accuracy}")

    print("Iniciando entrenamiento en el contenedor...")
    os.makedirs("modelos", exist_ok=True)

    # 3. GUARDAR EN EL VOLUMEN COMPARTIDO
    ruta_modelo = "/app/modelos/model.pkl"
    joblib.dump(model, ruta_modelo)
    print(f"✅ Modelo guardado exitosamente en {ruta_modelo}")

if __name__ == "__main__":
    train()