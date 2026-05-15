# 🚢 Titanic MLOps Project

Proyecto de Machine Learning y MLOps utilizando Docker, FastAPI y Streamlit para predecir la supervivencia de pasajeros del Titanic.

---

# 📌 Descripción

Este proyecto implementa un flujo completo de Machine Learning:

- Entrenamiento de modelo
- API de predicción
- Interfaz web interactiva
- Contenedores Docker
- Comunicación entre microservicios

El modelo predice si un pasajero sobrevivió o no al desastre del Titanic utilizando variables como:

- Edad
- Sexo
- Clase
- Tarifa
- Número de familiares
- Puerto de embarque

---

# 🧠 Algoritmo utilizado

Se utilizó:

- RandomForestClassifier (Scikit-learn)

El modelo fue entrenado utilizando el dataset Titanic en formato CSV.

---

# 🏗️ Arquitectura

```text
Streamlit Frontend
        ↓
FastAPI Backend
        ↓
Modelo ML RandomForest