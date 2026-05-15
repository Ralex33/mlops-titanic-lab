import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(page_title="MLOps Titanic", layout="wide")

st.title("🚢 Titanic ML Predictor")

st.sidebar.header("Configuración")
st.sidebar.info(
    "Esta aplicación se conecta a un modelo de Machine Learning usando FastAPI y Docker."
)

uploaded_file = st.file_uploader(
    "Sube un archivo CSV",
    type="csv"
)

if uploaded_file is not None:

    df_original = pd.read_csv(uploaded_file)

    st.subheader("Vista previa del CSV")
    st.dataframe(df_original.head())

    if st.button("Magia! ✨"):

        URL_API = "http://api:8000/predict-csv"

        try:

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "text/csv"
                )
            }

            response = requests.post(URL_API, files=files)

            if response.status_code == 200:

                data = response.json()

                st.success("Predicciones realizadas correctamente")

                df_resultado = pd.DataFrame(data)

                st.subheader("Resultados")
                st.dataframe(df_resultado)

                grafico = px.histogram(
                    df_resultado,
                    x="prediccion",
                    title="Distribución de Predicciones"
                )

                st.plotly_chart(
                    grafico,
                    use_container_width=True
                )

            else:
                st.error(
                    "Error en el servidor de IA."
                )

        except Exception as e:
            st.error(f"Error de conexión: {e}")