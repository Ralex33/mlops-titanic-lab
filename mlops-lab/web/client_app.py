import streamlit as st
import pandas as pd
import requests
import plotly.express as px # Nueva librería para gráficos
import io

st.set_page_config(page_title="MLOps", layout="wide")

st.title("MLops ")
st.sidebar.header("Configuración")
st.sidebar.info("Este portal se conecta a un microservicio de IA desplegado en Docker.")

uploaded_file = st.file_uploader(type="csv")

if uploaded_file:
    df_original = pd.read_csv(uploaded_file)
    
    if st.button("Magia!"):
        URL_API = "http://api:8000/predict-csv"
        
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
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

                st.plotly_chart(grafico, use_container_width=True)

            else:
                st.error("Error en el servidor de IA. Revisa el formato del CSV.")

        except Exception as e:
            st.error(f"Error de conexión: {e}")