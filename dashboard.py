import streamlit as st
import numpy as np
import requests
import json
import time
import os

# CONFIGURACIÓN DE LA APLICACIÓN
APP_VERSION = "1.0.2" 

# CONFIGURACIÓN DE LA API Y MODELO
API_URL = "http://127.0.0.1:8000"
PREDICT_ENDPOINT = f"{API_URL}/predict_sleep_stage"
HEALTH_ENDPOINT = f"{API_URL}/health"
TIME_STEPS = 3000

# FUNCIONES DE INTERFAZ Y SIMULACIÓN

def generate_simulated_eeg(signal_type):
    """
    Genera una señal simulada para la prueba, con parámetros y ruido 
    calibrados para cada etapa del sueño.
    """
    fs = 100 
    
    if signal_type == "WAKE":
        frequency = 20  
        amplitude = 0.5
        noise_factor = 0.1
    elif signal_type == "N1 (Drowsiness)":
        frequency = 8
        amplitude = 0.7
        noise_factor = 0.15
    elif signal_type == "N2":
        frequency = 6  
        amplitude = 1.0
        noise_factor = 0.2
    elif signal_type == "N3 (Slow Wave)":
        frequency = 1 
        amplitude = 1.5
        noise_factor = 0.25 
    elif signal_type == "REM":
        frequency = 12 
        amplitude = 0.4
        noise_factor = 0.15
    else:
        return np.zeros(TIME_STEPS).tolist()

    simulated_eeg = []
    for i in range(TIME_STEPS):
        value = amplitude * np.sin(2 * np.pi * frequency * (i / fs))
        
        noise = (np.random.randn() * noise_factor) 
        simulated_eeg.append(value + noise)
        
    return simulated_eeg

def call_api(eeg_data):
    """Llama al endpoint de predicción de FastAPI."""
    payload = {"eeg_signal": eeg_data}
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(PREDICT_ENDPOINT, headers=headers, data=json.dumps(payload))
        response.raise_for_status() 
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"ERROR: No se pudo conectar a la API ({API_URL}). Asegúrese de que Uvicorn esté ejecutándose.")
        st.code(f"Detalle del error: {e}")
        return None

# ESTRUCTURA DEL DASHBOARD

st.set_page_config(
    page_title="BrainWaveNet Demo",
    layout="wide"
)

st.title("BrainWaveNet: Clasificación de Etapas del Sueño")
st.markdown("### Demo de Inferencia en Tiempo Real (MLOps)")

st.sidebar.header("Estado del Servicio (FastAPI)")
st.sidebar.markdown(f"Dashboard Versión: `{APP_VERSION}`")

try:
    health_response = requests.get(HEALTH_ENDPOINT)
    if health_response.status_code == 200:
        health_data = health_response.json()
        st.sidebar.success(f"API Activa (Code 200)")
        st.sidebar.markdown(f"**Versión de la API:** `{health_data.get('api_version', 'N/A')}`")
        st.sidebar.markdown(f"**Modelo Cargado:** `{health_data.get('model_loaded')}`")
        st.sidebar.markdown(f"**Último Entrenamiento:** `{health_data.get('model_last_trained', 'N/A')}`")
    else:
        st.sidebar.error(f"API no responde ({health_response.status_code})")
except requests.exceptions.ConnectionError:
    st.sidebar.error("Error de Conexión. Ejecute Uvicorn primero.")


# Cuerpo Principal (Inferencia)
st.header("1. Selección de la Señal de Prueba")

signal_choice = st.selectbox(
    "Seleccione el tipo de señal EEG a simular:",
    ("WAKE", "N1 (Drowsiness)", "N2", "N3 (Slow Wave)", "REM", "Señal Nula (Ceros)") 
)

eeg_data = generate_simulated_eeg(signal_choice)
st.subheader(f"Señal Simulada ({len(eeg_data)} puntos)")

# Se grafica la señal
st.line_chart(eeg_data[:500])

# Botón de Predicción
if st.button("ENVIAR A FASTAPI PARA CLASIFICACIÓN", type="primary"):
    
    st.header("2. Resultados de la Predicción (BrainWaveNet)")
    
    with st.spinner('Clasificando...'):
        time.sleep(1) 
        result = call_api(eeg_data)

    if result:
        st.subheader(f"Etapa Predicha: {result['predicted_stage']}")
        st.metric(
            label="Confianza del Modelo", 
            value=f"{float(result['confidence']) * 100:.2f}%"
        )
        
        st.markdown("**Probabilidades de Clase:**")
        
        prob_data_list = [{'Clase': k, 'Probabilidad': float(v)} for k, v in result['probabilities'].items()]
        
        st.dataframe(prob_data_list, use_container_width=True, hide_index=True)
        
        st.success("Predicción completada exitosamente.")