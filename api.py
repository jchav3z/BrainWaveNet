from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import os
from src.model import BrainWaveNet 

# --- CONFIGURACIÓN ---
WEIGHTS_PATH = os.path.join('models', 'brainwavenet_weights.pth')
NUM_CLASSES = 5
INPUT_CHANNELS = 1

SLEEP_STAGES = {0: "WAKE", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
INPUT_TIME_STEPS = 3000

# 1. Carga del modelo
model = None
try:
    model = BrainWaveNet(input_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device('cpu')))
    model.eval()
    print(f"INFO: Modelo cargado exitosamente desde {WEIGHTS_PATH}.")
except Exception as e:
    print(f"ERROR: Fallo al cargar el modelo. La API no podrá predecir. Error: {e}")
    model = None

# Inicialización de la aplicación FastAPI
app = FastAPI(
    title="BrainWaveNet Sleep Classification API",
    version="1.0.0"
)

class EEGData(BaseModel):
    eeg_signal: list[float]
    
@app.post("/predict_sleep_stage")
def predict_sleep_stage(data: EEGData):
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo no está disponible o no se pudo cargar.")
        
    if len(data.eeg_signal) != INPUT_TIME_STEPS:
        raise HTTPException(status_code=400, detail=f"Señal de entrada incorrecta. Se esperaban {INPUT_TIME_STEPS} puntos, se recibieron {len(data.eeg_signal)}.")

    try:
        signal_array = np.array(data.eeg_signal, dtype=np.float32)
        input_tensor = torch.from_numpy(signal_array).unsqueeze(0).unsqueeze(0) 

        with torch.no_grad():
            output = model(input_tensor)
            
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        predicted_index = torch.argmax(output, dim=1).item()
        confidence = probabilities[predicted_index].item()
        predicted_stage = SLEEP_STAGES.get(predicted_index, "UNKNOWN")

        return {
            "predicted_stage": predicted_stage,
            "stage_index": predicted_index,
            "confidence": f"{confidence:.4f}",
            "probabilities": {SLEEP_STAGES[i]: f"{probabilities[i].item():.4f}" for i in range(NUM_CLASSES)}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la inferencia: {str(e)}")

# Endpoint de salud
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None, "api_version": app.version}