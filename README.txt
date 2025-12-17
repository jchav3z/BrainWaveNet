BrainWaveNet: Clasificación de Señales EEG

Este proyecto implementa y despliega un modelo de Deep Learning para la clasificación automática de las etapas del sueño (WAKE, N1, N2, N3, REM) a partir de segmentos de señales EEG. La solución final se sirve como una API REST funcional mediante FastAPI, demostrando prácticas MLOps.

Estructura y Componentes Clave del proyecto.

- src/model.py: Define la arquitectura BrainWaveNet (CNN 1D).
- src/train.py: Contiene la lógica de entrenamiento, preprocesamiento (Filtro Pasa-Banda) y validación (F1-score).
- api.py: Implementa la API REST (FastAPI/Uvicorn) para servir las predicciones en tiempo real.
- dashboard.py: Interfaz gráfica Streamlit para la demostración y uso del software.
- models/: Contiene el artefacto del modelo entrenado: brainwavenet_weights.pth.
- requirements.txt: Listado de dependencias para la instalación del entorno.

Requisitos del Sistema y Ejecución

Se requiere Python 3.9+ y el entorno virtual activo.

Instalación de Dependencias

pip install -r requirements.txt

Despliegue del Servicio

Abrir el primer terminal y levantar el back-end de la API:
uvicorn api:app --reload

El servicio de inferencia está activo en http://127.0.0.1:8000. (local)

Interfaz de Demostración

Abrir el segundo terminal y levantar el Dashboard Streamlit:
streamlit run dashboard.py

Verificación (Software Funcional): Acceda a la interfaz en http://localhost:8501. La API clasifica la etapa de sueño basándose en la señal simulada.

Detalles de Implementación (MLOps - RA 3.1.4)

Arquitectura: El modelo BrainWaveNet utiliza capas nn.Conv1d y nn.Dropout(0.5) para manejar series de tiempo y optimizar contra overfitting.
Gestión de Artefactos: La API carga los pesos del modelo (.pth) al inicio, lo cual es verificado por el endpoint /health.
Validación: La API impone una validación estricta de la entrada, esperando exactamente 3000 puntos por segmento de señal EEG.