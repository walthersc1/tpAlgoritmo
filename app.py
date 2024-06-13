from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi.responses import PlainTextResponse
import logging

logger = logging.getLogger(__name__)

app = FastAPI()

# Configurar CORS
origins = [
    "http://localhost:3000/**",  # Origen de desarrollo local
    "https://proyecto-6j9n8jmjo-walthersc11s-projects.vercel.app", 
    "https://tpalgoritmo-production.up.railway.app/mapri/", # Reemplaza con tu dominio de Vercel
    # Añade otros orígenes permitidos aquí
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definir el modelo de datos para la entrada
class TextInput(BaseModel):
    text: str

@app.options("/{rest_of_path:path}")
async def preflight_check():
    return {"message": "This is a preflight response"}

@app.get("/mapri/")
def ga():
    return {"mapri"}

@app.post("/predict/")
async def predict(text: str):
    try:
        # Cargar el modelo y el vectorizador desde los archivos .pkl
        model = joblib.load('model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')

        # Transformar los datos de entrada usando el vectorizador
        transformed_data = vectorizer.transform([text])
        # Realizar predicciones usando el modelo cargado
        prediction = model.predict(transformed_data)
        # Retornar la predicción como respuesta
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error during prediction")
	
    return {prediction}

