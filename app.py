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
def predict(input_data: TextInput):
    # Aquí deberías procesar input_data.text según tus necesidades
    return {"message": f"Received text: {input_data.text}"}


