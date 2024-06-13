from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


# Cargar el modelo y el vectorizador desde los archivos .pkl
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


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

class TextInput(BaseModel):
    text: str


@app.get("/predict/")
def predict(text: str):
 
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    # Transformar los datos de entrada usando el vectorizador
    transformed_data = vectorizer.transform([text])
    # Realizar predicciones usando el modelo cargado
    prediction = model.predict(transformed_data)
    # Retornar la predicción como respuesta
    return {str(prediction)}

