from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import re

# Cargar el modelo, vectorizador y otros objetos necesarios
model = joblib.load('modelo_gradient_boosting_mejorado.pkl')
text_vectorizer = joblib.load('vectorizador_tfidf.pkl')
cat_encoder = joblib.load('encoder_categorico.pkl')
scaler = joblib.load('escalador_numerico.pkl')

app = FastAPI()

# Configurar CORS
origins = [
    "http://localhost:3000/**",  # Origen de desarrollo local
    "https://proyecto-6j9n8jmjo-walthersc11s-projects.vercel.app", 
    "https://tpalgoritmo-production.up.railway.app/mapri/",  # Reemplaza con tu dominio de Vercel
    # Añade otros orígenes permitidos aquí
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir cualquier origen
    allow_credentials=True,
    allow_methods=["*"],  # Permitir cualquier método
    allow_headers=["*"],  # Permitir cualquier cabecera
)

# Función para limpiar el texto (puedes reutilizar la que ya tienes)
def clean_text(text):
    # Reemplazar saltos de línea, tabulaciones y retornos de carro por espacios
    text = re.sub(r'[\n\t\r]', ' ', text)
    # Reemplazar múltiples espacios por uno solo
    text = re.sub(r'\s+', ' ', text)
    # Remover caracteres especiales (no alfanuméricos)
    text = re.sub(r'\W', ' ', text)
    # Pasar todo a minúsculas
    text = text.lower()
    # Eliminar stopwords (ajustar según sea necesario)
    stop_words = set(["a", "de", "la", "y", "en", "el"])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

@app.options("/{rest_of_path:path}")
async def preflight_check():
    # Esta función maneja las peticiones OPTIONS para CORS
    return {"message": "This is a preflight response"}

@app.get("/mapri/")  # Endpoint de ejemplo para verificar si la API está funcionando
def ga():
    return {"mapri": "Bienvenido"}

# Definir el esquema de entrada de los datos
class InputData(BaseModel):
    all_responses: str  # Respuestas abiertas que se limpiarán y analizarán
    gender: str         # Género
    age: int            # Edad
    major: str          # Especialidad

@app.post("/predict/")
async def predict(input_data: InputData, request: Request):
    # Obtener el cuerpo de la solicitud (JSON) para verificar antes de procesar
    body = await request.json()
    print("Cuerpo de la solicitud:", body)

    # Limpiar el texto de la entrada
    cleaned_text = clean_text(input_data.all_responses)

    # Transformar el texto usando el vectorizador TF-IDF
    text_features = text_vectorizer.transform([cleaned_text]).toarray()

    # Codificar las variables categóricas (gender y major)
    cat_features = cat_encoder.transform([[input_data.gender, input_data.major]]).toarray()

    # Escalar la edad
    age_scaled = scaler.transform([[input_data.age]])

    # Combinar todas las características en un solo array
    X_input = np.hstack((text_features, cat_features, age_scaled))

    # Realizar la predicción
    prediction = model.predict(X_input)

    # Retornar la predicción como respuesta
    return {"prediction": str(prediction[0])}