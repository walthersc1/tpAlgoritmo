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
    "http://localhost:3000",  # Origen de desarrollo local
    "https://proyecto-6j9n8jmjo-walthersc11s-projects.vercel.app",  # Reemplaza con tu dominio de Vercel
    # Añade otros orígenes permitidos aquí
]


# Definir el modelo de datos para la entrada
class TextInput(BaseModel):
    text: str

@app.options("/{rest_of_path:path}")
async def preflight_check():
    return {"message": "This is a preflight response"}

@app.get("/mapri/")
def ga ():
    return {"mapri"}

@app.post("/predict/")
def predict(input_data: TextInput):
    # Transformar los datos de entrada usando el vectorizador
    transformed_data = vectorizer.transform([input_data.text])
    # Realizar predicciones usando el modelo cargado
    prediction = model.predict(transformed_data)
    # Retornar la predicción como respuesta
    return {"prediction": prediction.tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
