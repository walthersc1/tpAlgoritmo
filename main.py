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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Permitir solicitudes solo desde el frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definir el modelo de datos para la entrada
class TextInput(BaseModel):
    text: str

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