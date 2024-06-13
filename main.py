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
    "https://proyecto-eetmxget9-walthers-projects-76c0417b.vercel.app",  
    "https://proyecto-eetmxget9-walthers-projects-76c0417b.vercel.app/Chatbot",
    "https://proyecto-tp-chi.vercel.app",
    "https://tpalgoritmo-production.up.railway.app",# Reemplaza con tu dominio de Vercel 
    # Añade otros orígenes permitidos aquí
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
# Definir el modelo de datos para la entrada
class TextInput(BaseModel):
    text: str



# Definir tus rutas y funciones de manejo aquí
@app.options("/{rest_of_path:path}")  # Ruta de preflight OPTIONS
async def preflight_check():
    return {"message": "This is a preflight response"}

@app.post("/predict/")  # Ejemplo de ruta POST
def predict(input_data: TextInput):
    # Lógica de tu función predict aquí
    return {"prediction": "dummy prediction"}

"""
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""