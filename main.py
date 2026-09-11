from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(
    title="Synapse-LangID API",
    description="API de identificação de idioma do modelo da Comunidade Synapse BR"
)

# Carrega o modelo da Hugging Face diretamente na inicialização (pipeline otimizado para CPU)
MODEL_ID = "Comunidade-Synapse-BR/Synapse-LangID"

try:
    classifier = pipeline(
        "text-classification",
        model=MODEL_ID,
        device=-1 # Garante que o modelo rode na CPU
    )
except Exception as e:
    classifier = None
    print(f"Erro ao carregar o modelo da Hugging Face: {e}")

class PredictRequest(BaseModel):
    text: str

@app.get("/")
def health_check():
    return {
        "status": "online",
        "model": MODEL_ID,
        "loaded": classifier is not None
    }

@app.post("/predict")
def predict_language(payload: PredictRequest):
    if not classifier:
        raise HTTPException(status_code=500, detail="Modelo não carregado.")
    
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="O texto não pode ser vazio.")

    try:
        results = classifier(payload.text)
        return {
            "input": payload.text,
            "prediction": results[0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na inferência: {str(e)}")
