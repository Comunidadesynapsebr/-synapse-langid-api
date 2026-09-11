from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field
from typing import List
from transformers import pipeline

# Bibliotecas para proteção Anti-Spam/Flood (Rate Limiting)
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Configura o limitador para bloquear pelo IP do usuário
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Synapse-LangID API",
    description="API protegida contra Flood para identificação de idioma da Comunidade Synapse BR",
    version="1.2.0"
)

# Adiciona o escudo na aplicação para retornar Erro 429 se alguém fizer spam
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Carrega o pipeline do modelo otimizado para CPU
MODEL_ID = "Comunidade-Synapse-BR/Synapse-LangID"

try:
    classifier = pipeline(
        "text-classification",
        model=MODEL_ID,
        device=-1
    )
except Exception as e:
    classifier = None
    print(f"Erro ao carregar o modelo: {e}")

# Schemas com limitação de "Tokens/Caracteres" para evitar travamento da RAM
class SinglePredictRequest(BaseModel):
    # Bloqueia textos maiores que 2000 caracteres
    text: str = Field(..., max_length=2000, description="Texto com no máximo 2000 caracteres")

class BatchPredictRequest(BaseModel):
    # Permite no máximo 20 frases de uma vez no envio em lote
    texts: List[str] = Field(..., max_length=20, description="Máximo de 20 textos por requisição")


@app.get("/")
@limiter.limit("20/minute") # Permite 20 acessos por minuto no ping
def health_check(request: Request):
    return {
        "status": "online",
        "model": MODEL_ID,
        "loaded": classifier is not None,
        "security": "Rate Limiting ATIVADO"
    }


@app.post("/predict")
@limiter.limit("15/minute") # Limite: 15 predições por minuto por IP
def predict_language(request: Request, payload: SinglePredictRequest):
    if not classifier:
        raise HTTPException(status_code=500, detail="Modelo não carregado.")
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="O texto não pode ser vazio.")

    try:
        results = classifier(payload.text)
        return {
            # Corta a resposta visual se o texto for gigante (para economizar banda)
            "input": payload.text[:50] + "..." if len(payload.text) > 50 else payload.text,
            "prediction": results[0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na inferência: {str(e)}")


@app.post("/predict/batch")
@limiter.limit("5/minute") # Limite rígido (5 por minuto) pois processa muitos dados
def predict_batch(request: Request, payload: BatchPredictRequest):
    if not classifier:
        raise HTTPException(status_code=500, detail="Modelo não carregado.")
    
    clean_texts = [t for t in payload.texts if t.strip()]
    if not clean_texts:
        raise HTTPException(status_code=400, detail="A lista de textos não pode ser vazia.")

    try:
        results = classifier(clean_texts)
        output = [
            {"input": text[:50] + "..." if len(text) > 50 else text, "prediction": res}
            for text, res in zip(clean_texts, results)
        ]
        return {
            "total": len(output),
            "results": output
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na inferência em lote: {str(e)}")


@app.get("/detect")
@limiter.limit("15/minute") # Limite: 15 predições rápidas por minuto por IP
def detect_quick(request: Request, text: str = Query(..., max_length=2000, description="Texto para identificar")):
    if not classifier:
        raise HTTPException(status_code=500, detail="Modelo não carregado.")
    if not text.strip():
        raise HTTPException(status_code=400, detail="O parâmetro 'text' não pode estar vazio.")

    try:
        results = classifier(text)
        return {
            "input": text[:50] + "..." if len(text) > 50 else text,
            "prediction": results[0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na inferência: {str(e)}")
