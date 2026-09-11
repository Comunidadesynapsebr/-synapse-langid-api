import os
import secrets
import psycopg2
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.security import APIKeyQuery, APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import pipeline
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# ==========================================
# 🗄️ CONEXÃO COM O BANCO NEON
# ==========================================
DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    if not DATABASE_URL:
        return None
    return psycopg2.connect(DATABASE_URL)

def init_db():
    conn = get_db_connection()
    if conn:
        with conn.cursor() as cur:
            # Cria a tabela de chaves automaticamente
            cur.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    ip TEXT PRIMARY KEY,
                    key TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
        conn.close()

# Tenta criar a tabela ao iniciar a API
try:
    init_db()
except Exception as e:
    print(f"Erro ao inicializar banco de dados: {e}")

# ==========================================
# 🛡️ SEGURANÇA E AUTENTICAÇÃO
# ==========================================
MASTER_KEY = os.getenv("SYNAPSE_MASTER_KEY", "master-123")

api_key_query = APIKeyQuery(name="key", auto_error=False)
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

def get_real_ip(request: Request) -> str:
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip: return cf_ip
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded: return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"

def verify_api_key(api_key_query: str = Depends(api_key_query), api_key_header: str = Depends(api_key_header)):
    key = api_key_query or api_key_header
    if not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Acesso Negado: Chave ausente.",
        )
    
    # Permite acesso irrestrito para o administrador
    if key == MASTER_KEY:
        return key

    # Verifica se a chave do usuário existe no banco Neon
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Banco de dados indisponível.")

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT key FROM api_keys WHERE key = %s;", (key,))
            row = cur.fetchone()
            if row:
                return key
    finally:
        conn.close()

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Acesso Negado: Chave inválida ou não registrada.",
    )

limiter = Limiter(key_func=get_real_ip)

# ==========================================
# 🚀 INICIALIZAÇÃO DA API
# ==========================================
app = FastAPI(title="Synapse-LangID Enterprise Auth", version="4.0.0")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carrega o modelo de IA
MODEL_ID = "Comunidade-Synapse-BR/Synapse-LangID"
try:
    classifier = pipeline("text-classification", model=MODEL_ID, device=-1)
except Exception:
    classifier = None

class PredictRequest(BaseModel):
    text: str = Field(..., max_length=1500)

# ==========================================
# 🌐 ENDPOINTS
# ==========================================
@app.get("/")
@limiter.limit("10/minute")
def health_check(request: Request):
    return {"status": "Online", "database": "Neon PostgreSQL Conectado"}

@app.post("/gerar-chave")
@limiter.limit("5/minute")
def gerar_chave(request: Request):
    ip = get_real_ip(request)
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Banco de dados indisponível.")

    try:
        with conn.cursor() as cur:
            # Trava 1: Verifica se o IP já criou uma chave antes
            cur.execute("SELECT key FROM api_keys WHERE ip = %s;", (ip,))
            row = cur.fetchone()
            if row:
                raise HTTPException(
                    status_code=400,
                    detail=f"Este IP já possui uma chave ativa: {row[0]}"
                )

            # Gera a chave e salva no banco definitivamente
            nova_chave = "syn_" + secrets.token_hex(8)
            cur.execute("INSERT INTO api_keys (ip, key) VALUES (%s, %s);", (ip, nova_chave))
            conn.commit()

        return {"message": "Chave gerada e salva com sucesso!", "api_key": nova_chave}
    finally:
        conn.close()

@app.post("/v1/predict", dependencies=[Depends(verify_api_key)])
@limiter.limit("15/minute")
def predict_secure(request: Request, payload: PredictRequest):
    if not classifier:
        raise HTTPException(status_code=500, detail="Modelo Offline.")
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Texto vazio.")

    try:
        result = classifier(payload.text)[0]
        return {
            "prediction": {
                "language": result["label"],
                "confidence": result["score"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
