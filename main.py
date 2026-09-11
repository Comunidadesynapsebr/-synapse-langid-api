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
# 🗄️ CONEXÃO COM O BANCO DE DADOS RENDER
# ==========================================
def get_db_connection():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("Aviso: DATABASE_URL não configurada.")
        return None
    
    # Garante o prefixo correto para o driver do Python
    db_url = db_url.replace("postgres://", "postgresql://")
    
    try:
        return psycopg2.connect(db_url)
    except Exception as e:
        print(f"Erro fatal de conexão com o banco: {e}")
        return None

def init_db():
    conn = get_db_connection()
    if conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    ip TEXT PRIMARY KEY,
                    key TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
        conn.close()
        print("Tabela 'api_keys' verificada/criada com sucesso.")

init_db()

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
        raise HTTPException(status_code=401, detail="Acesso Negado: Chave ausente.")
    
    if key == MASTER_KEY:
        return key

    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Banco de dados temporariamente indisponível.")

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT key FROM api_keys WHERE key = %s;", (key,))
            if cur.fetchone():
                return key
    finally:
        conn.close()

    raise HTTPException(status_code=401, detail="Acesso Negado: Chave inválida ou revogada.")

limiter = Limiter(key_func=get_real_ip)

# ==========================================
# 🚀 INICIALIZAÇÃO DA API
# ==========================================
app = FastAPI(title="Synapse-LangID Enterprise Auth", version="4.1.0")
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
    text: str = Field(..., max_length=1500, description="Texto para identificar o idioma")

# ==========================================
# 🌐 ENDPOINTS PÚBLICOS
# ==========================================
@app.get("/")
@limiter.limit("10/minute")
def health_check(request: Request):
    db_status = "Conectado" if get_db_connection() else "Desconectado"
    return {"status": "Online", "database": db_status, "model": "Carregado" if classifier else "Offline"}

@app.post("/gerar-chave")
@limiter.limit("5/minute")
def gerar_chave(request: Request):
    ip = get_real_ip(request)
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Banco de dados indisponível.")

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT key FROM api_keys WHERE ip = %s;", (ip,))
            row = cur.fetchone()
            if row:
                raise HTTPException(status_code=400, detail=f"Este IP já possui uma chave ativa: {row[0]}")

            nova_chave = "syn_" + secrets.token_hex(8)
            cur.execute("INSERT INTO api_keys (ip, key) VALUES (%s, %s);", (ip, nova_chave))
            conn.commit()

        return {"message": "Chave gerada com sucesso!", "api_key": nova_chave}
    finally:
        conn.close()

# ==========================================
# 🔒 ENDPOINTS PROTEGIDOS (REQUER CHAVE)
# ==========================================

@app.post("/v1/predict", dependencies=[Depends(verify_api_key)])
@limiter.limit("15/minute")
def predict_secure(request: Request, payload: PredictRequest):
    """Analisa o texto e retorna o idioma detectado."""
    if not classifier:
        raise HTTPException(status_code=500, detail="Modelo Offline.")
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Texto vazio.")

    try:
        result = classifier(payload.text)[0]
        return {
            "prediction": {
                "language": result["label"],
                "confidence": round(result["score"], 4)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/v1/key-info", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
def info_chave(request: Request, key: str = Depends(verify_api_key)):
    """Retorna os dados da chave de API atual (quando foi criada)."""
    if key == MASTER_KEY:
        return {"key": "MASTER_KEY", "type": "admin", "status": "active"}

    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Banco indisponível.")

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT ip, created_at FROM api_keys WHERE key = %s;", (key,))
            row = cur.fetchone()
            if row:
                return {
                    "key_prefix": key[:7] + "...",
                    "registered_ip": row[0],
                    "created_at": row[1].strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "active"
                }
    finally:
        conn.close()
    
    raise HTTPException(status_code=404, detail="Informações não encontradas.")

@app.delete("/v1/revoke-key", dependencies=[Depends(verify_api_key)])
@limiter.limit("3/minute")
def revogar_chave(request: Request, key: str = Depends(verify_api_key)):
    """Deleta a chave atual do banco de dados, liberando o IP para gerar uma nova."""
    if key == MASTER_KEY:
        raise HTTPException(status_code=403, detail="A MASTER_KEY não pode ser revogada.")

    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Banco indisponível.")

    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM api_keys WHERE key = %s;", (key,))
            conn.commit()
        return {"message": "Sua chave de API foi revogada com sucesso. Seu IP está livre para gerar uma nova chave."}
    finally:
        conn.close()

@app.get("/v1/languages")
def listar_idiomas(request: Request):
    """Lista de idiomas que o modelo Synapse-LangID consegue detectar (Endpoint Público)."""
    return {
        "model": MODEL_ID,
        "supported_languages_count": 20,
        "examples": ["pt (Português)", "en (Inglês)", "es (Espanhol)", "fr (Francês)", "de (Alemão)"],
        "note": "O modelo detecta automaticamente os principais idiomas globais baseados no dataset de treinamento."
    }
