import os
import time
import secrets
import psycopg2
from psycopg2 import pool
import torch
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import APIKeyQuery, APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
from transformers import pipeline
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# ==========================================
# ⚙️ OTIMIZAÇÃO EXTREMA DE CPU (PYTORCH)
# ==========================================
# Trava o uso de núcleos para evitar fila na Render e desliga o motor de treino
torch.set_num_threads(2)
torch.set_grad_enabled(False)

# ==========================================
# 🗄️ POOL DE CONEXÕES PERSISTENTE
# ==========================================
DATABASE_URL = os.getenv("DATABASE_URL", "").strip().replace("\r", "").replace("\n", "").replace(" ", "").replace("postgres://", "postgresql://")

db_pool = None
if DATABASE_URL:
    try:
        db_pool = psycopg2.pool.ThreadedConnectionPool(minconn=1, maxconn=5, dsn=DATABASE_URL)
        print("Pool de conexões PostgreSQL iniciado com sucesso.")
    except Exception as e:
        print(f"Erro crítico ao inicializar pool do banco: {e}")

def get_db_conn():
    if db_pool:
        return db_pool.getconn()
    return None

def release_db_conn(conn):
    if db_pool and conn:
        db_pool.putconn(conn)

def init_db():
    conn = get_db_conn()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS api_keys (
                        ip TEXT PRIMARY KEY,
                        key TEXT UNIQUE NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.commit()
            print("Tabela api_keys verificada.")
        except Exception as e:
            print(f"Erro ao inicializar tabela: {e}")
        finally:
            release_db_conn(conn)

init_db()

# ==========================================
# ⚡ CACHE EM RAM (ZERO OVERHEAD DE BANCO)
# ==========================================
KEY_CACHE = {}
CACHE_TTL = 300  # 5 minutos

def is_key_cached(key: str) -> bool:
    exp = KEY_CACHE.get(key)
    return bool(exp and exp > time.time())

def cache_key(key: str):
    KEY_CACHE[key] = time.time() + CACHE_TTL

def invalidate_cache(key: str):
    KEY_CACHE.pop(key, None)

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

    # 1. Checa RAM primeiro (0ms)
    if is_key_cached(key):
        return key

    # 2. Se não estiver na RAM, vai no Banco usando o Pool
    conn = get_db_conn()
    if not conn:
        raise HTTPException(status_code=500, detail="Banco de dados indisponível.")

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT key FROM api_keys WHERE key = %s;", (key,))
            if cur.fetchone():
                cache_key(key)  # Salva na RAM para as próximas chamadas
                return key
    finally:
        release_db_conn(conn)

    raise HTTPException(status_code=401, detail="Chave inválida ou revogada.")

limiter = Limiter(key_func=get_real_ip)

# ==========================================
# 🚀 INICIALIZAÇÃO DA API
# ==========================================
app = FastAPI(title="Synapse-LangID Enterprise Auth", version="4.2.0")
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

MODEL_ID = "Comunidade-Synapse-BR/Synapse-LangID"
try:
    classifier = pipeline("text-classification", model=MODEL_ID, device=-1)
except Exception:
    classifier = None

class PredictRequest(BaseModel):
    text: str = Field(..., max_length=1500, description="Texto único para identificação")

class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., max_items=50, description="Lista de até 50 textos")

# ==========================================
# 🌐 ENDPOINTS PÚBLICOS
# ==========================================
@app.get("/")
@limiter.limit("20/minute")
def health_check(request: Request):
    db_status = "Conectado" if db_pool else "Desconectado"
    return {"status": "Online", "database": db_status, "model": "Carregado" if classifier else "Offline"}

@app.post("/gerar-chave")
@limiter.limit("5/minute")
def gerar_chave(request: Request):
    ip = get_real_ip(request)
    conn = get_db_conn()
    if not conn:
        raise HTTPException(status_code=500, detail="Banco indisponível.")

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT key FROM api_keys WHERE ip = %s;", (ip,))
            row = cur.fetchone()
            if row:
                raise HTTPException(status_code=400, detail=f"IP já possui chave: {row[0]}")

            nova_chave = "syn_" + secrets.token_hex(8)
            cur.execute("INSERT INTO api_keys (ip, key) VALUES (%s, %s);", (ip, nova_chave))
            conn.commit()

        cache_key(nova_chave)
        return {"message": "Chave gerada!", "api_key": nova_chave}
    finally:
        release_db_conn(conn)

# ==========================================
# 🔒 ENDPOINTS PROTEGIDOS
# ==========================================
@app.post("/v1/predict", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
def predict_secure(request: Request, payload: PredictRequest):
    if not classifier:
        raise HTTPException(status_code=500, detail="Modelo Offline.")
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Texto vazio.")

    # Inferência isolada e ultrarrápida
    with torch.inference_mode():
        result = classifier(payload.text)[0]

    return {
        "prediction": {
            "language": result["label"],
            "confidence": round(result["score"], 4)
        }
    }

@app.post("/v1/predict-batch", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
def predict_batch(request: Request, payload: BatchPredictRequest):
    """Processa vários textos em uma única requisição (Alta Performance)."""
    if not classifier:
        raise HTTPException(status_code=500, detail="Modelo Offline.")
    if not payload.texts:
        raise HTTPException(status_code=400, detail="Lista vazia.")

    clean_texts = [t[:1500] for t in payload.texts if t.strip()]
    if not clean_texts:
        raise HTTPException(status_code=400, detail="Textos inválidos.")

    with torch.inference_mode():
        # O modelo processa a lista inteira de uma só vez na memória
        raw_results = classifier(clean_texts, batch_size=8)

    results = [{"language": res["label"], "confidence": round(res["score"], 4)} for res in raw_results]
    return {"count": len(results), "predictions": results}

@app.get("/v1/key-info", dependencies=[Depends(verify_api_key)])
@limiter.limit("15/minute")
def info_chave(request: Request, key: str = Depends(verify_api_key)):
    if key == MASTER_KEY:
        return {"key": "MASTER_KEY", "type": "admin", "status": "active"}

    conn = get_db_conn()
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
        release_db_conn(conn)
    
    raise HTTPException(status_code=404, detail="Não encontrado.")

@app.delete("/v1/revoke-key", dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")
def revogar_chave(request: Request, key: str = Depends(verify_api_key)):
    if key == MASTER_KEY:
        raise HTTPException(status_code=403, detail="A MASTER_KEY não pode ser revogada.")

    conn = get_db_conn()
    if not conn:
        raise HTTPException(status_code=500, detail="Banco indisponível.")

    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM api_keys WHERE key = %s;", (key,))
            conn.commit()
        invalidate_cache(key)
        return {"message": "Chave revogada. Seu IP está liberado."}
    finally:
        release_db_conn(conn)

@app.get("/v1/languages")
def listar_idiomas(request: Request):
    return {
        "model": MODEL_ID,
        "batch_supported": True,
        "max_batch_size": 50
    }
