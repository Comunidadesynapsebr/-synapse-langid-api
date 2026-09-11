import os
import json
import secrets
import numpy as np
import onnxruntime as ort
import psycopg2
from psycopg2 import pool
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, constr, conlist
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from cachetools import TTLCache
from transformers import AutoTokenizer
from contextlib import asynccontextmanager

# ==============================================================================
# 1. CONFIGURAÇÕES E INFRAESTRUTURA
# ==============================================================================

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/synapse")
MAX_TOKENS_MONTH = 10_000_000
MIN_TOKEN_COST = 10
MODEL_DIR = "./synapse_langid_onnx"  # Pasta onde o modelo ONNX foi exportado

auth_cache = TTLCache(maxsize=1000, ttl=300)
db_pool = None

# Variáveis globais para o ONNX
tokenizer = None
ort_session = None
id2label = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool, tokenizer, ort_session, id2label
    
    # 1. Conecta ao PostgreSQL
    db_pool = psycopg2.pool.ThreadedConnectionPool(1, 10, dsn=DATABASE_URL)
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    api_key VARCHAR(64) PRIMARY KEY,
                    ip_address VARCHAR(45) UNIQUE NOT NULL,
                    monthly_quota BIGINT DEFAULT 10000000,
                    tokens_used BIGINT DEFAULT 0,
                    period_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status VARCHAR(20) DEFAULT 'active'
                )
            """)
            conn.commit()
    finally:
        db_pool.putconn(conn)
        
    # 2. Carrega o Tokenizer e o Config
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    
    with open(os.path.join(MODEL_DIR, "config.json"), "r") as f:
        config = json.load(f)
        # O id2label JSON tem chaves como string, precisamos converter para inteiro
        id2label = {int(k): v for k, v in config.get("id2label", {}).items()}
    
    # 3. Inicializa o ONNX Runtime (Otimizado para CPU da Render)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    ort_session = ort.InferenceSession(
        os.path.join(MODEL_DIR, "model.onnx"), 
        sess_options=opts, 
        providers=["CPUExecutionProvider"]
    )
    
    yield
    
    if db_pool:
        db_pool.closeall()

# Rate Limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(lifespan=lifespan, title="Synapse-LangID API", version="4.3.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ==============================================================================
# 2. MIDDLEWARES E DEPENDÊNCIAS DE AUTENTICAÇÃO
# ==============================================================================

@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    if request.headers.get("content-length"):
        if int(request.headers["content-length"]) > 256000:
            return JSONResponse(status_code=413, content={"detail": "Payload Too Large (Max 256KB)"})
    return await call_next(request)

def get_db_connection():
    conn = db_pool.getconn()
    try:
        yield conn
    finally:
        db_pool.putconn(conn)

def estimar_tokens(text: str) -> int:
    chars = len(text)
    estimated = max(MIN_TOKEN_COST, chars // 4 + (1 if chars % 4 else 0))
    return estimated

def verificar_autenticacao(
    request: Request, 
    x_api_key: str = Header(None, alias="x-api-key"),
    key: str = None,
    conn = Depends(get_db_connection)
):
    api_key = x_api_key or key
    if not api_key:
        raise HTTPException(status_code=401, detail="API Key ausente")
        
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE api_keys 
            SET tokens_used = 0, period_start = CURRENT_TIMESTAMP
            WHERE api_key = %s AND period_start + INTERVAL '30 days' <= CURRENT_TIMESTAMP
            RETURNING tokens_used, monthly_quota, status
        """, (api_key,))
        conn.commit()

    if api_key in auth_cache:
        account = auth_cache[api_key]
    else:
        with conn.cursor() as cur:
            cur.execute("SELECT tokens_used, monthly_quota, status FROM api_keys WHERE api_key = %s", (api_key,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=401, detail="Chave inválida")
            account = {"tokens_used": row[0], "monthly_quota": row[1], "status": row[2]}
            auth_cache[api_key] = account

    if account["status"] != "active":
        raise HTTPException(status_code=403, detail="Chave inativa ou revogada")
        
    if account["tokens_used"] >= account["monthly_quota"]:
        raise HTTPException(status_code=429, detail="Cota mensal excedida (10.000.000 tokens)")

    return api_key

# ==============================================================================
# 3. VALIDAÇÃO DE PAYLOAD E MOTOR DE INFERÊNCIA
# ==============================================================================

class PredictRequest(BaseModel):
    text: constr(min_length=2, max_length=1500)

class BatchPredictRequest(BaseModel):
    texts: conlist(constr(min_length=2, max_length=1500), min_length=1, max_length=30)

def rodar_inferencia_onnx(texts: list[str]):
    inputs = tokenizer(
        texts, padding=True, truncation=True, max_length=256, return_tensors="np"
    )
    ort_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }
    
    logits = ort_session.run(None, ort_inputs)[0]
    
    # Softmax otimizado via NumPy
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    predictions = []
    for prob in probs:
        idx = int(np.argmax(prob))
        score = float(prob[idx])
        label = id2label.get(idx, str(idx))
        predictions.append({"language": label, "confidence": round(score, 4)})
        
    return predictions

# ==============================================================================
# 4. ENDPOINTS DA API
# ==============================================================================

@app.get("/")
@limiter.limit("20/minute")
async def health_check(request: Request):
    return {
        "status": "Online",
        "database": "Conectado" if db_pool else "Desconectado",
        "model": "Carregado (ONNX Runtime)" if ort_session else "Carregando"
    }

@app.post("/gerar-chave")
@limiter.limit("3/hour")
async def gerar_chave(request: Request, conn = Depends(get_db_connection)):
    ip_addr = request.client.host
    new_key = f"syn_{secrets.token_hex(8)}"
    
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO api_keys (api_key, ip_address) VALUES (%s, %s)",
                (new_key, ip_addr)
            )
            conn.commit()
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute("SELECT api_key FROM api_keys WHERE ip_address = %s", (ip_addr,))
            existing_key = cur.fetchone()[0]
        raise HTTPException(status_code=400, detail=f"IP já possui chave: {existing_key}")

    return {"message": "Chave gerada!", "api_key": new_key}

@app.post("/v1/predict")
@limiter.limit("30/minute")
async def predict(
    request: Request, 
    payload: PredictRequest, 
    conn = Depends(get_db_connection),
    api_key: str = Depends(verificar_autenticacao)
):
    cost = estimar_tokens(payload.text)
    
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE api_keys SET tokens_used = tokens_used + %s 
            WHERE api_key = %s RETURNING tokens_used, monthly_quota
        """, (cost, api_key))
        updated_used, quota = cur.fetchone()
        conn.commit()
        
    if updated_used > quota:
        raise HTTPException(status_code=429, detail="Requisição excede a cota mensal restante.")

    auth_cache.pop(api_key, None)

    # Inferencia super rápida na CPU
    predictions = rodar_inferencia_onnx([payload.text])

    response = JSONResponse({
        "prediction": predictions[0],
        "usage": {
            "prompt_tokens": cost,
            "remaining_tokens": quota - updated_used
        }
    })
    response.headers["x-ratelimit-remaining-tokens"] = str(quota - updated_used)
    return response

@app.post("/v1/predict-batch")
@limiter.limit("6/minute")
async def predict_batch(
    request: Request, 
    payload: BatchPredictRequest,
    conn = Depends(get_db_connection),
    api_key: str = Depends(verificar_autenticacao)
):
    total_chars = sum(len(t) for t in payload.texts)
    if total_chars > 15000:
        raise HTTPException(status_code=413, detail="Lote excede o limite cumulativo de 15.000 caracteres.")
        
    cost = sum(estimar_tokens(t) for t in payload.texts)
    
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE api_keys SET tokens_used = tokens_used + %s 
            WHERE api_key = %s RETURNING tokens_used, monthly_quota
        """, (cost, api_key))
        updated_used, quota = cur.fetchone()
        conn.commit()
        
    if updated_used > quota:
        raise HTTPException(status_code=429, detail="Requisição excede a cota mensal restante.")
        
    auth_cache.pop(api_key, None)

    predictions = rodar_inferencia_onnx(payload.texts)

    response = JSONResponse({
        "count": len(payload.texts),
        "predictions": predictions,
        "usage": {
            "prompt_tokens": cost,
            "remaining_tokens": quota - updated_used
        }
    })
    response.headers["x-ratelimit-remaining-tokens"] = str(quota - updated_used)
    return response

@app.get("/v1/key-info")
@limiter.limit("15/minute")
async def key_info(request: Request, conn = Depends(get_db_connection), api_key: str = Depends(verificar_autenticacao)):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT ip_address, monthly_quota, tokens_used, period_start, status 
            FROM api_keys WHERE api_key = %s
        """, (api_key,))
        row = cur.fetchone()
        
    return {
        "key_prefix": f"{api_key[:8]}...",
        "status": row[4],
        "monthly_quota": row[1],
        "tokens_used": row[2],
        "tokens_remaining": row[1] - row[2],
        "period_start": row[3].isoformat(),
        "registered_ip": row[0]
    }

@app.delete("/v1/revoke-key")
@limiter.limit("5/hour")
async def revoke_key(request: Request, conn = Depends(get_db_connection), api_key: str = Depends(verificar_autenticacao)):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM api_keys WHERE api_key = %s", (api_key,))
        conn.commit()
        
    auth_cache.pop(api_key, None)
    return {"message": "Chave revogada com sucesso. O IP está liberado."}

@app.get("/v1/languages")
async def languages():
    return {
        "model": "Comunidade-Synapse-BR/Synapse-LangID (ONNX)",
        "batch_supported": True,
        "max_batch_size": 30
    }
langid_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool, langid_pipeline
    
    # 1. Conecta ao Banco
    db_pool = psycopg2.pool.ThreadedConnectionPool(1, 10, dsn=DATABASE_URL)
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    api_key VARCHAR(64) PRIMARY KEY,
                    ip_address VARCHAR(45) UNIQUE NOT NULL,
                    monthly_quota BIGINT DEFAULT 10000000,
                    tokens_used BIGINT DEFAULT 0,
                    period_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status VARCHAR(20) DEFAULT 'active'
                )
            """)
            conn.commit()
    finally:
        db_pool.putconn(conn)
        
    # 2. Carrega o Modelo (em modo de inferência)
    langid_pipeline = pipeline(
        "text-classification", 
        model="papluca/xlm-roberta-base-language-detection", # Substitua pelo modelo Synapse exato
        truncation=True, 
        max_length=256
    )
    
    yield
    
    # Encerramento
    if db_pool:
        db_pool.closeall()

# Rate Limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(lifespan=lifespan, title="Synapse-LangID API", version="4.2.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ==============================================================================
# 2. MIDDLEWARES E DEPENDÊNCIAS DE AUTENTICAÇÃO
# ==============================================================================

@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    # Proteção de RAM: Rejeita payloads maiores que 256KB
    if request.headers.get("content-length"):
        if int(request.headers["content-length"]) > 256000:
            return JSONResponse(status_code=413, content={"detail": "Payload Too Large (Max 256KB)"})
    return await call_next(request)

def get_db_connection():
    conn = db_pool.getconn()
    try:
        yield conn
    finally:
        db_pool.putconn(conn)

def estimar_tokens(text: str) -> int:
    # 1 token ~= 4 caracteres. Aplica custo mínimo.
    chars = len(text)
    estimated = max(MIN_TOKEN_COST, chars // 4 + (1 if chars % 4 else 0))
    return estimated

def verificar_autenticacao e_cota(
    request: Request, 
    x_api_key: str = Header(None, alias="x-api-key"),
    key: str = None,
    conn = Depends(get_db_connection)
):
    api_key = x_api_key or key
    if not api_key:
        raise HTTPException(status_code=401, detail="API Key ausente")
        
    # Verifica renovação de ciclo de 30 dias
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE api_keys 
            SET tokens_used = 0, period_start = CURRENT_TIMESTAMP
            WHERE api_key = %s AND period_start + INTERVAL '30 days' <= CURRENT_TIMESTAMP
            RETURNING tokens_used, monthly_quota, status
        """, (api_key,))
        conn.commit()

    # Checa Cache
    if api_key in auth_cache:
        account = auth_cache[api_key]
    else:
        with conn.cursor() as cur:
            cur.execute("SELECT tokens_used, monthly_quota, status FROM api_keys WHERE api_key = %s", (api_key,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=401, detail="Chave inválida")
            account = {"tokens_used": row[0], "monthly_quota": row[1], "status": row[2]}
            auth_cache[api_key] = account

    if account["status"] != "active":
        raise HTTPException(status_code=403, detail="Chave inativa ou revogada")
        
    if account["tokens_used"] >= account["monthly_quota"]:
        raise HTTPException(status_code=429, detail="Cota mensal excedida (10.000.000 tokens)")

    request.state.api_key = api_key
    request.state.account = account
    return api_key

# ==============================================================================
# 3. SCHEMAS PYDANTIC (VALIDAÇÃO DE PAYLOAD)
# ==============================================================================

class PredictRequest(BaseModel):
    text: constr(min_length=2, max_length=1500)

class BatchPredictRequest(BaseModel):
    texts: conlist(constr(min_length=2, max_length=1500), min_length=1, max_length=30)

# ==============================================================================
# 4. ENDPOINTS DA API
# ==============================================================================

@app.get("/")
@limiter.limit("20/minute")
async def health_check(request: Request):
    return {
        "status": "Online",
        "database": "Conectado" if db_pool else "Desconectado",
        "model": "Carregado" if langid_pipeline else "Carregando"
    }

@app.post("/gerar-chave")
@limiter.limit("3/hour")
async def gerar_chave(request: Request, conn = Depends(get_db_connection)):
    ip_addr = request.client.host
    new_key = f"syn_{secrets.token_hex(8)}"
    
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO api_keys (api_key, ip_address) VALUES (%s, %s)",
                (new_key, ip_addr)
            )
            conn.commit()
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute("SELECT api_key FROM api_keys WHERE ip_address = %s", (ip_addr,))
            existing_key = cur.fetchone()[0]
        raise HTTPException(status_code=400, detail=f"IP já possui chave: {existing_key}")

    return {"message": "Chave gerada!", "api_key": new_key}

@app.post("/v1/predict")
@limiter.limit("30/minute")
async def predict(
    request: Request, 
    payload: PredictRequest, 
    conn = Depends(get_db_connection),
    api_key: str = Depends(verificar_autenticacao_e_cota)
):
    cost = estimar_tokens(payload.text)
    
    # Atualiza consumo no banco atomicamente
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE api_keys SET tokens_used = tokens_used + %s 
            WHERE api_key = %s RETURNING tokens_used, monthly_quota
        """, (cost, api_key))
        updated_used, quota = cur.fetchone()
        conn.commit()
        
    if updated_used > quota:
        raise HTTPException(status_code=429, detail="Requisição excede a cota mensal restante.")

    # Atualiza cache
    auth_cache.pop(api_key, None)

    # Inferência isolada no processador
    with torch.inference_mode():
        result = langid_pipeline(payload.text)[0]

    response = JSONResponse({
        "prediction": {
            "language": result["label"],
            "confidence": round(result["score"], 4)
        },
        "usage": {
            "prompt_tokens": cost,
            "remaining_tokens": quota - updated_used
        }
    })
    response.headers["x-ratelimit-remaining-tokens"] = str(quota - updated_used)
    return response

@app.post("/v1/predict-batch")
@limiter.limit("6/minute")
async def predict_batch(
    request: Request, 
    payload: BatchPredictRequest,
    conn = Depends(get_db_connection),
    api_key: str = Depends(verificar_autenticacao_e_cota)
):
    total_chars = sum(len(t) for t in payload.texts)
    if total_chars > 15000:
        raise HTTPException(status_code=413, detail="Lote excede o limite cumulativo de 15.000 caracteres.")
        
    cost = sum(estimar_tokens(t) for t in payload.texts)
    
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE api_keys SET tokens_used = tokens_used + %s 
            WHERE api_key = %s RETURNING tokens_used, monthly_quota
        """, (cost, api_key))
        updated_used, quota = cur.fetchone()
        conn.commit()
        
    if updated_used > quota:
        raise HTTPException(status_code=429, detail="Requisição excede a cota mensal restante.")
        
    auth_cache.pop(api_key, None)

    with torch.inference_mode():
        results = langid_pipeline(payload.texts)

    predictions = [
        {"language": res["label"], "confidence": round(res["score"], 4)}
        for res in results
    ]

    response = JSONResponse({
        "count": len(payload.texts),
        "predictions": predictions,
        "usage": {
            "prompt_tokens": cost,
            "remaining_tokens": quota - updated_used
        }
    })
    response.headers["x-ratelimit-remaining-tokens"] = str(quota - updated_used)
    return response

@app.get("/v1/key-info")
@limiter.limit("15/minute")
async def key_info(request: Request, conn = Depends(get_db_connection), api_key: str = Depends(verificar_autenticacao_e_cota)):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT ip_address, monthly_quota, tokens_used, period_start, status 
            FROM api_keys WHERE api_key = %s
        """, (api_key,))
        row = cur.fetchone()
        
    return {
        "key_prefix": f"{api_key[:8]}...",
        "status": row[4],
        "monthly_quota": row[1],
        "tokens_used": row[2],
        "tokens_remaining": row[1] - row[2],
        "period_start": row[3].isoformat(),
        "registered_ip": row[0]
    }

@app.delete("/v1/revoke-key")
@limiter.limit("5/hour")
async def revoke_key(request: Request, conn = Depends(get_db_connection), api_key: str = Depends(verificar_autenticacao_e_cota)):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM api_keys WHERE api_key = %s", (api_key,))
        conn.commit()
        
    auth_cache.pop(api_key, None)
    return {"message": "Chave revogada com sucesso. O IP está liberado."}
================================================================================

--------------------------------------------------------------------------------
[GET] /
--------------------------------------------------------------------------------
Health Check do servico. Verifica se a API esta no ar, se o pool com o banco de
dados esta conectado e se o modelo Transformer esta carregado na memoria.

- Autenticacao: Nao requer
- Rate Limit: 20 chamadas/minuto
- Exemplo de Resposta (Status 200):
  {
    "status": "Online",
    "database": "Conectado",
    "model": "Carregado"
  }

--------------------------------------------------------------------------------
[POST] /gerar-chave
--------------------------------------------------------------------------------
Gera e armazena uma credencial unica (prefixo 'syn_') associada ao IP de origem.

- Autenticacao: Nao requer
- Rate Limit: 5 chamadas/minuto
- Exemplo de Resposta (Status 200 - Sucesso):
  {
    "message": "Chave gerada!",
    "api_key": "syn_85b21db93babe995"
  }
- Exemplo de Resposta (Status 400 - IP ja cadastrado):
  {
    "detail": "IP já possui chave: syn_85b21db93babe995"
  }

--------------------------------------------------------------------------------
[POST] /v1/predict
--------------------------------------------------------------------------------
Realiza a identificacao de idioma de uma unica string de texto.

- Autenticacao: Obrigatoria (via param 'key' ou header 'x-api-key')
- Rate Limit: 30 chamadas/minuto
- Limite de caracteres: Maximo de 1.500 caracteres por texto
- Content-Type: application/json
- Formato do Payload (Body):
  {
    "text": "Esta frase confirma se a deteccao de idioma esta funcionando."
  }
- Exemplo de Resposta (Status 200):
  {
    "prediction": {
      "language": "pt",
      "confidence": 0.9986
    }
  }

--------------------------------------------------------------------------------
[POST] /v1/predict-batch
--------------------------------------------------------------------------------
Processa multiplas frases em uma unica requisicao de rede utilizando
inferencia paralela em memoria. Recomendado para processamento massivo.

- Autenticacao: Obrigatoria (via param 'key' ou header 'x-api-key')
- Rate Limit: 10 chamadas/minuto
- Limite de lote: Ate 50 textos por chamada
- Content-Type: application/json
- Formato do Payload (Body):
  {
    "texts": [
      "Ola, tudo bem com voce?",
      "Artificial intelligence is transforming software engineering.",
      "Bonjour tout le monde."
    ]
  }
- Exemplo de Resposta (Status 200):
  {
    "count": 3,
    "predictions": [
      { "language": "pt", "confidence": 0.9985 },
      { "language": "en", "confidence": 0.9794 },
      { "language": "fr", "confidence": 0.9912 }
    ]
  }

--------------------------------------------------------------------------------
[GET] /v1/key-info
--------------------------------------------------------------------------------
Consulta as informacoes cadastrais e timestamp de criacao da chave atual.

- Autenticacao: Obrigatoria (via param 'key' ou header 'x-api-key')
- Rate Limit: 15 chamadas/minuto
- Exemplo de Resposta (Status 200):
  {
    "key_prefix": "syn_85b...",
    "registered_ip": "136.114.188.162",
    "created_at": "2026-09-11 14:15:42",
    "status": "active"
  }

--------------------------------------------------------------------------------
[DELETE] /v1/revoke-key
--------------------------------------------------------------------------------
Exclui a chave ativa do PostgreSQL e expira o cache em memoria imediatamente,
liberando o IP para gerar uma credencial nova.

- Autenticacao: Obrigatoria (via param 'key' ou header 'x-api-key')
- Rate Limit: 5 chamadas/minuto
- Exemplo de Resposta (Status 200):
  {
    "message": "Chave revogada. Seu IP está liberado."
  }

--------------------------------------------------------------------------------
[GET] /v1/languages
--------------------------------------------------------------------------------
Retorna os metadados do modelo de IA e recursos suportados.

- Autenticacao: Nao requer
- Exemplo de Resposta (Status 200):
  {
    "model": "Comunidade-Synapse-BR/Synapse-LangID",
    "batch_supported": true,
    "max_batch_size": 50
  }


================================================================================
3. EXEMPLOS DE IMPLEMENTACAO EM CODIGO
================================================================================

----------------------------------------
Exemplo em Python (requests):
----------------------------------------
import requests

BASE_URL = "https://synapse-langid-api.onrender.com"
API_KEY = "SUA_CHAVE_AQUI"

headers = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}

# 1. Chamada Individual
payload_individual = {"text": "Texto em portugues para ser analisado."}
res1 = requests.post(f"{BASE_URL}/v1/predict", json=payload_individual, headers=headers)
print("Individual:", res1.json())

# 2. Chamada em Lote (Batch)
payload_lote = {
    "texts": [
        "Frase de teste em portugues.",
        "A sample sentence written in English."
    ]
}
res2 = requests.post(f"{BASE_URL}/v1/predict-batch", json=payload_lote, headers=headers)
print("Lote:", res2.json())


----------------------------------------
Exemplo em JavaScript / Node.js (fetch):
----------------------------------------
const API_KEY = "SUA_CHAVE_AQUI";

async function classificarTexto(texto) {
  const url = "https://synapse-langid-api.onrender.com/v1/predict";
  const resposta = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": API_KEY
    },
    body: JSON.stringify({ text: texto })
  });
  
  const dados = await resposta.json();
  console.log(dados);
}

classificarTexto("Checking the language identification API.");


----------------------------------------
Exemplo em cURL (Terminal / Bash):
----------------------------------------
curl -X POST "https://synapse-langid-api.onrender.com/v1/predict" \
     -H "x-api-key: SUA_CHAVE_AQUI" \
     -H "Content-Type: application/json" \
     -d '{"text": "Exemplo simples de execucao via cURL no terminal."}'


================================================================================
4. ARQUITETURA INTERNA E OTIMIZACOES APLICADAS
================================================================================

- Fast Response Cache:
  As chaves autenticadas sao salvas em um dicionario local em RAM com TTL de
  5 minutos. Consultas subsequentes de um mesmo usuario nao precisam consultar o
  disco ou o banco de dados (0 ms de overhead de autenticacao).

- Threaded Connection Pool:
  Utiliza o pool nativo do psycopg2 para manter conexoes quentes abertas com o
  PostgreSQL da Render, eliminando atrasos de handshake TCP/TLS.

- CPU Inference Optimization:
  O pipeline roda sob 'torch.inference_mode()' com numero de threads restrito a 2,
  impedindo disputas de contexto e superaquecimento da vCPU compartilhada.

- Protecao por SlowAPI:
  Rate limiting configurado nas bordas para evitar travamento da instancia por
  denial-of-service (DDoS/spam).
