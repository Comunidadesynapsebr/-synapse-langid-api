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
from huggingface_hub import hf_hub_download
from contextlib import asynccontextmanager

# ==============================================================================
# 1. CONFIGURAÇÕES
# ==============================================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://user:pass@localhost:5432/synapse"
)

MAX_TOKENS_MONTH = 10_000_000
MIN_TOKEN_COST = 10

# Repositório ONNX da Synapse
HF_REPO = "Comunidade-Synapse-BR/Synapse-LangID-ONNX"
HF_MODEL_FILE = "model-int8.onnx"

# Modelo-base responsável pelo tokenizer
# IMPORTANTE:
# troque pelo repositório do modelo original caso seu tokenizer
# tenha vindo de outro lugar.
TOKENIZER_REPO = os.getenv(
    "TOKENIZER_REPO",
    HF_REPO
)

MODEL_DIR = "./synapse_langid_onnx"
MODEL_PATH = os.path.join(MODEL_DIR, HF_MODEL_FILE)

auth_cache = TTLCache(maxsize=1000, ttl=300)

db_pool = None
tokenizer = None
ort_session = None
id2label = {}

# ==============================================================================
# 2. DOWNLOAD DO MODELO ONNX
# ==============================================================================

def preparar_modelo():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print("📥 Baixando Synapse-LangID INT8 do Hugging Face...")

        downloaded_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=HF_MODEL_FILE,
            local_dir=MODEL_DIR
        )

        print(f"✅ Modelo baixado: {downloaded_path}")
    else:
        print(f"✅ Modelo já existe: {MODEL_PATH}")

# ==============================================================================
# 3. LIFESPAN
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    global tokenizer
    global ort_session
    global id2label

    print("=" * 70)
    print("🚀 INICIANDO SYNAPSE-LANGID API")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # Banco de dados
    # --------------------------------------------------------------------------

    db_pool = psycopg2.pool.ThreadedConnectionPool(
        1,
        10,
        dsn=DATABASE_URL
    )

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

    print("✅ PostgreSQL conectado")

    # --------------------------------------------------------------------------
    # Modelo
    # --------------------------------------------------------------------------

    preparar_modelo()

    # --------------------------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------------------------

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_REPO
        )

        print("✅ Tokenizer carregado")

    except Exception as e:
        print("❌ Erro ao carregar tokenizer")
        print(e)
        raise

    # --------------------------------------------------------------------------
    # Config / Labels
    # --------------------------------------------------------------------------

    config_path = os.path.join(MODEL_DIR, "config.json")

    if os.path.exists(config_path):

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        id2label = {
            int(k): v
            for k, v in config.get("id2label", {}).items()
        }

    # --------------------------------------------------------------------------
    # ONNX Runtime
    # --------------------------------------------------------------------------

    print("⚡ Inicializando ONNX Runtime...")

    opts = ort.SessionOptions()

    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    ort_session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=opts,
        providers=[
            "CPUExecutionProvider"
        ]
    )

    print("✅ Synapse-LangID INT8 carregado")
    print(f"📦 Modelo: {MODEL_PATH}")
    print(f"🧠 Inputs: {[x.name for x in ort_session.get_inputs()]}")
    print(f"🎯 Outputs: {[x.name for x in ort_session.get_outputs()]}")

    yield

    # --------------------------------------------------------------------------
    # Shutdown
    # --------------------------------------------------------------------------

    if db_pool:
        db_pool.closeall()

    print("🛑 API encerrada")


# ==============================================================================
# 4. FASTAPI
# ==============================================================================

limiter = Limiter(
    key_func=get_remote_address
)

app = FastAPI(
    lifespan=lifespan,
    title="Synapse-LangID API",
    version="5.0.0"
)

app.state.limiter = limiter

app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)

# ==============================================================================
# 5. MIDDLEWARE
# ==============================================================================

@app.middleware("http")
async def limit_body_size(
    request: Request,
    call_next
):
    content_length = request.headers.get("content-length")

    if content_length:

        if int(content_length) > 256_000:

            return JSONResponse(
                status_code=413,
                content={
                    "detail":
                    "Payload Too Large (Max 256KB)"
                }
            )

    return await call_next(request)


# ==============================================================================
# 6. DATABASE
# ==============================================================================

def get_db_connection():

    conn = db_pool.getconn()

    try:
        yield conn

    finally:
        db_pool.putconn(conn)


# ==============================================================================
# 7. TOKEN ESTIMATION
# ==============================================================================

def estimar_tokens(text: str) -> int:

    chars = len(text)

    estimated = (
        chars // 4
        + (1 if chars % 4 else 0)
    )

    return max(
        MIN_TOKEN_COST,
        estimated
    )


# ==============================================================================
# 8. AUTENTICAÇÃO
# ==============================================================================

def verificar_autenticacao(

    request: Request,

    x_api_key: str = Header(
        None,
        alias="x-api-key"
    ),

    key: str = None,

    conn=Depends(get_db_connection)
):

    api_key = x_api_key or key

    if not api_key:

        raise HTTPException(
            status_code=401,
            detail="API Key ausente"
        )

    # --------------------------------------------------------------------------
    # Renovação automática após 30 dias
    # --------------------------------------------------------------------------

    with conn.cursor() as cur:

        cur.execute("""
            UPDATE api_keys
            SET
                tokens_used = 0,
                period_start = CURRENT_TIMESTAMP

            WHERE
                api_key = %s
                AND period_start + INTERVAL '30 days'
                    <= CURRENT_TIMESTAMP

            RETURNING
                tokens_used,
                monthly_quota,
                status
        """, (api_key,))

        conn.commit()

    # --------------------------------------------------------------------------
    # Cache
    # --------------------------------------------------------------------------

    if api_key in auth_cache:

        account = auth_cache[api_key]

    else:

        with conn.cursor() as cur:

            cur.execute("""
                SELECT
                    tokens_used,
                    monthly_quota,
                    status

                FROM api_keys

                WHERE api_key = %s
            """, (api_key,))

            row = cur.fetchone()

        if not row:

            raise HTTPException(
                status_code=401,
                detail="Chave inválida"
            )

        account = {
            "tokens_used": row[0],
            "monthly_quota": row[1],
            "status": row[2]
        }

        auth_cache[api_key] = account

    if account["status"] != "active":

        raise HTTPException(
            status_code=403,
            detail="Chave inativa ou revogada"
        )

    if account["tokens_used"] >= account["monthly_quota"]:

        raise HTTPException(
            status_code=429,
            detail="Cota mensal excedida"
        )

    return api_key


# ==============================================================================
# 9. PAYLOADS
# ==============================================================================

class PredictRequest(BaseModel):

    text: constr(
        min_length=2,
        max_length=1500
    )


class BatchPredictRequest(BaseModel):

    texts: conlist(
        constr(
            min_length=2,
            max_length=1500
        ),
        min_length=1,
        max_length=30
    )


# ==============================================================================
# 10. INFERÊNCIA ONNX
# ==============================================================================

def rodar_inferencia_onnx(
    texts: list[str]
):

    if tokenizer is None:

        raise RuntimeError(
            "Tokenizer não carregado"
        )

    if ort_session is None:

        raise RuntimeError(
            "Modelo ONNX não carregado"
        )

    # --------------------------------------------------------------------------
    # Tokenização
    # --------------------------------------------------------------------------

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="np"
    )

    # --------------------------------------------------------------------------
    # Inputs ONNX
    # --------------------------------------------------------------------------

    ort_inputs = {}

    input_names = {
        item.name
        for item in ort_session.get_inputs()
    }

    if "input_ids" in input_names:

        ort_inputs["input_ids"] = (
            inputs["input_ids"].astype(np.int64)
        )

    if "attention_mask" in input_names:

        ort_inputs["attention_mask"] = (
            inputs["attention_mask"].astype(np.int64)
        )

    if "token_type_ids" in input_names:

        ort_inputs["token_type_ids"] = (
            inputs["token_type_ids"].astype(np.int64)
        )

    # --------------------------------------------------------------------------
    # Inferência
    # --------------------------------------------------------------------------

    outputs = ort_session.run(
        None,
        ort_inputs
    )

    logits = outputs[0]

    # --------------------------------------------------------------------------
    # Softmax
    # --------------------------------------------------------------------------

    logits = logits - np.max(
        logits,
        axis=-1,
        keepdims=True
    )

    exp_logits = np.exp(logits)

    probs = (
        exp_logits
        /
        np.sum(
            exp_logits,
            axis=-1,
            keepdims=True
        )
    )

    # --------------------------------------------------------------------------
    # Resultado
    # --------------------------------------------------------------------------

    predictions = []

    for prob in probs:

        idx = int(
            np.argmax(prob)
        )

        score = float(
            prob[idx]
        )

        label = id2label.get(
            idx,
            str(idx)
        )

        predictions.append({
            "language": label,
            "confidence": round(
                score,
                4
            )
        })

    return predictions


# ==============================================================================
# 11. HEALTH CHECK
# ==============================================================================

@app.get("/")
@limiter.limit("20/minute")
async def health_check(
    request: Request
):

    return {

        "status": "Online",

        "database":
            "Conectado"
            if db_pool
            else "Desconectado",

        "model":
            "Synapse-LangID INT8 carregado"
            if ort_session
            else "Carregando",

        "runtime":
            "ONNX Runtime",

        "quantization":
            "INT8"

    }


# ==============================================================================
# 12. GERAR API KEY
# ==============================================================================

@app.post("/gerar-chave")
@limiter.limit("3/hour")
async def gerar_chave(
    request: Request,
    conn=Depends(get_db_connection)
):

    ip_addr = request.client.host

    new_key = (
        f"syn_{secrets.token_hex(8)}"
    )

    try:

        with conn.cursor() as cur:

            cur.execute(
                """
                INSERT INTO api_keys
                    (api_key, ip_address)

                VALUES
                    (%s, %s)
                """,
                (
                    new_key,
                    ip_addr
                )
            )

            conn.commit()

    except psycopg2.errors.UniqueViolation:

        conn.rollback()

        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT api_key
                FROM api_keys
                WHERE ip_address = %s
                """,
                (ip_addr,)
            )

            existing_key = cur.fetchone()[0]

        raise HTTPException(
            status_code=400,
            detail=f"IP já possui chave: {existing_key}"
        )

    return {
        "message": "Chave gerada!",
        "api_key": new_key
    }


# ==============================================================================
# 13. PREDICT
# ==============================================================================

@app.post("/v1/predict")
@limiter.limit("30/minute")
async def predict(
    request: Request,

    payload: PredictRequest,

    conn=Depends(get_db_connection),

    api_key: str = Depends(
        verificar_autenticacao
    )
):

    cost = estimar_tokens(
        payload.text
    )

    with conn.cursor() as cur:

        cur.execute(
            """
            UPDATE api_keys

            SET tokens_used =
                tokens_used + %s

            WHERE api_key = %s

            RETURNING
                tokens_used,
                monthly_quota
            """,
            (
                cost,
                api_key
            )
        )

        row = cur.fetchone()

        if row is None:

            raise HTTPException(
                status_code=401,
                detail="Chave inválida"
            )

        updated_used, quota = row

        conn.commit()

    if updated_used > quota:

        raise HTTPException(
            status_code=429,
            detail=
                "Requisição excede "
                "a cota mensal restante."
        )

    auth_cache.pop(
        api_key,
        None
    )

    predictions = rodar_inferencia_onnx(
        [payload.text]
    )

    remaining = (
        quota - updated_used
    )

    response = JSONResponse({

        "prediction":
            predictions[0],

        "usage": {
            "prompt_tokens": cost,
            "remaining_tokens": remaining
        }

    })

    response.headers[
        "x-ratelimit-remaining-tokens"
    ] = str(remaining)

    return response


# ==============================================================================
# 14. PREDICT BATCH
# ==============================================================================

@app.post("/v1/predict-batch")
@limiter.limit("6/minute")
async def predict_batch(

    request: Request,

    payload: BatchPredictRequest,

    conn=Depends(get_db_connection),

    api_key: str = Depends(
        verificar_autenticacao
    )

):

    total_chars = sum(
        len(t)
        for t in payload.texts
    )

    if total_chars > 15_000:

        raise HTTPException(
            status_code=413,
            detail=
                "Lote excede o limite "
                "cumulativo de 15.000 caracteres."
        )

    cost = sum(
        estimar_tokens(t)
        for t in payload.texts
    )

    with conn.cursor() as cur:

        cur.execute(
            """
            UPDATE api_keys

            SET tokens_used =
                tokens_used + %s

            WHERE api_key = %s

            RETURNING
                tokens_used,
                monthly_quota
            """,
            (
                cost,
                api_key
            )
        )

        row = cur.fetchone()

        if row is None:

            raise HTTPException(
                status_code=401,
                detail="Chave inválida"
            )

        updated_used, quota = row

        conn.commit()

    if updated_used > quota:

        raise HTTPException(
            status_code=429,
            detail=
                "Requisição excede "
                "a cota mensal restante."
        )

    auth_cache.pop(
        api_key,
        None
    )

    predictions = rodar_inferencia_onnx(
        payload.texts
    )

    remaining = (
        quota - updated_used
    )

    response = JSONResponse({

        "count":
            len(payload.texts),

        "predictions":
            predictions,

        "usage": {
            "prompt_tokens": cost,
            "remaining_tokens": remaining
        }

    })

    response.headers[
        "x-ratelimit-remaining-tokens"
    ] = str(remaining)

    return response


# ==============================================================================
# 15. KEY INFO
# ==============================================================================

@app.get("/v1/key-info")
@limiter.limit("15/minute")
async def key_info(

    request: Request,

    conn=Depends(get_db_connection),

    api_key: str = Depends(
        verificar_autenticacao
    )
):

    with conn.cursor() as cur:

        cur.execute(
            """
            SELECT
                ip_address,
                monthly_quota,
                tokens_used,
                period_start,
                status

            FROM api_keys

            WHERE api_key = %s
            """,
            (api_key,)
        )

        row = cur.fetchone()

    if not row:

        raise HTTPException(
            status_code=404,
            detail="Chave não encontrada"
        )

    return {

        "key_prefix":
            f"{api_key[:8]}...",

        "status":
            row[4],

        "monthly_quota":
            row[1],

        "tokens_used":
            row[2],

        "tokens_remaining":
            row[1] - row[2],

        "period_start":
            row[3].isoformat(),

        "registered_ip":
            row[0]

    }


# ==============================================================================
# 16. REVOKE
# ==============================================================================

@app.delete("/v1/revoke-key")
@limiter.limit("5/hour")
async def revoke_key(

    request: Request,

    conn=Depends(get_db_connection),

    api_key: str = Depends(
        verificar_autenticacao
    )
):

    with conn.cursor() as cur:

        cur.execute(
            """
            DELETE FROM api_keys
            WHERE api_key = %s
            """,
            (api_key,)
        )

        conn.commit()

    auth_cache.pop(
        api_key,
        None
    )

    return {
        "message":
            "Chave revogada com sucesso. "
            "O IP está liberado."
    }


# ==============================================================================
# 17. LANGUAGES
# ==============================================================================

@app.get("/v1/languages")
async def languages():

    return {

        "model":
            "Comunidade-Synapse-BR/Synapse-LangID-ONNX",

        "file":
            "model-int8.onnx",

        "runtime":
            "ONNX Runtime",

        "quantization":
            "INT8",

        "batch_supported":
            True,

        "max_batch_size":
            30

        }
