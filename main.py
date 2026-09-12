
import os
import json
import secrets
from contextlib import asynccontextmanager
from urllib.parse import urlsplit, urlunsplit

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


# ==============================================================================
# 1. CONFIGURAÇÕES
# ==============================================================================

def normalizar_env(nome: str, obrigatoria: bool = False):
    valor = os.getenv(nome)

    if valor is not None:
        valor = valor.strip()

    if obrigatoria and not valor:
        raise RuntimeError(
            f"{nome} não configurada. "
            f"Configure a variável {nome} no Render."
        )

    return valor


DATABASE_URL = normalizar_env("DATABASE_URL", obrigatoria=True)

# Proteção extra: rejeita URL de banco com newline/caracteres de controle.
if any(ord(char) < 32 and char not in ("\t",) for char in DATABASE_URL):
    raise RuntimeError(
        "DATABASE_URL contém caracteres de controle inválidos. "
        "Copie novamente a Internal Database URL do Render."
    )


# ------------------------------------------------------------------------------
# COTA
# ------------------------------------------------------------------------------

MAX_TOKENS_MONTH = 10_000_000
MIN_TOKEN_COST = 10


# ------------------------------------------------------------------------------
# HUGGING FACE
# ------------------------------------------------------------------------------

HF_REPO = normalizar_env(
    "HF_REPO"
) or "Comunidade-Synapse-BR/Synapse-LangID-ONNX"

HF_MODEL_FILE = "model-int8.onnx"

TOKENIZER_REPO = normalizar_env(
    "TOKENIZER_REPO",
    obrigatoria=True
)


# ------------------------------------------------------------------------------
# DIRETÓRIOS
# ------------------------------------------------------------------------------

MODEL_DIR = "./synapse_langid_onnx"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    HF_MODEL_FILE
)


# ------------------------------------------------------------------------------
# CACHE
# ------------------------------------------------------------------------------

auth_cache = TTLCache(
    maxsize=1000,
    ttl=300
)


# ------------------------------------------------------------------------------
# GLOBAIS
# ------------------------------------------------------------------------------

db_pool = None
tokenizer = None
ort_session = None
id2label = {}


# ==============================================================================
# 2. DOWNLOAD DO MODELO INT8
# ==============================================================================

def preparar_modelo():
    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    if not os.path.isfile(MODEL_PATH):
        print("📥 Baixando Synapse-LangID INT8...")

        downloaded_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=HF_MODEL_FILE,
            local_dir=MODEL_DIR
        )

        print(
            f"✅ Modelo baixado: {downloaded_path}"
        )
    else:
        print(
            f"✅ Modelo já existe: {MODEL_PATH}"
        )


# ==============================================================================
# 3. BANCO
# ==============================================================================

def inicializar_banco():
    global db_pool

    print("🗄️ Conectando ao PostgreSQL...")

    try:
        db_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=DATABASE_URL,
        )

    except psycopg2.Error as e:
        print("❌ Falha ao conectar ao PostgreSQL")
        print(f"Tipo: {type(e).__name__}")
        print(f"Erro: {e}")
        raise RuntimeError(
            "Não foi possível conectar ao PostgreSQL. "
            "Verifique DATABASE_URL no Render."
        ) from e

    conn = None

    try:
        conn = db_pool.getconn()

        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS api_keys (
                    api_key VARCHAR(64) PRIMARY KEY,
                    ip_address VARCHAR(45) UNIQUE NOT NULL,
                    monthly_quota BIGINT NOT NULL DEFAULT 10000000,
                    tokens_used BIGINT NOT NULL DEFAULT 0,
                    period_start TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    status VARCHAR(20) NOT NULL DEFAULT 'active'
                )
                """
            )

        conn.commit()

    except Exception:
        if conn:
            conn.rollback()
        raise

    finally:
        if conn and db_pool:
            db_pool.putconn(conn)

    print("✅ PostgreSQL conectado")


# ==============================================================================
# 4. TOKENIZER
# ==============================================================================

def carregar_tokenizer():
    global tokenizer

    print(
        f"🔤 Carregando tokenizer: {TOKENIZER_REPO}"
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_REPO
        )

    except Exception as e:
        print("❌ Erro ao carregar tokenizer")
        print(str(e))

        raise RuntimeError(
            "Não foi possível carregar o tokenizer. "
            "Verifique TOKENIZER_REPO."
        ) from e

    print("✅ Tokenizer carregado")


# ==============================================================================
# 5. CONFIG / LABELS
# ==============================================================================

def carregar_labels():
    global id2label

    config_path = os.path.join(
        MODEL_DIR,
        "config.json"
    )

    if not os.path.isfile(config_path):
        print(
            "⚠️ config.json não encontrado. "
            "Os labels serão retornados como IDs."
        )

        id2label = {}
        return

    try:
        with open(
            config_path,
            "r",
            encoding="utf-8"
        ) as f:
            config = json.load(f)

        id2label = {
            int(k): v
            for k, v in config.get(
                "id2label",
                {}
            ).items()
        }

        print(
            f"✅ {len(id2label)} labels carregados"
        )

    except Exception as e:
        print("⚠️ Erro ao carregar labels")
        print(str(e))
        id2label = {}


# ==============================================================================
# 6. ONNX RUNTIME
# ==============================================================================

def carregar_onnx():
    global ort_session

    print("⚡ Inicializando ONNX Runtime...")

    if not os.path.isfile(MODEL_PATH):
        raise RuntimeError(
            f"Modelo ONNX não encontrado: {MODEL_PATH}"
        )

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
        providers=["CPUExecutionProvider"],
    )

    print("✅ Synapse-LangID INT8 carregado")
    print("📦 Modelo:", MODEL_PATH)

    print(
        "🧠 Inputs:",
        [
            x.name
            for x in ort_session.get_inputs()
        ]
    )

    print(
        "🎯 Outputs:",
        [
            x.name
            for x in ort_session.get_outputs()
        ]
    )


# ==============================================================================
# 7. LIFESPAN
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool

    print("=" * 70)
    print("🚀 INICIANDO SYNAPSE-LANGID API")
    print("=" * 70)

    try:
        print("1/5 🗄️ Banco")
        inicializar_banco()

        print("2/5 📥 Modelo")
        preparar_modelo()

        print("3/5 🔤 Tokenizer")
        carregar_tokenizer()

        print("4/5 🏷️ Labels")
        carregar_labels()

        print("5/5 ⚡ ONNX")
        carregar_onnx()

        print("=" * 70)
        print("✅ SYNAPSE-LANGID API ONLINE")
        print("=" * 70)

        yield

    except Exception:
        print("=" * 70)
        print("❌ FALHA DURANTE O STARTUP")
        print("=" * 70)

        if db_pool:
            db_pool.closeall()
            db_pool = None

        raise

    finally:
        if db_pool:
            db_pool.closeall()
            db_pool = None
            print("🗄️ Pool PostgreSQL fechado")

        auth_cache.clear()
        print("🛑 API encerrada")


# ==============================================================================
# 8. FASTAPI
# ==============================================================================

limiter = Limiter(
    key_func=get_remote_address
)

app = FastAPI(
    lifespan=lifespan,
    title="Synapse-LangID API",
    version="5.1.1",
)

app.state.limiter = limiter

app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)


# ==============================================================================
# 9. MIDDLEWARE — LIMITE DE PAYLOAD
# ==============================================================================

@app.middleware("http")
async def limit_body_size(
    request: Request,
    call_next
):
    content_length = request.headers.get(
        "content-length"
    )

    if content_length:
        try:
            size = int(content_length)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={
                    "detail": "Content-Length inválido"
                }
            )

        if size < 0:
            return JSONResponse(
                status_code=400,
                content={
                    "detail": "Content-Length inválido"
                }
            )

        if size > 256_000:
            return JSONResponse(
                status_code=413,
                content={
                    "detail": "Payload Too Large (Max 256KB)"
                }
            )

    return await call_next(request)


# ==============================================================================
# 10. DATABASE DEPENDENCY
# ==============================================================================

def get_db_connection():
    if db_pool is None:
        raise HTTPException(
            status_code=503,
            detail="Banco de dados indisponível"
        )

    conn = None

    try:
        conn = db_pool.getconn()

        if conn.closed:
            raise RuntimeError(
                "Conexão PostgreSQL fechada."
            )

        yield conn

    except HTTPException:
        raise

    except Exception as e:
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass

        raise HTTPException(
            status_code=503,
            detail="Erro ao obter conexão com o banco"
        ) from e

    finally:
        if conn and db_pool:
            try:
                db_pool.putconn(conn)
            except Exception:
                pass


# ==============================================================================
# 11. ESTIMATIVA DE TOKENS
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
# 12. AUTENTICAÇÃO
# ==============================================================================

def verificar_autenticacao(
    request: Request,
    x_api_key: str = Header(
        default=None,
        alias="x-api-key"
    ),
    key: str = None,
    conn=Depends(get_db_connection)
):
    api_key = (
        x_api_key.strip()
        if x_api_key
        else key.strip() if key else None
    )

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API Key ausente"
        )

    # --------------------------------------------------------------------------
    # RENOVAÇÃO
    # --------------------------------------------------------------------------

    with conn.cursor() as cur:
        cur.execute(
            """
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
            """,
            (api_key,)
        )

        renewed = cur.fetchone()

    if renewed:
        conn.commit()
        auth_cache.pop(api_key, None)
    else:
        conn.rollback()

    # --------------------------------------------------------------------------
    # CACHE
    # --------------------------------------------------------------------------

    cached = auth_cache.get(api_key)

    if cached:
        account = cached

    else:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    tokens_used,
                    monthly_quota,
                    status
                FROM api_keys
                WHERE api_key = %s
                """,
                (api_key,)
            )

            row = cur.fetchone()

        if not row:
            raise HTTPException(
                status_code=401,
                detail="Chave inválida"
            )

        account = {
            "tokens_used": int(row[0]),
            "monthly_quota": int(row[1]),
            "status": row[2],
        }

        auth_cache[api_key] = account

    # --------------------------------------------------------------------------
    # STATUS
    # --------------------------------------------------------------------------

    if account["status"] != "active":
        raise HTTPException(
            status_code=403,
            detail="Chave inativa ou revogada"
        )

    # --------------------------------------------------------------------------
    # COTA
    # --------------------------------------------------------------------------

    if account["tokens_used"] >= account["monthly_quota"]:
        raise HTTPException(
            status_code=429,
            detail="Cota mensal excedida"
        )

    return api_key


# ==============================================================================
# 13. PAYLOADS
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
# 14. INFERÊNCIA ONNX
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

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="np"
    )

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
        if "token_type_ids" in inputs:
            ort_inputs["token_type_ids"] = (
                inputs["token_type_ids"].astype(np.int64)
            )

    missing = input_names - set(ort_inputs.keys())

    if missing:
        raise RuntimeError(
            f"Inputs ONNX ausentes após tokenização: {sorted(missing)}"
        )

    outputs = ort_session.run(
        None,
        ort_inputs
    )

    if not outputs:
        raise RuntimeError(
            "ONNX Runtime não retornou outputs."
        )

    logits = np.asarray(outputs[0])

    if logits.ndim != 2:
        raise RuntimeError(
            f"Formato inesperado dos logits: {logits.shape}"
        )

    logits = (
        logits
        - np.max(
            logits,
            axis=-1,
            keepdims=True
        )
    )

    exp_logits = np.exp(logits)

    denominator = np.sum(
        exp_logits,
        axis=-1,
        keepdims=True
    )

    probs = exp_logits / denominator

    predictions = []

    for prob in probs:
        idx = int(np.argmax(prob))
        score = float(prob[idx])

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
# 15. HEALTH CHECK
# ==============================================================================

@app.get("/")
@limiter.limit("20/minute")
async def health_check(
    request: Request
):
    return {
        "status": "Online",
        "database": (
            "Conectado"
            if db_pool
            else "Desconectado"
        ),
        "model": (
            "Synapse-LangID INT8 carregado"
            if ort_session
            else "Carregando"
        ),
        "runtime": "ONNX Runtime",
        "quantization": "INT8",
    }


# ==============================================================================
# 16. GERAR API KEY
# ==============================================================================

@app.post("/gerar-chave")
@limiter.limit("3/hour")
async def gerar_chave(
    request: Request,
    conn=Depends(get_db_connection)
):
    ip_addr = (
        request.client.host
        if request.client
        else "unknown"
    )

    new_key = f"syn_{secrets.token_hex(16)}"

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO api_keys (
                    api_key,
                    ip_address,
                    monthly_quota
                )
                VALUES (%s, %s, %s)
                RETURNING api_key
                """,
                (
                    new_key,
                    ip_addr,
                    MAX_TOKENS_MONTH
                )
            )

            cur.fetchone()
            conn.commit()

    except psycopg2.errors.UniqueViolation:
        conn.rollback()

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT api_key
                FROM api_keys
                WHERE
                    ip_address = %s
                    AND status = 'active'
                """,
                (ip_addr,)
            )

            existing = cur.fetchone()

        if existing:
            raise HTTPException(
                status_code=400,
                detail="IP já possui uma chave ativa."
            )

        raise HTTPException(
            status_code=409,
            detail="Não foi possível gerar uma nova chave."
        )

    except psycopg2.Error as e:
        conn.rollback()

        print(
            f"❌ Erro PostgreSQL ao gerar chave: {e}"
        )

        raise HTTPException(
            status_code=503,
            detail="Erro interno ao gerar a chave."
        ) from e

    return {
        "message": "Chave gerada!",
        "api_key": new_key,
        "monthly_quota": MAX_TOKENS_MONTH
    }


# ==============================================================================
# 17. RESERVA ATÔMICA DE COTA
# ==============================================================================

def consumir_cota(
    conn,
    api_key: str,
    cost: int
):
    if cost <= 0:
        raise HTTPException(
            status_code=400,
            detail="Custo de tokens inválido."
        )

    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE api_keys
            SET
                tokens_used = tokens_used + %s
            WHERE
                api_key = %s
                AND status = 'active'
                AND tokens_used + %s <= monthly_quota
            RETURNING
                tokens_used,
                monthly_quota
            """,
            (
                cost,
                api_key,
                cost
            )
        )

        row = cur.fetchone()

        if row is None:
            conn.rollback()

            raise HTTPException(
                status_code=429,
                detail="Requisição excede a cota mensal restante."
            )

        conn.commit()

        return int(row[0]), int(row[1])


# ==============================================================================
# 18. PREDICT
# ==============================================================================

@app.post("/v1/predict")
@limiter.limit("30/minute")
async def predict(
    request: Request,
    payload: PredictRequest,
    conn=Depends(get_db_connection),
    api_key: str = Depends(verificar_autenticacao)
):
    cost = estimar_tokens(payload.text)

    updated_used, quota = consumir_cota(
        conn,
        api_key,
        cost
    )

    auth_cache.pop(
        api_key,
        None
    )

    try:
        predictions = rodar_inferencia_onnx(
            [payload.text]
        )

    except Exception as e:
        # Melhor não cobrar a requisição se a inferência falhar.
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE api_keys
                SET tokens_used =
                    GREATEST(tokens_used - %s, 0)
                WHERE api_key = %s
                """,
                (cost, api_key)
            )

        conn.commit()
        auth_cache.pop(api_key, None)

        print("❌ Erro na inferência /v1/predict:")
        print(str(e))

        raise HTTPException(
            status_code=500,
            detail="Erro interno na inferência."
        ) from e

    remaining = quota - updated_used

    response = JSONResponse({
        "prediction": predictions[0],
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
# 19. PREDICT BATCH
# ==============================================================================

@app.post("/v1/predict-batch")
@limiter.limit("6/minute")
async def predict_batch(
    request: Request,
    payload: BatchPredictRequest,
    conn=Depends(get_db_connection),
    api_key: str = Depends(verificar_autenticacao)
):
    total_chars = sum(
        len(text)
        for text in payload.texts
    )

    if total_chars > 15_000:
        raise HTTPException(
            status_code=413,
            detail=(
                "Lote excede o limite cumulativo "
                "de 15.000 caracteres."
            )
        )

    cost = sum(
        estimar_tokens(text)
        for text in payload.texts
    )

    updated_used, quota = consumir_cota(
        conn,
        api_key,
        cost
    )

    auth_cache.pop(
        api_key,
        None
    )

    try:
        predictions = rodar_inferencia_onnx(
            payload.texts
        )

    except Exception as e:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE api_keys
                SET tokens_used =
                    GREATEST(tokens_used - %s, 0)
                WHERE api_key = %s
                """,
                (cost, api_key)
            )

        conn.commit()
        auth_cache.pop(api_key, None)

        print("❌ Erro na inferência /v1/predict-batch:")
        print(str(e))

        raise HTTPException(
            status_code=500,
            detail="Erro interno na inferência do lote."
        ) from e

    remaining = quota - updated_used

    response = JSONResponse({
        "count": len(payload.texts),
        "predictions": predictions,
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
# 20. KEY INFO
# ==============================================================================

@app.get("/v1/key-info")
@limiter.limit("15/minute")
async def key_info(
    request: Request,
    conn=Depends(get_db_connection),
    api_key: str = Depends(verificar_autenticacao)
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
        "key_prefix": f"{api_key[:8]}...",
        "status": row[4],
        "monthly_quota": int(row[1]),
        "tokens_used": int(row[2]),
        "tokens_remaining": max(
            0,
            int(row[1]) - int(row[2])
        ),
        "period_start": row[3].isoformat(),
        "registered_ip": row[0]
    }


# ==============================================================================
# 21. REVOKE KEY
# ==============================================================================

@app.delete("/v1/revoke-key")
@limiter.limit("5/hour")
async def revoke_key(
    request: Request,
    conn=Depends(get_db_connection),
    api_key: str = Depends(verificar_autenticacao)
):
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE api_keys
            SET status = 'revoked'
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
        "message": "Chave revogada com sucesso."
    }


# ==============================================================================
# 22. LANGUAGES
# ==============================================================================

@app.get("/v1/languages")
async def languages():
    return {
        "model": HF_REPO,
        "file": HF_MODEL_FILE,
        "runtime": "ONNX Runtime",
        "quantization": "INT8",
        "batch_supported": True,
        "max_batch_size": 30
    }


# Tokenizer deve vir do modelo/base original.
# NÃO coloque automaticamente o repositório ONNX aqui,
# porque ele pode não conter tokenizer.json/config/tokenizer_config.
TOKENIZER_REPO = os.getenv("TOKENIZER_REPO")

if not TOKENIZER_REPO:
    raise RuntimeError(
        "TOKENIZER_REPO não configurado. "
        "Defina o repositório do tokenizer original no Render."
    )


# --------------------------------------------------------------------------
# DIRETÓRIOS
# --------------------------------------------------------------------------

MODEL_DIR = "./synapse_langid_onnx"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    HF_MODEL_FILE
)


# --------------------------------------------------------------------------
# CACHE
# --------------------------------------------------------------------------

auth_cache = TTLCache(
    maxsize=1000,
    ttl=300
)


# --------------------------------------------------------------------------
# GLOBAIS
# --------------------------------------------------------------------------

db_pool = None
tokenizer = None
ort_session = None
id2label = {}


# ==============================================================================
# 2. DOWNLOAD DO MODELO INT8
# ==============================================================================

def preparar_modelo():

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    if not os.path.exists(MODEL_PATH):

        print("📥 Baixando Synapse-LangID INT8...")

        downloaded_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=HF_MODEL_FILE,
            local_dir=MODEL_DIR
        )

        print(
            f"✅ Modelo baixado: {downloaded_path}"
        )

    else:

        print(
            f"✅ Modelo já existe: {MODEL_PATH}"
        )


# ==============================================================================
# 3. BANCO
# ==============================================================================

def inicializar_banco():

    global db_pool

    print("🗄️ Conectando ao PostgreSQL...")

    try:

        db_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=DATABASE_URL
        )

    except Exception as e:

        print("❌ Falha ao conectar ao PostgreSQL")
        print(str(e))

        raise


    conn = db_pool.getconn()

    try:

        with conn.cursor() as cur:

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS api_keys (

                    api_key VARCHAR(64)
                    PRIMARY KEY,

                    ip_address VARCHAR(45)
                    UNIQUE NOT NULL,

                    monthly_quota BIGINT
                    NOT NULL
                    DEFAULT 10000000,

                    tokens_used BIGINT
                    NOT NULL
                    DEFAULT 0,

                    period_start TIMESTAMP
                    NOT NULL
                    DEFAULT CURRENT_TIMESTAMP,

                    status VARCHAR(20)
                    NOT NULL
                    DEFAULT 'active'
                )
                """
            )

            conn.commit()

    finally:

        db_pool.putconn(conn)

    print("✅ PostgreSQL conectado")


# ==============================================================================
# 4. TOKENIZER
# ==============================================================================

def carregar_tokenizer():

    global tokenizer

    print(
        f"🔤 Carregando tokenizer: {TOKENIZER_REPO}"
    )

    try:

        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_REPO
        )

    except Exception as e:

        print("❌ Erro ao carregar tokenizer")
        print(str(e))

        raise RuntimeError(
            "Não foi possível carregar o tokenizer. "
            "Verifique TOKENIZER_REPO."
        ) from e

    print("✅ Tokenizer carregado")


# ==============================================================================
# 5. CONFIG / LABELS
# ==============================================================================

def carregar_labels():

    global id2label

    config_path = os.path.join(
        MODEL_DIR,
        "config.json"
    )

    if not os.path.exists(config_path):

        print(
            "⚠️ config.json não encontrado. "
            "Os labels serão retornados como IDs."
        )

        id2label = {}

        return


    try:

        with open(
            config_path,
            "r",
            encoding="utf-8"
        ) as f:

            config = json.load(f)


        id2label = {
            int(k): v
            for k, v in config.get(
                "id2label",
                {}
            ).items()
        }

        print(
            f"✅ {len(id2label)} labels carregados"
        )

    except Exception as e:

        print("⚠️ Erro ao carregar labels")
        print(str(e))

        id2label = {}


# ==============================================================================
# 6. ONNX RUNTIME
# ==============================================================================

def carregar_onnx():

    global ort_session

    print("⚡ Inicializando ONNX Runtime...")


    opts = ort.SessionOptions()

    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1

    opts.execution_mode = (
        ort.ExecutionMode.ORT_SEQUENTIAL
    )

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

    print(
        "📦 Modelo:",
        MODEL_PATH
    )

    print(
        "🧠 Inputs:",
        [
            x.name
            for x in ort_session.get_inputs()
        ]
    )

    print(
        "🎯 Outputs:",
        [
            x.name
            for x in ort_session.get_outputs()
        ]
    )


# ==============================================================================
# 7. LIFESPAN
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("=" * 70)
    print("🚀 INICIANDO SYNAPSE-LANGID API")
    print("=" * 70)


    inicializar_banco()

    preparar_modelo()

    carregar_tokenizer()

    carregar_labels()

    carregar_onnx()


    print("=" * 70)
    print("✅ SYNAPSE-LANGID API ONLINE")
    print("=" * 70)


    yield


    # --------------------------------------------------------------------------
    # SHUTDOWN
    # --------------------------------------------------------------------------

    global db_pool

    if db_pool:

        db_pool.closeall()

        print("🗄️ Pool PostgreSQL fechado")


    print("🛑 API encerrada")


# ==============================================================================
# 8. FASTAPI
# ==============================================================================

limiter = Limiter(
    key_func=get_remote_address
)


app = FastAPI(

    lifespan=lifespan,

    title="Synapse-LangID API",

    version="5.1.0"
)


app.state.limiter = limiter


app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)


# ==============================================================================
# 9. MIDDLEWARE — LIMITE DE PAYLOAD
# ==============================================================================

@app.middleware("http")
async def limit_body_size(
    request: Request,
    call_next
):

    content_length = request.headers.get(
        "content-length"
    )

    if content_length:

        try:

            size = int(content_length)

        except ValueError:

            return JSONResponse(
                status_code=400,
                content={
                    "detail":
                    "Content-Length inválido"
                }
            )


        if size > 256_000:

            return JSONResponse(
                status_code=413,
                content={
                    "detail":
                    "Payload Too Large "
                    "(Max 256KB)"
                }
            )


    return await call_next(request)


# ==============================================================================
# 10. DATABASE DEPENDENCY
# ==============================================================================

def get_db_connection():

    if db_pool is None:

        raise HTTPException(
            status_code=503,
            detail="Banco de dados indisponível"
        )


    conn = db_pool.getconn()

    try:

        yield conn

    finally:

        db_pool.putconn(conn)


# ==============================================================================
# 11. ESTIMATIVA DE TOKENS
# ==============================================================================

def estimar_tokens(
    text: str
) -> int:

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
# 12. AUTENTICAÇÃO
# ==============================================================================

def verificar_autenticacao(

    request: Request,

    x_api_key: str = Header(
        default=None,
        alias="x-api-key"
    ),

    key: str = None,

    conn=Depends(
        get_db_connection
    )

):

    api_key = (
        x_api_key
        or key
    )


    if not api_key:

        raise HTTPException(
            status_code=401,
            detail="API Key ausente"
        )


    # --------------------------------------------------------------------------
    # BUSCA DIRETA NO BANCO
    # --------------------------------------------------------------------------

    with conn.cursor() as cur:

        cur.execute(
            """
            UPDATE api_keys

            SET
                tokens_used = 0,
                period_start = CURRENT_TIMESTAMP

            WHERE
                api_key = %s

                AND period_start
                    + INTERVAL '30 days'
                    <= CURRENT_TIMESTAMP

            RETURNING
                tokens_used,
                monthly_quota,
                status
            """,
            (api_key,)
        )

        renewed = cur.fetchone()

        if renewed:

            conn.commit()

            # Depois da renovação, não usar cache antigo.
            auth_cache.pop(
                api_key,
                None
            )

        else:

            conn.rollback()


    # --------------------------------------------------------------------------
    # CACHE
    # --------------------------------------------------------------------------

    cached = auth_cache.get(
        api_key
    )

    if cached:

        account = cached

    else:

        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT
                    tokens_used,
                    monthly_quota,
                    status

                FROM api_keys

                WHERE api_key = %s
                """,
                (api_key,)
            )

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


    # --------------------------------------------------------------------------
    # STATUS
    # --------------------------------------------------------------------------

    if account["status"] != "active":

        raise HTTPException(
            status_code=403,
            detail="Chave inativa ou revogada"
        )


    # --------------------------------------------------------------------------
    # COTA
    # --------------------------------------------------------------------------

    if (
        account["tokens_used"]
        >= account["monthly_quota"]
    ):

        raise HTTPException(
            status_code=429,
            detail="Cota mensal excedida"
        )


    return api_key


# ==============================================================================
# 13. PAYLOADS
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
# 14. INFERÊNCIA ONNX
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
    # TOKENIZAÇÃO
    # --------------------------------------------------------------------------

    inputs = tokenizer(

        texts,

        padding=True,

        truncation=True,

        max_length=256,

        return_tensors="np"
    )


    # --------------------------------------------------------------------------
    # INPUTS ONNX
    # --------------------------------------------------------------------------

    ort_inputs = {}


    input_names = {
        item.name
        for item in ort_session.get_inputs()
    }


    if "input_ids" in input_names:

        ort_inputs["input_ids"] = (
            inputs["input_ids"]
            .astype(np.int64)
        )


    if "attention_mask" in input_names:

        ort_inputs["attention_mask"] = (
            inputs["attention_mask"]
            .astype(np.int64)
        )


    if "token_type_ids" in input_names:

        if "token_type_ids" in inputs:

            ort_inputs["token_type_ids"] = (
                inputs["token_type_ids"]
                .astype(np.int64)
            )


    # --------------------------------------------------------------------------
    # INFERÊNCIA
    # --------------------------------------------------------------------------

    outputs = ort_session.run(
        None,
        ort_inputs
    )


    logits = outputs[0]


    # --------------------------------------------------------------------------
    # SOFTMAX
    # --------------------------------------------------------------------------

    logits = (
        logits
        - np.max(
            logits,
            axis=-1,
            keepdims=True
        )
    )


    exp_logits = np.exp(
        logits
    )


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
    # RESULTADOS
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

            "language":
                label,

            "confidence":
                round(
                    score,
                    4
                )

        })


    return predictions


# ==============================================================================
# 15. HEALTH CHECK
# ==============================================================================

@app.get("/")
@limiter.limit("20/minute")
async def health_check(
    request: Request
):

    return {

        "status":
            "Online",

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
# 16. GERAR API KEY
# ==============================================================================

@app.post("/gerar-chave")
@limiter.limit("3/hour")
async def gerar_chave(

    request: Request,

    conn=Depends(
        get_db_connection
    )

):

    ip_addr = (
        request.client.host
        if request.client
        else "unknown"
    )


    new_key = (
        f"syn_{secrets.token_hex(16)}"
    )


    try:

        with conn.cursor() as cur:

            cur.execute(

                """
                INSERT INTO api_keys
                    (
                        api_key,
                        ip_address,
                        monthly_quota
                    )

                VALUES
                    (
                        %s,
                        %s,
                        %s
                    )

                RETURNING
                    api_key
                """,

                (
                    new_key,
                    ip_addr,
                    MAX_TOKENS_MONTH
                )
            )

            cur.fetchone()

            conn.commit()


    except psycopg2.errors.UniqueViolation:

        conn.rollback()


        with conn.cursor() as cur:

            cur.execute(

                """
                SELECT
                    api_key

                FROM api_keys

                WHERE
                    ip_address = %s

                AND
                    status = 'active'
                """,

                (ip_addr,)
            )

            existing = cur.fetchone()


        if existing:

            raise HTTPException(

                status_code=400,

                detail=
                    "IP já possui uma "
                    "chave ativa."
            )


        raise HTTPException(

            status_code=409,

            detail=
                "Não foi possível gerar "
                "uma nova chave."
        )


    return {

        "message":
            "Chave gerada!",

        "api_key":
            new_key,

        "monthly_quota":
            MAX_TOKENS_MONTH

    }


# ==============================================================================
# 17. RESERVA ATÔMICA DE COTA
# ==============================================================================

def consumir_cota(
    conn,
    api_key: str,
    cost: int
):

    with conn.cursor() as cur:

        cur.execute(

            """
            UPDATE api_keys

            SET
                tokens_used =
                    tokens_used + %s

            WHERE
                api_key = %s

                AND status = 'active'

                AND tokens_used + %s
                    <= monthly_quota

            RETURNING
                tokens_used,
                monthly_quota
            """,

            (
                cost,
                api_key,
                cost
            )
        )


        row = cur.fetchone()


        if row is None:

            conn.rollback()

            raise HTTPException(

                status_code=429,

                detail=
                    "Requisição excede "
                    "a cota mensal restante."
            )


        conn.commit()


        return row[0], row[1]


# ==============================================================================
# 18. PREDICT
# ==============================================================================

@app.post("/v1/predict")
@limiter.limit("30/minute")
async def predict(

    request: Request,

    payload: PredictRequest,

    conn=Depends(
        get_db_connection
    ),

    api_key: str = Depends(
        verificar_autenticacao
    )

):

    cost = estimar_tokens(
        payload.text
    )


    updated_used, quota = consumir_cota(

        conn,

        api_key,

        cost

    )


    auth_cache.pop(
        api_key,
        None
    )


    predictions = rodar_inferencia_onnx(

        [payload.text]

    )


    remaining = (
        quota
        - updated_used
    )


    response = JSONResponse({

        "prediction":
            predictions[0],

        "usage": {

            "prompt_tokens":
                cost,

            "remaining_tokens":
                remaining

        }

    })


    response.headers[
        "x-ratelimit-remaining-tokens"
    ] = str(remaining)


    return response


# ==============================================================================
# 19. PREDICT BATCH
# ==============================================================================

@app.post("/v1/predict-batch")
@limiter.limit("6/minute")
async def predict_batch(

    request: Request,

    payload: BatchPredictRequest,

    conn=Depends(
        get_db_connection
    ),

    api_key: str = Depends(
        verificar_autenticacao
    )

):

    total_chars = sum(
        len(text)
        for text in payload.texts
    )


    if total_chars > 15_000:

        raise HTTPException(

            status_code=413,

            detail=
                "Lote excede o limite "
                "cumulativo de 15.000 "
                "caracteres."
        )


    cost = sum(

        estimar_tokens(text)

        for text in payload.texts

    )


    updated_used, quota = consumir_cota(

        conn,

        api_key,

        cost

    )


    auth_cache.pop(
        api_key,
        None
    )


    predictions = rodar_inferencia_onnx(

        payload.texts

    )


    remaining = (
        quota
        - updated_used
    )


    response = JSONResponse({

        "count":
            len(payload.texts),

        "predictions":
            predictions,

        "usage": {

            "prompt_tokens":
                cost,

            "remaining_tokens":
                remaining

        }

    })


    response.headers[
        "x-ratelimit-remaining-tokens"
    ] = str(remaining)


    return response


# ==============================================================================
# 20. KEY INFO
# ==============================================================================

@app.get("/v1/key-info")
@limiter.limit("15/minute")
async def key_info(

    request: Request,

    conn=Depends(
        get_db_connection
    ),

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

            WHERE
                api_key = %s
            """,

            (api_key,)
        )


        row = cur.fetchone()


    if not row:

        raise HTTPException(

            status_code=404,

            detail=
                "Chave não encontrada"
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
            max(
                0,
                row[1] - row[2]
            ),

        "period_start":
            row[3].isoformat(),

        "registered_ip":
            row[0]

    }


# ==============================================================================
# 21. REVOKE KEY
# ==============================================================================

@app.delete("/v1/revoke-key")
@limiter.limit("5/hour")
async def revoke_key(

    request: Request,

    conn=Depends(
        get_db_connection
    ),

    api_key: str = Depends(
        verificar_autenticacao
    )

):

    with conn.cursor() as cur:

        cur.execute(

            """
            UPDATE api_keys

            SET
                status = 'revoked'

            WHERE
                api_key = %s
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
            "Chave revogada "
            "com sucesso."

    }


# ==============================================================================
# 22. LANGUAGES
# ==============================================================================

@app.get("/v1/languages")
async def languages():

    return {

        "model":
            HF_REPO,

        "file":
            HF_MODEL_FILE,

        "runtime":
            "ONNX Runtime",

        "quantization":
            "INT8",

        "batch_supported":
            True,

        "max_batch_size":
            30

    }
