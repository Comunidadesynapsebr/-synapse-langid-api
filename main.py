import asyncio
import json
import os
import secrets
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import onnxruntime as ort
import psycopg2
from psycopg2 import pool

from cachetools import TTLCache
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, ConfigDict, constr
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from tokenizers import Tokenizer


# ==============================================================================
# CONFIG
# ==============================================================================

def env(name: str, required: bool = False, default: Optional[str] = None):
    value = os.getenv(name, default)

    if value is not None:
        value = value.strip()

    if required and not value:
        raise RuntimeError(
            f"{name} não configurada no Render."
        )

    if value and any(ord(char) < 32 for char in value):
        raise RuntimeError(
            f"{name} contém caracteres de controle inválidos."
        )

    return value


DATABASE_URL = env(
    "DATABASE_URL",
    required=True,
)

HF_REPO = env(
    "HF_REPO"
) or "Comunidade-Synapse-BR/Synapse-LangID-ONNX"

TOKENIZER_REPO = env(
    "TOKENIZER_REPO"
) or "Comunidade-Synapse-BR/Synapse-LangID"


# ==============================================================================
# LIMITES
# ==============================================================================

MAX_TOKENS_MONTH = 10_000_000
MIN_TOKEN_COST = 10

MAX_TEXT_CHARS = 1500
MAX_BATCH_SIZE = 30
MAX_BATCH_CHARS = 15_000
MAX_BODY_BYTES = 256_000

MODEL_MAX_LENGTH = 128


# ==============================================================================
# ARQUIVOS
# ==============================================================================

MODEL_DIR = "/tmp/synapse_langid"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "model-int8.onnx",
)

TOKENIZER_PATH = os.path.join(
    MODEL_DIR,
    "tokenizer.json",
)

CONFIG_PATH = os.path.join(
    MODEL_DIR,
    "config.json",
)


# ==============================================================================
# GLOBAIS
# ==============================================================================

db_pool = None
tokenizer = None
ort_session = None
id2label = {}

startup_state = {
    "status": "loading",
    "error": None,
}

startup_task = None

auth_cache = TTLCache(
    maxsize=1000,
    ttl=300,
)


# ==============================================================================
# HUGGING FACE
# ==============================================================================

def baixar_arquivos():
    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------------------------
    # MODELO
    # --------------------------------------------------------------------------

    if not os.path.isfile(MODEL_PATH):
        print(
            "📥 Baixando Synapse-LangID INT8...",
            flush=True,
        )

        downloaded = hf_hub_download(
            repo_id=HF_REPO,
            filename="model-int8.onnx",
            local_dir=MODEL_DIR,
        )

        print(
            f"✅ Modelo baixado: {downloaded}",
            flush=True,
        )

    else:
        print(
            "✅ Modelo ONNX já existe.",
            flush=True,
        )

    # --------------------------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------------------------

    if not os.path.isfile(TOKENIZER_PATH):
        print(
            "📥 Baixando tokenizer.json...",
            flush=True,
        )

        downloaded = hf_hub_download(
            repo_id=TOKENIZER_REPO,
            filename="tokenizer.json",
            local_dir=MODEL_DIR,
        )

        print(
            f"✅ Tokenizer baixado: {downloaded}",
            flush=True,
        )

    else:
        print(
            "✅ tokenizer.json já existe.",
            flush=True,
        )

    # --------------------------------------------------------------------------
    # CONFIG / LABELS
    # --------------------------------------------------------------------------

    if not os.path.isfile(CONFIG_PATH):
        try:
            print(
                "📥 Baixando config.json...",
                flush=True,
            )

            downloaded = hf_hub_download(
                repo_id=TOKENIZER_REPO,
                filename="config.json",
                local_dir=MODEL_DIR,
            )

            print(
                f"✅ config.json baixado: {downloaded}",
                flush=True,
            )

        except Exception as exc:
            print(
                f"⚠️ config.json não disponível: {exc}",
                flush=True,
            )


# ==============================================================================
# BANCO
# ==============================================================================

def inicializar_banco():
    global db_pool

    print(
        "🗄️ Conectando ao PostgreSQL...",
        flush=True,
    )

    try:
        db_pool = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=5,
            dsn=DATABASE_URL,
        )

    except psycopg2.Error as exc:
        print(
            f"❌ PostgreSQL: {exc}",
            flush=True,
        )

        raise RuntimeError(
            "Não foi possível conectar ao PostgreSQL."
        ) from exc

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

        print(
            "✅ PostgreSQL conectado.",
            flush=True,
        )

    except Exception:
        if conn:
            conn.rollback()

        raise

    finally:
        if conn and db_pool:
            db_pool.putconn(conn)


def get_db_connection():
    if db_pool is None:
        raise HTTPException(
            status_code=503,
            detail="Banco de dados ainda está inicializando.",
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

    except Exception as exc:
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass

        raise HTTPException(
            status_code=503,
            detail="Erro ao obter conexão com o banco.",
        ) from exc

    finally:
        if conn and db_pool:
            try:
                db_pool.putconn(conn)
            except Exception:
                pass


# ==============================================================================
# TOKENIZER
# ==============================================================================

def carregar_tokenizer():
    global tokenizer

    print(
        "🔤 Carregando tokenizer...",
        flush=True,
    )

    tokenizer = Tokenizer.from_file(
        TOKENIZER_PATH
    )

    tokenizer.enable_truncation(
        max_length=MODEL_MAX_LENGTH
    )

    tokenizer.enable_padding()

    print(
        "✅ Tokenizer carregado.",
        flush=True,
    )


# ==============================================================================
# LABELS
# ==============================================================================

def carregar_labels():
    global id2label

    if not os.path.isfile(CONFIG_PATH):
        print(
            "⚠️ config.json não encontrado.",
            flush=True,
        )

        id2label = {}
        return

    try:
        with open(
            CONFIG_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            config = json.load(file)

        id2label = {
            int(key): value
            for key, value in config.get(
                "id2label",
                {},
            ).items()
        }

        print(
            f"✅ {len(id2label)} labels carregados.",
            flush=True,
        )

    except Exception as exc:
        print(
            f"⚠️ Falha nos labels: {exc}",
            flush=True,
        )

        id2label = {}


# ==============================================================================
# ONNX
# ==============================================================================

def carregar_onnx():
    global ort_session

    if not os.path.isfile(MODEL_PATH):
        raise RuntimeError(
            f"Modelo não encontrado: {MODEL_PATH}"
        )

    print(
        "⚡ Inicializando ONNX Runtime...",
        flush=True,
    )

    options = ort.SessionOptions()

    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1

    options.execution_mode = (
        ort.ExecutionMode.ORT_SEQUENTIAL
    )

    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    ort_session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=options,
        providers=[
            "CPUExecutionProvider"
        ],
    )

    print(
        "✅ Synapse-LangID INT8 carregado.",
        flush=True,
    )

    print(
        "📦 Inputs:",
        [
            item.name
            for item in ort_session.get_inputs()
        ],
        flush=True,
    )

    print(
        "🎯 Outputs:",
        [
            item.name
            for item in ort_session.get_outputs()
        ],
        flush=True,
    )


# ==============================================================================
# INICIALIZAÇÃO EM BACKGROUND
# ==============================================================================

def inicializar_recursos():
    global startup_state

    try:
        print(
            "=" * 70,
            flush=True,
        )

        print(
            "🚀 INICIANDO RECURSOS SYNAPSE-LANGID",
            flush=True,
        )

        print(
            "=" * 70,
            flush=True,
        )

        # 1
        print(
            "1/5 🗄️ Banco",
            flush=True,
        )

        inicializar_banco()

        # 2
        print(
            "2/5 📥 Arquivos Hugging Face",
            flush=True,
        )

        baixar_arquivos()

        # 3
        print(
            "3/5 🔤 Tokenizer",
            flush=True,
        )

        carregar_tokenizer()

        # 4
        print(
            "4/5 🏷️ Labels",
            flush=True,
        )

        carregar_labels()

        # 5
        print(
            "5/5 ⚡ ONNX",
            flush=True,
        )

        carregar_onnx()

        startup_state = {
            "status": "ready",
            "error": None,
        }

        print(
            "=" * 70,
            flush=True,
        )

        print(
            "✅ SYNAPSE-LANGID API PRONTA",
            flush=True,
        )

        print(
            "=" * 70,
            flush=True,
        )

    except Exception as exc:
        startup_state = {
            "status": "error",
            "error": str(exc),
        }

        print(
            "=" * 70,
            flush=True,
        )

        print(
            "❌ FALHA NA INICIALIZAÇÃO",
            flush=True,
        )

        print(
            str(exc),
            flush=True,
        )

        print(
            "=" * 70,
            flush=True,
        )


# ==============================================================================
# LIFESPAN
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global startup_task
    global db_pool

    # O Uvicorn pode subir imediatamente.
    startup_state["status"] = "loading"
    startup_state["error"] = None

    startup_task = asyncio.create_task(
        asyncio.to_thread(
            inicializar_recursos
        )
    )

    # IMPORTANTE:
    # liberamos o startup imediatamente para o Render detectar a porta.
    yield

    # Espera a tarefa finalizar somente no shutdown.
    if startup_task:
        try:
            await startup_task
        except Exception:
            pass

    if db_pool:
        db_pool.closeall()
        db_pool = None

    auth_cache.clear()

    print(
        "🛑 API encerrada.",
        flush=True,
    )


# ==============================================================================
# FASTAPI
# ==============================================================================

limiter = Limiter(
    key_func=get_remote_address
)

app = FastAPI(
    lifespan=lifespan,
    title="Synapse-LangID API",
    version="5.2.0",
)

app.state.limiter = limiter

app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler,
)


# ==============================================================================
# BODY SIZE
# ==============================================================================

@app.middleware("http")
async def limit_body_size(
    request: Request,
    call_next,
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
                },
            )

        if size > MAX_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={
                    "detail": "Payload Too Large (Max 256KB)"
                },
            )

    return await call_next(request)


# ==============================================================================
# AUTH
# ==============================================================================

def verificar_autenticacao(
    request: Request,
    x_api_key: Optional[str] = Header(
        default=None,
        alias="x-api-key",
    ),
    key: Optional[str] = Query(
        default=None,
    ),
    conn=Depends(get_db_connection),
):
    api_key = None

    if x_api_key:
        api_key = x_api_key.strip()

    elif key:
        api_key = key.strip()

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API Key ausente.",
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
            (api_key,),
        )

        renewed = cur.fetchone()

    if renewed:
        conn.commit()
        auth_cache.pop(
            api_key,
            None,
        )

    else:
        conn.rollback()

    # --------------------------------------------------------------------------
    # CACHE
    # --------------------------------------------------------------------------

    cached = auth_cache.get(
        api_key
    )

    if cached is None:
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
                (api_key,),
            )

            row = cur.fetchone()

        if not row:
            raise HTTPException(
                status_code=401,
                detail="Chave inválida.",
            )

        cached = {
            "tokens_used": int(row[0]),
            "monthly_quota": int(row[1]),
            "status": row[2],
        }

        auth_cache[api_key] = cached

    # --------------------------------------------------------------------------
    # STATUS
    # --------------------------------------------------------------------------

    if cached["status"] != "active":
        raise HTTPException(
            status_code=403,
            detail="Chave inativa ou revogada.",
        )

    # --------------------------------------------------------------------------
    # COTA
    # --------------------------------------------------------------------------

    if cached["tokens_used"] >= cached["monthly_quota"]:
        raise HTTPException(
            status_code=429,
            detail="Cota mensal excedida.",
        )

    return api_key


# ==============================================================================
# SCHEMAS
# ==============================================================================

class PredictRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    text: constr(
        min_length=2,
        max_length=MAX_TEXT_CHARS,
    )


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    texts: list[
        constr(
            min_length=2,
            max_length=MAX_TEXT_CHARS,
        )
    ]


# ==============================================================================
# TOKEN ESTIMATION
# ==============================================================================

def estimar_tokens(
    text: str,
) -> int:
    return max(
        MIN_TOKEN_COST,
        (len(text) + 3) // 4,
    )


# ==============================================================================
# INFERENCE
# ==============================================================================

def garantir_modelo_pronto():
    if startup_state["status"] == "loading":
        raise HTTPException(
            status_code=503,
            detail="Modelo ainda está inicializando. Tente novamente em alguns segundos.",
        )

    if startup_state["status"] == "error":
        raise HTTPException(
            status_code=503,
            detail="Falha ao inicializar o modelo.",
        )

    if tokenizer is None or ort_session is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo indisponível.",
        )


def rodar_inferencia_onnx(
    texts: list[str],
):
    garantir_modelo_pronto()

    encoded = tokenizer.encode_batch(
        texts
    )

    batch_size = len(encoded)

    max_len = min(
        MODEL_MAX_LENGTH,
        max(
            len(item.ids)
            for item in encoded
        ),
    )

    input_ids = np.zeros(
        (batch_size, max_len),
        dtype=np.int64,
    )

    attention_mask = np.zeros(
        (batch_size, max_len),
        dtype=np.int64,
    )

    token_type_ids = np.zeros(
        (batch_size, max_len),
        dtype=np.int64,
    )

    pad_id = tokenizer.token_to_id(
        "[PAD]"
    )

    if pad_id is None:
        pad_id = 0

    input_ids.fill(pad_id)

    for index, item in enumerate(encoded):
        ids = item.ids[:max_len]
        mask = item.attention_mask[:max_len]

        input_ids[
            index,
            :len(ids)
        ] = ids

        attention_mask[
            index,
            :len(mask)
        ] = mask

        if item.type_ids:
            types = item.type_ids[:max_len]

            token_type_ids[
                index,
                :len(types)
            ] = types

    input_names = {
        item.name
        for item in ort_session.get_inputs()
    }

    ort_inputs = {}

    if "input_ids" in input_names:
        ort_inputs["input_ids"] = input_ids

    if "attention_mask" in input_names:
        ort_inputs["attention_mask"] = attention_mask

    if "token_type_ids" in input_names:
        ort_inputs["token_type_ids"] = (
            token_type_ids
        )

    missing = (
        input_names
        - set(ort_inputs.keys())
    )

    if missing:
        raise RuntimeError(
            "Inputs ONNX ausentes: "
            + str(sorted(missing))
        )

    outputs = ort_session.run(
        None,
        ort_inputs,
    )

    if not outputs:
        raise RuntimeError(
            "ONNX não retornou outputs."
        )

    logits = np.asarray(
        outputs[0],
        dtype=np.float32,
    )

    if (
        logits.ndim != 2
        or logits.shape[0] != batch_size
    ):
        raise RuntimeError(
            f"Formato inesperado dos logits: {logits.shape}"
        )

    logits -= np.max(
        logits,
        axis=1,
        keepdims=True,
    )

    exp_logits = np.exp(
        logits
    )

    probs = (
        exp_logits
        / np.sum(
            exp_logits,
            axis=1,
            keepdims=True,
        )
    )

    predictions = []

    for prob in probs:
        index = int(
            np.argmax(prob)
        )

        confidence = float(
            prob[index]
        )

        predictions.append(
            {
                "language": id2label.get(
                    index,
                    str(index),
                ),
                "confidence": round(
                    confidence,
                    4,
                ),
            }
        )

    return predictions


# ==============================================================================
# QUOTA
# ==============================================================================

def consumir_cota(
    conn,
    api_key: str,
    cost: int,
):
    if cost <= 0:
        raise HTTPException(
            status_code=400,
            detail="Custo inválido.",
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
                cost,
            ),
        )

        row = cur.fetchone()

        if row is None:
            conn.rollback()

            raise HTTPException(
                status_code=429,
                detail="Requisição excede a cota mensal restante.",
            )

        conn.commit()

    return (
        int(row[0]),
        int(row[1]),
    )


def estornar_cota(
    conn,
    api_key: str,
    cost: int,
):
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE api_keys
            SET
                tokens_used = GREATEST(
                    tokens_used - %s,
                    0
                )
            WHERE
                api_key = %s
            """,
            (
                cost,
                api_key,
            ),
        )

    conn.commit()

    auth_cache.pop(
        api_key,
        None,
    )


# ==============================================================================
# HEALTH
# ==============================================================================

@app.get("/")
@limiter.limit("20/minute")
async def health(
    request: Request,
):
    return {
        "status": startup_state["status"],
        "database": db_pool is not None,
        "tokenizer": tokenizer is not None,
        "model": ort_session is not None,
        "runtime": "ONNX Runtime",
        "quantization": "INT8",
    }


@app.get("/health")
async def health_detailed():
    return {
        "status": startup_state["status"],
        "error": startup_state["error"],
        "database": db_pool is not None,
        "tokenizer": tokenizer is not None,
        "model": ort_session is not None,
        "runtime": "onnxruntime",
        "quantization": "int8",
    }


# ==============================================================================
# GERAR CHAVE
# ==============================================================================

@app.post("/gerar-chave")
@limiter.limit("3/hour")
async def gerar_chave(
    request: Request,
    conn=Depends(get_db_connection),
):
    if startup_state["status"] != "ready":
        raise HTTPException(
            status_code=503,
            detail="API ainda está inicializando.",
        )

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
                    MAX_TOKENS_MONTH,
                ),
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
                (ip_addr,),
            )

            existing = cur.fetchone()

        if existing:
            raise HTTPException(
                status_code=400,
                detail="IP já possui uma chave ativa.",
            )

        raise HTTPException(
            status_code=409,
            detail="Não foi possível gerar uma nova chave.",
        )

    except psycopg2.Error as exc:
        conn.rollback()

        print(
            f"❌ PostgreSQL /gerar-chave: {exc}",
            flush=True,
        )

        raise HTTPException(
            status_code=503,
            detail="Erro interno ao gerar a chave.",
        ) from exc

    return {
        "message": "Chave gerada!",
        "api_key": new_key,
        "monthly_quota": MAX_TOKENS_MONTH,
    }


# ==============================================================================
# PREDICT
# ==============================================================================

@app.post("/v1/predict")
@limiter.limit("30/minute")
async def predict(
    request: Request,
    payload: PredictRequest,
    conn=Depends(get_db_connection),
    api_key: str = Depends(verificar_autenticacao),
):
    garantir_modelo_pronto()

    cost = estimar_tokens(
        payload.text
    )

    updated_used, quota = consumir_cota(
        conn,
        api_key,
        cost,
    )

    auth_cache.pop(
        api_key,
        None,
    )

    try:
        prediction = rodar_inferencia_onnx(
            [payload.text]
        )[0]

    except Exception as exc:
        try:
            estornar_cota(
                conn,
                api_key,
                cost,
            )
        except Exception as refund_exc:
            print(
                f"⚠️ Falha ao estornar cota: {refund_exc}",
                flush=True,
            )

        print(
            f"❌ Inferência /v1/predict: {exc}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail="Erro interno na inferência.",
        ) from exc

    remaining = (
        quota
        - updated_used
    )

    response = JSONResponse(
        {
            "prediction": prediction,
            "usage": {
                "prompt_tokens": cost,
                "remaining_tokens": remaining,
            },
        }
    )

    response.headers[
        "x-ratelimit-remaining-tokens"
    ] = str(remaining)

    return response


# ==============================================================================
# PREDICT BATCH
# ==============================================================================

@app.post("/v1/predict-batch")
@limiter.limit("6/minute")
async def predict_batch(
    request: Request,
    payload: BatchPredictRequest,
    conn=Depends(get_db_connection),
    api_key: str = Depends(verificar_autenticacao),
):
    garantir_modelo_pronto()

    if not payload.texts:
        raise HTTPException(
            status_code=400,
            detail="texts não pode estar vazio.",
        )

    if len(payload.texts) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Máximo de {MAX_BATCH_SIZE} "
                "textos por lote."
            ),
        )

    total_chars = sum(
        len(text)
        for text in payload.texts
    )

    if total_chars > MAX_BATCH_CHARS:
        raise HTTPException(
            status_code=413,
            detail=(
                "Lote excede o limite "
                "cumulativo de 15.000 caracteres."
            ),
        )

    cost = sum(
        estimar_tokens(text)
        for text in payload.texts
    )

    updated_used, quota = consumir_cota(
        conn,
        api_key,
        cost,
    )

    auth_cache.pop(
        api_key,
        None,
    )

    try:
        predictions = rodar_inferencia_onnx(
            payload.texts
        )

    except Exception as exc:
        try:
            estornar_cota(
                conn,
                api_key,
                cost,
            )
        except Exception as refund_exc:
            print(
                f"⚠️ Falha ao estornar cota: {refund_exc}",
                flush=True,
            )

        print(
            f"❌ Inferência batch: {exc}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail="Erro interno na inferência do lote.",
        ) from exc

    remaining = (
        quota
        - updated_used
    )

    response = JSONResponse(
        {
            "count": len(payload.texts),
            "predictions": predictions,
            "usage": {
                "prompt_tokens": cost,
                "remaining_tokens": remaining,
            },
        }
    )

    response.headers[
        "x-ratelimit-remaining-tokens"
    ] = str(remaining)

    return response


# ==============================================================================
# KEY INFO
# ==============================================================================

@app.get("/v1/key-info")
@limiter.limit("15/minute")
async def key_info(
    request: Request,
    conn=Depends(get_db_connection),
    api_key: str = Depends(verificar_autenticacao),
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
            (api_key,),
        )

        row = cur.fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail="Chave não encontrada.",
        )

    quota = int(row[1])
    used = int(row[2])

    return {
        "key_prefix": f"{api_key[:8]}...",
        "status": row[4],
        "monthly_quota": quota,
        "tokens_used": used,
        "tokens_remaining": max(
            0,
            quota - used,
        ),
        "period_start": row[3].isoformat(),
        "registered_ip": row[0],
    }


# ==============================================================================
# REVOKE KEY
# ==============================================================================

@app.delete("/v1/revoke-key")
@limiter.limit("5/hour")
async def revoke_key(
    request: Request,
    conn=Depends(get_db_connection),
    api_key: str = Depends(verificar_autenticacao),
):
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE api_keys
            SET status = 'revoked'
            WHERE api_key = %s
            """,
            (api_key,),
        )

        conn.commit()

    auth_cache.pop(
        api_key,
        None,
    )

    return {
        "message": "Chave revogada com sucesso.",
    }


# ==============================================================================
# LANGUAGES
# ==============================================================================

@app.get("/v1/languages")
async def languages():
    return {
        "model": HF_REPO,
        "file": "model-int8.onnx",
        "tokenizer": TOKENIZER_REPO,
        "runtime": "ONNX Runtime",
        "quantization": "INT8",
        "max_length": MODEL_MAX_LENGTH,
        "batch_supported": True,
        "max_batch_size": MAX_BATCH_SIZE,
        "labels_loaded": len(id2label),
    }
