import asyncio
import gc
import json
import os
import secrets
from contextlib import asynccontextmanager
from typing import Optional

# ------------------------------------------------------------------------------
# TUNING DE MEMÓRIA (precisa vir ANTES de importar numpy/onnxruntime, pois as
# libs de thread (OpenMP/OpenBLAS) e o allocator do glibc só leem essas
# variáveis na primeira inicialização):
#   - OMP_NUM_THREADS=1 / OMP_WAIT_POLICY=PASSIVE: evita que o runtime crie
#     pools de threads ociosas (cada thread OpenMP mantém sua própria arena
#     de memória, mesmo parada).
#   - MALLOC_ARENA_MAX=2: limita o número de arenas do malloc do glibc.
#     Sem isso, processos Python multi-thread frequentemente acumulam RSS
#     que nunca é devolvido ao SO (é a causa nº1 de "memory bloat" em APIs
#     Python com numpy/onnxruntime).
# ------------------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "65536")

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
# BANCO — CRIAÇÃO / MIGRAÇÃO DE ESQUEMA
# ==============================================================================
#
# Em vez de um único CREATE TABLE IF NOT EXISTS com todas as colunas fixas,
# criamos a tabela com o mínimo indispensável (chave primária) e garantimos
# cada coluna via ALTER TABLE ... ADD COLUMN IF NOT EXISTS. Isso permite
# evoluir o schema em produção (novas colunas, novas tabelas) sem migrations
# externas e sem risco de apagar dados já existentes.
# ==============================================================================

def migrar_schema(conn):
    with conn.cursor() as cur:

        # --------------------------------------------------------------------
        # TABELA: api_keys
        # --------------------------------------------------------------------
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS api_keys (
                api_key VARCHAR(64) PRIMARY KEY
            )
            """
        )

        cur.execute(
            "ALTER TABLE api_keys "
            "ADD COLUMN IF NOT EXISTS ip_address VARCHAR(45)"
        )

        # UNIQUE separado (em vez de inline) pelo mesmo motivo do FK
        # abaixo: mais seguro de rodar de forma idempotente e mais fácil
        # de depurar se algo falhar.
        cur.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1
                    FROM pg_constraint
                    WHERE conname = 'api_keys_ip_address_key'
                ) THEN
                    ALTER TABLE api_keys
                        ADD CONSTRAINT api_keys_ip_address_key
                        UNIQUE (ip_address);
                END IF;
            END $$;
            """
        )

        cur.execute(
            "ALTER TABLE api_keys "
            "ADD COLUMN IF NOT EXISTS monthly_quota BIGINT "
            "NOT NULL DEFAULT 10000000"
        )

        cur.execute(
            "ALTER TABLE api_keys "
            "ADD COLUMN IF NOT EXISTS tokens_used BIGINT "
            "NOT NULL DEFAULT 0"
        )

        cur.execute(
            "ALTER TABLE api_keys "
            "ADD COLUMN IF NOT EXISTS period_start TIMESTAMP "
            "NOT NULL DEFAULT CURRENT_TIMESTAMP"
        )

        cur.execute(
            "ALTER TABLE api_keys "
            "ADD COLUMN IF NOT EXISTS status VARCHAR(20) "
            "NOT NULL DEFAULT 'active'"
        )

        # --------------------------------------------------------------------
        # TABELA: usage_log (consumo do usuário por requisição)
        # --------------------------------------------------------------------
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_log (
                id BIGSERIAL PRIMARY KEY
            )
            """
        )

        # Coluna sem constraint inline — mais seguro/portável do que
        # "ADD COLUMN ... REFERENCES ..." numa única instrução.
        cur.execute(
            "ALTER TABLE usage_log "
            "ADD COLUMN IF NOT EXISTS api_key VARCHAR(64)"
        )

        # FK adicionada à parte, só se ainda não existir. Isso evita o
        # erro "column referenced in foreign key constraint does not
        # exist" que algumas instâncias de Postgres (ex.: atrás de
        # poolers como o do Render) disparam quando a coluna e a FK são
        # criadas na mesma instrução ADD COLUMN.
        cur.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1
                    FROM pg_constraint
                    WHERE conname = 'usage_log_api_key_fkey'
                ) THEN
                    ALTER TABLE usage_log
                        ADD CONSTRAINT usage_log_api_key_fkey
                        FOREIGN KEY (api_key)
                        REFERENCES api_keys (api_key)
                        ON DELETE CASCADE;
                END IF;
            END $$;
            """
        )

        # NOT NULL aplicado depois que a coluna e a FK já existem —
        # seguro mesmo em reexecuções (não falha se já não houver NULLs).
        cur.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM usage_log WHERE api_key IS NULL
                ) THEN
                    ALTER TABLE usage_log
                        ALTER COLUMN api_key SET NOT NULL;
                END IF;
            END $$;
            """
        )

        cur.execute(
            "ALTER TABLE usage_log "
            "ADD COLUMN IF NOT EXISTS endpoint VARCHAR(50) NOT NULL "
            "DEFAULT 'unknown'"
        )

        cur.execute(
            "ALTER TABLE usage_log "
            "ADD COLUMN IF NOT EXISTS tokens_cost INTEGER "
            "NOT NULL DEFAULT 0"
        )

        cur.execute(
            "ALTER TABLE usage_log "
            "ADD COLUMN IF NOT EXISTS item_count INTEGER "
            "NOT NULL DEFAULT 1"
        )

        cur.execute(
            "ALTER TABLE usage_log "
            "ADD COLUMN IF NOT EXISTS refunded BOOLEAN "
            "NOT NULL DEFAULT FALSE"
        )

        cur.execute(
            "ALTER TABLE usage_log "
            "ADD COLUMN IF NOT EXISTS created_at TIMESTAMP "
            "NOT NULL DEFAULT CURRENT_TIMESTAMP"
        )

        # ----------------------------------------------------------------
        # ÍNDICE: um único índice composto (api_key, created_at DESC) em
        # vez de dois índices separados. A query de /v1/usage filtra por
        # api_key E ordena por created_at — um índice composto atende os
        # dois casos sozinho. Cada índice extra é mais uma estrutura que o
        # Postgres mantém residente em cache (shared_buffers): menos
        # índices = menos memória do servidor de banco ocupada.
        # ----------------------------------------------------------------
        cur.execute(
            "DROP INDEX IF EXISTS idx_usage_log_api_key"
        )

        cur.execute(
            "DROP INDEX IF EXISTS idx_usage_log_created_at"
        )

        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_usage_log_key_created "
            "ON usage_log (api_key, created_at DESC)"
        )

        # ----------------------------------------------------------------
        # UNLOGGED TABLE: usage_log é um log de consumo, não um registro
        # crítico (a cota real fica em api_keys, que continua LOGGED).
        # Tabelas UNLOGGED não escrevem no WAL, o que reduz bastante o
        # tráfego de I/O e a memória usada pelos buffers de WAL — o
        # trade-off é perder as últimas linhas não commitadas em caso de
        # crash do Postgres (aceitável para dados de telemetria).
        # A checagem evita reescrever a tabela toda vez que a API sobe.
        # ----------------------------------------------------------------
        cur.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1
                    FROM pg_class c
                    WHERE c.relname = 'usage_log'
                      AND c.relpersistence = 'u'
                ) THEN
                    EXECUTE 'ALTER TABLE usage_log SET UNLOGGED';
                END IF;
            END $$;
            """
        )

    conn.commit()


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

        migrar_schema(conn)

        print(
            "✅ PostgreSQL conectado e schema migrado.",
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

    # 1 thread: evita pools de threads adicionais (cada thread reserva sua
    # própria pilha/arena). Com o modelo já em INT8 e cargas pequenas por
    # requisição, mais threads custam mais RAM do que ganham em latência.
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1

    options.execution_mode = (
        ort.ExecutionMode.ORT_SEQUENTIAL
    )

    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    # Desliga a arena de memória e o "mem pattern" do ONNX Runtime: por
    # padrão o ORT pré-aloca e retém blocos grandes de memória para reuso
    # entre execuções (bom para throughput, ruim para RSS). Com arena
    # desligada, o allocator libera memória de volta ao SO entre chamadas.
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False

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
    global manutencao_task
    global db_pool

    # O Uvicorn pode subir imediatamente.
    startup_state["status"] = "loading"
    startup_state["error"] = None

    startup_task = asyncio.create_task(
        asyncio.to_thread(
            inicializar_recursos
        )
    )

    manutencao_task = asyncio.create_task(
        manutencao_periodica()
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

    if manutencao_task:
        manutencao_task.cancel()

        try:
            await manutencao_task
        except (asyncio.CancelledError, Exception):
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
    version="5.4.0",
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


_inferencia_contador = 0
_GC_A_CADA_N_INFERENCIAS = 25


def rodar_inferencia_onnx(
    texts: list[str],
):
    global _inferencia_contador

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

    # Libera explicitamente os arrays grandes (entradas + logits + probs)
    # em vez de esperar o coletor de lixo do Python decidir sozinho.
    del (
        input_ids,
        attention_mask,
        token_type_ids,
        outputs,
        logits,
        exp_logits,
        probs,
    )

    # gc.collect() tem custo (percorre todas as gerações); não vale a pena
    # rodar em toda requisição. Fazemos isso periodicamente para devolver
    # memória ao SO sem penalizar a latência de cada chamada.
    _inferencia_contador += 1

    if _inferencia_contador % _GC_A_CADA_N_INFERENCIAS == 0:
        gc.collect()

    return predictions


# ==============================================================================
# QUOTA + CONSUMO (usage_log)
# ==============================================================================

def consumir_cota(
    conn,
    api_key: str,
    cost: int,
    endpoint: str = "unknown",
    item_count: int = 1,
):
    """
    Debita a cota da chave e registra o consumo em usage_log na mesma
    transação. Retorna (tokens_used, monthly_quota, usage_log_id) para
    permitir estorno posterior caso a inferência falhe.
    """

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

        # usage_log é telemetria, não a fonte de verdade da cota (isso é
        # api_keys.tokens_used, atualizado acima). SET LOCAL só vale para
        # esta transação: evitamos esperar o fsync do WAL para o insert de
        # log, sem afetar a durabilidade do débito de cota em si.
        cur.execute(
            "SET LOCAL synchronous_commit = OFF"
        )

        cur.execute(
            """
            INSERT INTO usage_log (
                api_key,
                endpoint,
                tokens_cost,
                item_count
            )
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (
                api_key,
                endpoint,
                cost,
                item_count,
            ),
        )

        log_id = cur.fetchone()[0]

        conn.commit()

    return (
        int(row[0]),
        int(row[1]),
        int(log_id),
    )


def estornar_cota(
    conn,
    api_key: str,
    cost: int,
    log_id: Optional[int] = None,
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

        if log_id is not None:
            cur.execute(
                """
                UPDATE usage_log
                SET refunded = TRUE
                WHERE id = %s
                """,
                (log_id,),
            )

    conn.commit()

    auth_cache.pop(
        api_key,
        None,
    )


# ==============================================================================
# MANUTENÇÃO PERIÓDICA DO BANCO (retenção + VACUUM de usage_log)
# ==============================================================================
#
# usage_log cresce a cada requisição. Sem limpeza, a tabela (e seu índice)
# fica cada vez maior, ocupando mais e mais memória do Postgres em cache
# (shared_buffers / OS page cache) mesmo que ninguém consulte dados antigos.
# Esta rotina roda em background, fora do ciclo de requisições:
#   1. Apaga linhas de usage_log mais antigas que USAGE_LOG_RETENTION_DAYS.
#   2. Roda VACUUM (ANALYZE) para devolver o espaço ao SO e manter as
#      estatísticas do planner atualizadas.
# ==============================================================================

USAGE_LOG_RETENTION_DAYS = 30
MANUTENCAO_INTERVALO_SEGUNDOS = 24 * 60 * 60

manutencao_task = None


def limpar_e_compactar_usage_log():
    if db_pool is None:
        return

    conn = db_pool.getconn()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM usage_log
                WHERE created_at < NOW() - make_interval(days => %s)
                """,
                (USAGE_LOG_RETENTION_DAYS,),
            )

            deleted = cur.rowcount

        conn.commit()

        print(
            f"🧹 usage_log: {deleted} linha(s) antiga(s) removida(s).",
            flush=True,
        )

    except Exception as exc:
        conn.rollback()

        print(
            f"⚠️ Falha ao limpar usage_log: {exc}",
            flush=True,
        )

        return

    finally:
        db_pool.putconn(conn)

    # VACUUM não pode rodar dentro de um bloco de transação normal, então
    # usamos uma conexão dedicada com autocommit=True fora do pool.
    vacuum_conn = None

    try:
        vacuum_conn = psycopg2.connect(DATABASE_URL)
        vacuum_conn.autocommit = True

        with vacuum_conn.cursor() as cur:
            cur.execute("VACUUM (ANALYZE) usage_log")

        print(
            "🧹 VACUUM (ANALYZE) usage_log concluído.",
            flush=True,
        )

    except Exception as exc:
        print(
            f"⚠️ Falha ao rodar VACUUM em usage_log: {exc}",
            flush=True,
        )

    finally:
        if vacuum_conn:
            vacuum_conn.close()


async def manutencao_periodica():
    while True:
        await asyncio.sleep(
            MANUTENCAO_INTERVALO_SEGUNDOS
        )

        try:
            await asyncio.to_thread(
                limpar_e_compactar_usage_log
            )

        except Exception as exc:
            print(
                f"⚠️ Falha na manutenção periódica: {exc}",
                flush=True,
            )


# ==============================================================================
# HEALTH
# ==============================================================================

@app.get("/")
@limiter.limit("60/minute")
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
@limiter.limit("5/hour")
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
@limiter.limit("120/minute")
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

    updated_used, quota, log_id = consumir_cota(
        conn,
        api_key,
        cost,
        endpoint="/v1/predict",
        item_count=1,
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
                log_id,
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
@limiter.limit("20/minute")
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

    updated_used, quota, log_id = consumir_cota(
        conn,
        api_key,
        cost,
        endpoint="/v1/predict-batch",
        item_count=len(payload.texts),
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
                log_id,
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
@limiter.limit("60/minute")
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
# USAGE HISTORY (consumo detalhado do usuário)
# ==============================================================================

@app.get("/v1/usage")
@limiter.limit("60/minute")
async def usage_history(
    request: Request,
    conn=Depends(get_db_connection),
    api_key: str = Depends(verificar_autenticacao),
    limit: int = Query(default=50, ge=1, le=200),
):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                endpoint,
                tokens_cost,
                item_count,
                refunded,
                created_at
            FROM usage_log
            WHERE api_key = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (api_key, limit),
        )

        rows = cur.fetchall()

    return {
        "api_key_prefix": f"{api_key[:8]}...",
        "count": len(rows),
        "entries": [
            {
                "endpoint": row[0],
                "tokens_cost": int(row[1]),
                "item_count": int(row[2]),
                "refunded": bool(row[3]),
                "created_at": row[4].isoformat(),
            }
            for row in rows
        ],
    }


# ==============================================================================
# REVOKE KEY
# ==============================================================================

@app.delete("/v1/revoke-key")
@limiter.limit("10/hour")
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
