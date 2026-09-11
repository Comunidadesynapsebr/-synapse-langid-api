================================================================================
SYNAPSE-LANGID API - DOCUMENTACAO TECNICA E GUIA DE INTEGRACAO
================================================================================

Descricao:
API corporativa para identificacao automatica de idioma em textos com base no
modelo Transformer "Comunidade-Synapse-BR/Synapse-LangID".
Inclui autenticacao persistente via PostgreSQL, cache em memoria RAM,
pooling de conexoes, rate limiting por IP/chave, cota mensal de 10 milhoes
de tokens com tarifacao dinamica e politica estrita de Zero Data Retention.

Base URL:
https://synapse-langid-api.onrender.com

Versao: 4.3.0


================================================================================
1. AUTENTICACAO E POLITICA DE PRIVACIDADE
================================================================================

A) Formas de Envio da API Key:
   - Query Parameter: ?key=SUA_CHAVE_AQUI
   - Header HTTP:    x-api-key: SUA_CHAVE_AQUI

B) Regra de Unicidade:
   Cada endereco IP tem direito a apenas 1 credencial ativa simultanea. Para
   gerar uma nova credencial, a anterior deve ser formalmente revogada.

C) Zero Data Retention (Retencao Zero):
   Os textos submetidos para inferencia trafegam exclusivamente pela memoria
   RAM e sao descartados imediatamente apos a geracao do payload de resposta.
   Nenhum dado textual e gravado em disco ou persistido em banco de dados.


================================================================================
2. POLITICA DE COTAS, LIMITES E CONSUMO DE TOKENS
================================================================================

A) Cota Mensal:
   - Franquia: 10.000.000 de tokens por mes por chave ativa.
   - Renovacao: Ciclo automatico a cada 30 dias contados a partir da emissao.
   - Esgotamento: Status HTTP 429 (Too Many Requests) ao ultrapassar o limite.

B) Tarifacao Dinamica:
   - Consumo: 1 token a cada ~4 caracteres enviados.
   - Custo Minimo: 10 tokens por requisicao (inibe loops e flood de rede).
   - Inferencia Individual: Tarifacao proporcional ao campo 'text'.
   - Inferencia em Lote: Soma dos tokens de todas as strings contidas no array.

C) Limites Rigidos de Seguranca (Protecao Anti-OOM):
   - Tamanho Maximo de Payload (Body): 256 KB (HTTP 413 se ultrapassado).
   - Limite por Texto: Minimo de 2 e maximo de 1.500 caracteres.
   - Limite por Lote (Batch): Maximo de 30 textos por chamada.
   - Limite Cumulativo por Lote: Maximo de 15.000 caracteres no total do lote.
   - Truncamento: Tokenizer restrito a max_length=256.


================================================================================
3. ENDPOINTS DA API
================================================================================

--------------------------------------------------------------------------------
[GET] /
--------------------------------------------------------------------------------
Health Check do servico.

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
Gera credencial com cota de 10M de tokens associada ao IP de origem.

- Autenticacao: Nao requer
- Rate Limit: 3 chamadas/hora por IP
- Exemplo de Resposta (Status 200 - Sucesso):
  {
    "message": "Chave gerada!",
    "api_key": "syn_85b21db93babe995"
  }
- Exemplo de Resposta (Status 400 - Conflito):
  {
    "detail": "IP já possui chave: syn_85b21db93babe995"
  }

--------------------------------------------------------------------------------
[POST] /v1/predict
--------------------------------------------------------------------------------
Identificacao de idioma para uma string individual.

- Autenticacao: Obrigatoria
- Rate Limit: 30 chamadas/minuto
- Payload:
  {
    "text": "Esta frase confirma se a deteccao de idioma esta funcionando."
  }
- Exemplo de Resposta (Status 200):
  {
    "prediction": {
      "language": "pt",
      "confidence": 0.9986
    },
    "usage": {
      "prompt_tokens": 16,
      "remaining_tokens": 9999984
    }
  }

--------------------------------------------------------------------------------
[POST] /v1/predict-batch
--------------------------------------------------------------------------------
Processamento paralelo de multiplos textos em lote.

- Autenticacao: Obrigatoria
- Rate Limit: 6 chamadas/minuto
- Restricoes: Maximo 30 itens e maximo 15.000 caracteres somados
- Payload:
  {
    "texts": [
      "Ola, tudo bem com voce?",
      "Artificial intelligence is transforming software engineering."
    ]
  }
- Exemplo de Resposta (Status 200):
  {
    "count": 2,
    "predictions": [
      { "language": "pt", "confidence": 0.9985 },
      { "language": "en", "confidence": 0.9794 }
    ],
    "usage": {
      "prompt_tokens": 23,
      "remaining_tokens": 9999961
    }
  }

--------------------------------------------------------------------------------
[GET] /v1/key-info
--------------------------------------------------------------------------------
Auditoria de saldo, consumo e ciclo de faturamento da credencial.

- Autenticacao: Obrigatoria
- Rate Limit: 15 chamadas/minuto
- Exemplo de Resposta (Status 200):
  {
    "key_prefix": "syn_85b...",
    "status": "active",
    "monthly_quota": 10000000,
    "tokens_used": 39,
    "tokens_remaining": 9999961,
    "period_start": "2026-09-11 14:15:42",
    "registered_ip": "136.114.188.162"
  }

--------------------------------------------------------------------------------
[DELETE] /v1/revoke-key
--------------------------------------------------------------------------------
Exclui a credencial, expira o cache em RAM e libera o IP de origem.

- Autenticacao: Obrigatoria
- Rate Limit: 5 chamadas/hora
- Exemplo de Resposta (Status 200):
  {
    "message": "Chave revogada com sucesso. O IP está liberado."
  }


================================================================================
4. EXEMPLOS DE INTEGRACAO
================================================================================

----------------------------------------
Python (requests):
----------------------------------------
import requests

BASE_URL = "https://synapse-langid-api.onrender.com"
API_KEY = "SUA_CHAVE_AQUI"
headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}

# Individual
res = requests.post(f"{BASE_URL}/v1/predict", json={"text": "Texto em portugues."}, headers=headers)
print(res.json())

# Lote
res_batch = requests.post(f"{BASE_URL}/v1/predict-batch", json={"texts": ["Texto 1", "Text 2"]}, headers=headers)
print(res_batch.json())


----------------------------------------
JavaScript / Node.js (fetch):
----------------------------------------
const res = await fetch("https://synapse-langid-api.onrender.com/v1/predict", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "x-api-key": "SUA_CHAVE_AQUI"
  },
  body: JSON.stringify({ text: "Texto a ser analisado." })
});
console.log(await res.json());


----------------------------------------
cURL (Terminal / Bash):
----------------------------------------
curl -X POST "https://synapse-langid-api.onrender.com/v1/predict" \
     -H "x-api-key: SUA_CHAVE_AQUI" \
     -H "Content-Type: application/json" \
     -d '{"text": "Exemplo de chamada via cURL."}'
- [FastAPI](https://fastapi.tiangolo.com/) - Framework web
- [Hugging Face Transformers](https://huggingface.co/) - Pipeline de IA
- PyTorch (CPU) - Motor de inferência

## ☁️ Hospedagem
O deploy desta API está configurado para ser feito na [Render](https://render.com), utilizando o motor interno de Python.

---
*Mantido pela Comunidade Synapse BR.*
