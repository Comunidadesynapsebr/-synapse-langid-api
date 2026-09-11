================================================================================
SYNAPSE-LANGID API - DOCUMENTACAO TECNICA E GUIA DE INTEGRACAO
================================================================================

Descricao:
API corporativa para identificacao automatica de idioma em textos com base no
modelo Transformer "Comunidade-Synapse-BR/Synapse-LangID".
Inclui autenticacao persistente via PostgreSQL, cache em memoria RAM,
pooling de conexoes, rate limiting por IP e suporte a inferencia em lote.

Base URL:
https://synapse-langid-api.onrender.com

Versao: 4.2.0


================================================================================
1. AUTENTICACAO
================================================================================

A maioria dos endpoints exige envio de uma API Key valida.
Voce pode enviar a chave de duas formas equivalentes:

A) Via Query Parameter:
   https://synapse-langid-api.onrender.com/v1/predict?key=SUA_CHAVE_AQUI

B) Via Header HTTP:
   x-api-key: SUA_CHAVE_AQUI

Regra de Geracao:
Cada endereco IP tem direito a apenas 1 chave ativa registrada no banco de
dados. Para emitir uma nova chave, a anterior deve ser revogada antes.


================================================================================
2. ENDPOINTS DA API
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
