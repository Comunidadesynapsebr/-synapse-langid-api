# Synapse-LangID API 🧠🌐

Esta é a API oficial para inferência do modelo **Synapse-LangID**, desenvolvido pela **Comunidade Synapse BR**. 

O modelo é capaz de identificar e classificar idiomas a partir de pequenos trechos de texto, rodando de forma leve e rápida via FastAPI.

## 🚀 Como usar (Endpoints)

A API possui uma rota principal para predição:

**`POST /predict`**

**Exemplo de requisição:**
\`\`\`json
{
  "text": "Este é um teste da comunidade Synapse BR."
}
\`\`\`

**Exemplo de resposta:**
\`\`\`json
{
  "input": "Este é um teste da comunidade Synapse BR.",
  "prediction": {
    "label": "pt",
    "score": 0.998
  }
}
\`\`\`

## 🛠️ Tecnologias Utilizadas
- [FastAPI](https://fastapi.tiangolo.com/) - Framework web
- [Hugging Face Transformers](https://huggingface.co/) - Pipeline de IA
- PyTorch (CPU) - Motor de inferência

## ☁️ Hospedagem
O deploy desta API está configurado para ser feito na [Render](https://render.com), utilizando o motor interno de Python.

---
*Mantido pela Comunidade Synapse BR.*
