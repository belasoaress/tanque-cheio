---
title: tanque-cheio
emoji: ⛽
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.30.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Perguntas sobre preços de combustíveis da ANP em SQL
models:
  - belasoares/tanque-cheio-qwen2.5-coder-1.5b-gguf
  - belasoares/tanque-cheio-qwen2.5-coder-1.5b-lora-v2
  - Qwen/Qwen2.5-Coder-1.5B-Instruct
datasets:
  - belasoares/tanque-cheio
---

# tanque-cheio · demo

Pergunte em português sobre os preços de combustíveis coletados pela ANP (2024-10 a
2026-09) e veja a SQL que um modelo de 1,5B parâmetros escreveu, a tabela com o
resultado e quanto tempo levou.

- **Modelo:** Qwen2.5-Coder-1.5B-Instruct ajustado com QLoRA (adapter v2,
  [`belasoares/tanque-cheio-qwen2.5-coder-1.5b-lora-v2`](https://huggingface.co/belasoares/tanque-cheio-qwen2.5-coder-1.5b-lora-v2)),
  fundido e quantizado em GGUF Q4_K_M
  ([`belasoares/tanque-cheio-qwen2.5-coder-1.5b-gguf`](https://huggingface.co/belasoares/tanque-cheio-qwen2.5-coder-1.5b-gguf)),
  rodando em CPU com o `llama-cpp-python`. No teste do projeto, o modelo em fp16
  acertou 80,7% das perguntas pela execução, empatado com o Qwen2.5-Coder-32B sem
  ajuste. A versão quantizada não foi avaliada no teste.
- **Como responde:** o mesmo prompt único, o mesmo guardrail (só um `SELECT` de
  leitura), a mesma execução no DuckDB só leitura e a mesma nova tentativa com o erro
  (1 vez) da avaliação, vindos do pacote `tanque` instalado no commit fixado em
  `requirements.txt`.
- **Dados:** série histórica de preços de combustíveis da ANP, com atribuição à ANP.
  Data de referência 30/09/2026; "último mês" = setembro de 2026. O banco não tem
  endereço dos postos, e as perguntas não são gravadas.
- **Código e avaliação:** <https://github.com/belasoaress/tanque-cheio>

## Como funciona o Space

Ao iniciar, o `app.py`:

1. baixa do GitHub, no mesmo commit do pacote (`TANQUE_COMMIT` em `tanque_demo.py`),
   `configs/settings.toml`, `data/manifest.json` e o snapshot Parquet (2,4 MB, com o
   SHA-256 conferido) e monta o banco DuckDB com `tanque.data.build`;
2. baixa o GGUF do HF (`hf_hub_download`, revisão na variável `GGUF_REVISION`) e o
   carrega com contexto de 2048 tokens e 2 threads;
3. avalia uma vez o prefixo fixo do prompt, que fica no cache do llama.cpp.

Uma pergunta por vez (fila do Gradio com 1 worker). Geração gulosa, até 384 tokens.

| Variável (Settings → Variables) | Padrão | Para quê |
|---|---|---|
| `GGUF_REVISION` | `main` | commit do repositório do GGUF; fixe o que o notebook 07 imprimiu |
| `GGUF_REPO` | `belasoares/tanque-cheio-qwen2.5-coder-1.5b-gguf` | repositório do modelo |
| `GGUF_FILE` | `tanque-cheio-qwen2.5-coder-1.5b-q4_k_m.gguf` | arquivo do modelo |
| `GGUF_PATH` | (vazio) | um `.gguf` local, só para ensaio fora do Space |

## Publicação 👤

Feita pela Isabela, no PowerShell, a partir da raiz do repositório. Passo a passo
completo (GGUF no Kaggle, criação do Space, conferência das 6 perguntas) em
[`docs/demo.md`](https://github.com/belasoaress/tanque-cheio/blob/main/docs/demo.md).

```powershell
uv run hf upload belasoares/tanque-cheio app . --repo-type space
```
