# Candidatos ao bake-off (M5): proposta para aprovação

**Status: aguardando a Isabela.** Nada foi executado no Kaggle. A escolha final do
modelo base sai de `results/bakeoff.md`, depois da execução.

Levantamento feito em **28/10/2026** na API pública do Hugging Face
(`https://huggingface.co/api/models/<repo>`) e nos model cards / `config.json` / `LICENSE`
de cada checkpoint. Os critérios vêm do plano (D8, M5):

- ≤ 3B parâmetros; variante instruct ou code-instruct;
- licença **Apache-2.0 ou MIT no checkpoint exato** (campo `license` do card, conferido
  também no arquivo `LICENSE` quando existe);
- suporte em `transformers` e `peft` (arquitetura nativa, sem `trust_remote_code`);
- cabe numa T4 de 16 GB: inferência em fp16 e QLoRA (4-bit) no M6.

Contagem de parâmetros = campo `safetensors.total` da API (inclui embeddings).
Memória fp16 ≈ 2 bytes × parâmetros, só os pesos.

## Proposta (3 candidatos)

| # | Checkpoint | Licença (verificada) | Parâmetros | Contexto | Publicado | Pesos fp16 |
|---|---|---|---|---|---|---|
| 1 | [Qwen/Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) | Apache-2.0 ([LICENSE](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct/blob/main/LICENSE)) | 1,54 B (1,31 B sem embeddings) | 32.768 | set/2024 | ~3,1 GB |
| 2 | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | Apache-2.0 ([LICENSE](https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/LICENSE)) | 2,03 B (1,4 B sem embeddings) | 32.768 | abr/2025 | ~4,1 GB |
| 3 | [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | Apache-2.0 (campo `license` do [card](https://huggingface.co/HuggingFaceTB/SmolLM3-3B); o repo não tem arquivo LICENSE) | 3,08 B | 65.536 | jul/2025 | ~6,2 GB |

Os três são `*ForCausalLM` nativos do transformers (`Qwen2ForCausalLM`, `Qwen3ForCausalLM`,
`SmolLM3ForCausalLM`), sem código remoto, e a LoRA do `peft` cobre as camadas de atenção
e MLP padrão. Nenhum é gated: o Kaggle baixa sem token do HF.

### 1. Qwen2.5-Coder-1.5B-Instruct: a aposta em código

- **Motivo:** é o único code-instruct pequeno com licença permissiva. O
  [relatório técnico](https://arxiv.org/abs/2409.12186) diz que o pré-treino e o
  ajuste reforçaram text-to-SQL e avalia a série em Spider e BIRD. Os números estão numa
  figura e não foram conferidos aqui. É também o mais leve dos três, o que dá o QLoRA
  mais rápido no M6.
- **Ressalvas:**
  - **Armadilha de licença:** o irmão maior, `Qwen2.5-Coder-3B-Instruct`, é
    `qwen-research` (não comercial; ver [LICENSE](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct/blob/main/LICENSE)).
    O mesmo vale para `Qwen2.5-3B-Instruct`. Só o 1.5B (e o 0.5B/7B/14B/32B) é Apache-2.0.
  - O card lista só `en`. A família Qwen2.5 declara mais de 29 idiomas, português
    entre eles (card do Qwen2.5-1.5B-Instruct), mas o Coder foi treinado sobretudo em
    código.
  - O `generation_config.json` traz amostragem (`temperature=0.7`, `top_p=0.8`) e
    `repetition_penalty=1.05`. O notebook força decodificação gulosa pura
    (`do_sample=False`, `repetition_penalty=1.0`).
  - O chat template aceita `system`, conferido renderizando o prompt único: 784
    tokens de entrada.

### 2. Qwen3-1.7B: a geração mais nova da Qwen ainda abaixo de 3B

- **Motivo:** modelo geral de 2025, com mais de 100 idiomas, bom em seguir instruções e
  Apache-2.0 em todos os tamanhos da família Qwen3. É o meio-termo entre o Coder e o
  SmolLM3.
- **Ressalvas:**
  - **Raciocínio híbrido, ligado por padrão.** O notebook passa
    `chat_template_kwargs={"enable_thinking": False}`, e o template põe um bloco
    `<think></think>` vazio no fim do prompt. Sem isso, os 256 tokens acabam no
    raciocínio.
  - O card desaconselha a decodificação gulosa *no modo de raciocínio*. Com o
    raciocínio desligado o problema não se aplica, mas o card recomenda amostragem
    também no modo sem raciocínio. A gulosa segue o plano e é a mesma para todos.
  - Prompt único com o template: 788 tokens; `system` aceito.

### 3. SmolLM3-3B: a aposta em português

- **Motivo:** o único dos três com **português entre os 6 idiomas nativos** do card,
  com resultados em português publicados (MLMM HellaSwag pt). Modelo e dados de treino
  são abertos. É o maior que ainda pode ser chamado de "3B".
- **Ressalvas:**
  - **Tamanho no limite:** 3,08 B de parâmetros. É "3B" no nome e no card, mas passa de
    3,0 B na contagem exata. Se o critério "≤ 3B" for estrito, troque pelo reserva A.
  - O repositório não tem arquivo `LICENSE`. A licença Apache-2.0 vem só do metadado
    do card e do link para apache.org na seção *License*.
  - Raciocínio ligado por padrão: o notebook usa `enable_thinking=False`.
  - **O template põe a data do dia no system** ("Today Date: 28 October 2026"). Isso
    muda o prompt conforme o dia da execução e contradiz a data de referência (D14).
    O notebook acrescenta `/system_override` ao system, um sinal que o template remove,
    para desligar esse cabeçalho. O prompt fica igual ao dos outros modelos.
  - Treinado em bf16. A T4 não tem bf16 nativo, então o notebook usa fp16. Se aparecer
    saída degenerada (NaN, repetição de um token), rode o candidato com
    `"dtype": "float32"` (~12 GB, cabe na T4) e registre isso no `bakeoff.md`.

## Reservas (se a Isabela quiser trocar um candidato)

| Checkpoint | Licença | Parâmetros | Por que não está na proposta |
|---|---|---|---|
| A. [ibm-granite/granite-3.3-2b-instruct](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct) | Apache-2.0 (card) | 2,53 B | Bom candidato "empresarial", mas sem destaque em código ou português. Substituto natural do SmolLM3 se o limite de 3B for estrito. |
| B. [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | Apache-2.0 ([LICENSE](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/LICENSE)) | 2,27 B (com o codificador de visão) | O mais novo (fev/2026, 201 idiomas, sem raciocínio por padrão), mas é multimodal (`Qwen3_5ForConditionalGeneration`) e usa atenção linear (Gated DeltaNet). Na T4, sem os kernels `flash-linear-attention`, a inferência cai no caminho lento em PyTorch, e o QLoRA nessa arquitetura é menos testado. Risco alto para o prazo. |

## Excluídos (verificados)

| Checkpoint | Motivo |
|---|---|
| [Qwen/Qwen2.5-Coder-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct), [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | Licença `qwen-research` (não comercial), embora os irmãos 1.5B sejam Apache-2.0 |
| [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | Licença `llama3.2` (não é Apache/MIT); gated |
| [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) | Licença `gemma`; gated |
| [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) | Apache-2.0, mas 5,12 B de parâmetros no checkpoint ("E2B" = 2B *efetivos*), e multimodal |
| [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) | MIT, mas 3,84 B |
| [mistralai/Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) | Apache-2.0, mas 3,85 B e multimodal |
| [ibm-granite/granite-4.1-3b](https://huggingface.co/ibm-granite/granite-4.1-3b), [ibm-granite/granite-4.2-3b](https://huggingface.co/ibm-granite/granite-4.2-3b) | Apache-2.0, mas 3,40 B e 3,66 B |
| [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) | Licença `lfm1.0` (própria); lançado em 23/10/2026 |

## O que a Isabela decide

1. Aprovar os 3 candidatos (ou trocar algum por um reserva). A lista está em
   `CANDIDATES`, na primeira célula de `notebooks/01_bakeoff.ipynb`.
2. Aprovar a execução no Kaggle (regra 5): cerca de 50 perguntas × 3 modelos, com
   estimativa de 20 a 40 min de GPU T4, dentro da cota gratuita.
3. Se o "≤ 3B" é estrito (então o SmolLM3-3B, com 3,08 B, sai).
