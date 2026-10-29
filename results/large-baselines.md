# Modelos maiores no dev2 (M14)

> Executado no Kaggle (2 T4) em 30/10/2026; os 5 sistemas rodaram as 113 perguntas,
> sem cortes por tempo e com os checkpoints oficiais (`model_source: official`). A tabela
> completa, com IC 95%, EX tolerante, latência, custo e comparações pareadas, é gerada
> por `uv run tanque eval report` e está em [`results/report.md`](report.md), seção
> "Referência: dev2". Os números abaixo são cópia dela.

## Resultado (dev2, 113 perguntas, sem nova tentativa)

| Sistema | Parâmetros | EX | IC 95% | EX tolerante | p50 | Memória de GPU | GPUs | US$ / 1.000* |
|---|---|---|---|---|---|---|---|---|
| 1,5B base | 1,5 B | 11,5% | 6,2–17,7% | 12,4% | 2,8 s | 3,1 GiB | 1 | 0,31 |
| 7B base (nf4) | 7,6 B | 31,9% | 23,9–40,7% | 38,9% | 9,5 s | 5,7 GiB | 1 | 0,93 |
| 14B base (nf4) | 14,8 B | 40,7% | 31,9–49,6% | 45,1% | 17,4 s | 10,2 GiB | 1 | 1,90 |
| 32B base (nf4) | 32,8 B | 72,6% | 63,7–80,5% | 76,1% | 29,1 s | 10,8 GiB/GPU | 2 | 6,26 |
| **1,5B ajustado (ft-v1)** | 1,5 B | **74,3%** | 66,4–82,3% | **78,8%** | **4,1 s** | **3,1 GiB** | 1 | **0,40** |

\* Estimativa: latência média × GPUs × US$ 0,35/h por T4.

**Leitura.**

- O 1,5B ajustado empata com o 32B sem ajuste: 74,3% contra 72,6%, sem diferença
  significativa (McNemar p = 0,86; 15 perguntas só o 32B acerta, 17 só o ajustado). O
  ajustado faz isso com 1/21 dos parâmetros, ~7× menos latência (4,1 s contra 29,1 s),
  1 GPU em vez de 2 e ~15× menos custo por pergunta.
- O ajustado supera com folga o 7B (+42,5 p.p.) e o 14B (+33,6 p.p.), ambos com p < 0,001.
- A autocorreção (D32) ajuda os modelos sem ajuste, que erram mais a sintaxe: +6,2 p.p.
  no 7B (p = 0,016), +5,3 no 14B (p = 0,031), +1,8 no 32B (n.s.). No ajustado, ela não
  muda nada: ele quase não gera SQL inválida (97% válida).
- No nível difícil, ajustado e 32B empatam em 50%: é ali que está a margem do treino v2
  (M15).

**Ressalvas.**

- Os grandes rodaram em 4 bits (D27): os números deles são um piso do que fariam em fp16.
- O dev2 inclui os 50 perguntas do dev da v1, que serviram para escolher o modelo e
  conferir o fine-tuning. Isso pode favorecer o ajustado em alguns pontos. O teste (M17)
  é a comparação limpa.
- O dev2 foi revisado por inteiro pela Isabela (M13), sem descartes: os números acima
  valem como estão.

## Configuração (fixada antes de ver os resultados)

- **Dataset:** `datasets/dev2.jsonl` (113 perguntas, D30). O teste não é usado (regra 0.6;
  ele só roda no M17).
- **Mesmas regras para todos (D26, D31, D32):** prompt único (`prompting.build_prompt`)
  pelo chat template, decodificação gulosa (`do_sample=False`, `num_beams=1`,
  `repetition_penalty=1.0`), `max_new_tokens = 384`, lote de 1, e autocorreção com 1
  nova tentativa. Cada execução grava `<sistema>__dev2.jsonl` (sem nova tentativa) e
  `<sistema>+retry__dev2.jsonl` (com), da mesma geração.
- **Precisão (D27):** o 1,5B (base e ft-v1) em fp16, como na v1; o 7B, o 14B e o 32B em
  4 bits (bitsandbytes nf4, quantização dupla, computação em fp16). A quantização pode
  tirar alguns pontos dos modelos grandes: os números deles são um piso do que o modelo
  em fp16 faria.
- **Hardware:** sessão do Kaggle com 2 T4 (16 GB cada). Os modelos que cabem numa GPU
  rodam com `device_map={"": 0}`; o 32B, com `device_map="auto"` nas 2
  (`max_memory = {0: 13 GiB, 1: 14 GiB}`).
- **Custo (D35):** latência média × `gpus` × US$ 0,35/h por T4 (GCP sob demanda; 2 T4 =
  US$ 0,70/h). `gpus` vem de cada previsão.

### Checkpoints fixados

Consultados em https://huggingface.co/api/models/<repo> em 29/10/2026.

| Sistema | Checkpoint | Commit |
|---|---|---|
| `base-qwen2.5-coder-1.5b-instruct` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | `2e1fd397ee46e1388853d2af2c993145b0f1098a` (o mesmo do M6/M7) |
| `ft-qwen2.5-coder-1.5b-instruct` | o mesmo + `belasoares/tanque-cheio-qwen2.5-coder-1.5b-lora`, fundido | `bd8079ff5467c339586bc7fc31f18a19b6509a9e` (o do `04_test_predict`; o commit atual do repo, `930942e…`, só mudou o model card no M8) |
| `base-qwen2.5-coder-7b-instruct` | `Qwen/Qwen2.5-Coder-7B-Instruct` | `c03e6d358207e414f1eca0bb1891e29f1db0e242` |
| `base-qwen2.5-coder-14b-instruct` | `Qwen/Qwen2.5-Coder-14B-Instruct` | `aedcc2d42b622764e023cf882b6652e646b95671` |
| `base-qwen2.5-coder-32b-instruct` | `Qwen/Qwen2.5-Coder-32B-Instruct` | `381fc969f78efac66bc87ff7ddeadb7e73c218a7` |
| (reserva do 32B, só se faltar disco) | `unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit` | `89e52be6bf3252ffa41a6db933aeba59230848dd` |

O 32B tem 61 GiB de pesos em bf16, que o notebook baixa e quantiza ao carregar. Se o
disco da sessão não comportar isso, ele usa a reserva: o mesmo modelo já salvo em nf4
com quantização dupla (18 GiB, Apache-2.0), carregado com computação em fp16. O run-info
diz qual foi usado (`model_source`: `official` ou `fallback`).

## Estimativa de tempo de GPU (antes de rodar)

Base: a v1 mediu, numa T4, o 1,5B em fp16 a ~29 tokens/s (2,9 s por pergunta, ~85
tokens de saída, ~800 de entrada). Os modelos em nf4 foram estimados pela vazão típica
do bitsandbytes numa T4 com lote de 1 (7B ≈ 12 tok/s, 14B ≈ 6,5 tok/s, 32B em 2 T4 ≈
3 tok/s), com ~90 tokens de saída e uma fração de perguntas com nova tentativa (o base
1,5B só gerou SQL válida em 38% do dev, então ~60% dele refaz; os grandes, ~20–30%).
Os 2 arquivos de cada sistema saem da mesma execução: o custo é 113 perguntas mais as
novas tentativas, e não 2 × 113.

| Sistema | s/pergunta (1ª tentativa) | Novas tentativas | Geração | Download + carga | Total |
|---|---|---|---|---|---|
| base 1,5B (fp16) | ~3 | ~60% | ~9 min | ~1 min | ~10 min |
| ft-v1 1,5B (fp16) | ~4 | ~5% | ~8 min | ~1 min | ~9 min |
| base 7B (nf4) | ~8,5 | ~30% | ~21 min | ~5 min | ~26 min |
| base 14B (nf4) | ~16 | ~25% | ~38 min | ~8 min | ~46 min |
| base 32B (nf4, 2 GPUs) | ~34 | ~20% | ~77 min | ~10–16 min | ~1,5 h |
| Instalação, clone e banco | | | | | ~5 min |
| **Total** | | | | | **~3,1 h** (pessimista: ~5 h) |

O total fica abaixo de ~6 h, então o padrão é **um kernel só** (`PART = "all"`), dentro
do limite de 12 h por sessão do Kaggle e de ~3 a 5 h da cota semanal de ~30 h. Se o
ensaio (`LIMIT = 3`) mostrar uma vazão bem menor, a execução se divide em 2 versões do
mesmo kernel: `PART = "a"` (1,5B, ft, 7B, 14B: ~1,5 h) e depois `PART = "b"` (32B: ~1,5
h). O 32B tem ainda o limite próprio de `TIME_BUDGET_32B_H = 3` h (D37), contado do
início dele, com download e carga.

## Resultados no dev2

*A preencher pelo comando de relatório do M12, depois do push das previsões.*

| Sistema | Quant. | GPUs | EX (IC 95%) | EX +retry (IC 95%) | EX tolerante | Latência p50 / média | Pico de memória | US$/1.000 perguntas |
|---|---|---|---|---|---|---|---|---|
| `base-qwen2.5-coder-1.5b-instruct` | fp16 | 1 | — | — | — | — | — | — |
| `ft-qwen2.5-coder-1.5b-instruct` | fp16 | 1 | — | — | — | — | — | — |
| `base-qwen2.5-coder-7b-instruct` | nf4 | 1 | — | — | — | — | — | — |
| `base-qwen2.5-coder-14b-instruct` | nf4 | 1 | — | — | — | — | — | — |
| `base-qwen2.5-coder-32b-instruct` | nf4 | 2 | — | — | — | — | — | — |

## Execução (D37)

*A preencher com o run-info de cada sistema:* tempo total, itens completos
(`completed`/`n_items`), `stopped_reason` (ex.: `time_budget`), `skipped`/`skip_reason`
(ex.: falta de memória, 1 GPU só, disco) e `model_source` do 32B. Se o 32B não rodar
ou parar no meio, a comparação segue com o 7B e o 14B, que já sustentam a história
(D37), e isso fica declarado aqui.
