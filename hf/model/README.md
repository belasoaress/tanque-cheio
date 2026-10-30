---
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
library_name: peft
license: apache-2.0
language:
- pt
pipeline_tag: text-generation
datasets:
- belasoares/tanque-cheio
tags:
- text-to-sql
- sql
- duckdb
- qlora
- lora
- peft
- anp
---

# tanque-cheio: adapter LoRA do Qwen2.5-Coder-1.5B-Instruct para text-to-SQL em português

LoRA adapter for `Qwen/Qwen2.5-Coder-1.5B-Instruct` that turns Portuguese questions about
Brazil's ANP fuel prices into DuckDB SQL. On a hand-reviewed 150-question test set,
execution accuracy goes from 12.0% (base model) to 76.0% (with this adapter).

Este adapter transforma perguntas em português sobre os preços de combustíveis da ANP numa
consulta SQL para DuckDB. Ele foi treinado com QLoRA na GPU gratuita do Kaggle e faz parte
do projeto [tanque-cheio](https://github.com/belasoaress/tanque-cheio), onde estão o
código, os dados de avaliação e o relatório completo.

> **Há uma versão nova:**
> [`belasoares/tanque-cheio-qwen2.5-coder-1.5b-lora-v2`](https://huggingface.co/belasoares/tanque-cheio-qwen2.5-coder-1.5b-lora-v2),
> com 80,7% de EX no mesmo teste (este adapter: 76,0%; diferença não significativa,
> McNemar p = 0,12). Na v2 do projeto, este adapter foi reavaliado no teste com as regras
> novas (`max_new_tokens = 384` e uma nova tentativa depois de erro): **76,0%** (IC 95%
> 68,7–82,7%), o mesmo número da v1, contra 78,0% do Qwen2.5-Coder-32B-Instruct sem
> ajuste (empate estatístico, p = 0,74). Os números abaixo são os da v1 (256 tokens).

## Resultados

Teste com 150 perguntas escritas e revisadas à mão (50 fáceis, 60 médias, 40 difíceis),
nunca vistas no treino nem nas escolhas de modelo. EX = execution accuracy: a SQL gerada é
executada e o resultado é comparado ao da SQL de referência. Mesma T4, mesma sessão,
decodificação gulosa, `max_new_tokens = 256`, adapter fundido no base antes de gerar.

| Sistema | EX | EX fácil | EX médio | EX difícil | SQL válida | Bloqueada | p50 (ms) | p95 (ms) | US$ / 1.000 |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B-Instruct (base) | 12,0% | 28,0% | 6,7% | 0,0% | 22,0% | 3,3% | 2.638 | 6.518 | 0,2896* |
| + este adapter | **76,0%** | 98,0% | 76,7% | 47,5% | 96,0% | 2,0% | 3.726 | 8.172 | 0,3660* |

\* Estimativa: latência média × US$ 0,35 por hora de uma GPU T4 sob demanda no Google
Cloud (só a GPU, sem a VM). No Kaggle a execução não custou nada.

No dev (50 perguntas, usado para escolher o modelo e conferir o treino): EX de 14,0% → 74,0%.

Relatório completo, com os 36 erros do teste classificados por tipo e 10 exemplos:
[`results/report.md`](https://github.com/belasoaress/tanque-cheio/blob/main/results/report.md).

## Dados de treino

[`belasoares/tanque-cheio`](https://huggingface.co/datasets/belasoares/tanque-cheio),
config `pairs`, split `train`: 8.217 pares pergunta → SQL gerados por templates
parametrizados com valores reais do banco (40 intenções; 34% fáceis, 40% médios, 26%
difíceis) e validados por execução. Pares parecidos com o dev ou o teste foram removidos
(`token_set_ratio ≥ 85` na pergunta ou SQL normalizada idêntica). Os preços vêm da Série
Histórica de Preços de Combustíveis da ANP, coletas de 2024-10-01 a 2026-09-30.

## Treino

| Grupo | Valor |
|---|---|
| Base | `Qwen/Qwen2.5-Coder-1.5B-Instruct`, commit `2e1fd397ee46e1388853d2af2c993145b0f1098a` |
| Quantização | 4-bit nf4, double quant, compute fp16 (bitsandbytes) |
| LoRA | r=16, alpha=32, dropout=0,05, bias none; 18,5 M parâmetros treináveis |
| Módulos-alvo | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Otimização | AdamW, lr 2e-4, cosine, warmup 3%, weight decay 0, grad norm 1,0 |
| Batch | 4 × 4 de acumulação = 16 efetivo |
| Épocas | 2 (976 passos) |
| Sequência | 1.152 tokens; nenhum par truncado |
| Perda | só na resposta (SQL + `<|im_end|>`) |
| Divisão | 7.806 treino / 411 validação da perda (5%, seed 42) |
| Perda final | treino 0,0093 (média), validação 0,0016 |
| Hardware | 1 GPU T4 do Kaggle, 3,5 h de treino, pico de 2,4 GiB |
| Versões | transformers 5.21.0, peft 0.21.2, trl 1.16.0, bitsandbytes 0.50.2 |

Notebook público do treino:
<https://www.kaggle.com/code/belasoares/tanque-cheio-finetune>. Hiperparâmetros em
[`configs/train.toml`](https://github.com/belasoaress/tanque-cheio/blob/main/configs/train.toml);
registro da rodada em
[`results/training.md`](https://github.com/belasoaress/tanque-cheio/blob/main/results/training.md).

## Como usar

O adapter espera o prompt do projeto: a instrução, o schema do banco, os valores fechados,
a data de referência e as convenções de resposta, montados por
`tanque.prompting.build_prompt`. Sem esse prompt o modelo não conhece as tabelas. O jeito
mais simples é rodar de dentro do repositório:

```
git clone https://github.com/belasoaress/tanque-cheio
cd tanque-cheio
uv sync --group infer
uv run tanque data build --from-parquet snapshot/coletas-2026-09.parquet   # banco DuckDB
```

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from tanque.prompting import build_prompt, extract_sql
from tanque.sql import execute

base_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
adapter_id = "belasoares/tanque-cheio-qwen2.5-coder-1.5b-lora"

tokenizer = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_id).merge_and_unload()

messages = build_prompt("Qual o preço médio do etanol em Curitiba no último mês?")
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
output = model.generate(**inputs, max_new_tokens=384, do_sample=False)
new_tokens = output[0, inputs["input_ids"].shape[1] :]
sql = extract_sql(tokenizer.decode(new_tokens, skip_special_tokens=True))
print(sql)

result = execute.run(sql)  # guardrail + conexão somente leitura, timeout e limite de linhas
print(result.columns, result.rows[:5], result.error)
```

Em CPU, troque `torch.float16` por `torch.float32` e tire `device_map`. Com transformers
anterior à 4.56, use `torch_dtype=` no lugar de `dtype=`. Cada pergunta leva
de segundos (T4) a dezenas de segundos (CPU), porque o prompt tem ~800 tokens.

"Último mês" e outras datas relativas são resolvidas contra a data de referência do
snapshot (2026-09-30), não contra a data de hoje.

## Limitações

- **Snapshot fixo.** O modelo conhece um banco com coletas de 2024-10 a 2026-09; o prompt
  fixa a data de referência em 2026-09-30.
- **Só combustíveis automotivos** (gasolina, gasolina aditivada, etanol, diesel, diesel
  S10, GNV) e só o schema de 3 tabelas do projeto. Não serve como text-to-SQL genérico.
- **Só português.**
- **Perguntas difíceis:** 47,5% de EX (comparação entre períodos, CTEs, razão
  etanol/gasolina, funções de janela). O erro mais comum é a forma do resultado (colunas a
  mais ou a menos que a referência).
- **SQLs longas cortadas:** 4 dos 36 erros do teste são SQLs que bateram no limite de
  `max_new_tokens = 256` e terminaram no meio. O exemplo acima usa 384.
- **Latência:** p50 de 3,7 s numa T4, contra 2,6 s do base, porque o modelo ajustado
  escreve SQLs completas, mais longas.
- **Treino sintético por templates:** a variedade de redação é a dos templates. Perguntas
  muito diferentes das do treino tendem a sair piores.
- **Avaliação feita pelo próprio projeto:** o dev e o teste foram escritos e revisados
  pela mesma autora que montou o treino (isolados do treino por código). Não há comparação
  com um LLM de fronteira.
- A SQL é executada com guardrail (só `SELECT`/`WITH`, conexão somente leitura, sem acesso
  a arquivos ou rede). Use o mesmo cuidado se ligar o modelo a outro banco.

## Licença

O adapter segue a licença do modelo base, Apache-2.0. Os dados de treino são dados abertos
da ANP; veja o card do dataset para a licença e a atribuição.
