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

# tanque-cheio v2: adapter LoRA do Qwen2.5-Coder-1.5B-Instruct para text-to-SQL em português

LoRA adapter (v2) for `Qwen/Qwen2.5-Coder-1.5B-Instruct` that turns Portuguese questions
about Brazil's ANP fuel prices into DuckDB SQL. On a hand-reviewed 150-question test set it
reaches 80.7% execution accuracy (95% CI 74.0–86.7%), statistically tied with the untuned
Qwen2.5-Coder-32B-Instruct (78.0%, McNemar p = 0.63), on one T4 instead of two.

Este adapter transforma perguntas em português sobre os preços de combustíveis da ANP numa
consulta SQL para DuckDB. É a segunda versão: a primeira está em
[`belasoares/tanque-cheio-qwen2.5-coder-1.5b-lora`](https://huggingface.co/belasoares/tanque-cheio-qwen2.5-coder-1.5b-lora).
Foi treinado com QLoRA na GPU gratuita do Kaggle e faz parte do projeto
[tanque-cheio](https://github.com/belasoaress/tanque-cheio), onde estão o código, os dados
de avaliação e o relatório completo.

## Resultados

Teste com 150 perguntas escritas e revisadas à mão (50 fáceis, 60 médias, 40 difíceis),
isoladas do treino. EX = execution accuracy: a SQL gerada é executada e o resultado é
comparado ao da SQL de referência. Todos os sistemas rodaram no Kaggle com o mesmo prompt,
decodificação gulosa e `max_new_tokens = 384`, numa T4 (o 32B em 2). Os ajustados rodam em
fp16, com o adapter fundido no base; os modelos grandes, em 4 bits (nf4), o que é um piso
do que fariam em fp16.

| Sistema | EX | IC 95% | EX difícil | p50 | GPUs | Pico GPU | US$ / 1.000* |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B-Instruct (base) | 12,0% | 7,3–17,3% | 0,0% | 3,1 s | 1 | 3,1 GiB | 0,34 |
| Qwen2.5-Coder-7B-Instruct (nf4) | 40,7% | 32,7–48,7% | 12,5% | 8,8 s | 1 | 5,8 GiB | 0,89 |
| Qwen2.5-Coder-14B-Instruct (nf4) | 60,0% | 52,0–68,0% | 25,0% | 16,2 s | 1 | 10,2 GiB | 1,87 |
| Qwen2.5-Coder-32B-Instruct (nf4) | 78,0% | 71,3–84,7% | 37,5% | 29,1 s | 2 | 10,7 GiB por GPU | 6,22 |
| 1,5B + adapter v1 | 76,0% | 68,7–82,7% | 47,5% | 4,3 s | 1 | 3,1 GiB | 0,42 |
| **1,5B + este adapter (v2)** | **80,7%** | 74,0–86,7% | 62,5% | 4,3 s | 1 | 3,1 GiB | 0,42 |

\* Estimativa: latência média × US$ 0,35 por hora de cada GPU T4 sob demanda no Google
Cloud (só a GPU, sem a VM). No Kaggle a execução não custou nada.

- **Contra o 32B sem ajuste:** +2,7 p.p., sem diferença significativa (McNemar p = 0,63;
  21 perguntas só o v2 acerta, 17 só o 32B). Empate estatístico, com ~7× menos latência,
  1 GPU em vez de 2, ~1/7 da memória de GPU e ~15× menos custo por pergunta.
- **Contra o 14B sem ajuste:** +20,7 p.p. (p < 0,001).
- **Contra o adapter v1:** +4,7 p.p. (11 só o v2, 4 só o v1; p = 0,12, não
  significativo). O v2 ganha no difícil (47,5% → 62,5%) e no médio (76,7% → 83,3%) e
  perde no fácil (98,0% → 92,0%).
- EX tolerante (aceita colunas a mais): 85,3%. SQL válida: 98,7%. Uma nova tentativa
  depois de erro não muda a EX (80,7%).

Relatório completo, com os 29 erros do teste classificados por tipo, 10 exemplos e todas
as comparações pareadas:
[`results/report.md`](https://github.com/belasoaress/tanque-cheio/blob/main/results/report.md).

## Dados de treino

[`belasoares/tanque-cheio`](https://huggingface.co/datasets/belasoares/tanque-cheio)
(a publicar), config `pairs`, split `train`: **10.448 pares** pergunta → SQL gerados por
templates parametrizados com valores reais do banco e validados por execução. São as 40
intenções da v1 mais 17 novas (D34), para os padrões que o adapter v1 errava no dev2:
posição em ranking, acumulado, "tanto em X quanto em Y", "todos os …", participação
percentual, região × estado, e reforço das convenções de colunas. Distribuição: 28%
fáceis, 36% médios, 36% difíceis (632 com função de janela). Pares parecidos com o dev, o
dev2 ou o teste foram removidos (`token_set_ratio ≥ 85` na pergunta ou SQL normalizada
idêntica). Os preços vêm da Série Histórica de Preços de Combustíveis da ANP, coletas de
2024-10-01 a 2026-09-30. Análise dos erros e mudanças:
[`results/training-v2.md`](https://github.com/belasoaress/tanque-cheio/blob/main/results/training-v2.md).

## Treino

Os mesmos hiperparâmetros do v1, com sequência maior:

| Grupo | Valor |
|---|---|
| Base | `Qwen/Qwen2.5-Coder-1.5B-Instruct`, commit `2e1fd397ee46e1388853d2af2c993145b0f1098a` |
| Quantização | 4-bit nf4, double quant, compute fp16 (bitsandbytes) |
| LoRA | r=16, alpha=32, dropout=0,05, bias none; 18,5 M parâmetros treináveis |
| Módulos-alvo | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Otimização | AdamW, lr 2e-4, cosine, warmup 3%, weight decay 0, grad norm 1,0 |
| Batch | 4 × 4 de acumulação = 16 efetivo |
| Épocas | 2 (1.242 passos) |
| Sequência | 1.280 tokens; nenhum par truncado |
| Perda | só na resposta (SQL + `<|im_end|>`) |
| Divisão | 9.926 treino / 522 validação da perda (5%, seed 42) |
| Perda final | treino 0,0089 (média), validação 0,0018 |
| Hardware | 1 GPU T4 do Kaggle, 5,3 h de treino, pico de 2,5 GiB |
| Versões | transformers 5.21.0, peft 0.21.2, trl 1.16.0, bitsandbytes 0.50.2 |

Hiperparâmetros em
[`configs/train-v2.toml`](https://github.com/belasoaress/tanque-cheio/blob/main/configs/train-v2.toml);
notebook em
[`notebooks/02_finetune.ipynb`](https://github.com/belasoaress/tanque-cheio/blob/main/notebooks/02_finetune.ipynb).

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
adapter_id = "belasoares/tanque-cheio-qwen2.5-coder-1.5b-lora-v2"

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

- **Sobreajuste ao dev2.** No dev2 (113 perguntas, usado nas decisões), este adapter
  acertava 96,5%; no teste, 80,7%. Os templates novos foram desenhados a partir dos erros
  do dev2, então o número do dev2 é otimista. O teste é a medida limpa.
- **Regressão no fácil:** 92,0% no teste, contra 98,0% do adapter v1.
- **Diferenças pequenas não são significativas.** Com 150 perguntas, o IC tem ~±7 p.p.:
  o v2 empata com o 32B e com o v1 no teste.
- **Modelos grandes em 4 bits:** a comparação usa o 7B, o 14B e o 32B em nf4. Em fp16
  eles talvez fizessem alguns pontos a mais.
- **Teste reaberto:** o mesmo teste foi usado na v1 e de novo na v2, uma vez por sistema.
  Nenhuma decisão do treino v2 saiu dele.
- **Uma só seed:** uma rodada de treino (seed 42); a variação entre rodadas não foi
  medida.
- **Perguntas difíceis:** 62,5% de EX. Os erros mais comuns são a forma do resultado
  (colunas a mais ou a menos que a referência) e a lógica.
- **Snapshot fixo.** O modelo conhece um banco com coletas de 2024-10 a 2026-09; o prompt
  fixa a data de referência em 2026-09-30.
- **Só combustíveis automotivos** (gasolina, gasolina aditivada, etanol, diesel, diesel
  S10, GNV) e só o schema de 3 tabelas do projeto. Não serve como text-to-SQL genérico.
- **Só português.**
- **Treino sintético por templates:** a variedade de redação é a dos templates. Perguntas
  muito diferentes das do treino tendem a sair piores.
- **Avaliação feita pelo próprio projeto:** o dev, o dev2 e o teste foram escritos e
  revisados pela mesma autora que montou o treino (isolados do treino por código, mas com
  as mesmas convenções). Não há comparação com um LLM de fronteira.
- A SQL é executada com guardrail (só `SELECT`/`WITH`, conexão somente leitura, sem acesso
  a arquivos ou rede). Use o mesmo cuidado se ligar o modelo a outro banco.

## Licença

O adapter segue a licença do modelo base, Apache-2.0. Os dados de treino são dados abertos
da ANP; veja o card do dataset para a licença e a atribuição.
