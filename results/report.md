# Relatório final

> Gerado por `uv run tanque eval report` a partir de `results/metrics.json`, das previsões em `results/predictions/` e de `results/error-annotations.json`. Não edite à mão: o CI confere com `tanque eval report --check`.

EX = execution accuracy (seção 5.6 do plano), a métrica principal; EX tolerante = aceita colunas a mais na previsão (D28); IC 95% = intervalo de confiança bootstrap (D29). Definições na nota de método. Percentuais sobre todos os itens do split. `*` = custo estimado de GPU.

## Tabela principal: teste (150 perguntas)

_Previsões do teste ainda não pontuadas. Passos 👤 em `notebooks/README.md` (seção "Teste final v2 (M17)")._

## Base → ajustado (teste)

_Pendente: precisa de `base-*` e `ft-*` pontuados no teste._

## Comparações pareadas (teste)

_Pendente: precisa de 2 sistemas pontuados no teste (`tanque eval check-metrics --write`)._

## Erros do modelo ajustado (teste)

_Pendente: precisa de `results/predictions/ft-*__test.jsonl`._

## Referência: dev (50 perguntas)

Usado na escolha do modelo e do fine-tuning; não é o resultado final.

| Sistema | Split | N | EX | IC 95% EX | EX tolerante | IC 95% tolerante | EX fácil | EX médio | EX difícil | SQL válida | Bloqueada | p50 (ms) | p95 (ms) | GPUs | Pico GPU (GiB) | US$ / 1.000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `base-qwen2.5-coder-1.5b-instruct` | dev | 50 | 14,0% | 6,0–24,0% | 16,0% | 6,0–26,0% | 40,0% | 5,0% | 0,0% | 38,0% | 2,0% | 2.836 | 5.078 | — | — | 0,2837* |
| `base-qwen3-1.7b` | dev | 50 | 8,0% | 2,0–16,0% | 10,0% | 2,0–18,0% | 20,0% | 0,0% | 6,7% | 24,0% | 12,0% | 7.499 | 11.593 | — | — | 0,6771* |
| `base-smollm3-3b` | dev | 50 | 8,0% | 2,0–16,0% | 10,0% | 2,0–18,0% | 20,0% | 5,0% | 0,0% | 42,0% | 6,0% | 4.968 | 8.676 | — | — | 0,5102* |
| `ft-qwen2.5-coder-1.5b-instruct` | dev | 50 | 74,0% | 62,0–86,0% | 76,0% | 64,0–86,0% | 93,3% | 75,0% | 53,3% | 96,0% | 0,0% | 8.328† | 15.833† | — | — | 0,7958* |

Comparações pareadas no dev:

| A | B | N | EX A | EX B | Δ EX (B − A) | Só A | Só B | p (McNemar) |
|---|---|---|---|---|---|---|---|---|
| `base-qwen2.5-coder-1.5b-instruct` | `base-qwen3-1.7b` | 50 | 14,0% | 8,0% | −6,0 p.p. | 4 | 1 | 0,375 |
| `base-qwen2.5-coder-1.5b-instruct` | `base-smollm3-3b` | 50 | 14,0% | 8,0% | −6,0 p.p. | 4 | 1 | 0,375 |
| `base-qwen2.5-coder-1.5b-instruct` | `ft-qwen2.5-coder-1.5b-instruct` | 50 | 14,0% | 74,0% | +60,0 p.p. | 0 | 30 | < 0,001 |
| `base-qwen3-1.7b` | `base-smollm3-3b` | 50 | 8,0% | 8,0% | ±0,0 p.p. | 1 | 1 | 1,000 |
| `base-qwen3-1.7b` | `ft-qwen2.5-coder-1.5b-instruct` | 50 | 8,0% | 74,0% | +66,0 p.p. | 1 | 34 | < 0,001 |
| `base-smollm3-3b` | `ft-qwen2.5-coder-1.5b-instruct` | 50 | 8,0% | 74,0% | +66,0 p.p. | 0 | 33 | < 0,001 |

## Referência: dev2 (113 perguntas)

Dev ampliado da v2 (D30), usado nas decisões da v2; não é o resultado final.

| Sistema | Split | N | EX | IC 95% EX | EX tolerante | IC 95% tolerante | EX fácil | EX médio | EX difícil | SQL válida | Bloqueada | p50 (ms) | p95 (ms) | GPUs | Pico GPU (GiB) | US$ / 1.000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `base-qwen2.5-coder-1.5b-instruct` | dev2 | 113 | 11,5% | 6,2–17,7% | 12,4% | 6,2–18,6% | 28,2% | 5,9% | 0,0% | 28,3% | 2,6% | 2.808 | 5.899 | 1 | 3,1 | 0,3102* |
| `base-qwen2.5-coder-1.5b-instruct+retry` | dev2 | 113 | 11,5% | 6,2–17,7% | 12,4% | 6,2–18,6% | 28,2% | 5,9% | 0,0% | 31,0% | 2,6% | 4.871 | 11.846 | 1 | 3,1 | 0,5740* |
| `base-qwen2.5-coder-7b-instruct` | dev2 | 113 | 31,9% | 23,9–40,7% | 38,9% | 30,1–47,8% | 35,9% | 44,1% | 17,5% | 67,3% | 3,5% | 9.514 | 17.996 | 1 | 5,7 | 0,9275* |
| `base-qwen2.5-coder-7b-instruct+retry` | dev2 | 113 | 38,0% | 29,2–46,9% | 45,1% | 36,3–54,0% | 53,8% | 44,1% | 17,5% | 87,6% | 3,5% | 11.128 | 36.473 | 1 | 5,7 | 1,3049* |
| `base-qwen2.5-coder-14b-instruct` | dev2 | 113 | 40,7% | 31,9–49,6% | 45,1% | 36,3–54,0% | 53,8% | 50,0% | 20,0% | 84,1% | 0,9% | 17.425 | 41.085 | 1 | 10,2 | 1,9038* |
| `base-qwen2.5-coder-14b-instruct+retry` | dev2 | 113 | 46,0% | 37,2–54,9% | 51,3% | 41,6–60,2% | 53,8% | 67,7% | 20,0% | 96,5% | 0,9% | 18.015 | 61.241 | 1 | 10,2 | 2,3272* |
| `base-qwen2.5-coder-32b-instruct` | dev2 | 113 | 72,6% | 63,7–80,5% | 76,1% | 68,1–84,1% | 92,3% | 79,4% | 47,5% | 92,9% | 3,5% | 29.128 | 71.973 | 2 | 10,8 | 6,2569* |
| `base-qwen2.5-coder-32b-instruct+retry` | dev2 | 113 | 74,3% | 66,4–82,3% | 77,9% | 69,9–85,0% | 92,3% | 82,3% | 50,0% | 96,5% | 3,5% | 30.102 | 86.583 | 2 | 10,8 | 7,1498* |
| `ft-qwen2.5-coder-1.5b-instruct` | dev2 | 113 | 74,3% | 66,4–82,3% | 78,8% | 70,8–85,8% | 97,4% | 76,5% | 50,0% | 96,5% | 0,0% | 4.077 | 8.619 | 1 | 3,1 | 0,4042* |
| `ft-qwen2.5-coder-1.5b-instruct+retry` | dev2 | 113 | 74,3% | 66,4–82,3% | 78,8% | 70,8–85,8% | 97,4% | 76,5% | 50,0% | 97,4% | 0,0% | 4.101 | 8.652 | 1 | 3,1 | 0,4218* |

Comparações pareadas no dev2:

| A | B | N | EX A | EX B | Δ EX (B − A) | Só A | Só B | p (McNemar) |
|---|---|---|---|---|---|---|---|---|
| `base-qwen2.5-coder-1.5b-instruct` | `base-qwen2.5-coder-1.5b-instruct+retry` | 113 | 11,5% | 11,5% | ±0,0 p.p. | 0 | 0 | 1,000 |
| `base-qwen2.5-coder-1.5b-instruct` | `base-qwen2.5-coder-7b-instruct` | 113 | 11,5% | 31,9% | +20,3 p.p. | 2 | 25 | < 0,001 |
| `base-qwen2.5-coder-1.5b-instruct` | `base-qwen2.5-coder-14b-instruct` | 113 | 11,5% | 40,7% | +29,2 p.p. | 1 | 34 | < 0,001 |
| `base-qwen2.5-coder-1.5b-instruct` | `base-qwen2.5-coder-32b-instruct` | 113 | 11,5% | 72,6% | +61,1 p.p. | 0 | 69 | < 0,001 |
| `base-qwen2.5-coder-1.5b-instruct` | `ft-qwen2.5-coder-1.5b-instruct` | 113 | 11,5% | 74,3% | +62,8 p.p. | 0 | 71 | < 0,001 |
| `base-qwen2.5-coder-7b-instruct` | `base-qwen2.5-coder-7b-instruct+retry` | 113 | 31,9% | 38,0% | +6,2 p.p. | 0 | 7 | 0,016 |
| `base-qwen2.5-coder-7b-instruct` | `base-qwen2.5-coder-14b-instruct` | 113 | 31,9% | 40,7% | +8,8 p.p. | 10 | 20 | 0,099 |
| `base-qwen2.5-coder-7b-instruct` | `base-qwen2.5-coder-32b-instruct` | 113 | 31,9% | 72,6% | +40,7 p.p. | 5 | 51 | < 0,001 |
| `base-qwen2.5-coder-7b-instruct` | `ft-qwen2.5-coder-1.5b-instruct` | 113 | 31,9% | 74,3% | +42,5 p.p. | 6 | 54 | < 0,001 |
| `base-qwen2.5-coder-14b-instruct` | `base-qwen2.5-coder-14b-instruct+retry` | 113 | 40,7% | 46,0% | +5,3 p.p. | 0 | 6 | 0,031 |
| `base-qwen2.5-coder-14b-instruct` | `base-qwen2.5-coder-32b-instruct` | 113 | 40,7% | 72,6% | +31,9 p.p. | 1 | 37 | < 0,001 |
| `base-qwen2.5-coder-14b-instruct` | `ft-qwen2.5-coder-1.5b-instruct` | 113 | 40,7% | 74,3% | +33,6 p.p. | 6 | 44 | < 0,001 |
| `base-qwen2.5-coder-32b-instruct` | `base-qwen2.5-coder-32b-instruct+retry` | 113 | 72,6% | 74,3% | +1,8 p.p. | 0 | 2 | 0,500 |
| `base-qwen2.5-coder-32b-instruct` | `ft-qwen2.5-coder-1.5b-instruct` | 113 | 72,6% | 74,3% | +1,8 p.p. | 15 | 17 | 0,860 |
| `ft-qwen2.5-coder-1.5b-instruct` | `ft-qwen2.5-coder-1.5b-instruct+retry` | 113 | 74,3% | 74,3% | ±0,0 p.p. | 0 | 0 | 1,000 |

† Latência medida com o adapter LoRA sem fundir (`PeftModel`, run-info com `adapter_merged` ausente ou falso): não é comparável à do base. No teste o adapter é fundido antes de gerar.

## Nota de método

- **Dados:** snapshot fixo `snapshot/coletas-2026-09.parquet` (SHA-256 `55a2b1884665284ac4fc8e770623abb211617ab98c2fed4980e7c475b8a35006`), coletas de 2024-10-01 a 2026-09-30; o CI reconstrói o banco a partir dele e confere contagem e MD5 de cada tabela (D25).
- **Data de referência:** `2026-09-30` (= `max(data_coleta)`, D14); "último mês" = mês do calendário que a contém (D19).
- **Prompt único** (`tanque.prompting.build_prompt`, seção 5.4) para todos os sistemas, pelo chat template de cada modelo; decodificação gulosa, lote de 1 pergunta, `max_new_tokens = 384` no teste v2 e no dev2 (D31; no dev da v1 eram 256).
- **Métrica:** EX = execution accuracy (seção 5.6): a SQL prevista é executada com o guardrail e comparada ao resultado da referência; bloqueio, erro ou falta de previsão contam como erro. É a métrica principal (EX rígida): o número de colunas tem de ser o mesmo da referência.
- **EX tolerante (secundária, D28; definida antes dos resultados da v2):** acerta quando as colunas da referência estão entre as colunas da previsão, que pode ter colunas a mais (até 6 colunas na previsão), em qualquer ordem, com a mesma escolha de colunas para todas as linhas; depois valem as regras da EX rígida (mesmas linhas, arredondamento a 2 casas, ordem quando a pergunta pede). Toda previsão certa na rígida é certa na tolerante. Mede quantos erros são só de colunas a mais (`results_match_relaxed`, seção 5.6 da v2).
- **IC 95% (D29):** bootstrap percentil sobre as perguntas do split, 10.000 reamostragens com reposição, seed 42 (`tanque.eval.stats`). Com 150 perguntas, a margem da EX é de cerca de ±7 p.p.
- **Comparações pareadas (D29):** todo par de sistemas sem nova tentativa pontuados no mesmo split, mais cada sistema × a sua versão `+retry`; pergunta a pergunta, na EX rígida. Só A / Só B = perguntas que só um dos dois acerta; p = teste de McNemar exato (binomial, bilateral) sobre esses pares discordantes. p < 0,05 indica diferença dificilmente explicada pelo acaso. Os destaques do teste orientam cada par a partir do modelo final (Δ = final − outro).
- **Modelos locais (D26):** todos da família Qwen2.5-Coder-Instruct (mesmo tokenizer), sem ajuste (`base-*`) ou com adapter LoRA fundido no base antes de gerar: `ft-*` = adapter v1 (M6), `ft2-*` = adapter v2 (M15). O 1,5B roda em fp16. Hardware das previsões do teste: —. A latência é medida por pergunta, do template à decodificação, numa T4 (o 32B, em 2). Pico GPU = maior memória alocada numa GPU durante a execução (no 32B, por GPU).
- **Modelos grandes em 4 bits (D27):** o 7B, o 14B e o 32B rodam quantizados em nf4 (bitsandbytes, quantização dupla, computação em fp16), o 7B e o 14B numa T4 e o 32B em 2 T4 (`device_map="auto"`), porque é o que cabe nas GPUs grátis do Kaggle. A quantização pode tirar alguns pontos deles: os números dos grandes são um piso do que fariam em fp16.
- **Nova tentativa (D32):** as linhas `<sistema>+retry` vêm da mesma execução que `<sistema>`: quando a primeira SQL é bloqueada ou dá erro, o modelo recebe a mensagem de erro e tenta uma única vez de novo. As duas linhas aparecem juntas, com e sem a nova tentativa, e a comparação pareada mostra o ganho (R21).
- **Custo por 1.000 perguntas (* = estimativa):** latência média × 1.000 ÷ 3.600.000 ms/h × US$ 0,35/h, o preço de NVIDIA T4 (1 GPU), Google Compute Engine sob demanda, us-central1 em https://cloud.google.com/products/compute/gpus-pricing (consultado em 2026-10-29); não inclui a VM (vCPU e memória), então é um piso. O preço é por GPU (D35): o 32B, em 2 T4, paga US$ 0,70/h. No Kaggle a execução não custou nada; as previsões têm `cost_usd = 0`.
- **Sem LLM de fronteira por enquanto (D24):** sem crédito de API, o relatório compara só os sistemas locais. `tanque infer claude` continua no repositório; se `results/predictions/claude-*__test.jsonl` for adicionado e pontuado, ele entra nas tabelas sozinho.
- **Gasto total do projeto em API:** US$ 0,00 (`results/spend.jsonl`; `tanque budget status`).
- **Teste reaberto na v2 (D33):** o teste (150) já tinha sido usado uma vez na v1 (M7), com o base e o ft-v1, `max_new_tokens = 256` e sem nova tentativa. Na v2 ele foi reaberto uma vez por sistema, e o base e o ft-v1 foram reexecutados com as regras da v2 (384 tokens, nova tentativa): os números do teste deste relatório substituem os da v1. Os da v1 (EX 12,0% → 76,0%, citados no README como v1) ficam no histórico do git e em `results/predictions/v1/`, fora do `check-metrics`. Todas as decisões da v2 (templates do treino v2, hiperparâmetros, prompt de correção, modelo final) vieram do dev2, nunca do teste.
- **Dev × dev2 × teste:** o dev (50) foi usado na escolha do modelo (M5) e do fine-tuning (M6). O dev2 (D30) é esse dev mais as perguntas da reserva: como o dev2 contém o dev da v1, que já orientou as escolhas da v1, ele pode favorecer os ajustados em alguns pontos. O teste é a comparação limpa.
