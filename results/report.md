# Relatório final

> Gerado por `uv run tanque eval report` a partir de `results/metrics.json`, das previsões em `results/predictions/` e de `results/error-annotations.json`. Não edite à mão: o CI confere com `tanque eval report --check`.

EX = execution accuracy (seção 5.6 do plano); percentuais sobre todos os itens do split. `*` = custo estimado de GPU (ver a nota de método).

## Tabela principal: teste (150 perguntas)

_Previsões do teste ainda não pontuadas. Passos 👤 em `notebooks/README.md` (seção M7)._

## Base → ajustado (teste)

_Pendente: precisa de `base-*` e `ft-*` pontuados no teste._

## Erros do modelo ajustado (teste)

_Pendente: precisa de `results/predictions/ft-*__test.jsonl`._

## Referência: dev (50 perguntas)

Usado na escolha do modelo e do fine-tuning; não é o resultado final.

| Sistema | Split | N | EX | EX fácil | EX médio | EX difícil | SQL válida | Bloqueada | p50 (ms) | p95 (ms) | US$ / 1.000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `base-qwen2.5-coder-1.5b-instruct` | dev | 50 | 14,0% | 40,0% | 5,0% | 0,0% | 38,0% | 2,0% | 2.836 | 5.078 | 0,2837* |
| `base-qwen3-1.7b` | dev | 50 | 8,0% | 20,0% | 0,0% | 6,7% | 24,0% | 12,0% | 7.499 | 11.593 | 0,6771* |
| `base-smollm3-3b` | dev | 50 | 8,0% | 20,0% | 5,0% | 0,0% | 42,0% | 6,0% | 4.968 | 8.676 | 0,5102* |
| `ft-qwen2.5-coder-1.5b-instruct` | dev | 50 | 74,0% | 93,3% | 75,0% | 53,3% | 96,0% | 0,0% | 8.328† | 15.833† | 0,7958* |

† Latência medida com o adapter LoRA sem fundir (`PeftModel`, run-info com `adapter_merged` ausente ou falso): não é comparável à do base. No teste o adapter é fundido antes de gerar.

## Nota de método

- **Dados:** snapshot fixo `snapshot/coletas-2026-09.parquet` (SHA-256 `55a2b1884665284ac4fc8e770623abb211617ab98c2fed4980e7c475b8a35006`), coletas de 2024-10-01 a 2026-09-30; o CI reconstrói o banco a partir dele e confere contagem e MD5 de cada tabela (D25).
- **Data de referência:** `2026-09-30` (= `max(data_coleta)`, D14); "último mês" = mês do calendário que a contém (D19).
- **Prompt único** (`tanque.prompting.build_prompt`, seção 5.4) para todos os sistemas, pelo chat template de cada modelo; decodificação gulosa, `max_new_tokens = 256`, lote de 1 pergunta.
- **Métrica:** EX = execution accuracy (seção 5.6): a SQL prevista é executada com o guardrail e comparada ao resultado da referência; bloqueio, erro ou falta de previsão contam como erro.
- **Modelos locais:** `base-*` (sem treino) e `ft-*` (mesmo base + adapter LoRA do M6, fundido no base antes de gerar), na mesma sessão do Kaggle e na mesma GPU, fp16. Hardware das previsões do teste: —. A latência é medida por pergunta, do template à decodificação, numa T4.
- **Custo por 1.000 perguntas (* = estimativa):** latência média × 1.000 ÷ 3.600.000 ms/h × US$ 0,35/h, o preço de NVIDIA T4 (1 GPU), Google Compute Engine sob demanda, us-central1 em https://cloud.google.com/products/compute/gpus-pricing (consultado em 2026-10-29); não inclui a VM (vCPU e memória), então é um piso. No Kaggle a execução não custou nada; as previsões têm `cost_usd = 0`.
- **Sem LLM de fronteira por enquanto (D24):** sem crédito de API, o relatório compara só os 2 sistemas locais. `tanque infer claude` continua no repositório; se `results/predictions/claude-*__test.jsonl` for adicionado e pontuado, ele entra nas tabelas sozinho.
- **Gasto total do projeto em API:** US$ 0,00 (`results/spend.jsonl`; `tanque budget status`).
- **Dev × teste:** o dev (50) foi usado na escolha do modelo (M5) e do fine-tuning (M6); o teste (150) só aqui, uma vez por sistema.
