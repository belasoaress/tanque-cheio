# tanque-cheio

Portuguese text-to-SQL over Brazil's ANP fuel price data, with a small open-source model fine-tuned via QLoRA.
The small model is benchmarked against a frontier LLM on accuracy, latency and cost, with reproducible numbers.
Status: early setup (milestone M0); see `plan.md` for the full plan.

---

Text-to-SQL em português sobre a série histórica de preços de combustíveis da
ANP. Um modelo open-source pequeno (até 3B parâmetros), ajustado com QLoRA, é
comparado a um LLM de fronteira em acurácia, latência e custo.

> Projeto em construção. O plano completo está em [`plan.md`](plan.md).

## Como rodar

Pré-requisito: [uv](https://docs.astral.sh/uv/) instalado. Os comandos são os
mesmos no Windows (PowerShell), Linux e macOS.

```
uv sync
uv run tanque --help
```

Para rodar lint e testes:

```
uv run ruff check
uv run pytest -q
```

### Chaves de API

Copie `.env.example` para `.env` e preencha as chaves. O `.env` nunca vai para
o repositório.

### Orçamento

Toda chamada paga à API passa pelo ledger em `results/spend.jsonl`. Para ver o
gasto por marco:

```
uv run tanque budget status
```

Os preços e os modelos usados ficam em `configs/pricing.toml`, com fonte e
data de consulta. A estimativa de gasto está em `results/budget-plan.md`.

## Licença e dados

Os dados de preços são da Agência Nacional do Petróleo, Gás Natural e
Biocombustíveis (ANP). A atribuição completa entra no M8.
