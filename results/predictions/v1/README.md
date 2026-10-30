# Previsões do teste da v1 (M7), só para referência

O base e o ft-v1 no `datasets/test.jsonl`, com `max_new_tokens = 256` e sem nova
tentativa (EX 12,0% → 76,0%). Na v2 o teste é reaberto uma vez por sistema, com as regras
da v2 (D31, D32, D33), e as previsões novas, com os mesmos nomes, ficam em
`results/predictions/`. Estes arquivos saíram de lá no M17 para não serem sobrescritos.

Ficam fora do `tanque eval check-metrics` e do relatório: o comando só lê
`results/predictions/*__{dev,dev2,test}.jsonl`, sem subpastas. Para pontuá-los de novo:

```powershell
uv run tanque eval score datasets/test.jsonl results/predictions/v1/ft-qwen2.5-coder-1.5b-instruct__test.jsonl --metrics $env:TEMP/metrics-v1.json
```
