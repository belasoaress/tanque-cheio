# Bake-off dos modelos base (M5)

> Execução no Kaggle (conta `belasoares`), 28/10/2026. A tabela é gerada por
> `uv run tanque eval bakeoff` a partir de `results/metrics.json`. Candidatos e licenças:
> `results/bakeoff-candidates.md`.

## Configuração

- Dataset: `datasets/dev.jsonl` (50 perguntas: 15 fáceis, 20 médias, 15 difíceis).
- Prompt único (`tanque.prompting.build_prompt`) pelo chat template de cada modelo;
  decodificação gulosa; `max_new_tokens=256`; fp16; lote de 1.
- Hardware: Kaggle, GPU T4 (`hardware = kaggle-t4`). Latência por pergunta, medida do
  template à decodificação, com `torch.cuda.synchronize()`.
- Notebook: `notebooks/01_bakeoff.ipynb`. Versões, commit do repositório e commit de cada
  checkpoint: `results/predictions/run-info__bakeoff.json`.

## Resultados no dev

EX = execution accuracy (seção 5.6 do plano), sobre os 50 itens.

<!-- bakeoff:begin (gerado por `tanque eval bakeoff`; não edite à mão) -->
| Sistema | N | EX | EX fácil | EX médio | EX difícil | SQL válida | Bloqueada | p50 (ms) | p95 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| `base-qwen2.5-coder-1.5b-instruct` | 50 | 14,0% | 40,0% | 5,0% | 0,0% | 38,0% | 2,0% | 2.836 | 5.078 |
| `base-qwen3-1.7b` | 50 | 8,0% | 20,0% | 0,0% | 6,7% | 24,0% | 12,0% | 7.499 | 11.593 |
| `base-smollm3-3b` | 50 | 8,0% | 20,0% | 5,0% | 0,0% | 42,0% | 6,0% | 4.968 | 8.676 |
<!-- bakeoff:end -->

## Observações

- Sem treino, os três erram sobretudo o **caminho dos joins**: procuram `municipio_id`
  ou `uf` direto em `coletas`/`postos` em vez de passar por `postos → municipios`, ou
  usam aliases de tabela inexistentes (`coleta.` em vez de `coletas.`). São erros de
  binder do DuckDB, não do pipeline: a extração da SQL funcionou nos três.
- O Qwen3-1.7B gera as saídas mais longas (mediana de 166 tokens) e tem mais SQL
  bloqueada (12%: mais de uma instrução e trechos que não fazem parse).
- O SmolLM3-3B tem a maior taxa de SQL válida (42%), mas acerta pouco o resultado.
- Nenhum erro de memória na T4.

## Recomendação

**`Qwen/Qwen2.5-Coder-1.5B-Instruct`.** Tem a maior EX (14% contra 8%), a menor latência
(p50 de 2,8 s, quase 3× mais rápido que o Qwen3) e é o menor dos três (1,5 B), o que deixa
o QLoRA mais rápido na T4 e a demo em CPU mais viável. O tipo de erro dominante (caminho
dos joins) é justamente o que os 8.217 pares de treino ensinam. A diferença de EX entre os
candidatos é pequena em 50 itens, mas latência e tamanho também apontam para ele.

## Escolha da Isabela

_(pendente)_
