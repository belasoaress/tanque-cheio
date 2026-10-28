# Bake-off dos modelos base (M5)

> **Esqueleto.** Os números ainda não existem: a tabela é preenchida por
> `uv run tanque eval bakeoff` depois da execução no Kaggle e do `tanque eval score` de
> cada candidato. Candidatos e licenças: `results/bakeoff-candidates.md`.

## Configuração

- Dataset: `datasets/dev.jsonl` (50 perguntas: 15 fáceis, 20 médias, 15 difíceis).
- Prompt único (`tanque.prompting.build_prompt`) pelo chat template de cada modelo;
  decodificação gulosa; `max_new_tokens=256`; fp16; lote de 1.
- Hardware: Kaggle, GPU T4 (`hardware = kaggle-t4`). Latência por pergunta, medida do
  template à decodificação, com `torch.cuda.synchronize()`.
- Notebook: `notebooks/01_bakeoff.ipynb`. Versões, commit do repositório e commit de cada
  checkpoint: `results/predictions/run-info__bakeoff.json` _(a preencher)_.

## Resultados no dev

EX = execution accuracy (seção 5.6 do plano), sobre os 50 itens.

<!-- bakeoff:begin (gerado por `tanque eval bakeoff`; não edite à mão) -->
_Nenhum `base-*` com métricas no dev ainda. Rode `tanque eval score` com as previsões do Kaggle e depois `tanque eval bakeoff`._
<!-- bakeoff:end -->

## Observações

_(a preencher depois da execução: saídas cortadas em 256 tokens, SQL vazia, erros de
memória, uso de fp32 em algum candidato etc.)_

## Recomendação

_(a preencher com os números: o candidato com maior EX, olhando também o EX no difícil e
a latência. O peso relativo é decisão da Isabela.)_

## Escolha da Isabela

_(pendente)_
