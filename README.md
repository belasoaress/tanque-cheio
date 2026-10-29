# tanque-cheio

[![CI](https://github.com/belasoaress/tanque-cheio/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/belasoaress/tanque-cheio/actions/workflows/ci.yml)

Portuguese text-to-SQL over Brazil's ANP fuel price data (DuckDB), with a 1.5B open-source model fine-tuned via QLoRA on Kaggle's free T4.
On a hand-reviewed 150-question test set, execution accuracy goes from 12% (base Qwen2.5-Coder-1.5B-Instruct) to 76% (fine-tuned), at an estimated US$ 0.37 per 1,000 questions on a T4.
Total API spend: US$ 0; every number below is recomputed by CI from saved predictions.

---

O tanque-cheio recebe uma pergunta em português sobre preços de combustíveis no Brasil
("Em quais capitais o etanol compensou em todos os meses de 2025?"), gera uma consulta SQL,
executa-a com segurança sobre um snapshot dos dados da ANP e devolve a tabela, com a SQL
visível. O modelo é o Qwen2.5-Coder-1.5B-Instruct, pequeno o bastante para rodar numa GPU
barata, ajustado com QLoRA.

**Resultado principal:** no teste de 150 perguntas, o fine-tuning levou a acurácia de
execução de **12,0% para 76,0%** (+64 p.p.) e a SQL válida de 22% para 96%, com 3,5 h de
GPU T4 gratuita no Kaggle e US$ 0 de API. Em troca, a latência mediana subiu de 2,6 s para
3,7 s numa T4.

## Resultados

Teste com 150 perguntas (50 fáceis, 60 médias, 40 difíceis), escritas e revisadas à mão,
isoladas do treino. EX = execution accuracy: a SQL gerada é executada e o resultado tem de
bater com o da SQL de referência. A tabela vem de `uv run tanque eval report`
([`results/report.md`](results/report.md)); o CI a recalcula a partir das previsões
salvas em `results/predictions/`.

| Sistema | N | EX | EX fácil | EX médio | EX difícil | SQL válida | Bloqueada | p50 (ms) | p95 (ms) | US$ / 1.000 |
|---|---|---|---|---|---|---|---|---|---|---|
| `base-qwen2.5-coder-1.5b-instruct` | 150 | 12,0% | 28,0% | 6,7% | 0,0% | 22,0% | 3,3% | 2.638 | 6.518 | 0,2896* |
| `ft-qwen2.5-coder-1.5b-instruct` | 150 | 76,0% | 98,0% | 76,7% | 47,5% | 96,0% | 2,0% | 3.726 | 8.172 | 0,3660* |

\* Custo estimado: latência média × US$ 0,35 por hora de uma GPU T4 sob demanda no Google
Cloud (só a GPU, sem a VM; é um piso). No Kaggle não custou nada.

Os dois sistemas rodaram na mesma sessão do Kaggle, na mesma T4, com o mesmo prompt,
decodificação gulosa e `max_new_tokens = 256`. No ajustado, o adapter LoRA foi fundido no
modelo base antes de gerar, para a latência ser comparável. O relatório tem também os
erros do ajustado por tipo, 10 exemplos comentados e os números do dev.

## Como rodar

Pré-requisito: [uv](https://docs.astral.sh/uv/). Os comandos são os mesmos no Windows
(PowerShell), Linux e macOS.

```
uv sync
uv run tanque data build --from-parquet snapshot/coletas-2026-09.parquet
uv run tanque eval report
```

O primeiro instala o ambiente. O segundo monta o banco DuckDB (`data/tanque.duckdb`) a
partir do snapshot versionado no repositório, sem baixar nada, e confere o SHA-256 e cada
tabela contra o registro original. O terceiro recalcula o relatório a partir das previsões
salvas. Para conferir as métricas do jeito que o CI confere:

```
uv run tanque eval check-metrics
uv run pytest -q
```

Outros comandos úteis: `uv run tanque data build` (refaz tudo a partir dos 48 CSVs da ANP),
`uv run tanque eval score datasets/test.jsonl <previsões>.jsonl`,
`uv run tanque gen templates` (regera o treino) e `uv run tanque --help`. A inferência e o
treino rodam em GPU, nos notebooks do Kaggle (`notebooks/`); o CI não roda modelos.

## Arquitetura

```
CSVs mensais da ANP ──► download ──► normalize ──► snapshot/coletas-2026-09.parquet (commitado)
                                                              │
                                                    build ──► data/tanque.duckdb (3 tabelas)
                                                              │
  templates + valores reais do banco ──► 8.217 pares ──► filtro de vazamento ──► datasets/train.jsonl
                                                              │
                         Kaggle T4: QLoRA no Qwen2.5-Coder-1.5B ──► adapter LoRA (HF)
                                                              │
pergunta ──► prompt único (schema + valores + data de referência + convenções)
                │
       modelo (base ou base + adapter) ──► SQL
                │
       guardrail (sqlglot) ──► execução read-only, timeout, limite de linhas
                │
       tabela ──► comparador ──► métricas ──► results/report.md
```

- `src/tanque/data/`: download, normalização, build do banco e do snapshot.
- `src/tanque/sql/`: guardrail e execução segura.
- `src/tanque/prompting.py`: o prompt único, igual para todos os sistemas.
- `src/tanque/gen/`: geração do treino por templates, validação e filtro de vazamento.
- `src/tanque/train/` e `src/tanque/infer/`: fine-tuning e previsões, usados pelos notebooks.
- `src/tanque/eval/`: comparador, métricas e relatório.
- `datasets/`: treino, dev e teste. `results/`: previsões, métricas e relatórios.

## Decisões e trade-offs

- **DuckDB (D4).** Roda dentro do processo, lê Parquet direto e não precisa de servidor. O
  mesmo banco funciona no notebook, no CI e numa demo.
- **Nomes de tabela e coluna em português (D13).** `coletas`, `valor_venda`, `municipios`.
  A pergunta é em português; com o schema no mesmo idioma o modelo tem menos tradução para
  fazer.
- **Data de referência fixa (D14, D19).** O snapshot vai até 30/09/2026, então "último
  mês" é setembro de 2026, não o mês de hoje. A data e a regra entram no prompt, para o
  modelo e a SQL de referência usarem a mesma leitura.
- **Convenções de resposta no prompt (D22).** O comparador exige o mesmo número de colunas.
  Sem dizer ao modelo que preço médio é `AVG(valor_venda)`, que "gasolina" é a comum e
  quais colunas devolver, respostas razoáveis contariam como erro para todos os sistemas.
- **Treino por templates, sem API paga (D23).** Cada uma das 40 intenções tem um template
  de SQL e várias redações de pergunta, preenchidos com valores reais do banco. A SQL sai
  certa por construção e o custo é zero. O preço é a variedade: as perguntas do treino
  seguem os templates.
- **Sem LLM de fronteira na comparação (D24).** Sem crédito de API, o relatório compara o
  modelo base e o ajustado. O código para rodar a Claude no teste continua no repositório
  e o relatório o inclui sozinho se as previsões aparecerem.
- **Snapshot commitado (D25).** O Parquet tem 2,4 MB e fica no repositório, com o SHA-256
  e a contagem + MD5 de cada tabela. O CI reconstrói o banco em ~10 s, sem download, e
  prova que é o mesmo banco usado na avaliação.
- **Dev e teste separados.** O dev (50) serviu para escolher o modelo base e conferir o
  fine-tuning; o teste (150) foi rodado uma única vez por sistema.

O registro das decisões da v1 (D1 a D25) está em [`docs/plan-v1.md`](docs/plan-v1.md); o plano da v2 (D26 em diante) está em [`plan.md`](plan.md).

## Limitações

- **Snapshot fixo:** coletas de outubro de 2024 a setembro de 2026. Os preços não são
  atualizados.
- **Só combustíveis automotivos** (gasolina, gasolina aditivada, etanol, diesel, diesel S10,
  GNV). Nada de GLP nem de volumes de venda.
- **Só português.**
- **Perguntas difíceis ainda erram muito:** 47,5% de EX. O erro mais comum do ajustado é a
  forma do resultado (colunas a mais ou a menos que a referência), seguido de agregação e
  lógica.
- **SQLs longas cortadas:** 4 dos 36 erros do ajustado no teste são SQLs que bateram no
  limite de 256 tokens e terminaram no meio (uma delas aparece como bloqueada, porque a SQL
  cortada não faz parse). Um limite maior provavelmente recupera parte delas, ao custo de
  latência.
- **Latência:** 3,7 s de mediana e 8,2 s no p95 numa T4. Em CPU seria bem mais lento.
- **Sem comparação com um LLM de fronteira** (D24).
- **Treino sintético por templates:** o modelo aprendeu as redações dos templates.
  Perguntas fora desse estilo tendem a sair piores.
- **Avaliação feita pelo próprio projeto:** as 200 perguntas de dev e teste foram
  rascunhadas com ajuda de um agente de código e revisadas por mim, uma a uma. Estão
  isoladas do treino por código, mas seguem as mesmas convenções.

## Links

- Notebook do fine-tuning (Kaggle, público):
  <https://www.kaggle.com/code/belasoares/tanque-cheio-finetune>
- Adapter LoRA (Hugging Face):
  <https://huggingface.co/belasoares/tanque-cheio-qwen2.5-coder-1.5b-lora>
- Dataset (Hugging Face, snapshot + treino/dev/teste):
  <https://huggingface.co/datasets/belasoares/tanque-cheio>
- Relatório final: [`results/report.md`](results/report.md) · bake-off dos modelos base:
  [`results/bakeoff.md`](results/bakeoff.md) · treino:
  [`results/training.md`](results/training.md) · schema do banco:
  [`docs/schema.md`](docs/schema.md)
- Plano do projeto: [`plan.md`](plan.md) (v2) e [`docs/plan-v1.md`](docs/plan-v1.md) (v1)

## Licença e dados

O código está sob a [licença MIT](LICENSE).

Os dados de preços são da **Agência Nacional do Petróleo, Gás Natural e Biocombustíveis
(ANP)**, da
[Série Histórica de Preços de Combustíveis](https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-abertos/serie-historica-de-precos-de-combustiveis),
publicada como dados abertos. Eles **não** estão sob a licença MIT: seguem a licença de
dados abertos indicada pela ANP (detalhes no card do dataset). Ao reusar, cite a ANP como
fonte. O snapshot em `snapshot/` é uma seleção normalizada
desses dados feita por este projeto, não uma publicação oficial da ANP.
