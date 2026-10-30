---
license: other
license_name: dados-abertos-anp
license_link: https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-abertos/serie-historica-de-precos-de-combustiveis
language:
- pt
pretty_name: tanque-cheio
task_categories:
- table-question-answering
- text-generation
tags:
- text-to-sql
- sql
- duckdb
- anp
- combustiveis
size_categories:
- 10K<n<100K
- 1M<n<10M
configs:
- config_name: pairs
  default: true
  data_files:
  - split: train
    path: data/train.jsonl
  - split: dev
    path: data/dev.jsonl
  - split: test
    path: data/test.jsonl
- config_name: coletas
  data_files:
  - split: train
    path: snapshot/coletas-2026-09.parquet
---

# tanque-cheio: text-to-SQL em português sobre os preços de combustíveis da ANP

Portuguese text-to-SQL pairs over a fixed snapshot of Brazil's ANP fuel price survey
(Oct 2024 to Sep 2026), plus the snapshot itself. 10,448 template-generated training pairs (v2)
and 200 hand-reviewed dev/test questions, all validated by execution on DuckDB.

Este dataset acompanha o projeto [tanque-cheio](https://github.com/belasoaress/tanque-cheio):
perguntas em português sobre preços de combustíveis, a SQL (DuckDB) que responde cada
uma e o snapshot dos dados da ANP sobre o qual as SQLs rodam. Foi usado para treinar e
avaliar os adapters
[`belasoares/tanque-cheio-qwen2.5-coder-1.5b-lora-v2`](https://huggingface.co/belasoares/tanque-cheio-qwen2.5-coder-1.5b-lora-v2)
(treinado com o `train` daqui) e
[`belasoares/tanque-cheio-qwen2.5-coder-1.5b-lora`](https://huggingface.co/belasoares/tanque-cheio-qwen2.5-coder-1.5b-lora)
(treinado com o treino v1, de 8.217 pares, que fica no histórico do repositório).

## Fonte e atribuição

Os preços vêm da **Série Histórica de Preços de Combustíveis** da Agência Nacional do
Petróleo, Gás Natural e Biocombustíveis (**ANP**), levantamento de preços feito em postos
de todo o país:
<https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-abertos/serie-historica-de-precos-de-combustiveis>.

Fonte dos dados: ANP, Série Histórica de Preços de Combustíveis. Os arquivos aqui são uma
seleção e uma normalização desses dados feitas pelo projeto tanque-cheio, não uma
publicação oficial da ANP.

**Licença.** Dados abertos da ANP, publicados com base na Lei 12.527/2011 e no Decreto
8.777/2016, com atribuição à ANP. A página do conjunto na ANP não traz um campo de
licença; o espelho CKAN do catálogo dados.gov.br
([data.amerigeoss.org](https://data.amerigeoss.org/dataset/serie-historica-de-precos-de-combustiveis-por-revenda))
indica "Creative Commons Attribution". **Essa licença ainda não foi confirmada na fonte:**
o dados.gov.br não estava acessível quando o card foi escrito, e a autora ainda vai
conferir o campo "Licença" lá. Até isso acontecer, o card declara `license: other`, com
link para a página da ANP, em vez de afirmar CC BY. Quem reusar os dados deve citar a ANP
como fonte. As
perguntas e SQLs (`data/*.jsonl`) foram escritas pelo projeto e seguem a mesma licença,
para não criar uma regra diferente para cada arquivo.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `data/train.jsonl` | 10.448 pares de treino (v2), gerados por templates (`source = synthetic`) |
| `data/dev.jsonl` | 50 perguntas de desenvolvimento, revisadas (`source = human_reviewed`) |
| `data/test.jsonl` | 150 perguntas de teste, revisadas (`source = human_reviewed`) |
| `snapshot/coletas-2026-09.parquet` | Snapshot normalizado: 1.652.562 coletas, 2024-10-01 a 2026-09-30 (~2,4 MB) |
| `snapshot/coletas-2026-09.json` | SHA-256 do Parquet e contagem + MD5 de cada tabela do banco |
| `docs/schema.md` | Schema do banco, valores fechados de `produto`, `regiao` e bandeiras |
| `manifest.json` | Os 48 CSVs mensais da ANP usados, com tamanho e SHA-256, e as linhas rejeitadas |

Configs: `pairs` (padrão; splits `train`, `dev`, `test`) e `coletas` (o snapshot).

```python
from datasets import load_dataset

pairs = load_dataset("belasoares/tanque-cheio")  # train, dev, test
coletas = load_dataset("belasoares/tanque-cheio", "coletas")  # snapshot
```

## Período e data de referência

- **Snapshot:** coletas de **2024-10-01 a 2026-09-30** (24 meses, 27 UFs, 419 municípios,
  11.051 postos). Os arquivos foram baixados da ANP em 28/10/2026.
- **Data de referência:** **2026-09-30** (= `max(data_coleta)`). Toda pergunta com data
  relativa é resolvida contra ela: "último mês" é o mês do calendário que contém a data de
  referência (setembro de 2026), não o mês atual. O snapshot é fixo e não é atualizado.

## Schema

O Parquet tem uma linha por coleta, já normalizada: `regiao`, `uf`, `municipio`, `cnpj`,
`razao_social`, `bandeira`, `bairro`, `cep`, `produto`, `data_coleta`, `valor_venda`,
`valor_compra`, `unidade`. As SQLs não rodam sobre ele direto: rodam sobre o banco DuckDB
de 3 tabelas que `tanque data build --from-parquet` monta a partir dele.

```sql
CREATE TABLE municipios (
    municipio_id  INTEGER PRIMARY KEY,
    nome          VARCHAR NOT NULL,      -- como na ANP, em maiúsculas e sem acento: 'SAO CARLOS'
    uf            VARCHAR(2) NOT NULL,   -- 'SP'
    regiao        VARCHAR(2) NOT NULL,   -- 'N','NE','CO','SE','S'
    capital       BOOLEAN NOT NULL       -- true para as 27 capitais
);
CREATE TABLE postos (
    posto_id      INTEGER PRIMARY KEY,
    cnpj          VARCHAR NOT NULL UNIQUE,
    razao_social  VARCHAR,
    bandeira      VARCHAR,               -- 'BRANCA' para sem bandeira
    bairro        VARCHAR,
    cep           VARCHAR,
    municipio_id  INTEGER NOT NULL REFERENCES municipios(municipio_id)
);
CREATE TABLE coletas (
    coleta_id     BIGINT PRIMARY KEY,
    posto_id      INTEGER NOT NULL REFERENCES postos(posto_id),
    produto       VARCHAR NOT NULL,      -- GASOLINA, GASOLINA ADITIVADA, ETANOL, DIESEL, DIESEL S10, GNV
    data_coleta   DATE NOT NULL,
    valor_venda   DECIMAL(6,3) NOT NULL, -- R$ por unidade
    valor_compra  DECIMAL(6,3),          -- quase sempre nulo
    unidade       VARCHAR NOT NULL       -- 'R$ / litro' ou 'R$ / m³' (GNV)
);
```

Regras de construção: só os 6 combustíveis automotivos (GLP fica de fora); `valor_venda`
nulo ou ≤ 0 é rejeitado; coletas repetidas exatamente entram uma vez; se o posto mudou de
bandeira ou endereço, vale a coleta mais recente. Nome da rua, número e complemento não
entram. Detalhes em `docs/schema.md`.

Cada linha dos `.jsonl`:

```json
{"id": "test-0001", "question": "Qual é o preço médio do etanol no Brasil, considerando todo o período?",
 "sql": "SELECT AVG(valor_venda) AS preco_medio FROM coletas WHERE produto = 'ETANOL'",
 "difficulty": "facil", "uses_window": false, "order_matters": false, "split": "test",
 "source": "human_reviewed", "reviewed": true, "tags": ["etanol", "brasil"]}
```

- `difficulty`: `facil` (uma tabela ou um join, filtro e agregação), `medio` (dois ou mais
  joins, datas, GROUP BY, ranking), `dificil` (comparação entre períodos, CTEs, razão
  etanol/gasolina, funções de janela).
- `order_matters`: `true` quando a pergunta pede ordem ou ranking; a avaliação compara a
  sequência de linhas em vez do conjunto.

## Como os splits foram feitos

| Split | N | Fácil / médio / difícil | Com janela | Origem |
|---|---|---|---|---|
| `train` | 10.448 | 2.937 / 3.787 / 3.724 | 632 | Templates com valores reais do banco (v2) |
| `dev` | 50 | 15 / 20 / 15 | 3 | Escrito e revisado à mão |
| `test` | 150 | 50 / 60 / 40 | 8 | Escrito e revisado à mão |

- **Dev e teste.** As perguntas foram rascunhadas pelo projeto (com a ajuda de um agente
  de código, sem API paga) e revisadas uma a uma pela autora: a pergunta é natural e sem
  ambiguidade, a SQL responde exatamente a ela, o nível e `order_matters` estão certos.
  Toda SQL de referência executa e devolve linhas. O dev foi usado
  para escolher o modelo base e conferir o fine-tuning; o teste, uma vez por sistema, no
  relatório final da v1 e de novo no da v2. As decisões da v2 usaram o dev2 (o dev mais
  63 perguntas revisadas, 113 no total), que fica no repositório do projeto
  (`datasets/dev2.jsonl`) e não entra aqui.
- **Treino (v2).** Gerado por templates parametrizados
  (`tanque gen templates --n 13000 --seed 42`): 57 intenções (as 40 da v1 mais 17 para os
  padrões que o adapter v1 errava no dev2), cada uma com um template de SQL e várias
  redações de pergunta, preenchidos com valores sorteados do banco. Cada par passou por
  parse, guardrail e execução (o resultado não pode ser vazio). Depois, dois filtros:
  **vazamento** contra dev, dev2 e teste (pergunta com `rapidfuzz.token_set_ratio ≥ 85` ou
  SQL normalizada idêntica: 2.552 dos 13.000 pares descartados) e duplicatas internas. O
  treino v1 (8.217 pares, 40 intenções; uma auditoria manual de 50 pares sorteados aprovou
  49) está no histórico do repositório. Estatísticas por intenção:
  [`results/gen-stats.md`](https://github.com/belasoaress/tanque-cheio/blob/main/results/gen-stats.md).

## Reproduzir

No repositório do projeto:

```
uv sync
uv run tanque data build --from-parquet snapshot/coletas-2026-09.parquet   # banco DuckDB, sem download
uv run tanque eval verify-gold datasets/test.jsonl                         # toda SQL do teste executa
uv run tanque data hf-bundle --out dist/hf-dataset                          # monta esta pasta
```

`tanque data build` sem a opção baixa os 48 CSVs da ANP e refaz o snapshot do zero.

## Limitações

- Snapshot fixo; os preços não são atualizados.
- Só combustíveis automotivos, sem GLP nem volumes de venda.
- Só português, e só o vocabulário das perguntas de um projeto: o treino vem de templates,
  então a variedade de redação é menor que a de perguntas reais.
- O dev e o teste foram escritos pelo mesmo projeto que treinou o modelo (isolados do
  treino por código, mas com o mesmo estilo e as mesmas convenções de resposta).
- As SQLs seguem convenções fixas (preço médio = `AVG(valor_venda)`, combustível comum por
  padrão, município por `nome` + `uf`, quais colunas devolver). Uma SQL "certa" com outra
  convenção pode dar resultado diferente do gold.
- `razao_social`, `cnpj`, `bairro` e `cep` identificam empresas (postos), como nos dados
  públicos da ANP; não há dados de pessoas físicas além do que a ANP já publica.
