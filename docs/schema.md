# Schema do banco `data/tanque.duckdb`

Gerado por `tanque data build` (src/tanque/data/schema_doc.py). Não edite à mão.

- **Data de referência:** `2026-09-30` (= `max(data_coleta)`). Datas relativas são resolvidas contra ela; "último mês" = mês do calendário que contém a data de referência (D19).
- **Período:** `2024-10-01` a `2026-09-30` (24 meses com coletas).
- **Arquivos brutos:** 48 CSVs da ANP (hashes em `data/manifest.json`).
- **Linhas:** 1.652.568 lidas, 1.652.562 aceitas, 6 rejeitadas.
- **UFs presentes:** 27 (AC, AL, AM, AP, BA, CE, DF, ES, GO, MA, MG, MS, MT, PA, PB, PE, PI, PR, RJ, RN, RO, RR, RS, SC, SE, SP, TO).

## Tabelas

### `municipios` (419 linhas)

| coluna | tipo | nulo | descrição |
|---|---|---|---|
| `municipio_id` | INTEGER | não | ID determinístico (ordem de `uf`, `nome`) |
| `nome` | VARCHAR | não | Nome como na ANP, em maiúsculas e sem acento (ex.: `SAO CARLOS`) |
| `uf` | VARCHAR | não | Sigla da UF (ex.: `SP`) |
| `regiao` | VARCHAR | não | Sigla da região: `N`, `NE`, `CO`, `SE`, `S` |
| `capital` | BOOLEAN | não | `true` para as 27 capitais (lista fixa no código, D18) |

### `postos` (11.051 linhas)

| coluna | tipo | nulo | descrição |
|---|---|---|---|
| `posto_id` | INTEGER | não | ID determinístico (ordem de `cnpj`) |
| `cnpj` | VARCHAR | não | CNPJ da revenda, como na ANP (ex.: `00.003.188/0001-21`) |
| `razao_social` | VARCHAR | sim | Razão social da revenda (da coleta mais recente) |
| `bandeira` | VARCHAR | sim | Bandeira da coleta mais recente; `BRANCA` = sem bandeira |
| `bairro` | VARCHAR | sim | Bairro (da coleta mais recente) |
| `cep` | VARCHAR | sim | CEP (da coleta mais recente) |
| `municipio_id` | INTEGER | não | Município do posto (da coleta mais recente) |

### `coletas` (1.652.562 linhas)

| coluna | tipo | nulo | descrição |
|---|---|---|---|
| `coleta_id` | BIGINT | não | ID determinístico (ordem de `data_coleta`, `cnpj`, `produto`, valores, `unidade`) |
| `posto_id` | INTEGER | não | Posto pesquisado |
| `produto` | VARCHAR | não | Combustível; valores fechados abaixo |
| `data_coleta` | DATE | não | Data da coleta |
| `valor_venda` | DECIMAL(6,3) | não | Preço de venda ao consumidor, R$ por unidade; sempre > 0 |
| `valor_compra` | DECIMAL(6,3) | sim | Preço de compra pelo posto; quase sempre nulo |
| `unidade` | VARCHAR | não | `R$ / litro` ou `R$ / m³` (GNV) |

## DDL (`src/tanque/data/schema.sql`)

```sql
-- Contrato do banco. Não mudar sem aprovação.
CREATE TABLE municipios (
    municipio_id  INTEGER PRIMARY KEY,
    nome          VARCHAR NOT NULL,      -- como na ANP, em maiúsculas, ex.: 'SAO CARLOS'
    uf            VARCHAR(2) NOT NULL,   -- 'SP'
    regiao        VARCHAR(2) NOT NULL,   -- 'N','NE','CO','SE','S'
    capital       BOOLEAN NOT NULL       -- true para as 27 capitais (D18)
);

CREATE TABLE postos (
    posto_id      INTEGER PRIMARY KEY,
    cnpj          VARCHAR NOT NULL UNIQUE,
    razao_social  VARCHAR,
    bandeira      VARCHAR,               -- normalizada; 'BRANCA' para sem bandeira
    bairro        VARCHAR,
    cep           VARCHAR,
    municipio_id  INTEGER NOT NULL REFERENCES municipios(municipio_id)
);

CREATE TABLE coletas (
    coleta_id     BIGINT PRIMARY KEY,
    posto_id      INTEGER NOT NULL REFERENCES postos(posto_id),
    produto       VARCHAR NOT NULL,      -- valores fechados, ver docs/schema.md
    data_coleta   DATE NOT NULL,
    valor_venda   DECIMAL(6,3) NOT NULL, -- R$ por unidade
    valor_compra  DECIMAL(6,3),          -- frequentemente nulo
    unidade       VARCHAR NOT NULL       -- 'R$ / litro' ou 'R$ / m³'
);
```

## Valores fechados

### `coletas.produto`

| produto | unidade | coletas | min valor_venda | max valor_venda |
|---|---|---|---|---|
| `DIESEL` | `R$ / litro` | 178.554 | 5.070 | 9.450 |
| `DIESEL S10` | `R$ / litro` | 299.686 | 4.730 | 9.990 |
| `ETANOL` | `R$ / litro` | 366.283 | 2.770 | 7.440 |
| `GASOLINA` | `R$ / litro` | 434.869 | 4.890 | 9.790 |
| `GASOLINA ADITIVADA` | `R$ / litro` | 337.483 | 5.190 | 9.890 |
| `GNV` | `R$ / m³` | 35.687 | 3.190 | 6.890 |

### `municipios.regiao`

| regiao | UFs | municípios |
|---|---|---|
| `CO` | 4 | 29 |
| `N` | 7 | 31 |
| `NE` | 9 | 91 |
| `S` | 3 | 77 |
| `SE` | 4 | 191 |

### `postos.bandeira`: as 15 mais frequentes (de 50 distintas, por número de postos)

| bandeira | postos |
|---|---|
| `BRANCA` | 3.777 |
| `VIBRA` | 2.377 |
| `IPIRANGA` | 2.336 |
| `RAIZEN` | 1.580 |
| `ALE` | 291 |
| `SABBÁ` | 111 |
| `RODOIL` | 67 |
| `RAIZEN MIME` | 51 |
| `ATEM' S` | 44 |
| `NEXTA` | 44 |
| `EQUADOR` | 40 |
| `CHARRUA` | 39 |
| `TAURUS` | 30 |
| `SP` | 27 |
| `DISLUB` | 21 |

### Capitais (`municipios.capital = true`): 27 de 27

RIO BRANCO/AC, MACEIO/AL, MANAUS/AM, MACAPA/AP, SALVADOR/BA, FORTALEZA/CE, BRASILIA/DF, VITORIA/ES, GOIANIA/GO, SAO LUIS/MA, BELO HORIZONTE/MG, CAMPO GRANDE/MS, CUIABA/MT, BELEM/PA, JOAO PESSOA/PB, RECIFE/PE, TERESINA/PI, CURITIBA/PR, RIO DE JANEIRO/RJ, NATAL/RN, PORTO VELHO/RO, BOA VISTA/RR, PORTO ALEGRE/RS, FLORIANOPOLIS/SC, ARACAJU/SE, SAO PAULO/SP, PALMAS/TO

## Regras de construção

- Só entram os 6 produtos automotivos acima; GLP e outros vão para o relatório de rejeição.
- `valor_venda` nulo, inválido ou ≤ 0 não entra no banco; as linhas rejeitadas são contadas por motivo em `data/manifest.json` (`rows.rejected`).
- Coletas repetidas exatamente (mesmo CNPJ, produto, data, valores e unidade) entram uma vez só (`linha_duplicada`).
- Se o posto mudou de bandeira (ou de endereço) no período, vale a coleta mais recente. Bandeira vazia vira `BRANCA`.
- Nome da rua, número e complemento não entram no banco.
- Nomes de município sem acento e em maiúsculas, como nos arquivos da ANP.

### Rejeitadas por motivo

| motivo | linhas |
|---|---|
| `linha_duplicada` | 6 |
