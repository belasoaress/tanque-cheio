# Relatório final

> Gerado por `uv run tanque eval report` a partir de `results/metrics.json`, das previsões em `results/predictions/` e de `results/error-annotations.json`. Não edite à mão: o CI confere com `tanque eval report --check`.

EX = execution accuracy (seção 5.6 do plano), a métrica principal; EX tolerante = aceita colunas a mais na previsão (D28); IC 95% = intervalo de confiança bootstrap (D29). Definições na nota de método. Percentuais sobre todos os itens do split. `*` = custo estimado de GPU.

## Tabela principal: teste (150 perguntas)

| Sistema | Split | N | EX | IC 95% EX | EX tolerante | IC 95% tolerante | EX fácil | EX médio | EX difícil | SQL válida | Bloqueada | p50 (ms) | p95 (ms) | GPUs | Pico GPU (GiB) | US$ / 1.000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `base-qwen2.5-coder-1.5b-instruct` | test | 150 | 12,0% | 7,3–17,3% | 12,7% | 8,0–18,0% | 28,0% | 6,7% | 0,0% | 22,0% | 3,3% | 3.087 | 7.630 | 1 | 3,1 | 0,3384* |
| `base-qwen2.5-coder-1.5b-instruct+retry` | test | 150 | 12,0% | 7,3–17,3% | 12,7% | 8,0–18,0% | 28,0% | 6,7% | 0,0% | 23,3% | 3,3% | 6.232 | 13.517 | 1 | 3,1 | 0,6417* |
| `base-qwen2.5-coder-7b-instruct` | test | 150 | 40,7% | 32,7–48,7% | 46,0% | 38,0–54,0% | 28,0% | 70,0% | 12,5% | 62,7% | 0,7% | 8.800 | 18.218 | 1 | 5,8 | 0,8919* |
| `base-qwen2.5-coder-7b-instruct+retry` | test | 150 | 51,3% | 43,3–59,3% | 56,7% | 48,7–64,7% | 50,0% | 75,0% | 17,5% | 90,0% | 1,3% | 10.270 | 32.904 | 1 | 5,8 | 1,2643* |
| `base-qwen2.5-coder-14b-instruct` | test | 150 | 60,0% | 52,0–68,0% | 63,3% | 55,3–71,3% | 62,0% | 81,7% | 25,0% | 87,3% | 1,3% | 16.228 | 46.047 | 1 | 10,2 | 1,8745* |
| `base-qwen2.5-coder-14b-instruct+retry` | test | 150 | 62,7% | 54,7–70,7% | 66,7% | 59,3–74,0% | 62,0% | 86,7% | 27,5% | 94,7% | 1,3% | 16.456 | 68.499 | 1 | 10,2 | 2,2981* |
| `base-qwen2.5-coder-32b-instruct` | test | 150 | 78,0% | 71,3–84,7% | 82,7% | 76,7–88,7% | 96,0% | 90,0% | 37,5% | 96,7% | 0,7% | 29.052 | 74.415 | 2 | 10,7 | 6,2203* |
| `base-qwen2.5-coder-32b-instruct+retry` | test | 150 | 78,7% | 72,0–85,3% | 84,0% | 78,0–89,3% | 96,0% | 90,0% | 40,0% | 99,3% | 0,7% | 29.052 | 79.602 | 2 | 10,7 | 6,6139* |
| `ft-qwen2.5-coder-1.5b-instruct` | test | 150 | 76,0% | 68,7–82,7% | 84,0% | 78,0–89,3% | 98,0% | 76,7% | 47,5% | 96,7% | 0,7% | 4.259 | 9.352 | 1 | 3,1 | 0,4206* |
| `ft-qwen2.5-coder-1.5b-instruct+retry` | test | 150 | 76,0% | 68,7–82,7% | 84,0% | 78,0–89,3% | 98,0% | 76,7% | 47,5% | 98,0% | 0,0% | 4.268 | 9.588 | 1 | 3,1 | 0,4396* |
| `ft2-qwen2.5-coder-1.5b-instruct` | test | 150 | 80,7% | 74,0–86,7% | 85,3% | 79,3–90,7% | 92,0% | 83,3% | 62,5% | 98,7% | 0,7% | 4.268 | 8.952 | 1 | 3,1 | 0,4165* |
| `ft2-qwen2.5-coder-1.5b-instruct+retry` | test | 150 | 80,7% | 74,0–86,7% | 85,3% | 79,3–90,7% | 92,0% | 83,3% | 62,5% | 99,3% | 0,7% | 4.324 | 8.952 | 1 | 3,1 | 0,4207* |

## Base → ajustado (teste)

`base-qwen2.5-coder-1.5b-instruct` → `ft-qwen2.5-coder-1.5b-instruct`:

| Métrica | Base | Ajustado | Δ |
|---|---|---|---|
| EX | 12,0% | 76,0% | +64,0 p.p. |
| EX tolerante | 12,7% | 84,0% | +71,3 p.p. |
| EX fácil | 28,0% | 98,0% | +70,0 p.p. |
| EX médio | 6,7% | 76,7% | +70,0 p.p. |
| EX difícil | 0,0% | 47,5% | +47,5 p.p. |
| EX tolerante fácil | 30,0% | 98,0% | +68,0 p.p. |
| EX tolerante médio | 6,7% | 88,3% | +81,7 p.p. |
| EX tolerante difícil | 0,0% | 60,0% | +60,0 p.p. |
| SQL válida | 22,0% | 96,7% | +74,7 p.p. |
| Bloqueada | 3,3% | 0,7% | −2,7 p.p. |
| Latência p50 (ms) | 3.087 | 4.259 | +1.172 |
| Latência p95 (ms) | 7.630 | 9.352 | +1.722 |
| US$ / 1.000 | 0,3384* | 0,4206* | +0,0821 |

`base-qwen2.5-coder-1.5b-instruct+retry` → `ft-qwen2.5-coder-1.5b-instruct+retry`:

| Métrica | Base | Ajustado | Δ |
|---|---|---|---|
| EX | 12,0% | 76,0% | +64,0 p.p. |
| EX tolerante | 12,7% | 84,0% | +71,3 p.p. |
| EX fácil | 28,0% | 98,0% | +70,0 p.p. |
| EX médio | 6,7% | 76,7% | +70,0 p.p. |
| EX difícil | 0,0% | 47,5% | +47,5 p.p. |
| EX tolerante fácil | 30,0% | 98,0% | +68,0 p.p. |
| EX tolerante médio | 6,7% | 88,3% | +81,7 p.p. |
| EX tolerante difícil | 0,0% | 60,0% | +60,0 p.p. |
| SQL válida | 23,3% | 98,0% | +74,7 p.p. |
| Bloqueada | 3,3% | 0,0% | −3,3 p.p. |
| Latência p50 (ms) | 6.232 | 4.268 | −1.964 |
| Latência p95 (ms) | 13.517 | 9.588 | −3.928 |
| US$ / 1.000 | 0,6417* | 0,4396* | −0,2020 |

`base-qwen2.5-coder-1.5b-instruct` → `ft2-qwen2.5-coder-1.5b-instruct`:

| Métrica | Base | Ajustado | Δ |
|---|---|---|---|
| EX | 12,0% | 80,7% | +68,7 p.p. |
| EX tolerante | 12,7% | 85,3% | +72,7 p.p. |
| EX fácil | 28,0% | 92,0% | +64,0 p.p. |
| EX médio | 6,7% | 83,3% | +76,7 p.p. |
| EX difícil | 0,0% | 62,5% | +62,5 p.p. |
| EX tolerante fácil | 30,0% | 92,0% | +62,0 p.p. |
| EX tolerante médio | 6,7% | 86,7% | +80,0 p.p. |
| EX tolerante difícil | 0,0% | 75,0% | +75,0 p.p. |
| SQL válida | 22,0% | 98,7% | +76,7 p.p. |
| Bloqueada | 3,3% | 0,7% | −2,7 p.p. |
| Latência p50 (ms) | 3.087 | 4.268 | +1.181 |
| Latência p95 (ms) | 7.630 | 8.952 | +1.322 |
| US$ / 1.000 | 0,3384* | 0,4165* | +0,0781 |

`base-qwen2.5-coder-1.5b-instruct+retry` → `ft2-qwen2.5-coder-1.5b-instruct+retry`:

| Métrica | Base | Ajustado | Δ |
|---|---|---|---|
| EX | 12,0% | 80,7% | +68,7 p.p. |
| EX tolerante | 12,7% | 85,3% | +72,7 p.p. |
| EX fácil | 28,0% | 92,0% | +64,0 p.p. |
| EX médio | 6,7% | 83,3% | +76,7 p.p. |
| EX difícil | 0,0% | 62,5% | +62,5 p.p. |
| EX tolerante fácil | 30,0% | 92,0% | +62,0 p.p. |
| EX tolerante médio | 6,7% | 86,7% | +80,0 p.p. |
| EX tolerante difícil | 0,0% | 75,0% | +75,0 p.p. |
| SQL válida | 23,3% | 99,3% | +76,0 p.p. |
| Bloqueada | 3,3% | 0,7% | −2,7 p.p. |
| Latência p50 (ms) | 6.232 | 4.324 | −1.907 |
| Latência p95 (ms) | 13.517 | 8.952 | −4.565 |
| US$ / 1.000 | 0,6417* | 0,4207* | −0,2209 |

## Comparações pareadas (teste)

Destaques: `ft2-qwen2.5-coder-1.5b-instruct` contra o maior modelo sem ajuste, o outro ajustado e o segundo maior (Δ = final − outro; Só final / Só outro = perguntas que só um dos dois acerta):

| Modelo final | Outro | N | EX final | EX outro | Δ EX (final − outro) | Só final | Só outro | p (McNemar) |
|---|---|---|---|---|---|---|---|---|
| `ft2-qwen2.5-coder-1.5b-instruct` | `base-qwen2.5-coder-32b-instruct` | 150 | 80,7% | 78,0% | +2,7 p.p. | 21 | 17 | 0,627 |
| `ft2-qwen2.5-coder-1.5b-instruct` | `ft-qwen2.5-coder-1.5b-instruct` | 150 | 80,7% | 76,0% | +4,7 p.p. | 11 | 4 | 0,118 |
| `ft2-qwen2.5-coder-1.5b-instruct` | `base-qwen2.5-coder-14b-instruct` | 150 | 80,7% | 60,0% | +20,7 p.p. | 47 | 16 | < 0,001 |

Todas as comparações:

| A | B | N | EX A | EX B | Δ EX (B − A) | Só A | Só B | p (McNemar) |
|---|---|---|---|---|---|---|---|---|
| `base-qwen2.5-coder-1.5b-instruct` | `base-qwen2.5-coder-1.5b-instruct+retry` | 150 | 12,0% | 12,0% | ±0,0 p.p. | 0 | 0 | 1,000 |
| `base-qwen2.5-coder-1.5b-instruct` | `base-qwen2.5-coder-7b-instruct` | 150 | 12,0% | 40,7% | +28,7 p.p. | 3 | 46 | < 0,001 |
| `base-qwen2.5-coder-1.5b-instruct` | `base-qwen2.5-coder-14b-instruct` | 150 | 12,0% | 60,0% | +48,0 p.p. | 1 | 73 | < 0,001 |
| `base-qwen2.5-coder-1.5b-instruct` | `base-qwen2.5-coder-32b-instruct` | 150 | 12,0% | 78,0% | +66,0 p.p. | 0 | 99 | < 0,001 |
| `base-qwen2.5-coder-1.5b-instruct` | `ft-qwen2.5-coder-1.5b-instruct` | 150 | 12,0% | 76,0% | +64,0 p.p. | 0 | 96 | < 0,001 |
| `base-qwen2.5-coder-1.5b-instruct` | `ft2-qwen2.5-coder-1.5b-instruct` | 150 | 12,0% | 80,7% | +68,7 p.p. | 2 | 105 | < 0,001 |
| `base-qwen2.5-coder-7b-instruct` | `base-qwen2.5-coder-7b-instruct+retry` | 150 | 40,7% | 51,3% | +10,7 p.p. | 0 | 16 | < 0,001 |
| `base-qwen2.5-coder-7b-instruct` | `base-qwen2.5-coder-14b-instruct` | 150 | 40,7% | 60,0% | +19,3 p.p. | 10 | 39 | < 0,001 |
| `base-qwen2.5-coder-7b-instruct` | `base-qwen2.5-coder-32b-instruct` | 150 | 40,7% | 78,0% | +37,3 p.p. | 5 | 61 | < 0,001 |
| `base-qwen2.5-coder-7b-instruct` | `ft-qwen2.5-coder-1.5b-instruct` | 150 | 40,7% | 76,0% | +35,3 p.p. | 9 | 62 | < 0,001 |
| `base-qwen2.5-coder-7b-instruct` | `ft2-qwen2.5-coder-1.5b-instruct` | 150 | 40,7% | 80,7% | +40,0 p.p. | 8 | 68 | < 0,001 |
| `base-qwen2.5-coder-14b-instruct` | `base-qwen2.5-coder-14b-instruct+retry` | 150 | 60,0% | 62,7% | +2,7 p.p. | 0 | 4 | 0,125 |
| `base-qwen2.5-coder-14b-instruct` | `base-qwen2.5-coder-32b-instruct` | 150 | 60,0% | 78,0% | +18,0 p.p. | 6 | 33 | < 0,001 |
| `base-qwen2.5-coder-14b-instruct` | `ft-qwen2.5-coder-1.5b-instruct` | 150 | 60,0% | 76,0% | +16,0 p.p. | 17 | 41 | 0,002 |
| `base-qwen2.5-coder-14b-instruct` | `ft2-qwen2.5-coder-1.5b-instruct` | 150 | 60,0% | 80,7% | +20,7 p.p. | 16 | 47 | < 0,001 |
| `base-qwen2.5-coder-32b-instruct` | `base-qwen2.5-coder-32b-instruct+retry` | 150 | 78,0% | 78,7% | +0,7 p.p. | 0 | 1 | 1,000 |
| `base-qwen2.5-coder-32b-instruct` | `ft-qwen2.5-coder-1.5b-instruct` | 150 | 78,0% | 76,0% | −2,0 p.p. | 20 | 17 | 0,743 |
| `base-qwen2.5-coder-32b-instruct` | `ft2-qwen2.5-coder-1.5b-instruct` | 150 | 78,0% | 80,7% | +2,7 p.p. | 17 | 21 | 0,627 |
| `ft-qwen2.5-coder-1.5b-instruct` | `ft-qwen2.5-coder-1.5b-instruct+retry` | 150 | 76,0% | 76,0% | ±0,0 p.p. | 0 | 0 | 1,000 |
| `ft-qwen2.5-coder-1.5b-instruct` | `ft2-qwen2.5-coder-1.5b-instruct` | 150 | 76,0% | 80,7% | +4,7 p.p. | 4 | 11 | 0,118 |
| `ft2-qwen2.5-coder-1.5b-instruct` | `ft2-qwen2.5-coder-1.5b-instruct+retry` | 150 | 80,7% | 80,7% | ±0,0 p.p. | 0 | 0 | 1,000 |

## Erros do modelo ajustado (teste)

Modelo final: `ft2-qwen2.5-coder-1.5b-instruct` (o `ft*` sem nova tentativa de maior EX no teste).

### `ft2-qwen2.5-coder-1.5b-instruct` (split `test`)

29 erros em 150 perguntas. Tipos (heurística por código, com a classificação manual de `results/error-annotations.json` quando houver):

| Tipo | N | % dos erros |
|---|---|---|
| Bloqueada pelo guardrail | 1 | 3,4% |
| SQL inválida | 1 | 3,4% |
| Coluna/forma do resultado | 10 | 34,5% |
| Agregação | 5 | 17,2% |
| Filtro de data | 2 | 6,9% |
| Filtro de valor | 2 | 6,9% |
| Lógica | 8 | 27,6% |

10 exemplos (um tipo por vez, em rodízio):

**1. `test-0108`** · medio · Bloqueada pelo guardrail (heurística)

> Qual o maior e o menor preço médio mensal da gasolina comum no Brasil em 2025?

Erro: `bloqueada pelo guardrail: SQL não faz parse no DuckDB: Parser Error: syntax error at or near "," LINE 1: ... < DATE '2026-01-01' GROUP BY mes ORDER BY preco_medio LIMIT 1, 1 ^`

Referência:

```sql
SELECT MAX(preco_medio) AS maior, MIN(preco_medio) AS menor FROM (SELECT date_trunc('month', c.data_coleta) AS mes, AVG(c.valor_venda) AS preco_medio FROM coletas c WHERE c.produto = 'GASOLINA' AND c.data_coleta >= DATE '2025-01-01' AND c.data_coleta < DATE '2026-01-01' GROUP BY mes) t
```

Prevista:

```sql
SELECT date_trunc('month', c.data_coleta) AS mes, AVG(c.valor_venda) AS preco_medio FROM coletas c WHERE c.produto = 'GASOLINA' AND c.data_coleta >= DATE '2025-01-01' AND c.data_coleta < DATE '2026-01-01' GROUP BY mes ORDER BY preco_medio LIMIT 1, 1
```

**2. `test-0146`** · dificil · SQL inválida (heurística)

> Qual a média, entre os estados, do preço médio estadual do diesel comum no último mês?

Erro: `BinderException: Binder Error: No function matches the given name and argument types 'avg(VARCHAR)'. You might need to add explicit type casts. Candidate functions: avg(x DECIMAL) -> DECIMAL avg(x SM…`

Referência:

```sql
WITH u AS (SELECT m.uf, AVG(c.valor_venda) AS p FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'DIESEL' AND c.data_coleta >= DATE '2026-09-01' AND c.data_coleta < DATE '2026-10-01' GROUP BY m.uf) SELECT AVG(p) AS media_das_ufs FROM u
```

Prevista:

```sql
SELECT AVG(m.uf) AS estado, AVG(c.valor_venda) AS preco_medio FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'DIESEL' AND c.data_coleta >= DATE '2026-09-01' AND c.data_coleta < DATE '2026-10-01' GROUP BY m.uf
```

**3. `test-0077`** · medio · Coluna/forma do resultado (heurística)

> Qual o preço médio do etanol nas cidades do Espírito Santo no último mês?

Referência:

```sql
SELECT m.nome, AVG(c.valor_venda) AS preco_medio FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'ETANOL' AND m.uf = 'ES' AND c.data_coleta >= DATE '2026-09-01' AND c.data_coleta < DATE '2026-10-01' GROUP BY m.nome
```

Prevista:

```sql
SELECT AVG(c.valor_venda) AS preco_medio FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'ETANOL' AND m.uf = 'ES' AND c.data_coleta >= DATE '2026-09-01' AND c.data_coleta < DATE '2026-10-01'
```

**4. `test-0084`** · medio · Agregação (heurística)

> Qual foi o maior preço de gasolina comum registrado em cada região no último mês?

Referência:

```sql
SELECT m.regiao, MAX(c.valor_venda) AS maior_preco FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'GASOLINA' AND c.data_coleta >= DATE '2026-09-01' AND c.data_coleta < DATE '2026-10-01' GROUP BY m.regiao
```

Prevista:

```sql
WITH precos AS (SELECT m.regiao, AVG(c.valor_venda) AS preco FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'GASOLINA' AND c.data_coleta >= DATE '2026-09-01' AND c.data_coleta < DATE '2026-10-01' GROUP BY m.regiao), r AS (SELECT *, RANK() OVER (PARTITION BY regiao ORDER BY preco DESC) AS pos FROM precos) SELECT regiao, preco FROM r WHERE pos = 1 ORDER BY regiao
```

**5. `test-0005`** · facil · Filtro de data (heurística)

> Qual o preço médio do GNV no estado do rio de janeiro?

Referência:

```sql
SELECT AVG(c.valor_venda) AS preco_medio FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'GNV' AND m.uf = 'RJ'
```

Prevista:

```sql
SELECT AVG(c.valor_venda) AS preco_medio FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'GNV' AND m.uf = 'RJ' AND c.data_coleta >= DATE '2026-01-01' AND c.data_coleta < DATE '2026-02-01'
```

**6. `test-0042`** · facil · Filtro de valor (heurística)

> Quantas coletas foram feitas em postos de Joinville?

Referência:

```sql
SELECT COUNT(*) AS coletas FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE m.nome = 'JOINVILLE' AND m.uf = 'SC'
```

Prevista:

```sql
SELECT COUNT(*) AS coletas FROM coletas c JOIN postos p ON p.posto_id = c.posto_id WHERE p.bairro = 'JOINVILLE' AND p.municipio_id = (SELECT m.municipio_id FROM municipios m WHERE m.nome = 'JOINVILLE' AND m.uf = 'SC')
```

**7. `test-0018`** · facil · Lógica (heurística)

> Quais bandeiras aparecem entre os postos de Manaus?

Referência:

```sql
SELECT DISTINCT p.bandeira FROM postos p JOIN municipios m ON m.municipio_id = p.municipio_id WHERE m.nome = 'MANAUS' AND m.uf = 'AM'
```

Prevista:

```sql
SELECT p.bandeira FROM postos p JOIN municipios m ON m.municipio_id = p.municipio_id WHERE m.nome = 'MANAUS' AND m.uf = 'AM'
```

**8. `test-0079`** · medio · Coluna/forma do resultado (heurística)

> Qual o preço médio do etanol nas capitais do Centro-Oeste no último mês, por capital?

Referência:

```sql
SELECT m.nome, AVG(c.valor_venda) AS preco_medio FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'ETANOL' AND m.capital AND m.regiao = 'CO' AND c.data_coleta >= DATE '2026-09-01' AND c.data_coleta < DATE '2026-10-01' GROUP BY m.nome
```

Prevista:

```sql
SELECT m.nome, m.capital, AVG(c.valor_venda) AS preco_medio FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'ETANOL' AND m.capital AND m.regiao = 'CO' AND c.data_coleta >= DATE '2026-09-01' AND c.data_coleta < DATE '2026-10-01' GROUP BY m.nome, m.capital
```

**9. `test-0121`** · dificil · Agregação (heurística)

> Por estado, quanto o preço médio do GNV em 2026 (jan a set) ficou acima ou abaixo do preço médio de 2025? Considere só estados com coletas nos dois anos.

Referência:

```sql
SELECT m.uf, AVG(c.valor_venda) FILTER (WHERE c.data_coleta >= DATE '2026-01-01' AND c.data_coleta < DATE '2027-01-01') - AVG(c.valor_venda) FILTER (WHERE c.data_coleta >= DATE '2025-01-01' AND c.data_coleta < DATE '2026-01-01') AS diferenca FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'GNV' GROUP BY m.uf HAVING COUNT(*) FILTER (WHERE c.data_coleta >= DATE '2026-01-01' AND c.data_coleta < DATE '2027-01-01') > 0 AND COUNT(*) FILTER (WHERE c.data_coleta >= DATE '2025-01-01' AND c.data_coleta < DATE '2026-01-01') > 0
```

Prevista:

```sql
WITH a AS (SELECT m.uf, AVG(c.valor_venda) AS p FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'GNV' AND c.data_coleta >= DATE '2025-01-01' AND c.data_coleta < DATE '2025-07-01' GROUP BY m.uf), b AS (SELECT m.uf, AVG(c.valor_venda) AS p FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'GNV' AND c.data_coleta >= DATE '2026-01-01' AND c.data_coleta < DATE '2026-07-01' GROUP BY m.uf) SELECT a.uf, b.p - a.p AS diferenca FROM a JOIN b ON a.uf = b.uf
```

**10. `test-0123`** · dificil · Filtro de data (heurística)

> Quanto a gasolina comum de Curitiba variou, em reais, entre agosto e setembro de 2026?

Referência:

```sql
SELECT AVG(c.valor_venda) FILTER (WHERE c.data_coleta >= DATE '2026-09-01' AND c.data_coleta < DATE '2026-10-01') - AVG(c.valor_venda) FILTER (WHERE c.data_coleta >= DATE '2026-08-01' AND c.data_coleta < DATE '2026-09-01') AS variacao FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'GASOLINA' AND m.nome = 'CURITIBA' AND m.uf = 'PR'
```

Prevista:

```sql
SELECT AVG(c.valor_venda) - AVG(c.valor_compra) AS diferenca FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'GASOLINA' AND m.nome = 'CURITIBA' AND m.uf = 'PR' AND c.data_coleta >= DATE '2026-08-01' AND c.data_coleta < DATE '2026-10-01'
```

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
| `ft2-qwen2.5-coder-1.5b-instruct` | dev2 | 113 | 96,5% | 92,9–99,1% | 97,4% | 93,8–100,0% | 100,0% | 94,1% | 95,0% | 100,0% | 0,0% | 3.929 | 7.833 | 1 | 3,1 | 0,3865* |
| `ft2-qwen2.5-coder-1.5b-instruct+retry` | dev2 | 113 | 96,5% | 92,9–99,1% | 97,4% | 93,8–100,0% | 100,0% | 94,1% | 95,0% | 100,0% | 0,0% | 3.929 | 7.833 | 1 | 3,1 | 0,3865* |

Comparações pareadas no dev2:

| A | B | N | EX A | EX B | Δ EX (B − A) | Só A | Só B | p (McNemar) |
|---|---|---|---|---|---|---|---|---|
| `base-qwen2.5-coder-1.5b-instruct` | `base-qwen2.5-coder-1.5b-instruct+retry` | 113 | 11,5% | 11,5% | ±0,0 p.p. | 0 | 0 | 1,000 |
| `base-qwen2.5-coder-1.5b-instruct` | `base-qwen2.5-coder-7b-instruct` | 113 | 11,5% | 31,9% | +20,3 p.p. | 2 | 25 | < 0,001 |
| `base-qwen2.5-coder-1.5b-instruct` | `base-qwen2.5-coder-14b-instruct` | 113 | 11,5% | 40,7% | +29,2 p.p. | 1 | 34 | < 0,001 |
| `base-qwen2.5-coder-1.5b-instruct` | `base-qwen2.5-coder-32b-instruct` | 113 | 11,5% | 72,6% | +61,1 p.p. | 0 | 69 | < 0,001 |
| `base-qwen2.5-coder-1.5b-instruct` | `ft-qwen2.5-coder-1.5b-instruct` | 113 | 11,5% | 74,3% | +62,8 p.p. | 0 | 71 | < 0,001 |
| `base-qwen2.5-coder-1.5b-instruct` | `ft2-qwen2.5-coder-1.5b-instruct` | 113 | 11,5% | 96,5% | +85,0 p.p. | 0 | 96 | < 0,001 |
| `base-qwen2.5-coder-7b-instruct` | `base-qwen2.5-coder-7b-instruct+retry` | 113 | 31,9% | 38,0% | +6,2 p.p. | 0 | 7 | 0,016 |
| `base-qwen2.5-coder-7b-instruct` | `base-qwen2.5-coder-14b-instruct` | 113 | 31,9% | 40,7% | +8,8 p.p. | 10 | 20 | 0,099 |
| `base-qwen2.5-coder-7b-instruct` | `base-qwen2.5-coder-32b-instruct` | 113 | 31,9% | 72,6% | +40,7 p.p. | 5 | 51 | < 0,001 |
| `base-qwen2.5-coder-7b-instruct` | `ft-qwen2.5-coder-1.5b-instruct` | 113 | 31,9% | 74,3% | +42,5 p.p. | 6 | 54 | < 0,001 |
| `base-qwen2.5-coder-7b-instruct` | `ft2-qwen2.5-coder-1.5b-instruct` | 113 | 31,9% | 96,5% | +64,6 p.p. | 1 | 74 | < 0,001 |
| `base-qwen2.5-coder-14b-instruct` | `base-qwen2.5-coder-14b-instruct+retry` | 113 | 40,7% | 46,0% | +5,3 p.p. | 0 | 6 | 0,031 |
| `base-qwen2.5-coder-14b-instruct` | `base-qwen2.5-coder-32b-instruct` | 113 | 40,7% | 72,6% | +31,9 p.p. | 1 | 37 | < 0,001 |
| `base-qwen2.5-coder-14b-instruct` | `ft-qwen2.5-coder-1.5b-instruct` | 113 | 40,7% | 74,3% | +33,6 p.p. | 6 | 44 | < 0,001 |
| `base-qwen2.5-coder-14b-instruct` | `ft2-qwen2.5-coder-1.5b-instruct` | 113 | 40,7% | 96,5% | +55,8 p.p. | 0 | 63 | < 0,001 |
| `base-qwen2.5-coder-32b-instruct` | `base-qwen2.5-coder-32b-instruct+retry` | 113 | 72,6% | 74,3% | +1,8 p.p. | 0 | 2 | 0,500 |
| `base-qwen2.5-coder-32b-instruct` | `ft-qwen2.5-coder-1.5b-instruct` | 113 | 72,6% | 74,3% | +1,8 p.p. | 15 | 17 | 0,860 |
| `base-qwen2.5-coder-32b-instruct` | `ft2-qwen2.5-coder-1.5b-instruct` | 113 | 72,6% | 96,5% | +23,9 p.p. | 2 | 29 | < 0,001 |
| `ft-qwen2.5-coder-1.5b-instruct` | `ft-qwen2.5-coder-1.5b-instruct+retry` | 113 | 74,3% | 74,3% | ±0,0 p.p. | 0 | 0 | 1,000 |
| `ft-qwen2.5-coder-1.5b-instruct` | `ft2-qwen2.5-coder-1.5b-instruct` | 113 | 74,3% | 96,5% | +22,1 p.p. | 2 | 27 | < 0,001 |
| `ft2-qwen2.5-coder-1.5b-instruct` | `ft2-qwen2.5-coder-1.5b-instruct+retry` | 113 | 96,5% | 96,5% | ±0,0 p.p. | 0 | 0 | 1,000 |

† Latência medida com o adapter LoRA sem fundir (`PeftModel`, run-info com `adapter_merged` ausente ou falso): não é comparável à do base. No teste o adapter é fundido antes de gerar.

## Nota de método

- **Dados:** snapshot fixo `snapshot/coletas-2026-09.parquet` (SHA-256 `55a2b1884665284ac4fc8e770623abb211617ab98c2fed4980e7c475b8a35006`), coletas de 2024-10-01 a 2026-09-30; o CI reconstrói o banco a partir dele e confere contagem e MD5 de cada tabela (D25).
- **Data de referência:** `2026-09-30` (= `max(data_coleta)`, D14); "último mês" = mês do calendário que a contém (D19).
- **Prompt único** (`tanque.prompting.build_prompt`, seção 5.4) para todos os sistemas, pelo chat template de cada modelo; decodificação gulosa, lote de 1 pergunta, `max_new_tokens = 384` no teste v2 e no dev2 (D31; no dev da v1 eram 256).
- **Métrica:** EX = execution accuracy (seção 5.6): a SQL prevista é executada com o guardrail e comparada ao resultado da referência; bloqueio, erro ou falta de previsão contam como erro. É a métrica principal (EX rígida): o número de colunas tem de ser o mesmo da referência.
- **EX tolerante (secundária, D28; definida antes dos resultados da v2):** acerta quando as colunas da referência estão entre as colunas da previsão, que pode ter colunas a mais (até 6 colunas na previsão), em qualquer ordem, com a mesma escolha de colunas para todas as linhas; depois valem as regras da EX rígida (mesmas linhas, arredondamento a 2 casas, ordem quando a pergunta pede). Toda previsão certa na rígida é certa na tolerante. Mede quantos erros são só de colunas a mais (`results_match_relaxed`, seção 5.6 da v2).
- **IC 95% (D29):** bootstrap percentil sobre as perguntas do split, 10.000 reamostragens com reposição, seed 42 (`tanque.eval.stats`). Com 150 perguntas, a margem da EX é de cerca de ±7 p.p.
- **Comparações pareadas (D29):** todo par de sistemas sem nova tentativa pontuados no mesmo split, mais cada sistema × a sua versão `+retry`; pergunta a pergunta, na EX rígida. Só A / Só B = perguntas que só um dos dois acerta; p = teste de McNemar exato (binomial, bilateral) sobre esses pares discordantes. p < 0,05 indica diferença dificilmente explicada pelo acaso. Os destaques do teste orientam cada par a partir do modelo final (Δ = final − outro).
- **Modelos locais (D26):** todos da família Qwen2.5-Coder-Instruct (mesmo tokenizer), sem ajuste (`base-*`) ou com adapter LoRA fundido no base antes de gerar: `ft-*` = adapter v1 (M6), `ft2-*` = adapter v2 (M15). O 1,5B roda em fp16. Hardware das previsões do teste: `kaggle-t4`. A latência é medida por pergunta, do template à decodificação, numa T4 (o 32B, em 2). Pico GPU = maior memória alocada numa GPU durante a execução (no 32B, por GPU).
- **Modelos grandes em 4 bits (D27):** o 7B, o 14B e o 32B rodam quantizados em nf4 (bitsandbytes, quantização dupla, computação em fp16), o 7B e o 14B numa T4 e o 32B em 2 T4 (`device_map="auto"`), porque é o que cabe nas GPUs grátis do Kaggle. A quantização pode tirar alguns pontos deles: os números dos grandes são um piso do que fariam em fp16.
- **Nova tentativa (D32):** as linhas `<sistema>+retry` vêm da mesma execução que `<sistema>`: quando a primeira SQL é bloqueada ou dá erro, o modelo recebe a mensagem de erro e tenta uma única vez de novo. As duas linhas aparecem juntas, com e sem a nova tentativa, e a comparação pareada mostra o ganho (R21).
- **Custo por 1.000 perguntas (* = estimativa):** latência média × 1.000 ÷ 3.600.000 ms/h × US$ 0,35/h, o preço de NVIDIA T4 (1 GPU), Google Compute Engine sob demanda, us-central1 em https://cloud.google.com/products/compute/gpus-pricing (consultado em 2026-10-29); não inclui a VM (vCPU e memória), então é um piso. O preço é por GPU (D35): o 32B, em 2 T4, paga US$ 0,70/h. No Kaggle a execução não custou nada; as previsões têm `cost_usd = 0`.
- **Sem LLM de fronteira por enquanto (D24):** sem crédito de API, o relatório compara só os sistemas locais. `tanque infer claude` continua no repositório; se `results/predictions/claude-*__test.jsonl` for adicionado e pontuado, ele entra nas tabelas sozinho.
- **Gasto total do projeto em API:** US$ 0,00 (`results/spend.jsonl`; `tanque budget status`).
- **Teste reaberto na v2 (D33):** o teste (150) já tinha sido usado uma vez na v1 (M7), com o base e o ft-v1, `max_new_tokens = 256` e sem nova tentativa. Na v2 ele foi reaberto uma vez por sistema, e o base e o ft-v1 foram reexecutados com as regras da v2 (384 tokens, nova tentativa): os números do teste deste relatório substituem os da v1. Os da v1 (EX 12,0% → 76,0%, citados no README como v1) ficam no histórico do git e em `results/predictions/v1/`, fora do `check-metrics`. Todas as decisões da v2 (templates do treino v2, hiperparâmetros, prompt de correção, modelo final) vieram do dev2, nunca do teste.
- **Dev × dev2 × teste:** o dev (50) foi usado na escolha do modelo (M5) e do fine-tuning (M6). O dev2 (D30) é esse dev mais as perguntas da reserva: como o dev2 contém o dev da v1, que já orientou as escolhas da v1, ele pode favorecer os ajustados em alguns pontos. O teste é a comparação limpa.
