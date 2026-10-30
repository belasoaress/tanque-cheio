# Relatório final

> Gerado por `uv run tanque eval report` a partir de `results/metrics.json`, das previsões em `results/predictions/` e de `results/error-annotations.json`. Não edite à mão: o CI confere com `tanque eval report --check`.

EX = execution accuracy (seção 5.6 do plano), a métrica principal; EX tolerante = aceita colunas a mais na previsão (D28); IC 95% = intervalo de confiança bootstrap (D29). Definições na nota de método. Percentuais sobre todos os itens do split. `*` = custo estimado de GPU.

## Tabela principal: teste (150 perguntas)

| Sistema | Split | N | EX | IC 95% EX | EX tolerante | IC 95% tolerante | EX fácil | EX médio | EX difícil | SQL válida | Bloqueada | p50 (ms) | p95 (ms) | US$ / 1.000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `base-qwen2.5-coder-1.5b-instruct` | test | 150 | 12,0% | 7,3–17,3% | 12,7% | 8,0–18,0% | 28,0% | 6,7% | 0,0% | 22,0% | 3,3% | 2.638 | 6.518 | 0,2896* |
| `ft-qwen2.5-coder-1.5b-instruct` | test | 150 | 76,0% | 68,7–82,7% | 83,3% | 77,3–89,3% | 98,0% | 76,7% | 47,5% | 96,0% | 2,0% | 3.726 | 8.172 | 0,3660* |

## Base → ajustado (teste)

`base-qwen2.5-coder-1.5b-instruct` → `ft-qwen2.5-coder-1.5b-instruct`:

| Métrica | Base | Ajustado | Δ |
|---|---|---|---|
| EX | 12,0% | 76,0% | +64,0 p.p. |
| EX tolerante | 12,7% | 83,3% | +70,7 p.p. |
| EX fácil | 28,0% | 98,0% | +70,0 p.p. |
| EX médio | 6,7% | 76,7% | +70,0 p.p. |
| EX difícil | 0,0% | 47,5% | +47,5 p.p. |
| EX tolerante fácil | 30,0% | 98,0% | +68,0 p.p. |
| EX tolerante médio | 6,7% | 88,3% | +81,7 p.p. |
| EX tolerante difícil | 0,0% | 57,5% | +57,5 p.p. |
| SQL válida | 22,0% | 96,0% | +74,0 p.p. |
| Bloqueada | 3,3% | 2,0% | −1,3 p.p. |
| Latência p50 (ms) | 2.638 | 3.726 | +1.089 |
| Latência p95 (ms) | 6.518 | 8.172 | +1.654 |
| US$ / 1.000 | 0,2896* | 0,3660* | +0,0763 |

## Comparações pareadas (teste)

| A | B | N | EX A | EX B | Δ EX (B − A) | Só A | Só B | p (McNemar) |
|---|---|---|---|---|---|---|---|---|
| `base-qwen2.5-coder-1.5b-instruct` | `ft-qwen2.5-coder-1.5b-instruct` | 150 | 12,0% | 76,0% | +64,0 p.p. | 0 | 96 | < 0,001 |

## Erros do modelo ajustado (teste)

Modelo final: `ft-qwen2.5-coder-1.5b-instruct` (o `ft*` sem nova tentativa de maior EX no teste).

### `ft-qwen2.5-coder-1.5b-instruct` (split `test`)

36 erros em 150 perguntas. Tipos (heurística por código, com a classificação manual de `results/error-annotations.json` quando houver):

| Tipo | N | % dos erros |
|---|---|---|
| Bloqueada pelo guardrail | 3 | 8,3% |
| SQL inválida | 3 | 8,3% |
| Coluna/forma do resultado | 17 | 47,2% |
| Agregação | 5 | 13,9% |
| Filtro de data | 2 | 5,6% |
| Filtro de valor | 2 | 5,6% |
| Lógica | 4 | 11,1% |

10 exemplos (um tipo por vez, em rodízio):

**1. `test-0128`** · dificil · Bloqueada pelo guardrail (heurística)

> Quais capitais tiveram o diesel S10 acima da média da própria região no último mês?

Erro: `bloqueada pelo guardrail: SQL não faz parse: Error tokenizing ' >= DATE '2026-09-01' AND data_coleta < DATE '202'`

Referência:

```sql
WITH reg AS (SELECT m.regiao, AVG(c.valor_venda) AS p FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'DIESEL S10' AND c.data_coleta >= DATE '2026-09-01' AND c.data_coleta < DATE '2026-10-01' GROUP BY m.regiao), cap AS (SELECT m.nome, m.uf, m.regiao, AVG(c.valor_venda) AS p FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'DIESEL S10' AND m.capital AND c.data_coleta >= DATE '2026-09-01' AND c.data_coleta < DATE '2026-10-01' GROUP BY m.nome, m.uf, m.regiao) SELECT cap.nome, cap.uf FROM cap JOIN reg ON reg.regiao = cap.regiao WHERE cap.p > reg.p
```

Prevista:

```sql
WITH precos AS (SELECT m.nome, m.uf, AVG(c.valor_venda) AS media FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'DIESEL S10' AND m.capital AND c.data_coleta >= DATE '2026-09-01' AND c.data_coleta < DATE '2026-10-01' GROUP BY m.nome, m.uf), por_regiao AS (SELECT m.nome, m.uf, AVG(p.media) AS media FROM precos p JOIN municipios m ON m.nome = p.nome AND m.uf = p.uf WHERE p.data_coleta >= DATE '2026-09-01' AND p.data_coleta < DATE '2026-10-01' GROUP BY m.nome, m.uf) SELECT nome, uf FROM por_regiao WHERE media > (SELECT media FROM precos WHERE nome = m.nome AND uf = m.uf AND data_coleta >= DATE '2026-09-01' AND data_coleta < DATE '2026
```

**2. `test-0132`** · dificil · SQL inválida (heurística)

> Qual a participação (%) de cada combustível no total de coletas do último mês?

Erro: `BinderException: Binder Error: No function matches the given name and argument types '/(VARCHAR, DECIMAL(4,1))'. You might need to add explicit type casts. Candidate functions: "/"("left" FLOAT, "rig…`

Referência:

```sql
SELECT c.produto, 100.0 * COUNT(*) / (SELECT COUNT(*) FROM coletas c WHERE c.data_coleta >= DATE '2026-09-01' AND c.data_coleta < DATE '2026-10-01') AS pct FROM coletas c WHERE c.data_coleta >= DATE '2026-09-01' AND c.data_coleta < DATE '2026-10-01' GROUP BY c.produto
```

Prevista:

```sql
SELECT c.produto, COUNT(*) AS coletas FROM coletas c WHERE c.data_coleta >= DATE '2026-09-01' AND c.data_coleta < DATE '2026-10-01' GROUP BY c.produto / 100.0 * 100.0 ORDER BY c.produto
```

**3. `test-0057`** · medio · Coluna/forma do resultado (heurística)

> Quais municípios de Santa Catarina tiveram mais de 700 coletas de gasolina comum em 2025?

Referência:

```sql
SELECT m.nome, COUNT(*) AS coletas FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'GASOLINA' AND m.uf = 'SC' AND c.data_coleta >= DATE '2025-01-01' AND c.data_coleta < DATE '2026-01-01' GROUP BY m.nome HAVING COUNT(*) > 700
```

Prevista:

```sql
SELECT m.nome FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'GASOLINA' AND m.uf = 'SC' AND c.data_coleta >= DATE '2025-01-01' AND c.data_coleta < DATE '2026-01-01' GROUP BY m.nome HAVING COUNT(*) > 700
```

**4. `test-0121`** · dificil · Agregação (heurística)

> Por estado, quanto o preço médio do GNV em 2026 (jan a set) ficou acima ou abaixo do preço médio de 2025? Considere só estados com coletas nos dois anos.

Referência:

```sql
SELECT m.uf, AVG(c.valor_venda) FILTER (WHERE c.data_coleta >= DATE '2026-01-01' AND c.data_coleta < DATE '2027-01-01') - AVG(c.valor_venda) FILTER (WHERE c.data_coleta >= DATE '2025-01-01' AND c.data_coleta < DATE '2026-01-01') AS diferenca FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'GNV' GROUP BY m.uf HAVING COUNT(*) FILTER (WHERE c.data_coleta >= DATE '2026-01-01' AND c.data_coleta < DATE '2027-01-01') > 0 AND COUNT(*) FILTER (WHERE c.data_coleta >= DATE '2025-01-01' AND c.data_coleta < DATE '2026-01-01') > 0
```

Prevista:

```sql
WITH inicio AS (SELECT m.uf, AVG(c.valor_venda) AS p FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'GNV' AND c.data_coleta >= DATE '2025-01-01' AND c.data_coleta < DATE '2026-01-01' GROUP BY m.uf), fim AS (SELECT m.uf, AVG(c.valor_venda) AS p FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'GNV' AND c.data_coleta >= DATE '2026-01-01' AND c.data_coleta < DATE '2027-01-01' GROUP BY m.uf) SELECT inicio.uf, CASE WHEN fim.p > inicio.p THEN 'acima' WHEN fim.p < inicio.p THEN 'abaixo' ELSE 'igual' END AS diferenca FROM inicio JOIN fim ON inicio.uf = fim.uf WHERE fim.p IS NOT NULL AND inicio.p
```

**5. `test-0123`** · dificil · Filtro de data (heurística)

> Quanto a gasolina comum de Curitiba variou, em reais, entre agosto e setembro de 2026?

Referência:

```sql
SELECT AVG(c.valor_venda) FILTER (WHERE c.data_coleta >= DATE '2026-09-01' AND c.data_coleta < DATE '2026-10-01') - AVG(c.valor_venda) FILTER (WHERE c.data_coleta >= DATE '2026-08-01' AND c.data_coleta < DATE '2026-09-01') AS variacao FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'GASOLINA' AND m.nome = 'CURITIBA' AND m.uf = 'PR'
```

Prevista:

```sql
WITH a AS (SELECT AVG(c.valor_venda) AS p FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'GASOLINA' AND m.nome = 'CURITIBA' AND m.uf = 'PR' AND c.data_coleta >= DATE '2026-08-01' AND c.data_coleta < DATE '2026-10-01'), b AS (SELECT AVG(c.valor_venda) AS p FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'GASOLINA' AND m.nome = 'CURITIBA' AND m.uf = 'PR' AND c.data_coleta >= DATE '2026-09-01' AND c.data_coleta < DATE '2026-10-01') SELECT b.p - a.p AS diferenca FROM a CROSS JOIN b
```

**6. `test-0124`** · dificil · Filtro de valor (heurística)

> Quais municípios do Rio Grande do Sul tiveram a gasolina comum acima da média da região Sul em 2025?

Referência:

```sql
SELECT m.nome FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'GASOLINA' AND m.uf = 'RS' AND c.data_coleta >= DATE '2025-01-01' AND c.data_coleta < DATE '2026-01-01' GROUP BY m.nome HAVING AVG(c.valor_venda) > (SELECT AVG(c.valor_venda) FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'GASOLINA' AND m.regiao = 'S' AND c.data_coleta >= DATE '2025-01-01' AND c.data_coleta < DATE '2026-01-01')
```

Prevista:

```sql
WITH nacional AS (SELECT AVG(valor_venda) AS media FROM coletas WHERE produto = 'GASOLINA' AND data_coleta >= DATE '2025-01-01' AND data_coleta < DATE '2026-01-01') SELECT m.nome FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'GASOLINA' AND m.uf = 'RS' AND c.data_coleta >= DATE '2025-01-01' AND c.data_coleta < DATE '2026-01-01' GROUP BY m.nome HAVING AVG(c.valor_venda) > (SELECT media FROM nacional) ORDER BY m.nome
```

**7. `test-0024`** · facil · Lógica (heurística)

> Em quais bairros de Florianópolis há postos pesquisados?

Referência:

```sql
SELECT DISTINCT p.bairro FROM postos p JOIN municipios m ON m.municipio_id = p.municipio_id WHERE m.nome = 'FLORIANOPOLIS' AND m.uf = 'SC' AND p.bairro IS NOT NULL
```

Prevista:

```sql
SELECT p.bairro FROM postos p JOIN municipios m ON m.municipio_id = p.municipio_id WHERE m.nome = 'FLORIANOPOLIS' AND m.uf = 'SC'
```

**8. `test-0129`** · dificil · Bloqueada pelo guardrail (heurística)

> Qual a diferença de preço médio do etanol entre a capital e o restante do estado, para cada UF do Sudeste, em 2025?

Erro: `bloqueada pelo guardrail: tabela desconhecida: m`

Referência:

```sql
SELECT m.uf, AVG(c.valor_venda) FILTER (WHERE m.capital) - AVG(c.valor_venda) FILTER (WHERE NOT m.capital) AS diferenca FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'ETANOL' AND m.regiao = 'SE' AND c.data_coleta >= DATE '2025-01-01' AND c.data_coleta < DATE '2026-01-01' GROUP BY m.uf
```

Prevista:

```sql
WITH precos AS (SELECT m.uf, AVG(c.valor_venda) AS preco FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'ETANOL' AND m.capital OR m.regiao = 'SE' AND c.data_coleta >= DATE '2025-01-01' AND c.data_coleta < DATE '2026-01-01' GROUP BY m.uf), por_uf AS (SELECT *, precos.preco AS preco_capital FROM precos JOIN m ON m.uf = precos.uf) SELECT uf, preco_capital - preco AS diferenca FROM por_uf WHERE uf <> 'SP' ORDER BY uf
```

**9. `test-0143`** · dificil · SQL inválida (heurística)

> Qual a diferença entre o preço médio do diesel S10 de cada estado do Sul e o da região, no último mês?

Erro: `BinderException: Binder Error: Referenced table "c" not found! Candidate tables: "por_uf" LINE 2: ...) AS preco FROM coletas WHERE produto = 'DIESEL S10' AND c.data_coleta >= DATE '2026-09-01' AND c.…`

Referência:

```sql
WITH u AS (SELECT m.uf, AVG(c.valor_venda) AS p, COUNT(*) AS n FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'DIESEL S10' AND m.regiao = 'S' AND c.data_coleta >= DATE '2026-09-01' AND c.data_coleta < DATE '2026-10-01' GROUP BY m.uf) SELECT uf, p - SUM(p * n) OVER () / SUM(n) OVER () AS diferenca FROM u
```

Prevista:

```sql
WITH por_uf AS (SELECT m.uf, AVG(c.valor_venda) AS preco FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'DIESEL S10' AND m.regiao = 'S' AND c.data_coleta >= DATE '2026-09-01' AND c.data_coleta < DATE '2026-10-01' GROUP BY m.uf) SELECT uf, preco - brasil AS diferenca FROM por_uf CROSS JOIN (SELECT AVG(valor_venda) AS preco FROM coletas WHERE produto = 'DIESEL S10' AND c.data_coleta >= DATE '2026-09-01' AND c.data_coleta < DATE '2026-10-01') brasil
```

**10. `test-0074`** · medio · Coluna/forma do resultado (heurística)

> Qual o preço médio da gasolina comum nas capitais do Sudeste em 2025, por capital?

Referência:

```sql
SELECT m.nome, AVG(c.valor_venda) AS preco_medio FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'GASOLINA' AND m.capital AND m.regiao = 'SE' AND c.data_coleta >= DATE '2025-01-01' AND c.data_coleta < DATE '2026-01-01' GROUP BY m.nome
```

Prevista:

```sql
SELECT m.nome, m.uf, AVG(c.valor_venda) AS preco_medio FROM coletas c JOIN postos p ON p.posto_id = c.posto_id JOIN municipios m ON m.municipio_id = p.municipio_id WHERE c.produto = 'GASOLINA' AND m.capital AND m.regiao = 'SE' AND c.data_coleta >= DATE '2025-01-01' AND c.data_coleta < DATE '2026-01-01' GROUP BY m.nome, m.uf
```

## Referência: dev (50 perguntas)

Usado na escolha do modelo e do fine-tuning; não é o resultado final.

| Sistema | Split | N | EX | IC 95% EX | EX tolerante | IC 95% tolerante | EX fácil | EX médio | EX difícil | SQL válida | Bloqueada | p50 (ms) | p95 (ms) | US$ / 1.000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `base-qwen2.5-coder-1.5b-instruct` | dev | 50 | 14,0% | 6,0–24,0% | 16,0% | 6,0–26,0% | 40,0% | 5,0% | 0,0% | 38,0% | 2,0% | 2.836 | 5.078 | 0,2837* |
| `base-qwen3-1.7b` | dev | 50 | 8,0% | 2,0–16,0% | 10,0% | 2,0–18,0% | 20,0% | 0,0% | 6,7% | 24,0% | 12,0% | 7.499 | 11.593 | 0,6771* |
| `base-smollm3-3b` | dev | 50 | 8,0% | 2,0–16,0% | 10,0% | 2,0–18,0% | 20,0% | 5,0% | 0,0% | 42,0% | 6,0% | 4.968 | 8.676 | 0,5102* |
| `ft-qwen2.5-coder-1.5b-instruct` | dev | 50 | 74,0% | 62,0–86,0% | 76,0% | 64,0–86,0% | 93,3% | 75,0% | 53,3% | 96,0% | 0,0% | 8.328† | 15.833† | 0,7958* |

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

| Sistema | Split | N | EX | IC 95% EX | EX tolerante | IC 95% tolerante | EX fácil | EX médio | EX difícil | SQL válida | Bloqueada | p50 (ms) | p95 (ms) | US$ / 1.000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `base-qwen2.5-coder-1.5b-instruct` | dev2 | 113 | 11,5% | 6,2–17,7% | 12,4% | 6,2–18,6% | 28,2% | 5,9% | 0,0% | 28,3% | 2,6% | 2.808 | 5.899 | 0,3102* |
| `base-qwen2.5-coder-1.5b-instruct+retry` | dev2 | 113 | 11,5% | 6,2–17,7% | 12,4% | 6,2–18,6% | 28,2% | 5,9% | 0,0% | 31,0% | 2,6% | 4.871 | 11.846 | 0,5740* |
| `base-qwen2.5-coder-7b-instruct` | dev2 | 113 | 31,9% | 23,9–40,7% | 38,9% | 30,1–47,8% | 35,9% | 44,1% | 17,5% | 67,3% | 3,5% | 9.514 | 17.996 | 0,9275* |
| `base-qwen2.5-coder-7b-instruct+retry` | dev2 | 113 | 38,0% | 29,2–46,9% | 45,1% | 36,3–54,0% | 53,8% | 44,1% | 17,5% | 87,6% | 3,5% | 11.128 | 36.473 | 1,3049* |
| `base-qwen2.5-coder-14b-instruct` | dev2 | 113 | 40,7% | 31,9–49,6% | 45,1% | 36,3–54,0% | 53,8% | 50,0% | 20,0% | 84,1% | 0,9% | 17.425 | 41.085 | 1,9038* |
| `base-qwen2.5-coder-14b-instruct+retry` | dev2 | 113 | 46,0% | 37,2–54,9% | 51,3% | 41,6–60,2% | 53,8% | 67,7% | 20,0% | 96,5% | 0,9% | 18.015 | 61.241 | 2,3272* |
| `base-qwen2.5-coder-32b-instruct` | dev2 | 113 | 72,6% | 63,7–80,5% | 76,1% | 68,1–84,1% | 92,3% | 79,4% | 47,5% | 92,9% | 3,5% | 29.128 | 71.973 | 6,2569* |
| `base-qwen2.5-coder-32b-instruct+retry` | dev2 | 113 | 74,3% | 66,4–82,3% | 77,9% | 69,9–85,0% | 92,3% | 82,3% | 50,0% | 96,5% | 3,5% | 30.102 | 86.583 | 7,1498* |
| `ft-qwen2.5-coder-1.5b-instruct` | dev2 | 113 | 74,3% | 66,4–82,3% | 78,8% | 70,8–85,8% | 97,4% | 76,5% | 50,0% | 96,5% | 0,0% | 4.077 | 8.619 | 0,4042* |
| `ft-qwen2.5-coder-1.5b-instruct+retry` | dev2 | 113 | 74,3% | 66,4–82,3% | 78,8% | 70,8–85,8% | 97,4% | 76,5% | 50,0% | 97,4% | 0,0% | 4.101 | 8.652 | 0,4218* |
| `ft2-qwen2.5-coder-1.5b-instruct` | dev2 | 113 | 96,5% | 92,9–99,1% | 97,4% | 93,8–100,0% | 100,0% | 94,1% | 95,0% | 100,0% | 0,0% | 3.929 | 7.833 | 0,3865* |
| `ft2-qwen2.5-coder-1.5b-instruct+retry` | dev2 | 113 | 96,5% | 92,9–99,1% | 97,4% | 93,8–100,0% | 100,0% | 94,1% | 95,0% | 100,0% | 0,0% | 3.929 | 7.833 | 0,3865* |

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
- **Prompt único** (`tanque.prompting.build_prompt`, seção 5.4) para todos os sistemas, pelo chat template de cada modelo; decodificação gulosa, `max_new_tokens = 256`, lote de 1 pergunta.
- **Métrica:** EX = execution accuracy (seção 5.6): a SQL prevista é executada com o guardrail e comparada ao resultado da referência; bloqueio, erro ou falta de previsão contam como erro. É a métrica principal (EX rígida): o número de colunas tem de ser o mesmo da referência.
- **EX tolerante (secundária, D28; definida antes dos resultados da v2):** acerta quando as colunas da referência estão entre as colunas da previsão, que pode ter colunas a mais (até 6 colunas na previsão), em qualquer ordem, com a mesma escolha de colunas para todas as linhas; depois valem as regras da EX rígida (mesmas linhas, arredondamento a 2 casas, ordem quando a pergunta pede). Toda previsão certa na rígida é certa na tolerante. Mede quantos erros são só de colunas a mais (`results_match_relaxed`, seção 5.6 da v2).
- **IC 95% (D29):** bootstrap percentil sobre as perguntas do split, 10.000 reamostragens com reposição, seed 42 (`tanque.eval.stats`). Com 150 perguntas, a margem da EX é de cerca de ±7 p.p.
- **Comparações pareadas (D29):** todo par de sistemas sem nova tentativa pontuados no mesmo split, mais cada sistema × a sua versão `+retry`; pergunta a pergunta, na EX rígida. Só A / Só B = perguntas que só um dos dois acerta; p = teste de McNemar exato (binomial, bilateral) sobre esses pares discordantes. p < 0,05 indica diferença dificilmente explicada pelo acaso.
- **Modelos locais:** `base-*` (sem treino) e `ft-*` (mesmo base + adapter LoRA do M6, fundido no base antes de gerar), na mesma sessão do Kaggle e na mesma GPU, fp16. Hardware das previsões do teste: `kaggle-t4`. A latência é medida por pergunta, do template à decodificação, numa T4.
- **Nova tentativa (D32):** as linhas `<sistema>+retry` vêm da mesma execução que `<sistema>`: quando a primeira SQL é bloqueada ou dá erro, o modelo recebe a mensagem de erro e tenta uma única vez de novo. As duas linhas aparecem juntas, com e sem a nova tentativa, e a comparação pareada mostra o ganho (R21).
- **Custo por 1.000 perguntas (* = estimativa):** latência média × 1.000 ÷ 3.600.000 ms/h × US$ 0,35/h, o preço de NVIDIA T4 (1 GPU), Google Compute Engine sob demanda, us-central1 em https://cloud.google.com/products/compute/gpus-pricing (consultado em 2026-10-29); não inclui a VM (vCPU e memória), então é um piso. No Kaggle a execução não custou nada; as previsões têm `cost_usd = 0`.
- **Sem LLM de fronteira por enquanto (D24):** sem crédito de API, o relatório compara só os 2 sistemas locais. `tanque infer claude` continua no repositório; se `results/predictions/claude-*__test.jsonl` for adicionado e pontuado, ele entra nas tabelas sozinho.
- **Gasto total do projeto em API:** US$ 0,00 (`results/spend.jsonl`; `tanque budget status`).
- **Dev × teste:** o dev (50) foi usado na escolha do modelo (M5) e do fine-tuning (M6); o teste (150) só aqui, uma vez por sistema.
