# Guia de revisão dos conjuntos de dev e teste (M3)

Este guia é para a revisão humana, item a item, de `datasets/dev.jsonl` (50
itens) e `datasets/test.jsonl` (150 itens). Os itens foram rascunhados por um
agente e validados por execução no snapshot (`tanque eval verify-gold`). A
revisão decide se cada pergunta e a sua SQL de referência valem como gabarito.

Ordem dos lotes:

| Lote | Arquivo | Itens | Observação |
|---|---|---|---|
| 1 | `datasets/dev.jsonl` | 50 | prioridade: destrava o M5 |
| 2 | `datasets/test.jsonl` | 1–50 | |
| 3 | `datasets/test.jsonl` | 51–100 | |
| 4 | `datasets/test.jsonl` | 101–150 | |

A ferramenta segue a ordem do arquivo e retoma de onde parou, então os lotes
2 a 4 são só sessões sucessivas sobre o mesmo arquivo.

**Isolamento.** O conteúdo do teste não sai de `datasets/`: não copie perguntas
do teste para issues, PRs, prompts de outros agentes nem para o gerador de
treino (M4).

---

## 1. Como usar `scripts/review.py`

```
uv run python scripts/review.py datasets/dev.jsonl
uv run python scripts/review.py datasets/test.jsonl
```

Opções: `--db <caminho>` (padrão: `data/tanque.duckdb`). Funciona no
PowerShell; não usa curses, só `Enter` depois de cada tecla.

Para cada item ainda não revisado, a ferramenta mostra o id, o nível,
`order_matters`, `uses_window`, as tags, a pergunta, a SQL e as 5 primeiras
linhas do resultado (com o total de linhas e o tempo). Se o item tiver algum
problema de formato ou a SQL falhar ou vier vazia, aparece um aviso e a
aprovação fica bloqueada até corrigir.

| Tecla | Ação |
|---|---|
| `a` | aprovar: grava `reviewed: true` |
| `e` | editar: abre o item como JSON no editor (`$VISUAL`/`$EDITOR`; no Windows, o Bloco de Notas). Salve e feche o editor; o item volta à tela, reexecutado, para você decidir |
| `d` | descartar: pede um motivo, remove o item do arquivo e acrescenta `{from, reason, item}` em `datasets/discarded.jsonl` |
| `s` | pular: deixa para depois (continua pendente na próxima sessão) |
| `q` | sair |

O arquivo é regravado de forma atômica depois de cada decisão (nada se perde
se o terminal fechar). Ao abrir de novo, os itens já aprovados são pulados.

Para mudar a SQL, o nível, `order_matters` ou as tags, use `e`. Ao editar, o
item volta a `reviewed: false` até você aprovar a versão nova. Não mude o `id`
nem o `split`.

### Depois de descartar

As cotas são exatas (R5): dev 15/20/15 e teste 50/60/40 (fácil/médio/difícil),
no máximo 8 itens com janela no teste. Cada descarte deixa o arquivo abaixo da
cota e o `tests/test_datasets.py` fica vermelho até a reposição. Os substitutos
vêm de `datasets/candidates_reserve.jsonl` (63 itens já validados, `split:
"reserve"`): escolha um do mesmo nível, mova-o para o arquivo com o próximo id
livre (`dev-0051`, `test-0151`...) e `split` correto, e revise-o. Um agente pode
fazer a reposição a partir do `discarded.jsonl`. Reserva por nível: 24 fáceis,
14 médios, 25 difíceis (2 com janela).

### Auditoria por amostra (M4)

```
uv run python scripts/review.py datasets/train.jsonl --audit 50 --seed 7
```

Sorteia N itens com a seed, pergunta só `a` (aprovado) ou `r` (reprovado) e
acrescenta cada veredito em `results/audit-<nome do arquivo>.jsonl`. Não
altera o dataset. Retoma pulando os ids já auditados e no fim imprime o % de
reprovados.

---

## 2. Checklist por item

Responda "sim" a todas para aprovar.

1. **A pergunta é natural?** Uma pessoa perguntaria assim. Erros leves de
   digitação e falta de acento são intencionais (~10% dos itens, tag `typo`);
   aprove-os se a intenção continuar clara.
2. **A pergunta não tem ambiguidade?** Só existe uma leitura razoável, dadas
   as convenções da seção 3. Se duas SQLs razoáveis dariam resultados
   diferentes, edite a pergunta para fechar a leitura ou descarte.
3. **A SQL responde exatamente à pergunta?** Produto, lugar, período, métrica
   e colunas certos (seção 3). Nada a mais (filtro extra) e nada a menos.
4. **O resultado faz sentido?** Olhe as 5 linhas: valores plausíveis (R$ 3 a
   R$ 10 por litro, R$ 3 a R$ 7 por m³ de GNV), número de linhas coerente
   com a pergunta, sem NULL estranho.
5. **`order_matters` está certo?** `true` só quando a pergunta pede ordem ou
   ranking ("os 5 mais baratos", "do mais caro para o mais barato", "qual o
   mais..."). Séries mensais e listas sem ordem pedida são `false`.
6. **O nível está certo?** Ver seção 4.
7. **`uses_window` está certo?** `true` se e só se a SQL usa função de janela
   (`... OVER (...)`). O teste automatizado também confere isso.
8. **Datas relativas coerentes com D19?** "último mês" = setembro de 2026
   (seção 3.2). Perguntas com "entre março e junho" trazem o ano no texto.
9. **Não é quase duplicata** de outro item do mesmo arquivo nem do outro split.

---

## 3. Convenções das SQLs de referência

### 3.1 Métricas e colunas

- **"Preço médio"** = `AVG(c.valor_venda)` sobre as coletas do recorte (média
  simples das coletas, sem ponderar por posto nem por município).
- **Sem arredondamento na SQL.** O comparador (seção 5.6 do plano) arredonda
  números a 2 casas.
- **"Mais barato/mais caro"** = menor/maior preço médio no recorte pedido.
  "Menor/maior preço registrado" = `MIN`/`MAX(valor_venda)`.
- **Colunas do resultado.** A SQL devolve as colunas que identificam a
  entidade pedida e, quando a pergunta é sobre um valor (preço médio, razão,
  variação, contagem por grupo) ou um ranking por valor, também esse valor.
  Perguntas de pertencimento ("em quais estados...", "quais capitais...")
  sem pedir o valor devolvem só a identificação. Contagem simples devolve um
  número. O comparador exige o mesmo número de colunas, então esta regra
  importa: se discordar dela num item, edite o item.
- **Município** é identificado por `nome` + `uf` (há homônimos, por exemplo
  `VALENCA` na BA e no RJ). O filtro usa sempre os dois. No resultado, dentro
  de uma única UF basta `nome`; quando o resultado cobre várias UFs, `nome,
  uf`. Os nomes são os da ANP, em maiúsculas e sem acento (`SAO CARLOS`).
- **Capital** = `municipios.capital` (D18). Capitais aparecem como `nome` ou
  `nome, uf` pela mesma regra acima.
- **Combustíveis.** "gasolina" ou "gasolina comum" = `'GASOLINA'`;
  "aditivada" só quando dito. "diesel" ou "diesel comum" = `'DIESEL'`;
  "S10" = `'DIESEL S10'`. "etanol" = `'ETANOL'`; "GNV" = `'GNV'` (R$/m³).
- **Bandeira** = `postos.bandeira` (a da coleta mais recente do posto).
  "Bandeira branca"/"sem bandeira" = `'BRANCA'`. Grafia exata do banco:
  `'SABBÁ'`, `'ATEM'' S'`, `'RAIZEN'` (sem acento).
- **Contagem de postos**: `COUNT(*)` em `postos`; quando condicionada a
  coletas (ex. "postos que tiveram GNV"), `COUNT(DISTINCT c.posto_id)`.
- **Bairro**: rankings por bairro excluem `bairro IS NULL`.
- **Postos por nome**: identificados por `razao_social`, agrupando por
  `posto_id` (dois postos podem ter a mesma razão social).

### 3.2 Datas (data de referência = 2026-09-30)

| Expressão | Intervalo |
|---|---|
| "último mês" (D19) | setembro de 2026: `data_coleta >= '2026-09-01' AND < '2026-10-01'` |
| "em 2025" | ano civil de 2025 |
| "em 2026" | o que há de 2026: janeiro a setembro |
| "entre março e junho de AAAA" | 1º de março a 30 de junho de AAAA, inclusive (4 meses) |
| "primeiro semestre de AAAA" | janeiro a junho de AAAA |
| "últimos 12 meses" | outubro de 2025 a setembro de 2026 |
| "setembro de 2025 × setembro de 2026" | os dois meses civis |
| sem período | todo o snapshot (outubro de 2024 a setembro de 2026) |

- O rascunho sempre escreve o ano em "entre março e junho". Se alguma
  pergunta ficar sem ano, a leitura é o ano da data de referência (2026).
- **Mês** = `date_trunc('month', data_coleta)` (sai como data à meia-noite do
  dia 1º; o comparador casa `date` com `timestamp` à meia-noite, mas não com
  texto `'2026-08'`). Trimestre = `quarter(...)`; ano = `year(...)` (inteiros).
- **"Em relação ao mês anterior"**: `LAG` sobre a série mensal, incluindo o
  mês anterior ao período para que o primeiro mês tenha variação.

### 3.3 Regra dos 70% (D21)

- Razão = `AVG(valor_venda)` do `'ETANOL'` ÷ `AVG(valor_venda)` da
  `'GASOLINA'` comum, no mesmo recorte (lugar e período).
- "O etanol compensa" se a razão ≤ 0,7; "não compensa" se > 0,7.
- "Em todos os meses de AAAA" = a razão mensal ≤ 0,7 em cada um dos meses do
  ano com dados (12 em 2025; 9 em 2026).

### 3.4 Variações e diferenças

- Variação % = (novo − antigo) ÷ antigo × 100. Diferença em R$ = novo −
  antigo. "A mais que" = primeiro termo − segundo.
- "Maior alta" = maior variação; "maior queda" = menor variação (mais
  negativa); "menor alta" = menor variação.

### 3.5 Rankings e empates

- `ORDER BY <métrica>` com `LIMIT k`. O rascunho verificou no banco que não
  há empate exato entre a posição k e k+1 nem entre linhas vizinhas do
  resultado; itens com empate foram descartados. Se editar um ranking, rode a
  consulta sem o `LIMIT` e confira.

---

## 4. Níveis (seção 5.2 do plano)

- **Fácil**: uma agregação ou lista simples, com filtros diretos (produto,
  UF, município, região, capital, bandeira) e no máximo um filtro de período
  simples; pode usar o caminho padrão `coletas → postos → municipios`. Sem
  `GROUP BY` de ranking nem `ORDER BY`/`LIMIT`.
- **Médio**: `GROUP BY` por uma dimensão, `ORDER BY`/`LIMIT`, `HAVING`,
  filtros de data, séries mensais, comparações simples entre poucos lugares.
- **Difícil**: comparação entre períodos, CTEs ou subconsultas, razões entre
  produtos (regra dos 70%), "em todos os meses", diferença contra uma média,
  funções de janela. No máximo 20% dos difíceis com janela (≤ 3 no dev, ≤ 8
  no teste).

Na dúvida entre dois níveis, mantenha o do rascunho só se as cotas exigirem;
senão, ajuste e avise (mudar de nível também desequilibra as cotas).

---

## 5. Critérios de descarte

Descarte (`d`) quando, mesmo editando, o item não ficaria bom:

- a pergunta admite duas leituras razoáveis com resultados diferentes;
- a SQL não responde à pergunta e a correção mudaria a pergunta toda;
- o resultado é trivial ou degenerado (um NULL, zero, todas as linhas
  iguais) ou depende de empate;
- a resposta depende de algo fora do banco ou do escopo (GLP, volume de
  vendas, endereço, preço de compra, que é sempre nulo);
- é quase duplicata de outro item (mesmo template com outra UF conta como
  duplicata só se também repetir produto e período);
- a pergunta entrega a SQL (cita nomes de coluna ou valores internos de forma
  artificial).

Escreva o motivo no prompt de descarte: ele fica em `datasets/discarded.jsonl`
e ajuda a escolher o substituto da reserva.
