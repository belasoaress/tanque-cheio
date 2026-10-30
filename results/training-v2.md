# Treino v2 (M15)

> Tudo aqui vem do **dev2** (`datasets/dev2.jsonl`, 113 perguntas, D30). O teste não foi
> aberto nem consultado (regra 0.6): só o código de vazamento o lê, em tempo de execução,
> para tirar do treino as perguntas parecidas com ele.

## Análise de erros do ft-v1 no dev2

Base: `results/predictions/ft-qwen2.5-coder-1.5b-instruct__dev2.jsonl` (M14, gulosa, 384
tokens, sem nova tentativa). Cada item errado foi executado (gold e previsão, banco do
snapshot) e classificado à mão pela causa principal. Para comparação, o mesmo com
`base-qwen2.5-coder-32b-instruct__dev2.jsonl`.

| Sistema | EX | fácil (39) | médio (34) | difícil (40) | erros |
|---|---|---|---|---|---|
| ft-v1 (1,5B ajustado) | 74,3% (84) | 97,4% (38) | 76,5% (26) | 50,0% (20) | 29 |
| 32B sem ajuste (nf4) | 72,6% (82) | 92,3% (36) | 79,4% (27) | 47,5% (19) | 31 |

Dos 29 erros do ft-v1, 1 é fácil, 8 são médios e 20 são difíceis. Em 5 deles a resposta
está certa com colunas a mais (EX tolerante 78,8%).

### Erros por causa

| Causa principal | Erros | Itens (dev2) | O 32B acerta |
|---|---:|---|---|
| Forma / colunas | 7 | 0022, 0024, 0028, 0076, 0096, 0104, 0108 | 0022, 0024, 0028, 0076, 0104 |
| Filtro de valor | 4 | 0004, 0042, 0079, 0095 | 0095 |
| Agregação | 3 | 0078, 0103, 0110 | as 3 |
| Filtro de data | 3 | 0034, 0105, 0109 | 0105 |
| SQL inválida | 3 | 0031, 0044, 0101 | 0031, 0044 |
| Lógica: ranking por posição / por grupo (janela) | 2 | 0048, 0047 | 0048 |
| Lógica: "todos os …" | 2 | 0045, 0102 | — |
| Lógica: acumulado | 1 | 0049 | — |
| Lógica: "tanto … quanto" | 1 | 0050 | 0050 |
| Lógica: região × estado (capital × resto) | 1 | 0112 | 0112 |
| Lógica: comparação entre períodos | 1 | 0094 | — |
| Outros (comparação entre produtos virou regra dos 70%) | 1 | 0111 | — |
| Junção | 0 | — | — |
| Participação % | 0 (*) | — | — |
| **Total** | **29** | | **15** |

(*) A participação percentual (0104) saiu com o valor certo e duas colunas a mais; está
contada em "forma".

### O que cada erro mostra

- **Forma.** O ft-v1 aprendeu bem "colunas = entidade + valor", mas erra em três lugares:
  (a) capitais de uma região: devolve `nome, uf`, e o gold do dev2 usa só `nome` nos 4
  itens desse tipo (0028, 0076, 0096, 0108); (b) "municípios paulistas": o gentílico não
  aparece no treino v1, e ele devolve `nome, uf` (0024) ou troca de estado — "municípios
  gaúchos" virou `m.uf = 'MG'` (0079); (c) "o preço médio do diesel comum e do S10 no
  Centro-Oeste" foi para duas colunas (`AVG(CASE ...)`), porque o template
  `lado_a_lado_produtos` da v1 ensinava isso, e o gold quer uma linha por produto (0022).
  Também a entidade implícita: "Qual a gasolina aditivada mais barata entre as capitais
  do Sul?" voltou só a média, sem a capital (0028).
- **Agregação.** Conta coletas onde a pergunta pede postos: "quantos postos com gasolina
  acima de R$ 7" (0078, `COUNT(*)` em vez de `COUNT(DISTINCT c.posto_id)`) e "postos
  abaixo da média da cidade" (0110, sem a média por posto); a "amplitude do preço médio
  mensal" usou o máximo e o mínimo das coletas (0103). O 32B acerta os três.
- **Filtro de valor.** Some a lista de bandeiras citadas (0042 e 0095: "entre Vibra,
  Ipiranga, Raízen e branca"); o bairro "Tatuapé" foi filtrado com acento e minúsculas,
  e o banco tem `'TATUAPE'` (0004; o 32B também erra).
- **Filtro de data.** "Por trimestre" saiu como `date_trunc('quarter', ...)`, e a
  convenção é `quarter(...)` (0034, o 32B também erra); "comparando a média de junho com
  a de março" usou março a junho inteiro como início (0105); a maior alta mensal "em
  2025" olhou só setembro a dezembro (0109).
- **SQL inválida** (3, todas difíceis ou de `HAVING COUNT`): alias inexistente (0031),
  referência a uma CTE fora do `FROM` (0044), coluna que a CTE não tem (0101, "nunca
  tiveram coleta").
- **Lógica.** Os padrões que o treino v1 não tinha: posição num ranking (0048, sem
  `RANK()`), o item de pico de cada grupo (0047, bandeira com mais postos por região),
  acumulado (0049, sem `SUM() OVER`), "tanto em X quanto em Y" (0050, `JOIN` sem
  `DISTINCT` contou 2.168 em vez de 158), "todos os municípios" e "em todos os meses"
  com limiar em reais (0045, 0102), "compensou em 2026, mas não em 2025" com os períodos
  trocados (0094), capital × resto do estado (0112) e "GNV mais barato que o etanol",
  que virou a regra dos 70% (0111).

### Comparação com o 32B

- **O 32B acerta e o ft-v1 erra (15):** 0022, 0024, 0028, 0031, 0044, 0048, 0050, 0076,
  0078, 0095, 0103, 0104, 0105, 0110, 0112. São quase todos os casos de convenção de
  forma no médio (4), contagem e agregação sobre a série (3), CTEs válidas (2) e os
  padrões novos que um modelo maior compõe sem ter visto (posição, "tanto … quanto",
  capital × resto).
- **O ft-v1 acerta e o 32B erra (17):** sobretudo forma (o 32B põe `uf` ou o valor a
  mais), regra dos 70% e SQL bloqueada ou inválida.
- **Os dois erram (14):** 0004, 0034, 0042, 0045, 0047, 0049, 0079, 0094, 0096, 0101,
  0102, 0108, 0109, 0111 — o núcleo difícil: "todos os", acumulado, pico por grupo, dois
  períodos e convenções que só o gold conhece (trimestre, bairro, capitais da região).

Tipos da heurística de `tanque.eval.errors` (para conferência): ft-v1 = forma 10,
agregação 5, SQL inválida 4, filtro de valor 3, filtro de data 3, lógica 3, junção 1;
32B = forma 9, agregação 7, filtro de data 4, bloqueada 4, SQL inválida 4, filtro de
valor 2, lógica 1.

## Mudanças no treino (D34)

Cada mudança está ligada a um erro acima. Nenhuma pergunta do dev2 entrou no treino: o
filtro de vazamento (token_set_ratio ≥ 85 ou SQL normalizada idêntica contra dev, dev2,
teste e reserva) roda no pipeline, e `tanque gen check-leakage` dá 0.

### Intenções novas (17)

| Intenção | Nível | Padrão | Erros do dev2 |
|---|---|---|---|
| `postos_bairro` | fácil | bairro escrito por uma pessoa, filtrado com a grafia da ANP | 0004 |
| `filtro_contagem` | médio | grupos com mais/menos de N postos ou coletas (entidade + contagem) | 0031, 0079 |
| `serie_trimestral` | médio | por trimestre (`quarter`) e por ano (`year`) | 0034 |
| `postos_condicao_preco` | médio | postos com preço acima/abaixo de R$ X (`COUNT(DISTINCT c.posto_id)`) | 0078 |
| `tanto_quanto` | difícil | dois recortes ao mesmo tempo (`INTERSECT`) | 0050 |
| `todos_os` | difícil | todos os municípios / todos os meses (regra dos 70% ou limiar) | 0045, 0102 |
| `participacao_pct` | difícil | % de postos de uma bandeira ou de coletas de um produto | 0104 |
| `regiao_vs_estado` | difícil | capital × interior; estado × média da região | 0112 |
| `media_da_serie` | difícil | meses acima da média do ano; amplitude das médias mensais; postos × média da cidade | 0044, 0103, 0110 |
| `nunca_coleta` | difícil | municípios/postos sem coleta de um produto (`NOT IN`) | 0101 |
| `variacao_grupos` | difícil | variação entre períodos por região, bandeiras citadas ou produto | 0042, 0105 |
| `regra70_dois_periodos` | difícil | compensou em X mas não em Y; nos dois; em nenhum | 0094 |
| `compara_produtos_grupo` | difícil | onde um produto foi mais barato que outro (sem 70%) | 0111 |
| `razao_por_grupo` | difícil | razão etanol/gasolina por bandeira citada, capital, região, cidade | 0095, 0096 |
| `posicao_ranking` | difícil, janela | posição de uma entidade (`RANK() OVER`) | 0048 |
| `acumulado` | difícil, janela | soma acumulada mês a mês (`SUM() OVER`) | 0049 |
| `pico_por_grupo` | difícil, janela | mês de pico / bandeira com mais postos em cada grupo | 0047, 0108 |

### Reforços em intenções da v1

- **Capitais de uma região só com `nome`** (`ranking_capitais`, `lado_a_lado_produtos`,
  e as intenções novas): o gold do dev2 faz assim nos 4 itens desse tipo; as capitais do
  país inteiro continuam com `nome, uf`, como nos outros 5 itens do dev2. É uma decisão
  tomada no dev2 que diverge da letra do guia de revisão (seção 3.1, "várias UFs →
  `nome, uf`") e fica registrada aqui.
- **"O preço médio de A e de B" num só lugar = uma linha por produto**
  (`lado_a_lado_produtos`, variante de um lugar): `SELECT c.produto, AVG(...) GROUP BY
  c.produto` em vez de duas colunas (0022). As variantes por UF e por capital continuam
  com duas colunas.
- **Gentílicos** ("municípios paulistas", "cidades gaúchas", "nordestinos") em
  `ranking_municipios`, `municipios_acima_limite` e nas intenções novas (0024, 0079).
- **Entidade implícita** em `ranking_capitais`: "Qual a gasolina mais barata entre as
  capitais do Sul?" devolve a capital e o valor (0028).
- O 0109 (janela de meses da maior alta) não ganhou template: `janela_variacao_mensal`
  já cobre o padrão.

### Distribuição nova

Comando (reprodutível; mesma seed da v1): `uv run tanque gen templates --n 13000 --seed
42`. Cada intenção tem o seu próprio sorteador (`seed:intenção`), então as intenções da
v1 que não mudaram geram os mesmos pares de antes, na mesma ordem: dos 8.217 pares da
v1, **7.148 continuam no treino v2** (os outros saem porque a cota por intenção do médio
e do difícil ficou um pouco menor, ou porque o gerador da intenção mudou).

| Nível | v1 (M4) | v2 (M15) | alvo v2 |
|---|---|---|---|
| fácil | 2.810 (34,2%) | 2.937 (28,1%) | 28% |
| médio | 3.307 (40,2%) | 3.787 (36,2%) | 36% |
| difícil | 2.100 (25,6%) | 3.724 (35,6%) | 36% |
| **total** | **8.217** | **10.448** | |
| com janela | 311 (14,8% do difícil) | 632 (17,0% do difícil) | ≤ 20% |

Descartes por vazamento: 2.552 dos 13.000 candidatos (19,6%). Por intenção:
`results/gen-stats.md`.

## Configuração e tempo de GPU

`configs/train-v2.toml` = `configs/train.toml` com os mesmos hiperparâmetros (QLoRA nf4,
LoRA r=16/alpha=32, lr 2e-4 cosine, 2 épocas, batch efetivo 16, seed 42). Mudam só:

- `system_prefix = "ft2"` → sistema `ft2-qwen2.5-coder-1.5b-instruct` (5.3 v2);
- `revision` fixado no commit que a v1 usou (`2e1fd397…`, regra 0.8);
- `max_seq_len = 1280` (era 1152). Com o tokenizer do Qwen2.5 (mesmo commit), o treino v2
  tem pares de 780 a 1.170 tokens (média 924; prompt de 780-842 + SQL de até 356). Passam
  de 1152 só 47 pares, todos de `regra70_dois_periodos`; descartá-los tiraria 39% dessa
  intenção. Com 1280 cabem todos.

**Estimativa.** A v1 levou 12,91 s/passo (976 passos, 3,5 h de treino) numa T4. Na v2:
10.448 pares → 9.926 de treino + 522 de validação → 621 passos por época → **1.242
passos**. Com o mesmo s/passo, corrigido pelo tamanho médio dos pares (924 contra 920
tokens), dá **~4,5 h de treino**, mais ~10 min de previsões no dev2 (com nova tentativa)
e de carga: **~4,7 h**. Fica abaixo de ~6 h, então nada de `max_train_examples` nem
épocas a menos. O ensaio com `MAX_STEPS = 30` (`notebooks/README.md`, M15, passo 2)
confirma a vazão antes da rodada inteira.

## Rodadas

Rodada 1 executada no Kaggle (kernel `belasoares/tanque-cheio-finetune-v2`, 1 T4) em
30/10/2026: 1.242 passos, 9.926 pares de treino (5% para validação), nenhum descartado
por tamanho, 15,4 s/passo, **5,3 h de treino**, pico de 2,5 GiB, perda de validação final
0,0018 (`results/training/train-log__ft2-qwen2.5-coder-1.5b-instruct.json`).

| Rodada | Mudança | EX dev2 | fácil | médio | difícil | +retry | Tempo de GPU |
|---|---|---|---|---|---|---|---|
| ft-v1 (referência) | — | 74,3% | 97,4% | 76,5% | 50,0% | 74,3% | 3,5 h |
| **1** | templates v2 (PR #33) | **96,5%** (IC 92,9–99,1%) | 100,0% | 94,1% | 95,0% | 96,5% | 5,3 h |

**Cuidado na leitura.** Os templates novos foram desenhados olhando os erros do dev2.
O filtro de vazamento impede perguntas quase iguais às do dev2, mas o ganho no dev2 vai
ser otimista: o teste (M17) é a medida limpa.

**Resultado da rodada 1.** O ft-v2 supera o ft-v1 no dev2 por +22,1 p.p. (27 perguntas só
o v2 acerta, 2 só o v1; McNemar p < 0,001), e o 32B sem ajuste por +23,9 p.p. (p < 0,001).
O maior salto é no difícil (50% → 95%). A nova tentativa não muda nada: o v2 gerou SQL
válida em 100% do dev2. A latência fica igual à do v1 (p50 3,9 s). Pela regra do M15, a
rodada 1 basta (ganho acima do v1 e nenhum motivo para uma rodada 2), e o **ft-v2 é o
modelo candidato do teste final (M17)**. A ressalva acima vale com força aqui: parte do
salto vem de templates desenhados para os padrões do dev2.
