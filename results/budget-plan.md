# Plano de gasto com a API (M0)

Estimativa da seção 6 do `plan.md`, calculada com os preços oficiais em
`configs/pricing.toml`.

- Fonte: <https://platform.claude.com/docs/en/about-claude/pricing> (destino do
  redirecionamento de `docs.claude.com/en/docs/about-claude/pricing`) e
  <https://platform.claude.com/docs/en/about-claude/models/overview> para os IDs.
- Consultado em: 2026-10-28.
- Teto do projeto: **US$ 10,00** (`configs/settings.toml`, `[budget].budget_usd`).

## Modelos escolhidos

| Papel | Chave em `pricing.toml` | Modelo | ID da API | Por que |
|---|---|---|---|---|
| Geração do treino (M4) | `cheap` | Claude Haiku 5.5 | `claude-haiku-5-5` | Modelo atual mais barato; não está depreciado (aposentadoria não antes de 07/10/2027) |
| Previsões de fronteira (M7) | `frontier` | Claude Fable 5.1 | `claude-fable-5-1` | Modelo mais capaz com disponibilidade geral. O Claude Mythos 5.1 tem o mesmo preço, mas acesso limitado a organizações verificadas |

## Preços usados (US$ por milhão de tokens)

| Modelo | Entrada | Saída | Escrita no cache (5 min) | Leitura do cache | Batch |
|---|---|---|---|---|---|
| Claude Haiku 5.5 (prompt ≤ 100 mil tokens) | 0,10 | 0,50 | 0,125 | 0,01 | −50% |
| Claude Fable 5.1 | 10,00 | 50,00 | 12,50 | 0,25 | −50% |

O desconto de 50% da Batch API vale para entrada e saída e se acumula com os
multiplicadores de cache. Com Batch:

| Modelo | Entrada | Saída | Escrita no cache | Leitura do cache |
|---|---|---|---|---|
| Claude Haiku 5.5 | 0,05 | 0,25 | 0,0625 | 0,005 |
| Claude Fable 5.1 | 5,00 | 25,00 | 6,25 | 0,125 |

## Conta

### M4: geração do treino (Haiku 5.5, Batch)

500 requisições × ~2.000 tokens de entrada × ~1.500 tokens de saída. Suposição
para o cache: ~1.800 tokens de prefixo comum (instruções + schema) e ~200
tokens novos por requisição.

| Cenário | Conta | US$ |
|---|---|---|
| Saída (igual em todos) | 500 × 1.500 = 750.000 × 0,25 / 1M | 0,1875 |
| Otimista: cache funciona (5 escritas, 495 leituras arredondadas para 500) | 100.000 × 0,05 + 900.000 × 0,005 + 9.000 × 0,0625, tudo / 1M, + saída | 0,1976 |
| Sem cache | 1.000.000 × 0,05 / 1M + saída | 0,2375 |
| **Pior caso: toda requisição grava no cache** | 100.000 × 0,05 + 900.000 × 0,0625, tudo / 1M, + saída | **0,2488** |

Em Batch o cache é de melhor esforço (as requisições rodam em paralelo e
podem não se reaproveitar), então o total usa o pior caso.

### M7: previsões de fronteira (Fable 5.1, Batch)

200 perguntas × ~2.000 tokens de entrada × ~150 tokens de saída.

| Componente | Conta | US$ |
|---|---|---|
| Entrada (sem cache) | 200 × 2.000 = 400.000 × 5,00 / 1M | 2,0000 |
| Saída | 200 × 150 = 30.000 × 25,00 / 1M | 0,7500 |
| **Total M7** | | **2,7500** |

Com cache do prefixo comum (200 novos + 1.800 lidos por pergunta) o M7 cairia
para ~US$ 1,00. O total abaixo não conta com isso.

### Total

| Item | US$ |
|---|---|
| M4 (pior caso) | 0,2488 |
| M7 (sem cache) | 2,7500 |
| Subtotal | 2,9988 |
| Margem de retrabalho (+30%) | 0,8996 |
| **Total estimado** | **3,8984** |

**US$ 3,90 de US$ 10,00 (39% do teto).** Folga de ~US$ 6,10. No cenário
otimista (cache funcionando nos dois marcos) o total com margem é ~US$ 1,55.

## Riscos para a estimativa

1. **Raciocínio do Fable 5.1.** O modelo usa raciocínio adaptativo sempre
   ligado, e os tokens de raciocínio são cobrados como saída. Os ~150 tokens de
   saída da seção 6 não incluem isso. Com a margem de 30%, o teto aguenta até
   **~938 tokens de raciocínio por pergunta** no M7; com 1.000 o total iria
   para ~US$ 10,40 e estouraria. Mitigação no M7: definir `effort` baixo,
   medir antes num lote pequeno do dev set e usar `budget.reserve()` com a
   estimativa medida.
2. **Tokenizador.** Os modelos a partir da geração 4.7 usam um tokenizador que
   gera ~30% mais tokens para o mesmo texto. Se os "~2 mil tokens" da seção 6
   foram estimados com outra contagem, o total sobe para ~US$ 5,07, ainda
   abaixo do teto.
3. **Haiku 5.5 acima de 100 mil tokens de prompt** paga 5x mais. Não se aplica
   aqui (prompts de ~2 mil tokens), mas o `budget.py` já cobra o preço maior
   se acontecer.
4. **Aprovação.** Pela regra 5 do plano, cada execução acima de US$ 0,50 exige
   aprovação da Isabela. O M7 (~US$ 2,75) precisa dessa aprovação; o M4 inteiro
   fica abaixo de US$ 0,50.

Se o teto apertar, a ordem do plano vale: reduzir primeiro o número de pares
gerados no M4, nunca o tamanho do teste. Uma alternativa para o M7 é o
Claude Opus 5.5 (Batch: US$ 2 / US$ 10 por milhão), que custaria ~US$ 1,10
sem cache, mas deixaria de ser o modelo de fronteira.
