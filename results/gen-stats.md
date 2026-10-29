# Estatísticas do dataset de treino (M4; v2 no M15)

Gerado por `tanque gen stats`. Não edite à mão.

Treino v2 (M15, D34): as 40 intenções da v1 mais 17 novas, ligadas aos erros do ft-v1 no dev2 (`results/training-v2.md`); distribuição alvo 28/36/36 (a v1 tinha 35/40/25 e 8.217 pares).

- **Comando:** `uv run tanque gen templates --n 13000 --seed 42`
- **Pares no treino:** 10448 (`datasets/train.jsonl`; meta do plano: ≥ 3.000)
- **Com função de janela:** 632 (17.0% do nível difícil; teto de 20%)

## Por nível

| nível | pares | % | alvo | dentro de ±5 p.p. |
|---|---:|---:|---:|---|
| facil | 2937 | 28.1% | 28% | sim |
| medio | 3787 | 36.2% | 36% | sim |
| dificil | 3724 | 35.6% | 36% | sim |

## Por intenção

Contagens por intenção: crus = pares que entraram no pipeline; os motivos são os do
pipeline (validação por execução, vazamento contra dev/teste, duplicado interno).

| intenção | nível | pares | crus | mantidos | vazamento | duplicado | outros |
|---|---|---:|---:|---:|---:|---:|---:|
| `media_produto_uf` | facil | 176 | 251 | 70.1% | 75 | 0 | 0 |
| `media_produto_municipio` | facil | 189 | 251 | 75.3% | 62 | 0 | 0 |
| `media_produto_regiao` | facil | 184 | 251 | 73.3% | 67 | 0 | 0 |
| `media_nacional_periodo` | facil | 181 | 251 | 72.1% | 70 | 0 | 0 |
| `extremo_preco` | facil | 212 | 251 | 84.5% | 39 | 0 | 0 |
| `conta_postos_local` | facil | 216 | 251 | 86.1% | 35 | 0 | 0 |
| `conta_coletas` | facil | 221 | 251 | 88.0% | 30 | 0 | 0 |
| `conta_postos_bandeira` | facil | 195 | 251 | 77.7% | 56 | 0 | 0 |
| `media_bandeira` | facil | 161 | 251 | 64.1% | 90 | 0 | 0 |
| `media_capitais` | facil | 204 | 251 | 81.3% | 47 | 0 | 0 |
| `conta_municipios` | facil | 251 | 251 | 100.0% | 0 | 0 | 0 |
| `media_ultimo_mes` | facil | 167 | 251 | 66.5% | 84 | 0 | 0 |
| `amplitude_local` | facil | 216 | 251 | 86.1% | 35 | 0 | 0 |
| `datas_coleta` | facil | 250 | 251 | 99.6% | 1 | 0 | 0 |
| `postos_bairro` | facil | 114 | 126 | 90.5% | 12 | 0 | 0 |
| `ranking_uf_preco` | medio | 187 | 246 | 76.0% | 59 | 0 | 0 |
| `ranking_municipios` | medio | 213 | 246 | 86.6% | 33 | 0 | 0 |
| `ranking_bandeiras` | medio | 221 | 247 | 89.5% | 26 | 0 | 0 |
| `media_por_regiao` | medio | 174 | 246 | 70.7% | 72 | 0 | 0 |
| `serie_mensal` | medio | 191 | 247 | 77.3% | 56 | 0 | 0 |
| `media_por_produto` | medio | 208 | 246 | 84.6% | 38 | 0 | 0 |
| `ranking_capitais` | medio | 191 | 246 | 77.6% | 55 | 0 | 0 |
| `compara_bandeiras` | medio | 174 | 246 | 70.7% | 72 | 0 | 0 |
| `postos_por_bandeira` | medio | 179 | 247 | 72.5% | 68 | 0 | 0 |
| `municipios_acima_limite` | medio | 229 | 246 | 93.1% | 17 | 0 | 0 |
| `postos_mais_baratos` | medio | 232 | 246 | 94.3% | 14 | 0 | 0 |
| `coletas_por_mes` | medio | 201 | 247 | 81.4% | 46 | 0 | 0 |
| `ranking_bairros` | medio | 234 | 247 | 94.7% | 13 | 0 | 0 |
| `amplitude_por_uf` | medio | 219 | 246 | 89.0% | 27 | 0 | 0 |
| `presenca_bandeira` | medio | 224 | 246 | 91.1% | 22 | 0 | 0 |
| `lado_a_lado_produtos` | medio | 140 | 246 | 56.9% | 106 | 0 | 0 |
| `filtro_contagem` | medio | 189 | 247 | 76.5% | 58 | 0 | 0 |
| `serie_trimestral` | medio | 150 | 246 | 61.0% | 96 | 0 | 0 |
| `postos_condicao_preco` | medio | 231 | 246 | 93.9% | 15 | 0 | 0 |
| `regra70_ufs` | dificil | 179 | 215 | 83.3% | 36 | 0 | 0 |
| `regra70_todos_meses` | dificil | 169 | 215 | 78.6% | 46 | 0 | 0 |
| `variacao_periodos` | dificil | 168 | 215 | 78.1% | 47 | 0 | 0 |
| `razao_mensal` | dificil | 182 | 215 | 84.7% | 33 | 0 | 0 |
| `acima_media_nacional` | dificil | 197 | 215 | 91.6% | 18 | 0 | 0 |
| `subiu_no_periodo` | dificil | 200 | 215 | 93.0% | 15 | 0 | 0 |
| `premio_bandeira` | dificil | 163 | 215 | 75.8% | 52 | 0 | 0 |
| `diferenca_produtos_uf` | dificil | 206 | 215 | 95.8% | 9 | 0 | 0 |
| `janela_variacao_mensal` | dificil | 112 | 162 | 69.1% | 50 | 0 | 0 |
| `janela_ranking_por_grupo` | dificil | 141 | 162 | 87.0% | 21 | 0 | 0 |
| `tanto_quanto` | dificil | 195 | 215 | 90.7% | 20 | 0 | 0 |
| `todos_os` | dificil | 167 | 215 | 77.7% | 48 | 0 | 0 |
| `participacao_pct` | dificil | 184 | 215 | 85.6% | 31 | 0 | 0 |
| `regiao_vs_estado` | dificil | 158 | 215 | 73.5% | 57 | 0 | 0 |
| `media_da_serie` | dificil | 163 | 215 | 75.8% | 52 | 0 | 0 |
| `nunca_coleta` | dificil | 172 | 215 | 80.0% | 43 | 0 | 0 |
| `variacao_grupos` | dificil | 160 | 215 | 74.4% | 55 | 0 | 0 |
| `regra70_dois_periodos` | dificil | 119 | 215 | 55.3% | 96 | 0 | 0 |
| `compara_produtos_grupo` | dificil | 196 | 215 | 91.2% | 19 | 0 | 0 |
| `razao_por_grupo` | dificil | 114 | 215 | 53.0% | 101 | 0 | 0 |
| `posicao_ranking` | dificil | 105 | 162 | 64.8% | 57 | 0 | 0 |
| `acumulado` | dificil | 144 | 162 | 88.9% | 18 | 0 | 0 |
| `pico_por_grupo` | dificil | 130 | 162 | 80.2% | 32 | 0 | 0 |

## Descartes

- Requisições com pares: 0
- Pares crus: 13000
- Válidos após execução: 13000
- No treino final: 10448 (80.4% dos crus)

| motivo | pares | % dos crus |
|---|---:|---:|
| `vazamento` | 2552 | 19.6% |

- Deduplicação interna: pergunta ou SQL normalizada idêntica (D23).
- Regra `resultado_repetido` desligada: a SQL é montada pelo template a partir dos parâmetros sorteados (D23), então não pode ser genérica.
- Vazamento contra dev/teste: token_set_ratio >= 85 ou SQL normalizada idêntica.

### Geração por templates (antes do pipeline)

- Seed: 42; pedidos: 13000; gerados: 13000
- Tentativas recusadas na geração (o candidato nem chega ao pipeline):

| motivo | tentativas |
|---|---:|
| `repetido` | 6154 |
| `empate` | 948 |
| `vazio` | 771 |
| `zero` | 768 |
| `poucas_linhas` | 374 |
| `nulo` | 339 |
| `sinal` | 148 |
| `sem_parametro` | 65 |
| `muitas_linhas` | 36 |

## Custo (ledger, marco M4)

- Origem do treino: templates parametrizados com valores reais do banco (D23), sem chamadas à API.
- Lotes enviados: 0; requisições: 0
- Chamadas cobradas: 0
- Tokens: entrada 0, leitura de cache 0, escrita de cache 0, saída 0
- **Gasto real: US$ 0.0000** (estimado no envio: US$ 0.0000)
- Por par no treino: US$ 0.000000
