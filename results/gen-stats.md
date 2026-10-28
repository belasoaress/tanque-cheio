# Estatísticas do dataset de treino (M4)

Gerado por `tanque gen stats`. Não edite à mão.

- **Pares no treino:** 8217 (`datasets/train.jsonl`; meta do plano: ≥ 3.000)
- **Com função de janela:** 311 (14.8% do nível difícil; teto de 20%)

## Por nível

| nível | pares | % | alvo | dentro de ±5 p.p. |
|---|---:|---:|---:|---|
| facil | 2810 | 34.2% | 35% | sim |
| medio | 3307 | 40.2% | 40% | sim |
| dificil | 2100 | 25.6% | 25% | sim |

## Por intenção

Contagens por intenção: crus = pares que entraram no pipeline; os motivos são os do
pipeline (validação por execução, vazamento contra dev/teste, duplicado interno).

| intenção | nível | pares | crus | mantidos | vazamento | duplicado | outros |
|---|---|---:|---:|---:|---:|---:|---:|
| `media_produto_uf` | facil | 175 | 250 | 70.0% | 75 | 0 | 0 |
| `media_produto_municipio` | facil | 189 | 250 | 75.6% | 61 | 0 | 0 |
| `media_produto_regiao` | facil | 183 | 250 | 73.2% | 67 | 0 | 0 |
| `media_nacional_periodo` | facil | 180 | 250 | 72.0% | 70 | 0 | 0 |
| `extremo_preco` | facil | 211 | 250 | 84.4% | 39 | 0 | 0 |
| `conta_postos_local` | facil | 215 | 250 | 86.0% | 35 | 0 | 0 |
| `conta_coletas` | facil | 220 | 250 | 88.0% | 30 | 0 | 0 |
| `conta_postos_bandeira` | facil | 194 | 250 | 77.6% | 56 | 0 | 0 |
| `media_bandeira` | facil | 160 | 250 | 64.0% | 90 | 0 | 0 |
| `media_capitais` | facil | 203 | 250 | 81.2% | 47 | 0 | 0 |
| `conta_municipios` | facil | 250 | 250 | 100.0% | 0 | 0 | 0 |
| `media_ultimo_mes` | facil | 166 | 250 | 66.4% | 84 | 0 | 0 |
| `amplitude_local` | facil | 215 | 250 | 86.0% | 35 | 0 | 0 |
| `datas_coleta` | facil | 249 | 250 | 99.6% | 1 | 0 | 0 |
| `ranking_uf_preco` | medio | 189 | 250 | 75.6% | 61 | 0 | 0 |
| `ranking_municipios` | medio | 237 | 250 | 94.8% | 13 | 0 | 0 |
| `ranking_bandeiras` | medio | 224 | 250 | 89.6% | 26 | 0 | 0 |
| `media_por_regiao` | medio | 177 | 250 | 70.8% | 73 | 0 | 0 |
| `serie_mensal` | medio | 194 | 250 | 77.6% | 56 | 0 | 0 |
| `media_por_produto` | medio | 212 | 250 | 84.8% | 38 | 0 | 0 |
| `ranking_capitais` | medio | 195 | 250 | 78.0% | 55 | 0 | 0 |
| `compara_bandeiras` | medio | 177 | 250 | 70.8% | 73 | 0 | 0 |
| `postos_por_bandeira` | medio | 180 | 250 | 72.0% | 70 | 0 | 0 |
| `municipios_acima_limite` | medio | 242 | 250 | 96.8% | 8 | 0 | 0 |
| `postos_mais_baratos` | medio | 236 | 250 | 94.4% | 14 | 0 | 0 |
| `coletas_por_mes` | medio | 203 | 250 | 81.2% | 47 | 0 | 0 |
| `ranking_bairros` | medio | 237 | 250 | 94.8% | 13 | 0 | 0 |
| `amplitude_por_uf` | medio | 223 | 250 | 89.2% | 27 | 0 | 0 |
| `presenca_bandeira` | medio | 228 | 250 | 91.2% | 22 | 0 | 0 |
| `lado_a_lado_produtos` | medio | 153 | 250 | 61.2% | 97 | 0 | 0 |
| `regra70_ufs` | dificil | 215 | 263 | 81.7% | 48 | 0 | 0 |
| `regra70_todos_meses` | dificil | 213 | 263 | 81.0% | 50 | 0 | 0 |
| `variacao_periodos` | dificil | 204 | 263 | 77.6% | 59 | 0 | 0 |
| `razao_mensal` | dificil | 220 | 263 | 83.7% | 43 | 0 | 0 |
| `acima_media_nacional` | dificil | 241 | 263 | 91.6% | 22 | 0 | 0 |
| `subiu_no_periodo` | dificil | 244 | 263 | 92.8% | 19 | 0 | 0 |
| `premio_bandeira` | dificil | 200 | 263 | 76.0% | 63 | 0 | 0 |
| `diferenca_produtos_uf` | dificil | 252 | 263 | 95.8% | 11 | 0 | 0 |
| `janela_variacao_mensal` | dificil | 138 | 198 | 69.7% | 60 | 0 | 0 |
| `janela_ranking_por_grupo` | dificil | 173 | 198 | 87.4% | 25 | 0 | 0 |

## Descartes

- Requisições com pares: 0
- Pares crus: 10000
- Válidos após execução: 10000
- No treino final: 8217 (82.2% dos crus)

| motivo | pares | % dos crus |
|---|---:|---:|
| `vazamento` | 1783 | 17.8% |

- Deduplicação interna: pergunta ou SQL normalizada idêntica (D23).
- Regra `resultado_repetido` desligada: a SQL é montada pelo template a partir dos parâmetros sorteados (D23), então não pode ser genérica.
- Vazamento contra dev/teste: token_set_ratio >= 85 ou SQL normalizada idêntica.

### Geração por templates (antes do pipeline)

- Seed: 42; pedidos: 10000; gerados: 10000
- Tentativas recusadas na geração (o candidato nem chega ao pipeline):

| motivo | tentativas |
|---|---:|
| `repetido` | 5549 |
| `empate` | 943 |
| `zero` | 627 |
| `vazio` | 409 |
| `poucas_linhas` | 358 |
| `nulo` | 325 |
| `sinal` | 158 |
| `sem_parametro` | 33 |
| `muitas_linhas` | 29 |

## Custo (ledger, marco M4)

- Origem do treino: templates parametrizados com valores reais do banco (D23), sem chamadas à API.
- Lotes enviados: 0; requisições: 0
- Chamadas cobradas: 0
- Tokens: entrada 0, leitura de cache 0, escrita de cache 0, saída 0
- **Gasto real: US$ 0.0000** (estimado no envio: US$ 0.0000)
- Por par no treino: US$ 0.000000
