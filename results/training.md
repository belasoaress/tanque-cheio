# Fine-tuning QLoRA (M6)

> Status: **rodada 1 concluída (29/10/2026): EX no dev de 14,0% → 74,0%.** O ganho passa
> muito dos 5 p.p., então não há rodadas extras. Pendente: publicar o adapter no HF e
> tornar o notebook público (👤, regra 5).

Modelo base: **`Qwen/Qwen2.5-Coder-1.5B-Instruct`** (recomendado no bake-off e confirmado
pela Isabela; `results/bakeoff.md`). Sistema nas previsões: `ft-qwen2.5-coder-1.5b-instruct`.
Notebook: `notebooks/02_finetune.ipynb` (kernel `belasoares/tanque-cheio-finetune`).
Código: `src/tanque/train/` (testado em CPU com um modelo minúsculo,
`tests/test_train_smoke.py`).

**Regra:** só o **dev** (50 perguntas) é usado aqui. Se o EX no dev subir menos de 5 p.p.
sobre o base (14,0%), no máximo **2 rodadas extras** de ajuste (lr, épocas, r). O teste
nunca é aberto neste marco.

## Hiperparâmetros (rodada 1 = `configs/train.toml`)

| Grupo | Valor |
|---|---|
| Quantização | 4-bit nf4, double quant, compute fp16 (bitsandbytes) |
| LoRA | r=16, alpha=32, dropout=0,05, bias none |
| Módulos-alvo | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (atenção + MLP do Qwen2) |
| Otimização | AdamW (`adamw_torch`), lr 2e-4, cosine, warmup 3%, weight decay 0, grad norm 1,0 |
| Batch | 4 por passo × 4 de acumulação = **16** efetivo, 1 GPU T4 |
| Épocas | 2 |
| Sequência | `max_seq_len = 1152` (ver desvio abaixo); pares maiores são descartados, não truncados |
| Perda | só na completion (SQL + `<|im_end|>`), `SFTConfig(completion_only_loss=True)` |
| Precisão | fp16 (a T4 não tem bf16); pesos LoRA em fp32; gradient checkpointing |
| Dados | `datasets/train.jsonl` (8.217 pares, D23); 5% para validação (seed 42) → 7.806 treino / 411 val |
| Seed | 42 (`seed` e `data_seed`) |
| Versões | transformers 5.21.0, peft 0.21.2, trl 1.16.0, accelerate 1.15.0, datasets 5.1.0, bitsandbytes 0.50.2 (torch da imagem do Kaggle) |

**Desvio do plano: `max_seq_len` 1152 em vez de 1024.** Com o tokenizer do Qwen2.5, o
prompt único tem 780 a 836 tokens e a SQL + fim de turno 11 a 285 (média 920 no total,
p95 1.026, máximo 1.104). Com 1024, 421 pares (5,1%, quase todos difíceis) seriam
truncados: o fim da SQL e o token de fim sairiam do treino. 1152 cabe todos.

## Tempo de GPU (estimativa)

- Passos: ceil(7.806 / 16) = 488 por época × 2 = **976 passos**.
- Tokens por passo: 16 × ~920 (mais o padding até a maior sequência de cada lote de 4)
  ≈ 15 a 16 mil.
- Custo por token com LoRA e checkpointing ≈ 6 × 1,5 B FLOPs (forward, recomputação e
  backward das ativações) → ~1,4 × 10¹⁴ FLOPs por passo. A T4 tem 65 TFLOPs fp16 de
  pico; com a desquantização do 4-bit, 10 a 20% de aproveitamento é o realista →
  **7 a 14 s por passo**.
- Treino: 976 × 7–14 s ≈ **1,9 a 3,8 h**. Mais ~10 avaliações da validação (~1 min cada),
  carga do modelo e previsões no dev (50 × ~3 s): **≈ 2,2 a 4,1 h no total**.
- Fica abaixo de ~6 h (sessão do Kaggle: 12 h; cota semanal: ~30 h), então a rodada 1 usa
  os 8.217 pares e 2 épocas.
- **Para medir antes:** rode com `MAX_STEPS = 30` (~5 min). O notebook imprime
  `s/passo` e a projeção da rodada completa. Se passar de 6 h, reduza em
  `train.toml` (ou nos parâmetros do notebook): `max_train_examples = 4000` (sorteio
  estratificado por nível e intenção) ou `epochs = 1`. Os dois juntos dão 238 passos.

## Rodadas

Para cada rodada: cole a tabela impressa pela célula "Curvas de perda" do notebook, os
números do `train-log__ft-<modelo>.json` e o EX de
`uv run tanque eval score datasets/dev.jsonl results/predictions/ft-qwen2.5-coder-1.5b-instruct__dev.jsonl`.

### Rodada 1: `configs/train.toml` como está

- Commit do repositório: `0ccbf83` (antes da reescrita do histórico de 29/10; o conteúdo é o mesmo) · kernel `belasoares/tanque-cheio-finetune`
- Pares: treino 7.806 / val 411 / descartados por tamanho 0; 13,1% dos tokens entram na perda
- Tempo de GPU: treino **3,5 h** (12,9 s/passo, 976 passos); total da sessão 3,55 h; pico de memória 2,4 GiB; 18,5 M parâmetros treináveis
- Perda final: treino 0,0093 (média da rodada) · validação **0,0016**

Curvas de perda (de `results/training/train-log__ft-qwen2.5-coder-1.5b-instruct.json`):

| Passo | 10 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 | 900 | 976 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Treino | 0,338 | — | 0,0055 | 0,0034 | 0,0028 | 0,0012 | 0,0017 | 0,0017 | 0,0007 | 0,0011 | 0,0014 |
| Validação | — | 0,0084 | 0,0041 | 0,0030 | 0,0033 | 0,0024 | 0,0019 | 0,0017 | 0,0017 | 0,0016 | 0,0016 |

A perda de validação cai até o passo ~700 e estabiliza; não há sinal de sobreajuste na
validação. Ela é quase zero porque a validação vem da mesma distribuição de templates do
treino; o número que importa é o EX no dev, escrito à mão e isolado do treino.

EX no dev:

| Sistema | EX | EX fácil | EX médio | EX difícil | SQL válida | p50 (ms) |
|---|---|---|---|---|---|---|
| `base-qwen2.5-coder-1.5b-instruct` | 14,0% | 40,0% | 5,0% | 0,0% | 38,0% | 2.836 |
| `ft-qwen2.5-coder-1.5b-instruct` (rodada 1) | **74,0%** | 93,3% | 75,0% | 53,3% | 96,0% | 8.328 |

Erros restantes no dev (13 de 50):

- **Forma do resultado (5):** colunas a mais ou a menos que o gold (`nome, uf` dentro de
  uma UF; preço sem a capital; formato largo em vez de uma linha por produto; ranking
  sem a coluna pedida) e `date_trunc('quarter')` em vez de `quarter()`.
- **Lógica nas difíceis (5):** ignorou a lista de bandeiras da pergunta, posição no
  ranking sem função de janela, acumulado sem soma, "tanto em … quanto em …" e "todos os
  municípios" interpretados de forma mais simples.
- **SQL inválida (2):** alias inexistente (`b.bandeira`) e CTE referenciada sem estar no
  `FROM`.
- **Valor do banco (1):** bairro com acento (`'Tatuapé'`); no banco ele está em maiúsculas
  e sem acento.

**Latência:** p50 de 8,3 s contra 2,8 s do base. A previsão usa o adapter sem fundir
(`PeftModel`) e o modelo ajustado escreve SQLs completas em vez de respostas curtas e
erradas. Para o M7: fundir o adapter (`merge_and_unload`) antes de medir a latência no
teste.

### Rodadas 2 e 3

Não executadas: a rodada 1 ganhou 60 p.p. no dev, muito acima do limite de 5 p.p.

## Escolha final

Rodada 1. Adapter no HF: `belasoares/tanque-cheio-qwen2.5-coder-1.5b-lora` _(publicação
pendente, 👤)_. Notebook público no Kaggle: _(pendente, 👤)_.
