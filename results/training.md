# Fine-tuning QLoRA (M6)

> Status: **notebook pronto, nenhuma rodada executada.** Executar no Kaggle e publicar no
> HF exigem a aprovação da Isabela (regra 5). Passos: `notebooks/README.md`.

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

- Commit do repositório: _(pendente)_ · versão do kernel no Kaggle: _(pendente)_
- Pares: treino _ / val _ / descartados por tamanho _
- Tempo de GPU: treino _ h (_ s/passo); total da sessão _ h; pico de memória _ GiB
- Perda final: treino _ · validação _

Curvas de perda:

_(pendente)_

EX no dev:

| Sistema | EX | EX fácil | EX médio | EX difícil | SQL válida | p50 (ms) |
|---|---|---|---|---|---|---|
| `base-qwen2.5-coder-1.5b-instruct` | 14,0% | 40,0% | 5,0% | 0,0% | 38,0% | 2.836 |
| `ft-qwen2.5-coder-1.5b-instruct` (rodada 1) | _ | _ | _ | _ | _ | _ |

### Rodada 2 (só se o ganho for < 5 p.p.)

- Mudança em relação à rodada 1: _(pendente)_
- Tempo de GPU, perdas, EX no dev: _(pendente)_

### Rodada 3 (só se a rodada 2 ainda ficar < 5 p.p.; última)

- Mudança: _(pendente)_
- Tempo de GPU, perdas, EX no dev: _(pendente)_

## Escolha final

_(pendente: rodada escolhida, link do adapter no HF e do notebook público no Kaggle)_
