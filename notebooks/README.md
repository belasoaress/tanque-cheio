# Notebooks do Kaggle

| Notebook | Kernel no Kaggle | O que faz |
|---|---|---|
| `01_bakeoff.ipynb` | `belasoares/tanque-cheio-bakeoff` (`kaggle/bakeoff/`) | Previsões dos candidatos aprovados no dev (M5) |
| `03_predict.ipynb` | `belasoares/tanque-cheio-predict` (`kaggle/predict/`) | Previsões de um modelo (com adapter opcional) em `dev` ou `test` (M5–M7) |
| `02_finetune.ipynb` | `belasoares/tanque-cheio-finetune` (`kaggle/finetune/`) | QLoRA do modelo base + previsões do adapter no dev (M6) |

Os notebooks instalam o pacote com
`pip install "tanque @ git+https://github.com/belasoaress/tanque-cheio@<REPO_REF>"`,
clonam o repositório no mesmo ref (para ter `datasets/`, `configs/` e
`data/manifest.json`) e chamam `tanque.infer.local.run_predictions`. Esse código é
testado sem GPU (`tests/test_infer_local.py`). O teste com um modelo minúsculo em CPU
(`tests/test_infer_local_hf.py`) precisa de `uv sync --group infer`.

Cada `kernel-metadata.json` aponta para o notebook com `code_file` relativo
(`../../01_bakeoff.ipynb`) e liga GPU T4 (`machine_shape: NvidiaTeslaT4`) e internet. Os
kernels ficam **privados**: só viram públicos no M6, com aprovação.

> **Regra 5:** executar no Kaggle exige a aprovação da Isabela. Os passos abaixo são para
> ela.

## Antes de rodar

1. Aprovar os candidatos em `results/bakeoff-candidates.md`. A lista está em
   `CANDIDATES`, na primeira célula de `01_bakeoff.ipynb`.
2. Conferir `REPO_REF` na primeira célula: o padrão é `main`. Para uma execução
   reproduzível, troque pelo commit exato.
3. A conta do Kaggle precisa ter o **telefone verificado**, ou o Kaggle não libera GPU
   nem internet.
4. Se o seu usuário do Kaggle **não** for `belasoares`, troque o dono no campo `id` dos
   dois `kernel-metadata.json` (ex.: `seu-usuario/tanque-cheio-bakeoff`).

## Opção 1: Kaggle CLI (Windows / PowerShell)

Na raiz do repositório:

```powershell
# 1. Instalar a CLI (uma vez)
uv tool install kaggle
kaggle --version

# 2. Credenciais (uma vez). Em https://www.kaggle.com/settings/api:
#    - "Generate New Token": salve o token no arquivo access_token, ou
#    - "Create Legacy API Key": baixa um kaggle.json.
New-Item -ItemType Directory -Force "$env:USERPROFILE\.kaggle" | Out-Null
# token novo:
Set-Content -NoNewline "$env:USERPROFILE\.kaggle\access_token" "<cole o token aqui>"
# ou kaggle.json legado:
Move-Item "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"
kaggle kernels list --mine --page-size 5   # confirma a autenticação

# 3. (Opcional) ensaio rápido: LIMIT = 3 na 1ª célula do 01_bakeoff.ipynb, push, conferir,
#    e voltar para LIMIT = None.

# 4. Enviar e executar o bake-off (cria o kernel privado na primeira vez)
kaggle kernels push -p notebooks/kaggle/bakeoff

# 5. Acompanhar até "COMPLETE" (ou "ERROR"); repita de tempos em tempos
kaggle kernels status belasoares/tanque-cheio-bakeoff

# 6. Baixar as previsões e o run-info para results/predictions/
kaggle kernels output belasoares/tanque-cheio-bakeoff -p results/predictions --file-pattern ".*\.jsonl?$" -o

# 7. Métricas de cada candidato e a tabela do bake-off
Get-ChildItem results/predictions/base-*__dev.jsonl | ForEach-Object {
    uv run tanque eval score datasets/dev.jsonl $_.FullName
}
uv run tanque eval bakeoff
```

Se o status for `ERROR`, o log vem junto com
`kaggle kernels output belasoares/tanque-cheio-bakeoff -p kaggle-log -o` (arquivo
`.log`). Não commite a pasta `kaggle-log`.

`03_predict` funciona do mesmo jeito: ajuste a 1ª célula (`MODEL_ID`, `DATASET`...), rode
`kaggle kernels push -p notebooks/kaggle/predict` e baixe com
`kaggle kernels output belasoares/tanque-cheio-predict -p results/predictions --file-pattern ".*\.jsonl?$" -o`.

## Opção 2: pela interface web

1. Em https://www.kaggle.com, clique em **Create → New Notebook**.
2. **File → Import Notebook** e escolha `notebooks/01_bakeoff.ipynb` do seu computador.
3. No painel lateral (**Settings**): **Accelerator → GPU T4 x2** e **Internet → On**.
   Deixe o notebook privado (o padrão).
4. Confira a 1ª célula (`REPO_REF`, `CANDIDATES`, `LIMIT = None`).
5. **Save Version → Save & Run All (Commit)**. A execução roda em segundo plano; dá para
   fechar a aba.
6. Quando a versão terminar, abra-a, vá na aba **Output** e baixe os
   `base-*__dev.jsonl` e o `run-info__bakeoff.json` para `results/predictions/`.
7. Rode o passo 7 da Opção 1 (`tanque eval score` + `tanque eval bakeoff`).

## Saída esperada

- `base-qwen2.5-coder-1.5b-instruct__dev.jsonl`, `base-qwen3-1.7b__dev.jsonl`,
  `base-smollm3-3b__dev.jsonl`: 50 linhas cada, formato 5.3 (`hardware = kaggle-t4`,
  `cost_usd = 0.0`).
- `run-info__bakeoff.json`: commit do repositório, versões de torch/transformers e o
  commit de cada checkpoint no HF.
- Tempo estimado: 20 a 40 min de GPU no total, com download dos modelos incluído.

## Fine-tuning (M6): `02_finetune.ipynb`

O notebook lê `configs/train.toml` (modelo base, QLoRA, LoRA, lr, épocas...), treina o
adapter com `tanque.train.finetune.run_finetune` e gera as previsões do **dev** com o
adapter pela mesma `local.run_predictions` do `03_predict` (sistema `ft-<modelo>`). O
código é testado em CPU com um modelo minúsculo, sem 4-bit:
`uv sync --group infer --group train` e `uv run pytest tests/test_train_smoke.py`.
Hiperparâmetros, estimativa de tempo e o registro de cada rodada: `results/training.md`.

> **Regra 5:** cada execução no Kaggle e a publicação no HF precisam da aprovação da
> Isabela. Os passos abaixo são para ela.

1. **(Opcional, recomendado) Medir o tempo.** Na 1ª célula, `MAX_STEPS = 30` e
   `DEV_LIMIT = 3`; depois push (passo 2). Leva ~10 min. No log, a linha
   "Projeção da rodada completa" diz quantas horas a rodada inteira leva. Se passar de
   6 h, reduza `max_train_examples` (ex.: 4000) ou `epochs = 1` em `configs/train.toml`,
   commite e siga. Volte para `MAX_STEPS = None` e `DEV_LIMIT = None`.
2. **Enviar e executar** (cria o kernel privado na primeira vez):

   ```powershell
   kaggle kernels push -p notebooks/kaggle/finetune
   kaggle kernels status belasoares/tanque-cheio-finetune   # repita até COMPLETE
   ```

3. **Baixar a saída** (o adapter vem junto, numa pasta à parte, fora do git):

   ```powershell
   kaggle kernels output belasoares/tanque-cheio-finetune -p kaggle-ft -o
   Copy-Item kaggle-ft/ft-*__dev.jsonl, kaggle-ft/run-info__ft-*.json results/predictions/
   New-Item -ItemType Directory -Force results/training | Out-Null
   Copy-Item kaggle-ft/train-log__ft-*.json results/training/
   ```

   Se o status for `ERROR`, o log está em `kaggle-ft/*.log`.
4. **EX no dev:**

   ```powershell
   uv run tanque eval score datasets/dev.jsonl results/predictions/ft-qwen2.5-coder-1.5b-instruct__dev.jsonl
   ```

5. **Registrar** a rodada em `results/training.md`: a tabela das curvas (impressa pela
   célula "Curvas de perda"), tempo de GPU (`train_seconds`, `seconds_per_step`,
   `total_seconds` do `train-log`), pico de memória e o EX do passo 4.
6. **O que commitar:** `results/predictions/ft-*__dev.jsonl`,
   `results/predictions/run-info__ft-*__dev.json`, `results/training/train-log__ft-*.json`,
   `results/metrics.json` (atualizado pelo `eval score`) e `results/training.md`. **Não**
   commite `kaggle-ft/` nem o adapter (`kaggle-ft/adapter/`): ele vai para o HF.
7. **Rodadas extras** (no máximo 2, só se o EX subir menos de 5 p.p. sobre o base): mude
   `configs/train.toml` (lr, épocas ou r), commite, repita os passos 2 a 6 e registre a
   rodada nova em `training.md`. Nunca use o teste.
8. **Publicar o adapter no HF** (com aprovação). Crie um token de escrita em
   https://huggingface.co/settings/tokens. No editor do kernel no Kaggle, **Add-ons →
   Secrets → Add Secret**, nome `HF_TOKEN`, e marque-o para este notebook (fica salvo
   para as próximas versões). Na 1ª célula: `HF_REPO = "<usuário-hf>/tanque-cheio-qwen2.5-coder-1.5b-instruct-lora"`
   (o plano sugere o dono `belasoaress`; confirme seu usuário no HF) e
   `PUSH_TO_HUB = True`. Rode de novo (passo 2). O repo nasce **privado**
   (`HF_PRIVATE = True`), com um model card provisório; torná-lo público é um clique em
   *Settings* no HF. Para publicar sem treinar de novo, suba a pasta baixada:
   `uvx hf upload <usuário-hf>/tanque-cheio-...-lora kaggle-ft/adapter . --private`.
9. **Tornar o notebook público** (com aprovação): na página do kernel, *Settings →
   Sharing → Public*, ou `"is_private": false` em `kaggle/finetune/kernel-metadata.json` e
   novo push.
