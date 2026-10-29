# Notebooks do Kaggle

| Notebook | Kernel no Kaggle | O que faz |
|---|---|---|
| `01_bakeoff.ipynb` | `belasoares/tanque-cheio-bakeoff` (`kaggle/bakeoff/`) | Previsões dos candidatos aprovados no dev (M5) |
| `03_predict.ipynb` | `belasoares/tanque-cheio-predict` (`kaggle/predict/`) | Previsões de um modelo (com adapter opcional) em `dev` ou `test` (M5–M7) |
| `02_finetune.ipynb` | `belasoares/tanque-cheio-finetune` (`kaggle/finetune/`) | QLoRA do modelo base + previsões do adapter no dev (M6) |
| `04_test_predict.ipynb` | `belasoares/tanque-cheio-test-predict` (`kaggle/test-predict/`) | Previsões finais no teste: base e base + adapter fundido, na mesma sessão (M7) |

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

## Teste final (M7): `04_test_predict.ipynb`

Roda os **2 sistemas do relatório final** (D24) no `datasets/test.jsonl`, numa única
sessão e na mesma T4: `base-qwen2.5-coder-1.5b-instruct` e
`ft-qwen2.5-coder-1.5b-instruct` (o mesmo base + o adapter do HF, **fundido** antes de
gerar, para a latência ser justa). Decodificação gulosa, `max_new_tokens = 256`, fp16. O
checkpoint do base e o adapter ficam fixados por commit na 1ª célula (`MODEL_REVISION`,
o mesmo do M6, e `ADAPTER_REVISION`).

Por que um notebook novo, e não o `03_predict`: o 03 roda um modelo num split; aqui os dois
sistemas precisam sair da mesma sessão, com o mesmo hardware, as mesmas versões e o
checkpoint e o adapter fixados. Juntar isso no 03 encheria a célula de parâmetros de
casos especiais, e o teste roda uma vez só.

Saída (`/kaggle/working/`): `base-qwen2.5-coder-1.5b-instruct__test.jsonl`,
`ft-qwen2.5-coder-1.5b-instruct__test.jsonl` (150 linhas cada, formato 5.3) e um
`run-info__<sistema>__test.json` para cada um (`adapter_merged: true` no ft). Tempo
estimado: 15 a 25 min de GPU.

> **Regra 5:** executar no Kaggle exige a aprovação da Isabela. Os passos abaixo (👤) são
> para ela, no PowerShell, na raiz do repositório, com a `main` atualizada
> (`git pull`).

**(a) Publicar o adapter no HF: ✅ feito em 29/10/2026.** O repo
[`belasoares/tanque-cheio-qwen2.5-coder-1.5b-lora`](https://huggingface.co/belasoares/tanque-cheio-qwen2.5-coder-1.5b-lora)
está público (commit `bd8079f…`, o `ADAPTER_REVISION` do notebook). Para referência, ou
para publicar de novo a partir da pasta `kaggle-ft/adapter` baixada no M6:

```powershell
uv run hf auth login        # cole um token com permissão de escrita (huggingface.co/settings/tokens)
uv run hf upload belasoares/tanque-cheio-qwen2.5-coder-1.5b-lora kaggle-ft/adapter . --no-private --commit-message "Adapter LoRA do M6"
```

`--no-private` só vale quando o `hf upload` cria o repo; num repo que já existe, a
visibilidade se muda em *Settings* no HF. Se publicar uma versão nova do adapter, troque
`ADAPTER_REVISION` na 1ª célula do `04_test_predict.ipynb` pelo commit novo (ou `None`).

**(b) Rodar o notebook do teste no Kaggle** (cria o kernel privado
`belasoares/tanque-cheio-test-predict` na primeira vez):

```powershell
# (opcional) ensaio: LIMIT = 3 na 1ª célula, push, conferir, voltar para LIMIT = None
kaggle kernels push -p notebooks/kaggle/test-predict
kaggle kernels status belasoares/tanque-cheio-test-predict   # repita até COMPLETE
```

Pela interface web: importe `notebooks/04_test_predict.ipynb`, **Accelerator → GPU T4**,
**Internet → On**, *Save Version → Save & Run All*. Se o status for `ERROR`, o log vem
junto no passo (c) (arquivo `.log`).

**(c) Baixar, pontuar e commitar as previsões:**

```powershell
kaggle kernels output belasoares/tanque-cheio-test-predict -p kaggle-test -o
Copy-Item kaggle-test/*__test.jsonl, kaggle-test/run-info__*__test.json results/predictions/

# banco a partir do snapshot versionado (se ainda não tiver data/tanque.duckdb)
uv run tanque data build --from-parquet snapshot/coletas-2026-09.parquet

uv run tanque eval score datasets/test.jsonl results/predictions/base-qwen2.5-coder-1.5b-instruct__test.jsonl
uv run tanque eval score datasets/test.jsonl results/predictions/ft-qwen2.5-coder-1.5b-instruct__test.jsonl
uv run tanque eval check-metrics   # o mesmo que o CI confere
uv run tanque eval report          # gera results/report.md (tabelas, base → ft, 10 erros)
uv run pytest -q

git switch -c m7-test-predictions
git add results/predictions/*__test.jsonl results/predictions/run-info__*__test.json results/metrics.json results/report.md
git commit -m "chore(results): previsões do teste e relatório final (M7)"
git push -u origin m7-test-predictions   # e abra o PR
```

Não commite a pasta `kaggle-test/`. Opcional: revisar a classificação dos 10 erros no
`results/report.md` e corrigir à mão em `results/error-annotations.json` (ex.:
`"test-0042": {"type": "join", "note": "passou por postos.uf"}`), depois rodar de novo
`uv run tanque eval report` e commitar os dois arquivos.

**(d) Tornar público o notebook do fine-tuning (R9):** na página do kernel
`belasoares/tanque-cheio-finetune` no Kaggle, *Settings → Sharing → Public*. Use a
interface web: um `kaggle kernels push` com `"is_private": false` criaria uma versão
nova e treinaria tudo de novo (~3,5 h de GPU). Depois, cole o link em
`results/training.md` ("Escolha final"). O `tanque-cheio-test-predict` pode ficar público
do mesmo jeito, se quiser.
