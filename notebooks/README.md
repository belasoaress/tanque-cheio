# Notebooks do Kaggle

| Notebook | Kernel no Kaggle | O que faz |
|---|---|---|
| `01_bakeoff.ipynb` | `belasoares/tanque-cheio-bakeoff` (`kaggle/bakeoff/`) | Previsões dos candidatos aprovados no dev (M5) |
| `03_predict.ipynb` | `belasoares/tanque-cheio-predict` (`kaggle/predict/`) | Previsões de um modelo (com adapter opcional) em `dev` ou `test` (M5–M7) |
| `02_finetune.ipynb` | `belasoares/tanque-cheio-finetune` (`kaggle/finetune/`) | QLoRA do modelo base + previsões do adapter no dev (M6) |
| `04_test_predict.ipynb` | `belasoares/tanque-cheio-test-predict` (`kaggle/test-predict/`) | Previsões finais no teste: base e base + adapter fundido, na mesma sessão (M7) |
| `05_large_baselines.ipynb` | `belasoares/tanque-cheio-large` (`kaggle/large/`) | 1,5B (base e ft-v1), 7B, 14B e 32B no dev2, com autocorreção, em 2 T4 (M14) |
| `06_final_v2.ipynb` | `belasoares/tanque-cheio-final-v2a` (`kaggle/final-v2a/`) e `belasoares/tanque-cheio-final-v2b` (`kaggle/final-v2b/`) | Teste final v2: (a) 1,5B base, ft-v1, ft-v2, 7B e 14B numa T4; (b) 32B em 2 T4; com autocorreção (M17) |

Os notebooks instalam o pacote com
`pip install "tanque @ git+https://github.com/belasoaress/tanque-cheio@<REPO_REF>"`,
clonam o repositório no mesmo ref (para ter `datasets/`, `configs/` e
`data/manifest.json`) e chamam `tanque.infer.local.run_predictions`. Esse código é
testado sem GPU (`tests/test_infer_local.py`). O teste com um modelo minúsculo em CPU
(`tests/test_infer_local_hf.py`) precisa de `uv sync --group infer`.

Cada `kernel-metadata.json` aponta para o notebook com `code_file` relativo
(`../../01_bakeoff.ipynb`) e liga GPU (`machine_shape: NvidiaTeslaT4`, que na Kaggle CLI é
o acelerador "GPU T4 ×2": a sessão recebe 2 T4) e internet. Os
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

> Desde o M15 a 1ª célula vem com o padrão do treino v2 (`CONFIG =
> "configs/train-v2.toml"`, dev2, retry); para refazer a v1, veja a seção "Treino v2".

O notebook lê `configs/train.toml` (modelo base, QLoRA, LoRA, lr, épocas...), treina o
adapter com `tanque.train.finetune.run_finetune` e gera as previsões do **dev** com o
adapter pela mesma `local.run_predictions` do `03_predict` (sistema `ft-<modelo>`). O
código é testado em CPU com um modelo minúsculo, sem 4-bit:
`uv sync --group infer --group train` e `uv run pytest tests/test_train_smoke.py`.
Hiperparâmetros, estimativa de tempo e o registro de cada rodada: `results/training.md`.

> **Regra 5:** cada execução no Kaggle e a publicação no HF precisam da aprovação da
> Isabela. Os passos abaixo são para ela.

1. **(Opcional, recomendado) Medir o tempo.** Na 1ª célula, `MAX_STEPS = 30` e
   `EVAL_LIMIT = 3`; depois push (passo 2). Leva ~10 min. No log, a linha
   "Projeção da rodada completa" diz quantas horas a rodada inteira leva. Se passar de
   6 h, reduza `max_train_examples` (ex.: 4000) ou `epochs = 1` em `configs/train.toml`,
   commite e siga. Volte para `MAX_STEPS = None` e `EVAL_LIMIT = None`.
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

## Modelos maiores (M14): `05_large_baselines.ipynb`

Roda, numa sessão com **2 GPUs T4**, os 5 sistemas no `datasets/dev2.jsonl` (113
perguntas), nesta ordem: `base-qwen2.5-coder-1.5b-instruct` (fp16, 1 GPU),
`ft-qwen2.5-coder-1.5b-instruct` (adapter v1 do HF, fundido, 1 GPU),
`base-qwen2.5-coder-7b-instruct` e `base-qwen2.5-coder-14b-instruct` (nf4, 1 GPU) e, por
último, `base-qwen2.5-coder-32b-instruct` (nf4, 2 GPUs). Gulosa, `max_new_tokens = 384`,
autocorreção ligada (`RETRY = True`): cada sistema grava `<sistema>__dev2.jsonl`,
`<sistema>+retry__dev2.jsonl` e um `run-info__...json` para cada. Checkpoints fixados por
commit na 1ª célula (`MODELS`); configuração, commits e estimativa de tempo em
`results/large-baselines.md`.

- **Tempo estimado:** ~3 h de GPU (pessimista: ~5 h). O 32B tem limite próprio de
  `TIME_BUDGET_32B_H = 3` h; se estourar, faltar memória, faltar disco ou a sessão tiver
  1 GPU só, ele para ou é pulado e o run-info registra (`stopped_reason`, `skipped`,
  `skip_reason`), sem derrubar os outros (D37).
- **Checkpoint por item:** cada previsão é gravada assim que sai. Se a sessão cair, rode
  de novo com a saída da versão anterior como input (abaixo, "Se parar no meio").
- **Dividir em 2 execuções** (se o ensaio mostrar que não cabe numa sessão): `PART = "a"`
  (1,5B, ft, 7B, 14B), push, baixar; depois `PART = "b"` (32B), push, baixar.

### Como escolher "GPU T4 x2"

- **Pela CLI:** o `kernel-metadata.json` já tem `"machine_shape": "NvidiaTeslaT4"`, que
  na documentação da Kaggle CLI é exatamente **"GPU T4 ×2"** (o acelerador padrão de
  GPU). Não há um valor separado para 2 T4. Para deixar explícito no push:
  `kaggle kernels push -p notebooks/kaggle/large --accelerator NvidiaTeslaT4` (se a sua
  versão da CLI não conhecer `--accelerator`, rode `uv tool upgrade kaggle` ou omita a
  opção: o `machine_shape` basta).
- **Pela interface web:** no editor do notebook, painel lateral **Settings →
  Accelerator → GPU T4 x2** (não "GPU P100" nem "GPU T4"), **Internet → On**. Depois
  *Save Version → Save & Run All (Commit)*.
- **Para conferir:** a célula de checagem imprime `GPUs visíveis: 2 ['Tesla T4', 'Tesla
  T4']`. Com 1 só, ela avisa e o 32B é pulado.

### Passos 👤 (PowerShell, na raiz do repositório)

> **Regra 5:** executar no Kaggle exige a aprovação da Isabela.

```powershell
# 1. main atualizada (com o M14 já no merge)
git switch main
git pull

# 2. (opcional, recomendado) ensaio de ~15 min: LIMIT = 3 na 1ª célula do
#    notebooks/05_large_baselines.ipynb, push, conferir o log e os 10 JSONL, e voltar
#    para LIMIT = None (sem commitar o LIMIT = 3).

# 3. enviar e executar (cria o kernel privado belasoares/tanque-cheio-large na 1ª vez)
kaggle kernels push -p notebooks/kaggle/large --accelerator NvidiaTeslaT4

# 4. acompanhar até COMPLETE (ou ERROR); repita de tempos em tempos (~3 h)
kaggle kernels status belasoares/tanque-cheio-large

# 5. baixar a saída (JSONL, run-info e o log) numa pasta fora do git
kaggle kernels output belasoares/tanque-cheio-large -p kaggle-large -o
Get-ChildItem kaggle-large

# 6. copiar as previsões e os run-info do dev2 para results/predictions/
Copy-Item kaggle-large/*__dev2.jsonl, kaggle-large/run-info__*__dev2.json results/predictions/

# 7. branch, commit e push (sem a pasta kaggle-large/)
git switch -c m14-results
git add results/predictions/*__dev2.jsonl results/predictions/run-info__*__dev2.json
git commit -m "chore(results): previsões dos modelos maiores no dev2 (M14)"
git push -u origin m14-results
```

Depois do push, abra o PR (ou avise o agente): a pontuação e a tabela de
`results/large-baselines.md` saem do comando de relatório do M12, feitas pelo agente.
**Não** rode `tanque eval score` nos arquivos do dev2 por conta própria: o
`results/metrics.json` é recalculado e conferido pelo CI. Se o status for `ERROR`, o log
está em `kaggle-large/*.log`. Se você dividiu em `PART = "a"` e `"b"`, faça os passos 3 a
6 para cada parte antes do 7 (o `kernels output` baixa só a última versão).

**Se parar no meio** (sessão caiu, limite de 12 h): na página do kernel, abra o editor,
**Add Input → Notebook Output Files → `tanque-cheio-large`** (a versão anterior) e, na 1ª
célula, `RESUME_DIR = "/kaggle/input/tanque-cheio-large"` (confira o caminho no painel
*Input*). *Save & Run All*: os itens já prontos são pulados, e só o que falta roda.
Volte depois para `RESUME_DIR = None`.

## Treino v2 (M15): `02_finetune.ipynb` com `configs/train-v2.toml`

O mesmo notebook do M6, agora com padrão v2 na 1ª célula: `CONFIG =
"configs/train-v2.toml"` (mesmos hiperparâmetros da v1, sistema
`ft2-qwen2.5-coder-1.5b-instruct`), treino com o `datasets/train.jsonl` v2 (templates
novos para os erros do ft-v1 no dev2; ver `results/training-v2.md`) e, depois do treino,
previsões no **dev2** com `MAX_NEW_TOKENS = 384` (D31) e `RETRY = True` (D32). O banco é
montado do snapshot dentro do notebook (~6 s), para a nova tentativa executar as SQLs.
Defesas do M6 mantidas: `CUDA_VISIBLE_DEVICES = "0"` antes do torch, `torchao`
desinstalado e `is_torchao_available()` checado.

**Kernel novo:** `belasoares/tanque-cheio-finetune-v2`
(`notebooks/kaggle/finetune-v2/`), para não criar uma versão nova do notebook do v1
(`belasoares/tanque-cheio-finetune`), que fica como está. **Não** rode
`kaggle kernels push -p notebooks/kaggle/finetune`: com o padrão novo da 1ª célula, ele
treinaria o v2 por cima do kernel do v1.

Saída (`/kaggle/working/`): `adapter/`, `train-log__ft2-qwen2.5-coder-1.5b-instruct.json`,
`ft2-qwen2.5-coder-1.5b-instruct__dev2.jsonl`,
`ft2-qwen2.5-coder-1.5b-instruct+retry__dev2.jsonl` e os dois `run-info__...json`.
Tempo estimado: ~5 h de GPU (treino) + ~10 min (previsões); a conta está em
`results/training-v2.md`.

### Passos 👤 (PowerShell, na raiz do repositório)

> **Regra 5:** executar no Kaggle e publicar no HF exigem a aprovação da Isabela.

```powershell
# 1. main atualizada (com o M15 já no merge)
git switch main
git pull

# 2. (opcional, recomendado) ensaio de ~15 min: MAX_STEPS = 30 e EVAL_LIMIT = 3 na 1ª
#    célula do notebooks/02_finetune.ipynb, push (passo 3), conferir no log a linha
#    "Projeção da rodada completa", e voltar para None (sem commitar o ensaio).

# 3. enviar e executar (cria o kernel privado belasoares/tanque-cheio-finetune-v2 na 1ª vez)
kaggle kernels push -p notebooks/kaggle/finetune-v2

# 4. acompanhar até COMPLETE (ou ERROR); repita de tempos em tempos (~5 h)
kaggle kernels status belasoares/tanque-cheio-finetune-v2

# 5. baixar a saída (o adapter vem junto) numa pasta fora do git
kaggle kernels output belasoares/tanque-cheio-finetune-v2 -p kaggle-ft2 -o
Get-ChildItem kaggle-ft2

# 6. copiar as previsões, os run-info e o log de treino para results/
Copy-Item kaggle-ft2/ft2-*__dev2.jsonl, kaggle-ft2/run-info__ft2-*__dev2.json results/predictions/
New-Item -ItemType Directory -Force results/training | Out-Null
Copy-Item kaggle-ft2/train-log__ft2-*.json results/training/

# 7. branch, commit e push (sem a pasta kaggle-ft2/ e sem o adapter)
git switch -c m15-run1
git add results/predictions/ft2-*__dev2.jsonl results/predictions/run-info__ft2-*__dev2.json results/training/train-log__ft2-*.json
git commit -m "chore(results): previsões do ft-v2 no dev2 e log do treino (M15, rodada 1)"
git push -u origin m15-run1
```

Depois do push, abra o PR (ou avise o agente): a pontuação no dev2, a comparação com o
ft-v1 (McNemar) e o registro da rodada em `results/training-v2.md` são feitos pelo
agente; `results/metrics.json` é conferido pelo CI. Se o status for `ERROR`, o log está
em `kaggle-ft2/*.log`.

**8. Publicar o adapter v2 no HF — só depois da avaliação no dev2** (e só se o ft-v2
superar o ft-v1; senão o ft-v1 continua sendo o modelo final, regra do M15):

```powershell
uv run hf auth whoami   # deve mostrar belasoares (se não: uv run hf auth login)
uv run hf upload belasoares/tanque-cheio-qwen2.5-coder-1.5b-lora-v2 kaggle-ft2/adapter .
```

O `hf upload` cria o repo se ele não existir. Anote o commit que ele imprime: é o
`ADAPTER_REVISION` do ft-v2 no teste final (M17).

## Teste final v2 (M17): `06_final_v2.ipynb`

Roda os **6 sistemas da v2** no `datasets/test.jsonl` (150 perguntas), com as regras da
v2 para todos: prompt único, gulosa, `max_new_tokens = 384` (D31) e autocorreção com 1
nova tentativa (D32). Cada sistema grava `<sistema>__test.jsonl`,
`<sistema>+retry__test.jsonl` e um `run-info__...json` para cada. O banco é montado do
snapshot dentro do notebook (~6 s). São **dois kernels do mesmo notebook**, escolhidos por
`PART` na 1ª célula (D27, D37):

| Kernel | `PART` | GPUs | Sistemas |
|---|---|---|---|
| `belasoares/tanque-cheio-final-v2a` | `"a"` (padrão) | 1 T4 (`CUDA_VISIBLE_DEVICES = "0"` antes do torch) | `base-qwen2.5-coder-1.5b-instruct` (fp16), `ft-qwen2.5-coder-1.5b-instruct` (adapter v1 `bd8079f…`, fundido), `ft2-qwen2.5-coder-1.5b-instruct` (adapter v2, fundido), `base-qwen2.5-coder-7b-instruct` e `base-qwen2.5-coder-14b-instruct` (nf4) |
| `belasoares/tanque-cheio-final-v2b` | `"b"` | 2 T4 (`device_map="auto"`) | `base-qwen2.5-coder-32b-instruct` (nf4), com `TIME_BUDGET_32B_H = 4` e checkpoint por item |

Os checkpoints são os mesmos commits do M14. O teste roda **uma vez por sistema** (regra
0.6, D33): o base e o ft-v1 já tinham passado por ele na v1 (256 tokens, sem nova
tentativa); as previsões da v1 foram movidas para `results/predictions/v1/` (fora do
`check-metrics`), e as da v2 entram com os mesmos nomes em `results/predictions/`.

**O commit do adapter v2.** A 1ª célula já traz
`FT2_ADAPTER_REVISION = "beeb19b5b4331ff048c6be0f03b9d53476d509ab"`: o commit do repo
público `belasoares/tanque-cheio-qwen2.5-coder-1.5b-lora-v2` no HF (publicado em
30/10/2026; adapter LoRA r=16, alpha=32 sobre o Qwen2.5-Coder-1.5B-Instruct). O ft-v2
venceu o ft-v1 no dev2 (96,5% contra 74,3%, `results/training-v2.md`) e é o candidato do
teste final. Se um dia publicar outra versão do adapter, troque esse valor na 1ª célula
pelo commit novo (40 caracteres hexadecimais, aba *Files and versions* do repo). Sem um
commit válido (ex.: um placeholder como `PREENCHER_COM_O_COMMIT_DO_ADAPTER_V2`), o kernel
(a) para na 2ª célula, antes de instalar qualquer coisa, com a mensagem
`FT2_ADAPTER_REVISION não foi preenchido`. O kernel (b) não usa o adapter.

### Estimativa de GPU (150 perguntas)

A partir dos tempos medidos no M14 (run-info dos `*__dev2.json`: tempo total, carga e 113
perguntas, já com as novas tentativas), escalados para 150 perguntas. O ft-v2 é estimado
igual ao ft-v1 (mesmo modelo e adapter do mesmo tamanho).

| Kernel | Sistema | s/pergunta no M14 (com retry) | Carga | Total para 150 |
|---|---|---|---|---|
| (a) | base 1,5B (fp16) | 5,95 | 24 s | ~15 min |
| (a) | ft-v1 1,5B (fp16) | 4,40 | 27 s | ~12 min |
| (a) | ft-v2 1,5B (fp16) | ~4,4 (estimado) | ~30 s | ~12 min |
| (a) | base 7B (nf4) | 13,5 | 1,9 min | ~36 min |
| (a) | base 14B (nf4) | 24,0 | 4,1 min | ~64 min |
| (a) | instalação, clone e banco | | | ~5 min |
| **(a)** | | | | **~2,4 h** (pessimista ~3,1 h) |
| (b) | base 32B (nf4, 2 T4) | 36,8 | 9,6 min | ~102 min |
| (b) | instalação, clone e banco | | | ~5 min |
| **(b)** | | | | **~1,8 h** (pessimista ~2,4 h) |
| **Total** | | | | **~4,2 h de cota** (pessimista ~5,5 h) |

Cabe na faixa de 4 a 6 h do plano (seção 6) e no limite de 12 h por sessão. Os dois
kernels podem rodar ao mesmo tempo, se a conta tiver 2 sessões de GPU livres; senão, um
depois do outro. O `TIME_BUDGET_32B_H = 4` dá folga de ~2× sobre a estimativa do 32B; se
estourar, ele para com o que já fez e o run-info registra (`stopped_reason`).

### Passos 👤 (PowerShell, na raiz do repositório)

> **Regra 5:** executar no Kaggle exige a aprovação da Isabela.

```powershell
# 1. main atualizada (com o M17 já no merge)
git switch main
git pull

# 2. conferir a 1ª célula: o commit do adapter v2 já vem preenchido, e PART = "a"
$nb = (Resolve-Path notebooks/06_final_v2.ipynb).Path
Select-String -Path $nb -Pattern 'FT2_ADAPTER_REVISION = |PART = '
#    deve mostrar FT2_ADAPTER_REVISION = \"beeb19b5b4331ff048c6be0f03b9d53476d509ab\"

# 3. (opcional, recomendado) ensaio de ~15 min: LIMIT = 3 na 1ª célula, push do (a),
#    conferir o log e os 10 JSONL, e voltar para LIMIT = None (sem commitar o ensaio).

# 4. kernel (a): PART = "a" (o padrão); cria o kernel privado na 1ª vez
kaggle kernels push -p notebooks/kaggle/final-v2a --accelerator NvidiaTeslaT4

# 5. kernel (b): troca PART para "b", push com GPU T4 x2, e volta PART para "a"
$text = [IO.File]::ReadAllText($nb)
[IO.File]::WriteAllText($nb, $text.Replace('PART = \"a\"', 'PART = \"b\"'))
Select-String -Path $nb -Pattern 'PART = '      # deve mostrar PART = \"b\"
kaggle kernels push -p notebooks/kaggle/final-v2b --accelerator NvidiaTeslaT4
[IO.File]::WriteAllText($nb, $text)             # volta para PART = "a"

# 6. acompanhar até COMPLETE (ou ERROR); repita de tempos em tempos (~2,4 h e ~1,8 h)
kaggle kernels status belasoares/tanque-cheio-final-v2a
kaggle kernels status belasoares/tanque-cheio-final-v2b

# 7. baixar as saídas (JSONL, run-info e o log) em pastas fora do git
kaggle kernels output belasoares/tanque-cheio-final-v2a -p kaggle-final-a -o
kaggle kernels output belasoares/tanque-cheio-final-v2b -p kaggle-final-b -o
Get-ChildItem kaggle-final-a, kaggle-final-b

# 8. copiar as previsões e os run-info do teste para results/predictions/
Copy-Item kaggle-final-a/*__test.jsonl, kaggle-final-a/run-info__*__test.json results/predictions/
Copy-Item kaggle-final-b/*__test.jsonl, kaggle-final-b/run-info__*__test.json results/predictions/

# 9. branch, commit e push (o notebook volta com PART = "a"; não precisa ir no commit)
git switch -c m17-results
git add results/predictions/*__test.jsonl results/predictions/run-info__*__test.json
git commit -m "chore(results): previsões do teste final v2 (M17)"
git push -u origin m17-results
```

`--accelerator NvidiaTeslaT4` é o "GPU T4 ×2" da Kaggle CLI (o mesmo do
`machine_shape` dos dois `kernel-metadata.json`); no kernel (a), o notebook usa só a 1ª
GPU. Pela interface web: importe o notebook, *Accelerator → GPU T4 x2*, *Internet → On*,
ajuste `PART` na 1ª célula e *Save Version → Save & Run All*.

Esperado: 10 JSONL do (a) (`<sistema>__test.jsonl` e `<sistema>+retry__test.jsonl` dos 5
sistemas) e 2 do (b), cada um com 150 linhas, e um `run-info` para cada. Depois
do push, abra o PR (ou avise o agente): a pontuação (`check-metrics --write`), o relatório
(`tanque eval report`) e o gráfico (`uv sync --group chart` e `tanque eval chart`, que
grava `results/ex-vs-cost.png`) são feitos pelo agente, e o CI confere os três. **Não**
rode `tanque eval score` por conta própria. Se o status for `ERROR`, o log está em
`kaggle-final-a/*.log` ou `kaggle-final-b/*.log`.

**Se parar no meio** (sessão caiu, limite de tempo): no editor do kernel, **Add Input →
Notebook Output Files** com a versão anterior do mesmo kernel e, na 1ª célula,
`RESUME_DIR = "/kaggle/input/tanque-cheio-final-v2a"` (ou `-v2b`; confira o caminho no
painel *Input*). *Save & Run All*: os itens já prontos são pulados. Volte depois para
`RESUME_DIR = None`.

## Publicação (M8)

> **Regra 5:** publicar no HF e mudar a visibilidade de um notebook exige a aprovação da
> Isabela. Os passos abaixo (👤) são para ela, no PowerShell, na raiz do repositório, com
> o branch do M8 (ou a `main`, depois do merge) atualizado.

Antes de tudo, **confira a licença dos dados da ANP**: abra o conjunto "Série Histórica de
Preços de Combustíveis" no [dados.gov.br](https://dados.gov.br) e leia o campo
**Licença**. Se ele disser, por exemplo, "Creative Commons Atribuição", troque no topo de
`hf/dataset/README.md` as três linhas `license: other`, `license_name` e `license_link`
por `license: cc-by-4.0` (ou o identificador certo, lista em
https://huggingface.co/docs/hub/repositories-licenses), ajuste o parágrafo "Licença" do
card e ponha `license_verified = true` em `configs/settings.toml`. Commite antes de
publicar.

Login no HF (uma vez; token com permissão de escrita em
https://huggingface.co/settings/tokens):

```powershell
uv run hf auth login
uv run hf auth whoami   # deve mostrar belasoares
```

**(a) Publicar o dataset** `belasoares/tanque-cheio`:

```powershell
# monta dist/hf-dataset/: card, data/{train,dev,test}.jsonl, snapshot/ (Parquet + JSON),
# docs/schema.md e manifest.json; confere o SHA-256 do snapshot. dist/ fica fora do git.
uv run tanque data hf-bundle --out dist/hf-dataset --force
Get-ChildItem -Recurse dist/hf-dataset | Select-Object FullName, Length

# cria o repo (público) na primeira vez e sobe a pasta inteira
uv run hf upload belasoares/tanque-cheio dist/hf-dataset . --repo-type dataset --commit-message "Dataset do tanque-cheio (M8)"
```

Para olhar antes de abrir ao público, acrescente `--private` no primeiro upload e depois
mude em *Settings → Change repo visibility* no HF. Na página do dataset, confira se o
*Dataset Viewer* mostra os configs `pairs` (splits `train`, `dev`, `test`) e `coletas`.

**(b) Atualizar o model card do adapter** (substitui o card provisório; os pesos não
mudam, então o `ADAPTER_REVISION` do `04_test_predict.ipynb` continua valendo):

```powershell
uv run hf upload belasoares/tanque-cheio-qwen2.5-coder-1.5b-lora hf/model/README.md README.md --commit-message "Model card final (M8)"
```

**(c) Opcional: tornar público o notebook do bake-off.** Hoje
`belasoares/tanque-cheio-bakeoff` é privado, por isso o README não o cita. Pela interface
web: página do kernel → *Settings → Sharing → Public* (não use `kaggle kernels push` com
`"is_private": false`: isso cria uma versão nova e roda tudo de novo). Se tornar público,
acrescente o link em `results/bakeoff.md` e na seção "Links" do `README.md`. O mesmo vale
para `belasoares/tanque-cheio-test-predict`.

Depois de (a) e (b), abra os links da seção "Links" do `README.md` e confira se todos
respondem.
