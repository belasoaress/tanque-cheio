# Demo no HF Spaces (M19) · passos da Isabela 👤

A demo roda o modelo final (Qwen2.5-Coder-1.5B-Instruct + adapter v2, EX de 80,7% no
teste) em GGUF Q4_K_M, na CPU grátis do Hugging Face Spaces (D36). O agente preparou
o notebook de exportação, o app e os testes; **publicar é com você**. São três etapas:

| Etapa | Onde | Tempo |
|---|---|---|
| (a) GGUF: fundir, converter, quantizar, conferir e publicar | Kaggle (CPU) + HF | ~25 min, quase tudo esperando |
| (b) Space: criar e subir a pasta `app/` | navegador + PowerShell | ~10 min (build ~5 min) |
| (c) Conferir as 6 perguntas e o tempo (R24, meta < 30 s) | navegador | ~5 min |

Todos os comandos são de PowerShell, na raiz do repositório, **depois do merge do PR do
M19** (o notebook usa `REPO_REF = "main"` e importa `app/tanque_demo.py` de lá):

```powershell
git switch main
git pull
uv sync
```

## O que fica fixado

| Peça | Versão |
|---|---|
| Modelo base | `Qwen/Qwen2.5-Coder-1.5B-Instruct` @ `2e1fd397ee46e1388853d2af2c993145b0f1098a` |
| Adapter v2 | `belasoares/tanque-cheio-qwen2.5-coder-1.5b-lora-v2` @ `beeb19b5b4331ff048c6be0f03b9d53476d509ab` |
| llama.cpp (conversor) | tag `b11312` (`0c1e570`), a mesma que o `llama-cpp-python` 0.3.36 embute |
| `llama-cpp-python` | 0.3.36, wheel CPU do índice oficial (`abetlen.github.io/llama-cpp-python/whl/cpu`) |
| Gradio | 6.30.0 (`sdk_version` do Space), Python 3.11 |
| Pacote `tanque` no Space | commit `78029d259dfb0a69ff5256dafd7f3c6cb34dd1a5` (o mesmo de `TANQUE_COMMIT`) |
| GGUF no Space | variável `GGUF_REVISION` (passo b.3) |

## (a) GGUF no Kaggle

O kernel `belasoares/tanque-cheio-export-gguf` roda **sem GPU** (não gasta a cota):
4 vCPU e ~30 GB de RAM bastam para fundir o 1,5B em fp32 (~7 GB) e converter. Saídas
em `/kaggle/working`: `tanque-cheio-qwen2.5-coder-1.5b-q4_k_m.gguf` (~0,92 GiB),
`README.md` (model card) e `export-info.json` (versões, SHA-256, tamanhos e a
conferência em 3 perguntas do dev2).

### a.1 Rodar sem publicar e conferir

```powershell
kaggle kernels list --mine --page-size 5            # confirma a autenticação
kaggle kernels push -p notebooks/kaggle/export-gguf  # sem --accelerator: CPU
kaggle kernels status belasoares/tanque-cheio-export-gguf   # repita até COMPLETE (~20 min)
kaggle kernels output belasoares/tanque-cheio-export-gguf -p kaggle-gguf -o
Get-ChildItem kaggle-gguf
(Get-Content kaggle-gguf/export-info.json -Raw | ConvertFrom-Json).check |
    Select-Object n, same_sql, same_result, fp16_correct, gguf_correct, same_prompt
```

O que olhar no `check`:

- `same_prompt` = `True`: o chat template do GGUF gera o mesmo texto que o do
  transformers.
- `same_result` (resultados que batem na execução, GGUF × fp16): o esperado é 3/3;
  2/3 ainda serve (a quantização pode trocar uma SQL equivalente). Com 0 ou 1 de 3,
  **não publique**: abra o log (`kaggle-gguf/*.log`) e me chame.
- `gguf_correct` e `fp16_correct`: acertos contra o gold do dev2 (só informativo).

Se o status for `ERROR`, o log está em `kaggle-gguf/*.log`. A pasta `kaggle-gguf/` não
vai para o git.

### a.2 Publicar o GGUF no HF

Escolha **um** dos dois caminhos. O B não precisa de secret no Kaggle.

**Caminho A, pelo Kaggle (com o secret `HF_TOKEN`):**

1. Crie um token de escrita em <https://huggingface.co/settings/tokens> (*Create new
   token* → *Write*, ou *Fine-grained* com escrita em `belasoares/*`).
2. Abra <https://www.kaggle.com/code/belasoares/tanque-cheio-export-gguf> → **Edit**.
   No editor: **Add-ons → Secrets → Add Secret**, *Label* `HF_TOKEN`, *Value* o token;
   salve e deixe a chave **ligada** para este notebook (ela fica para as próximas
   versões). Feche o editor sem rodar.
3. Ligue a publicação só para este push e volte o arquivo ao normal:

   ```powershell
   $nb = (Resolve-Path notebooks/07_export_gguf.ipynb).Path
   $text = [IO.File]::ReadAllText($nb)
   [IO.File]::WriteAllText($nb, $text.Replace('PUSH_TO_HUB = False', 'PUSH_TO_HUB = True'))
   Select-String -Path $nb -Pattern 'PUSH_TO_HUB = '   # deve mostrar PUSH_TO_HUB = True
   kaggle kernels push -p notebooks/kaggle/export-gguf
   [IO.File]::WriteAllText($nb, $text)                  # volta para PUSH_TO_HUB = False
   git status --short                                   # o notebook não deve aparecer
   ```

4. Espere `COMPLETE` e baixe o log de novo:

   ```powershell
   kaggle kernels status belasoares/tanque-cheio-export-gguf
   kaggle kernels output belasoares/tanque-cheio-export-gguf -p kaggle-gguf -o
   Select-String -Path kaggle-gguf/*.log -Pattern 'GGUF_REVISION para o app'
   ```

   Anote o commit (40 caracteres hexadecimais) da linha `GGUF_REVISION para o app:`.

**Caminho B, subir a saída do a.1 do seu computador:**

```powershell
uv run hf auth login        # cole o token de escrita (uma vez por máquina)
uv run hf upload belasoares/tanque-cheio-qwen2.5-coder-1.5b-gguf kaggle-gguf . `
    --include "*.gguf" --include "README.md" --include "export-info.json" `
    --commit-message "GGUF Q4_K_M do ft2 (beeb19b), llama.cpp b11312"
```

O `hf upload` cria o repositório (público) se ele não existir e imprime a URL do commit (`.../commit/<commit>`): anote o commit. O
arquivo tem ~1 GB; se cair no meio, rode o mesmo comando de novo.

Nos dois casos, confira em
<https://huggingface.co/belasoares/tanque-cheio-qwen2.5-coder-1.5b-gguf> o arquivo, o
model card e o SHA-256 (o mesmo de `export-info.json`).

## (b) Space

### b.1 Criar o Space

Em <https://huggingface.co/new-space>:

- **Owner** `belasoares`, **Space name** `tanque-cheio`, **License** `mit`;
- **Select the Space SDK:** *Gradio* (template *Blank*);
- **Space hardware:** *CPU basic · 2 vCPU · 16 GB · FREE*;
- **Public**; *Create Space*.

O título, o emoji, a versão do Gradio, o Python 3.11 e os links para o modelo e o
dataset vêm do front-matter de `app/README.md`, que sobrescreve o README gerado.

### b.2 Subir a pasta `app/`

```powershell
uv run hf auth login        # se ainda não fez no caminho B
uv run hf upload belasoares/tanque-cheio app . --repo-type space `
    --exclude "runtime/**" --exclude "__pycache__/**"
```

(`runtime/` só existe se você rodou o app no seu computador: é o banco montado e não
deve subir.)

### b.3 Fixar a revisão do GGUF

No Space: **Settings → Variables and secrets → New variable**, *Name* `GGUF_REVISION`,
*Value* o commit anotado no a.2. Salve: o Space reinicia sozinho. Sem a variável, o app
usa a revisão `main` do repositório do GGUF (funciona, mas não fica fixado).

### b.4 Acompanhar o build

Na página do Space, aba **Logs**: o *Build* instala o `requirements.txt` (~3–5 min; o
`llama-cpp-python` vem como wheel pronto, sem compilar) e o *Container* baixa o snapshot
do GitHub (2,4 MB), monta o banco (~10 s), baixa o GGUF (~1 GB) e aquece o prompt. O
Space fica *Running* quando aparece `Running on local URL: http://0.0.0.0:7860`.

Se o build falhar em `tanque @ git+https://...`, confira se o commit `78029d2` ainda
existe na `main` do GitHub (ele é público e não deve ser reescrito).

## (c) Conferir as 6 perguntas e o tempo (R24)

Em <https://huggingface.co/spaces/belasoares/tanque-cheio>, clique em cada um dos 6
exemplos (um clique já pergunta). A linha de status mostra o **tempo de resposta** e se
houve **nova tentativa**. A meta é **< 30 s** por pergunta. A primeira pergunta depois
de o Space acordar pode demorar mais; meça a partir da segunda rodada.

| # | Exemplo | Tempo (s) | Nova tentativa? | Resposta plausível? |
|---|---|---|---|---|
| 1 | O etanol compensou em São Carlos (SP) no último mês? | | | |
| 2 | Qual o preço médio da gasolina em Curitiba (PR) no último mês? | | | |
| 3 | Quantos postos de bandeira branca há em Belo Horizonte (MG)? | | | |
| 4 | Quais as 5 capitais com o diesel S10 mais barato em agosto de 2026? | | | |
| 5 | Como variou mês a mês o preço médio do GNV no Rio de Janeiro (RJ) em 2026? | | | |
| 6 | Em quais meses de 2026 o etanol compensou em Goiânia (GO)? | | | |

**Referência (ensaio do agente, 30/10/2026):** o notebook rodou fora do Kaggle (CPU, 4
vCPU, sem a célula de instalação) com o adapter real e gerou um GGUF de 0,918 GiB
(986.048.096 bytes); na conferência, SQL idêntica e resultado igual em 3/3, prompt
idêntico ao do transformers. O app com esse GGUF, `n_threads = 2` (como no CPU basic),
respondeu os 6 exemplos em 18,7 / 11,8 / 5,8 / 12,9 / 14,1 / 21,1 s, sem nova tentativa.
A CPU do Spaces pode ser mais lenta que a do ensaio: se passar de 30 s, me avise.

Para medir pela linha de comando (opcional), o app expõe a API `/perguntar`:

```powershell
@'
import sys, time
sys.path.insert(0, "app")
from gradio_client import Client
import tanque_demo

client = Client("belasoares/tanque-cheio")
for q in tanque_demo.EXAMPLES:
    t0 = time.perf_counter()
    sql, table, status, first = client.predict(q, api_name="/perguntar")
    print(f"{time.perf_counter() - t0:5.1f} s | {q}")
'@ | uv run --with gradio_client==2.7.2 python -
```

O tempo do cliente inclui a rede e a fila; o da interface é só o do modelo e do banco.

Para fechar o R24, cole a tabela preenchida e o link do Space no PR (ou num issue) e
avise o agente líder.

## Rodar o app no seu computador (opcional)

Com o GGUF baixado (ex.: `kaggle-gguf/`), num ambiente à parte para não mexer no
`uv.lock`:

```powershell
uv venv .venv-app --python 3.11
uv pip install --python .venv-app -r app/requirements.txt
$env:GGUF_PATH = (Resolve-Path kaggle-gguf/tanque-cheio-qwen2.5-coder-1.5b-q4_k_m.gguf).Path
.venv-app\Scripts\python app/app.py     # abre em http://127.0.0.1:7860
Remove-Item Env:GGUF_PATH
```

O banco fica em `app/runtime/` (fora do git). Apague `.venv-app/` e `app/runtime/` quando
terminar.
