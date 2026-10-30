"""Lógica da demo (M19, D36), separada da interface para ser testada sem o llama.cpp.

O fluxo de uma pergunta é o mesmo da avaliação v2:

1. `tanque.prompting.build_prompt` (o prompt único, 5.4);
2. geração gulosa, até 384 tokens (D31);
3. `tanque.sql.execute.run` (guardrail + DuckDB só leitura, 5.5);
4. se a SQL foi bloqueada ou deu erro, 1 nova tentativa com
   `tanque.prompting.build_repair_prompt` (D32), pela mesma
   `tanque.infer.repair.predict_item` usada nos notebooks.

Nada aqui importa `gradio` nem `llama_cpp`: o gerador e o executor são injetados.
`app.py` liga as peças reais. Nenhuma pergunta é gravada em disco ou em log.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

# --- parâmetros fixos da demo -----------------------------------------------------

# Commit do tanque-cheio de onde vêm o pacote (app/requirements.txt), a configuração,
# o manifest e o snapshot. Os dois precisam ser o MESMO commit.
TANQUE_COMMIT = "78029d259dfb0a69ff5256dafd7f3c6cb34dd1a5"
RAW_BASE = "https://raw.githubusercontent.com/belasoaress/tanque-cheio"
SNAPSHOT = "snapshot/coletas-2026-09.parquet"
SNAPSHOT_META = "snapshot/coletas-2026-09.json"
# Arquivos que o pacote `tanque` lê da "raiz do repositório" (TANQUE_ROOT).
ROOT_FILES = ("configs/settings.toml", "data/manifest.json", SNAPSHOT_META, SNAPSHOT)

GGUF_REPO = "belasoares/tanque-cheio-qwen2.5-coder-1.5b-gguf"
GGUF_FILE = "tanque-cheio-qwen2.5-coder-1.5b-q4_k_m.gguf"
# Revisão do GGUF no HF. O padrão "main" só serve até a 1ª publicação: depois, fixe o
# commit impresso pelo notebook 07 na variável de ambiente GGUF_REVISION do Space
# (Settings > Variables) ou aqui (docs/demo.md).
DEFAULT_GGUF_REVISION = "main"

MAX_TOKENS = 384  # D31
N_CTX = 2048  # prompt (~800 tokens) + nova tentativa (~1.000) + 384 de saída cabem
N_THREADS = 2  # CPU basic do Spaces: 2 vCPU
TIMEOUT_S = 10.0  # o mesmo de configs/settings.toml
MAX_TABLE_ROWS = 100
EXEC_MAX_ROWS = 1000  # teto do executor (configs/settings.toml); a tabela mostra 100
MAX_QUESTION_CHARS = 300

SYSTEM_NAME = "ft2-qwen2.5-coder-1.5b-instruct-q4_k_m"
HARDWARE = "hf-spaces-cpu-basic"

DISCLAIMER = (
    "Dados da ANP de 2024-10 a 2026-09; data de referência 30/09/2026; "
    "'último mês' = setembro de 2026."
)

# Perguntas de exemplo escritas para a demo (não vêm dos conjuntos de dev nem de teste).
EXAMPLES: list[str] = [
    "O etanol compensou em São Carlos (SP) no último mês?",
    "Qual o preço médio da gasolina em Curitiba (PR) no último mês?",
    "Quantos postos de bandeira branca há em Belo Horizonte (MG)?",
    "Quais as 5 capitais com o diesel S10 mais barato em agosto de 2026?",
    "Como variou mês a mês o preço médio do GNV no Rio de Janeiro (RJ) em 2026?",
    "Em quais meses de 2026 o etanol compensou em Goiânia (GO)?",
]


# --- montagem do banco ----------------------------------------------------------------


def raw_url(path: str, commit: str = TANQUE_COMMIT) -> str:
    return f"{RAW_BASE}/{commit}/{path}"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch_root(
    root: Path,
    *,
    commit: str = TANQUE_COMMIT,
    download: Callable[[str, Path], None] = _download,
) -> Path:
    """Baixa configuração, manifest e snapshot do commit fixado para `root`.

    Confere o SHA-256 do Parquet contra o JSON do snapshot (o mesmo do D25).
    """
    root = Path(root)
    for rel in ROOT_FILES:
        dest = root / rel
        if not dest.is_file():
            download(raw_url(rel, commit), dest)
    meta = json.loads((root / SNAPSHOT_META).read_text(encoding="utf-8"))
    digest = sha256_of(root / SNAPSHOT)
    if digest != meta["sha256"]:
        (root / SNAPSHOT).unlink(missing_ok=True)
        raise RuntimeError(f"snapshot corrompido: sha256 {digest} != {meta['sha256']}")
    return root


def build_db(root: Path) -> Path:
    """Monta o DuckDB do snapshot com o mesmo código de `tanque data build --from-parquet`."""
    os.environ["TANQUE_ROOT"] = str(Path(root).resolve())
    from tanque import config
    from tanque.data import build

    config.repo_root.cache_clear()
    db_path = config.setting_path("db_path")
    if not db_path.is_file():
        build.build_database(config.resolve_path(SNAPSHOT), db_path)
    return db_path


# --- geração com o llama.cpp ------------------------------------------------------------


@dataclass(frozen=True)
class Generation:
    """Mesmo formato de `tanque.infer.local.Generation` (texto e tokens)."""

    text: str
    input_tokens: int
    output_tokens: int


def llama_generator(llm: Any, max_tokens: int = MAX_TOKENS) -> Callable[[list[dict]], Generation]:
    """Gerador "mensagens -> Generation" sobre um `llama_cpp.Llama`.

    O chat template é o do próprio GGUF (o do Qwen2.5, copiado do tokenizer pelo
    conversor). Decodificação gulosa: temperature 0 vira o amostrador greedy do
    llama.cpp, sem penalidades de repetição.
    """

    def generate(messages: list[dict]) -> Generation:
        out = llm.create_chat_completion(
            messages=messages,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            repeat_penalty=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            max_tokens=max_tokens,
            seed=0,
        )
        text = out["choices"][0]["message"].get("content") or ""
        usage = out.get("usage") or {}
        return Generation(
            text, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
        )

    return generate


# --- uma pergunta ---------------------------------------------------------------------


@dataclass
class Answer:
    question: str
    sql: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    total_rows: int = 0
    seconds: float = 0.0
    retried: bool = False
    first_sql: str | None = None
    first_error: str | None = None
    message: str | None = None  # erro amigável (guardrail, banco, modelo)
    ok: bool = False

    @property
    def truncated(self) -> bool:
        return self.total_rows > len(self.rows)


def friendly_error(error: str | None, blocked: bool) -> str:
    """Mensagem curta para quem usa a demo, sem stack trace."""
    text = str(error or "").strip()
    if blocked:
        reason = text.removeprefix("bloqueada pelo guardrail:").strip()
        return (
            "A SQL gerada foi recusada pelo guardrail de segurança, que só aceita uma "
            f"consulta SELECT de leitura nas tabelas do banco. Motivo: {reason}"
        )
    if text.startswith("timeout"):
        return "A consulta passou do limite de tempo (10 s). Tente uma pergunta mais específica."
    return f"A SQL gerada deu erro no banco: {text[:300]}"


def cell_value(value: Any) -> Any:
    """Valor de célula que a tabela do Gradio mostra bem (Decimal, datas, nulos)."""
    if value is None:
        return ""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dt.datetime) and value.time() == dt.time(0):
        return value.date().isoformat()  # date_trunc('month', ...) devolve timestamp
    if isinstance(value, dt.datetime | dt.date):
        return value.isoformat()
    if isinstance(value, bool | int | float | str):
        return value
    return str(value)


def answer_question(
    question: str,
    generate: Callable[[list[dict]], Any],
    execute: Callable[[str], Any],
    *,
    max_rows: int = MAX_TABLE_ROWS,
    clock: Callable[[], float] = time.perf_counter,
) -> Answer:
    """Responde a uma pergunta: SQL, tabela (até `max_rows` linhas), tempo e nova tentativa.

    `generate` recebe mensagens no formato chat; `execute` recebe a SQL e devolve
    um `ExecResult` (colunas, linhas, erro, `blocked`). A nova tentativa é a de
    `tanque.infer.repair.predict_item`, igual à da avaliação.
    """
    from tanque.infer import repair

    question = " ".join(str(question or "").split())
    ans = Answer(question=question)
    if not question:
        ans.message = "Escreva uma pergunta sobre preços de combustíveis."
        return ans
    if len(question) > MAX_QUESTION_CHARS:
        ans.message = f"A pergunta é longa demais (máximo de {MAX_QUESTION_CHARS} caracteres)."
        return ans

    cache: dict[str, Any] = {}

    def run(sql: str) -> Any:
        if sql not in cache:
            cache[sql] = execute(sql)
        return cache[sql]

    start = clock()
    _, record = repair.predict_item(
        {"id": "demo", "question": question},
        generate,
        run,
        system=SYSTEM_NAME,
        hardware=HARDWARE,
        clock=clock,
    )
    ans.retried = record.get("attempts") == 2
    ans.first_sql = record.get("first_sql")
    ans.first_error = record.get("first_error")
    ans.sql = record.get("sql") or ""

    if record.get("error"):  # a geração falhou (ex.: falta de memória)
        ans.seconds = clock() - start
        ans.message = "O modelo não conseguiu gerar uma resposta agora. Tente de novo."
        return ans
    if not ans.sql.strip():
        ans.seconds = clock() - start
        ans.message = "O modelo não devolveu uma SQL. Tente reformular a pergunta."
        return ans

    res = run(ans.sql)
    ans.seconds = clock() - start
    if res.blocked or res.error:
        ans.message = friendly_error(res.error, res.blocked)
        return ans
    ans.ok = True
    ans.columns = [str(c) for c in res.columns]
    ans.total_rows = len(res.rows)
    ans.rows = [[cell_value(v) for v in row] for row in res.rows[:max_rows]]
    return ans


# --- textos da interface ----------------------------------------------------------------


def status_markdown(ans: Answer) -> str:
    """Linha de status: tempo, nova tentativa, linhas e erro amigável."""
    parts = [f"Tempo de resposta: **{ans.seconds:.1f} s**"]
    if ans.retried:
        parts.append(
            "houve **nova tentativa**: a 1ª SQL falhou e o modelo recebeu o erro para corrigir"
        )
    else:
        parts.append("sem nova tentativa")
    if ans.ok:
        if ans.total_rows == 0:
            parts.append("a consulta rodou, mas não retornou linhas")
        elif ans.truncated:
            total = f"{ans.total_rows}" + (" ou mais" if ans.total_rows >= EXEC_MAX_ROWS else "")
            parts.append(f"mostrando {len(ans.rows)} de {total} linhas")
        else:
            parts.append(f"{ans.total_rows} linha(s)")
    text = " · ".join(parts)
    if ans.message:
        text = f"{ans.message}\n\n{text}" if ans.question else ans.message
    return text


def first_attempt_markdown(ans: Answer) -> str:
    """Detalhe da 1ª tentativa, quando houve nova tentativa."""
    if not ans.retried:
        return ""
    return (
        f"**1ª tentativa** (falhou: {str(ans.first_error or '')[:300]})\n\n"
        f"```sql\n{ans.first_sql or ''}\n```"
    )


def table_value(ans: Answer) -> dict[str, Any]:
    """Valor do `gr.Dataframe` (cabeçalhos e linhas), sem depender do pandas."""
    if not ans.ok:
        return {"headers": ["resultado"], "data": []}
    return {"headers": ans.columns or ["resultado"], "data": ans.rows}


def respond(
    question: str,
    generate: Callable[[list[dict]], Any],
    execute: Callable[[str], Any],
) -> tuple[str, dict[str, Any], str, str]:
    """Saídas da interface: SQL, tabela, status e detalhe da 1ª tentativa."""
    ans = answer_question(question, generate, execute)
    return ans.sql, table_value(ans), status_markdown(ans), first_attempt_markdown(ans)
