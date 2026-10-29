"""Autocorreção com 1 nova tentativa (plan.md v2, D32; contratos 5.3 e 5.4 v2).

Para cada pergunta:

1. gera a SQL com o prompt único (`prompting.build_prompt`);
2. executa a SQL com o executor (guardrail + `execute.run` no banco montado do
   snapshot);
3. se a SQL foi **bloqueada pelo guardrail** ou deu **erro na execução**
   (inclui timeout e SQL vazia, que o guardrail recusa), gera uma única nova
   tentativa com `prompting.build_repair_prompt(pergunta, sql, erro)`.

Resultado vazio ou "errado" **não** dispara nova tentativa: o sistema não vê o
gold. Erro na *geração* (exceção do modelo, ex.: falta de memória) também não:
não há SQL nem erro de banco para mostrar ao modelo.

Cada pergunta rende dois registros (5.3), da mesma execução:

- o **single-shot**, igual ao da v1 (só a 1ª tentativa), para
  `<sistema>__<split>.jsonl`;
- o **+retry**, para `<sistema>+retry__<split>.jsonl`, com `system =
  "<sistema>+retry"`, `attempts` (1 ou 2), `first_sql` e `first_error` (nulos
  quando `attempts = 1`). Com nova tentativa, `sql`/`raw_output` são os da 2ª,
  `latency_ms` soma as duas gerações (a execução da 1ª SQL fica de fora, como a
  execução fica de fora da latência da v1) e `input_tokens`/`output_tokens`
  somam as duas gerações. Sem nova tentativa, o registro é o single-shot com
  `attempts = 1`.

Tudo aqui é injetável (gerador, executor, relógio), para testar sem modelo e
sem banco (`tests/test_repair.py`). O gerador recebe as mensagens (formato
chat) e devolve o texto, ou um objeto com `text`, `input_tokens` e
`output_tokens` (como `local.Generation`).

**Banco no Kaggle.** O executor precisa do banco: o notebook o monta do
snapshot versionado antes de carregar o modelo (~6 s, sem download):

    uv run tanque data build --from-parquet snapshot/coletas-2026-09.parquet

(ou `python -m tanque.cli data build --from-parquet ...`), que grava
`data/tanque.duckdb`; `make_executor()` usa esse caminho por padrão.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

RETRY_SUFFIX = "+retry"


class ExecOutcome(Protocol):
    """O que o executor devolve (como `execute.ExecResult`)."""

    error: str | None
    blocked: bool


Generate = Callable[[list[dict]], Any]
Execute = Callable[[str], ExecOutcome]
BuildPrompt = Callable[[str], list[dict]]
BuildRepairPrompt = Callable[[str, str, str], list[dict]]
ExtractSql = Callable[[str], str]


def _default_build_prompt(question: str) -> list[dict]:
    from tanque import prompting

    return prompting.build_prompt(question)


def _default_build_repair_prompt(question: str, sql: str, error: str) -> list[dict]:
    from tanque import prompting

    return prompting.build_repair_prompt(question, sql, error)


def _default_extract_sql(raw: str) -> str:
    from tanque import prompting

    return prompting.extract_sql(raw)


def retry_system(system: str) -> str:
    """Nome do sistema com nova tentativa: `<sistema>+retry` (5.3 v2)."""
    return f"{system}{RETRY_SUFFIX}"


def make_executor(
    db_path: str | Path | None = None,
    *,
    timeout_s: float | None = None,
    max_rows: int | None = None,
) -> Execute:
    """Executor real: `execute.run` (que aplica o guardrail) no banco `db_path`.

    Falha logo se o banco não existe: sem isso, toda SQL daria "banco não
    encontrado" e todas as perguntas ganhariam uma nova tentativa indevida.
    """
    from tanque.sql import execute

    db = Path(db_path) if db_path is not None else execute.default_db_path()
    if not db.is_file():
        raise FileNotFoundError(
            f"banco não encontrado: {db}. Monte com "
            "`tanque data build --from-parquet snapshot/coletas-2026-09.parquet`."
        )

    def run(sql: str) -> ExecOutcome:
        return execute.run(sql, timeout_s=timeout_s, max_rows=max_rows, db_path=db)

    return run


def failure_of(sql: str, execute: Execute) -> str | None:
    """Erro que justifica nova tentativa (guardrail ou execução), ou None se a SQL rodou."""
    if not str(sql or "").strip():
        return "SQL vazia"
    res = execute(sql)
    if res.blocked or res.error:
        return str(res.error or "bloqueada pelo guardrail")
    return None


# --- uma geração ----------------------------------------------------------------


@dataclass(frozen=True)
class Attempt:
    raw: str
    sql: str
    latency_ms: int
    input_tokens: int
    output_tokens: int


def _attempt(
    generate: Generate,
    messages: list[dict],
    extract_sql: ExtractSql,
    sync: Callable[[], None],
    clock: Callable[[], float],
) -> Attempt:
    sync()
    start = clock()
    out = generate(messages)
    sync()
    latency = round((clock() - start) * 1000)
    if isinstance(out, str):
        text, tin, tout = out, 0, 0
    else:
        text = str(out.text)
        tin, tout = int(out.input_tokens), int(out.output_tokens)
    return Attempt(text, extract_sql(text) if text else "", latency, tin, tout)


def _record(
    item_id: str,
    *,
    system: str,
    hardware: str,
    raw: str,
    sql: str,
    latency_ms: int | None,
    input_tokens: int,
    output_tokens: int,
    error: str | None,
) -> dict[str, Any]:
    from tanque.infer.local import _record as local_record

    return local_record(
        item_id,
        system=system,
        hardware=hardware,
        raw=raw,
        sql=sql,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        error=error,
    )


def _error_text(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"


# --- uma pergunta ---------------------------------------------------------------


def predict_item(
    item: dict[str, Any],
    generate: Generate,
    execute: Execute,
    *,
    system: str,
    hardware: str,
    build_prompt: BuildPrompt = _default_build_prompt,
    build_repair_prompt: BuildRepairPrompt = _default_build_repair_prompt,
    extract_sql: ExtractSql = _default_extract_sql,
    sync: Callable[[], None] = lambda: None,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(registro single-shot, registro +retry) de uma pergunta (5.3 v2)."""
    item_id = str(item["id"])
    question = str(item["question"])
    retry_name = retry_system(system)
    common = {"hardware": hardware}

    try:
        first = _attempt(generate, build_prompt(question), extract_sql, sync, clock)
    except Exception as e:  # erro na geração: sem SQL para corrigir, sem nova tentativa
        fields = {
            "raw": "",
            "sql": "",
            "latency_ms": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "error": _error_text(e),
        }
        single = _record(item_id, system=system, **common, **fields)
        retried = _record(item_id, system=retry_name, **common, **fields)
        retried.update(attempts=1, first_sql=None, first_error=None)
        return single, retried

    first_fields = {
        "raw": first.raw,
        "sql": first.sql,
        "latency_ms": first.latency_ms,
        "input_tokens": first.input_tokens,
        "output_tokens": first.output_tokens,
        "error": None,
    }
    single = _record(item_id, system=system, **common, **first_fields)

    failure = failure_of(first.sql, execute)
    if failure is None:
        retried = _record(item_id, system=retry_name, **common, **first_fields)
        retried.update(attempts=1, first_sql=None, first_error=None)
        return single, retried

    repair_messages = build_repair_prompt(question, first.sql, failure)
    try:
        second = _attempt(generate, repair_messages, extract_sql, sync, clock)
    except Exception as e:  # a 2ª geração falhou: conta como erro, com a 1ª registrada
        retried = _record(
            item_id,
            system=retry_name,
            **common,
            raw="",
            sql="",
            latency_ms=None,
            input_tokens=first.input_tokens,
            output_tokens=first.output_tokens,
            error=_error_text(e),
        )
    else:
        retried = _record(
            item_id,
            system=retry_name,
            **common,
            raw=second.raw,
            sql=second.sql,
            latency_ms=first.latency_ms + second.latency_ms,
            input_tokens=first.input_tokens + second.input_tokens,
            output_tokens=first.output_tokens + second.output_tokens,
            error=None,
        )
    retried.update(attempts=2, first_sql=first.sql, first_error=failure)
    return single, retried


# --- laço -----------------------------------------------------------------------


def predict_with_retry(
    items: Iterable[dict[str, Any]],
    generate: Generate,
    execute: Execute,
    *,
    system: str,
    hardware: str,
    build_prompt: BuildPrompt = _default_build_prompt,
    build_repair_prompt: BuildRepairPrompt = _default_build_repair_prompt,
    extract_sql: ExtractSql = _default_extract_sql,
    sync: Callable[[], None] = lambda: None,
    clock: Callable[[], float] = time.perf_counter,
    echo: Callable[[str], None] = lambda _: None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(previsões single-shot, previsões +retry), na ordem dos itens."""
    items = list(items)
    singles: list[dict[str, Any]] = []
    retried: list[dict[str, Any]] = []
    for n, item in enumerate(items, start=1):
        single, retry = predict_item(
            item,
            generate,
            execute,
            system=system,
            hardware=hardware,
            build_prompt=build_prompt,
            build_repair_prompt=build_repair_prompt,
            extract_sql=extract_sql,
            sync=sync,
            clock=clock,
        )
        singles.append(single)
        retried.append(retry)
        note = "nova tentativa" if retry["attempts"] == 2 else "1 tentativa"
        status = f"erro {single['error'].split(':')[0]}" if single["error"] else note
        echo(f"[{n}/{len(items)}] {item['id']}: {status}, {retry['latency_ms']} ms")
    return singles, retried


def summarize_retry(retried: list[dict[str, Any]]) -> dict[str, Any]:
    """Contagens da nova tentativa para o log e o run-info."""
    second = [p for p in retried if p.get("attempts") == 2]
    blocked = sum(1 for p in second if str(p.get("first_error") or "").startswith("bloqueada"))
    return {
        "n": len(retried),
        "retried": len(second),
        "first_blocked": blocked,
        "first_exec_error": len(second) - blocked,
        "retry_generation_errors": sum(1 for p in second if p.get("error")),
    }
