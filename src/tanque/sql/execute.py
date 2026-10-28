"""Execução segura de SQL no DuckDB (seção 5.5).

Camadas, da primeira para a última:

1. `guard.check` (allowlist pelo sqlglot) com as tabelas reais do banco;
2. o parser do próprio DuckDB confirma que há exatamente uma instrução SELECT
   (defesa contra divergências entre os dois parsers);
3. conexão `read_only=True` com `enable_external_access=false`: mesmo se as
   camadas acima falharem, não há escrita nem leitura de arquivo ou rede;
4. timeout por `connection.interrupt()` disparado por um timer;
5. `LIMIT max_rows` aplicado por fora, numa subconsulta, quando a consulta não
   tem um LIMIT menor; `fetchmany(max_rows)` garante o teto de qualquer forma.

`run` nunca levanta exceção por causa da SQL: erros voltam em `ExecResult.error`.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.tokens import TokenType

from tanque import config
from tanque.sql import guard

_WRAP_ALIAS = "_tanque_q"


@dataclass
class ExecResult:
    columns: list[str] = field(default_factory=list)
    rows: list[tuple[Any, ...]] = field(default_factory=list)
    error: str | None = None
    elapsed_ms: float = 0.0
    # Extensão do contrato 5.5: True quando o guardrail recusou a SQL (não executou).
    blocked: bool = False


def default_timeout_s() -> float:
    return config.load_settings()["execution"]["timeout_s"]


def default_max_rows() -> int:
    return config.load_settings()["execution"]["max_rows"]


def default_db_path() -> Path:
    return config.setting_path("db_path")


def _strip_trailing(sql: str) -> str:
    """SQL sem `;` finais e sem comentários depois do último token.

    Necessário para envolver a consulta em `SELECT * FROM (...)`: um `;` ou um
    `-- comentário` no fim quebrariam a subconsulta.
    """
    tokens = [t for t in sqlglot.tokenize(sql, read=guard.DIALECT)]
    while tokens and tokens[-1].token_type == TokenType.SEMICOLON:
        tokens.pop()
    if not tokens:
        return sql
    return sql[: tokens[-1].end + 1]


def _top_level_limit(tree: exp.Expression) -> int | None:
    limit = tree.args.get("limit")
    if not isinstance(limit, exp.Limit):
        return None
    value = limit.expression
    if isinstance(value, exp.Literal) and not value.is_string:
        try:
            return int(value.this)
        except ValueError:
            return None
    return None


def _apply_limit(sql: str, max_rows: int) -> str:
    """Envolve a consulta em `SELECT * FROM (...) LIMIT max_rows`, se necessário.

    Se a SQL não faz parse como uma única consulta, devolve-a sem mudança: o
    DuckDB vai reportar o erro (e `fetchmany` mantém o teto de linhas).
    """
    try:
        trees = [
            t
            for t in sqlglot.parse(sql, read=guard.DIALECT)
            if t is not None and not isinstance(t, exp.Semicolon)
        ]
    except SqlglotError:
        return sql
    if len(trees) != 1 or not isinstance(trees[0], (exp.Select, exp.SetOperation)):
        return sql
    inner = _top_level_limit(trees[0])
    if inner is not None and inner <= max_rows:
        return sql
    body = _strip_trailing(sql)
    # Quebras de linha protegem contra comentários de linha dentro da consulta.
    return f"SELECT * FROM (\n{body}\n) AS {_WRAP_ALIAS} LIMIT {int(max_rows)}"


def _database_tables(con: duckdb.DuckDBPyConnection) -> set[str]:
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()
    return {r[0] for r in rows}


def _duckdb_single_select(con: duckdb.DuckDBPyConnection, sql: str) -> str | None:
    """Motivo do bloqueio segundo o parser do DuckDB, ou None se for um SELECT."""
    try:
        statements = con.extract_statements(sql)
    except duckdb.Error as e:
        return f"SQL não faz parse no DuckDB: {e}"
    if len(statements) != 1:
        return f"mais de uma instrução segundo o DuckDB ({len(statements)})"
    if statements[0].type != duckdb.StatementType.SELECT:
        return f"só SELECT é permitido; o DuckDB leu {statements[0].type.name}"
    return None


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def run(
    sql: str,
    timeout_s: float | None = None,
    max_rows: int | None = None,
    db_path: str | Path | None = None,
    *,
    check_guard: bool = True,
) -> ExecResult:
    """Executa `sql` com segurança e devolve colunas, linhas, erro e tempo.

    Os padrões de `timeout_s`, `max_rows` e `db_path` vêm de configs/settings.toml.
    `check_guard=False` pula o guardrail (só para testar as outras camadas).
    """
    timeout_s = default_timeout_s() if timeout_s is None else timeout_s
    max_rows = default_max_rows() if max_rows is None else max_rows
    db = Path(db_path) if db_path is not None else default_db_path()

    start = time.perf_counter()
    if not db.is_file():
        return ExecResult(error=f"banco não encontrado: {db}", elapsed_ms=_ms_since(start))

    try:
        con = duckdb.connect(str(db), read_only=True, config={"enable_external_access": False})
    except duckdb.Error as e:
        return ExecResult(error=f"falha ao abrir o banco: {e}", elapsed_ms=_ms_since(start))

    try:
        if check_guard:
            verdict = guard.check(sql, allowed_tables=_database_tables(con))
            reason = verdict.reason if not verdict.ok else _duckdb_single_select(con, sql)
            if reason:
                return ExecResult(
                    error=f"bloqueada pelo guardrail: {reason}",
                    elapsed_ms=_ms_since(start),
                    blocked=True,
                )

        final_sql = _apply_limit(sql, max_rows)
        timed_out = threading.Event()

        def _interrupt() -> None:
            timed_out.set()
            con.interrupt()

        timer = threading.Timer(timeout_s, _interrupt)
        start = time.perf_counter()
        timer.start()
        try:
            cursor = con.execute(final_sql)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(max_rows) if cursor.description else []
        except (duckdb.Error, RuntimeError) as e:
            if timed_out.is_set():
                return ExecResult(error=f"timeout após {timeout_s}s", elapsed_ms=_ms_since(start))
            return ExecResult(error=f"{type(e).__name__}: {e}", elapsed_ms=_ms_since(start))
        finally:
            timer.cancel()
        return ExecResult(columns=columns, rows=rows, elapsed_ms=_ms_since(start))
    finally:
        con.close()
