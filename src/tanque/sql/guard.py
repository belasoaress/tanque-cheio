"""Guardrail de SQL (seção 5.5): só uma instrução SELECT/WITH sobre tabelas conhecidas.

A análise usa o parser do `sqlglot` no dialeto `duckdb` e segue uma **allowlist**:

- a árvore de topo precisa ser um `SELECT` (inclui `WITH ... SELECT` e `FROM ...
  SELECT`) ou uma operação de conjunto (`UNION`, `INTERSECT`, `EXCEPT`) de SELECTs;
- toda referência de tabela no `FROM`/`JOIN` precisa ser uma tabela permitida
  (padrão: as tabelas da seção 5.1), o nome de uma CTE da própria consulta, ou
  uma função de tabela geradora sem acesso a arquivo (`range`, `generate_series`,
  `unnest`). Isso bloqueia `FROM 'arquivo.csv'` (o DuckDB lê arquivos citados por
  string no FROM), `read_csv(...)`, `glob(...)`, `query(...)`, catálogos anexados e
  `information_schema`, sem depender de uma lista de nomes perigosos;
- qualquer nó de escrita, DDL ou comando (`INSERT`, `COPY`, `PRAGMA`, `SET`,
  `ATTACH` etc.) em qualquer ponto da árvore bloqueia a consulta;
- como defesa extra, chamadas escalares a funções que leem arquivos ou o
  ambiente (`read_*`, `*_scan`, `glob`, `getenv` ...) também são bloqueadas.

A execução (`tanque.sql.execute`) ainda abre o banco somente leitura e sem acesso
externo; o guardrail é a primeira camada, não a única.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

DIALECT = "duckdb"

# Tabelas do contrato (seção 5.1). `execute.run` passa as tabelas reais do banco.
DEFAULT_TABLES: frozenset[str] = frozenset({"municipios", "postos", "coletas"})

# Esquema aceito como qualificador (`main.coletas`). Catálogos nunca são aceitos.
ALLOWED_SCHEMAS: frozenset[str] = frozenset({"main"})

# Funções de tabela que só geram valores (sem arquivo, rede ou metadados).
ALLOWED_TABLE_FUNCTIONS: frozenset[str] = frozenset({"range", "generate_series", "unnest"})

# Funções escalares bloqueadas mesmo fora do FROM.
_BLOCKED_FUNCTION_RE = re.compile(
    r"^(read_\w*|\w*_scan|glob|sniff_csv|getenv|query|query_table|parquet_\w+|"
    r"iceberg_\w+|delta_\w+|load_\w+|install_\w+)$"
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Nós que nunca podem aparecer, em nenhum nível da árvore.
_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = tuple(
    getattr(exp, name)
    for name in (
        "Insert",
        "Delete",
        "Update",
        "Merge",
        "Create",
        "Drop",
        "Alter",
        "TruncateTable",
        "Copy",
        "Command",
        "Pragma",
        "Set",
        "Attach",
        "Detach",
        "Use",
        "Transaction",
        "Commit",
        "Rollback",
        "Describe",
        "Summarize",
        "Show",
        "Install",
        "Export",
        "Cache",
        "Uncache",
        "Refresh",
        "Analyze",
        "Grant",
        "Revoke",
        "Comment",
        "LoadData",
    )
    if hasattr(exp, name)
)

_TOP_LEVEL_OK: tuple[type[exp.Expression], ...] = (exp.Select, exp.SetOperation)


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    reason: str | None = None


def _block(reason: str) -> GuardResult:
    return GuardResult(ok=False, reason=reason)


def _node_name(node: exp.Expression) -> str:
    return type(node).__name__.upper()


def _function_name(func: exp.Func) -> str:
    if isinstance(func, exp.Anonymous):
        return str(func.name).lower()
    return func.sql_name().lower()


def _parse(sql: str) -> list[exp.Expression | None]:
    # O sqlglot avisa via logging quando cai para `Command`; o guardrail já trata isso.
    logger = logging.getLogger("sqlglot")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    try:
        return sqlglot.parse(sql, read=DIALECT)
    finally:
        logger.setLevel(previous)


def _check_table(
    table: exp.Table, allowed: frozenset[str], cte_names: frozenset[str]
) -> str | None:
    """Motivo do bloqueio da referência de tabela, ou None se permitida."""
    target = table.this
    if isinstance(target, exp.Func):
        name = _function_name(target)
        if name in ALLOWED_TABLE_FUNCTIONS:
            return None
        return f"função de tabela não permitida: {name}()"
    if not isinstance(target, exp.Identifier):
        return f"referência de tabela não reconhecida: {table.sql(dialect=DIALECT)}"

    name = str(target.this)
    if table.catalog:
        return f"catálogo não permitido: {table.catalog}"
    schema = table.db.lower() if table.db else ""
    if schema and schema not in ALLOWED_SCHEMAS:
        return f"esquema não permitido: {table.db}"
    if not _IDENTIFIER_RE.match(name):
        # Ex.: FROM 'dados.csv' — o DuckDB leria o arquivo.
        return f"leitura de arquivo ou nome inválido no FROM: {name!r}"
    lowered = name.lower()
    if not schema and lowered in cte_names:
        return None
    if lowered in allowed:
        return None
    return f"tabela desconhecida: {name}"


def check(sql: str, allowed_tables: Iterable[str] | None = None) -> GuardResult:
    """Valida `sql`; `ok=False` vem sempre com o motivo em `reason`.

    `allowed_tables` substitui as tabelas permitidas (padrão: `DEFAULT_TABLES`).
    Um único `;` no fim é aceito.
    """
    if not isinstance(sql, str) or not sql.strip():
        return _block("SQL vazia")

    try:
        # `None` e `Semicolon` são trechos vazios (só `;` ou só comentário).
        statements = [s for s in _parse(sql) if s is not None and not isinstance(s, exp.Semicolon)]
    except SqlglotError as e:
        first_line = str(e).splitlines()[0] if str(e) else type(e).__name__
        return _block(f"SQL não faz parse: {first_line}")

    if not statements:
        return _block("SQL vazia")
    if len(statements) > 1:
        return _block(f"mais de uma instrução ({len(statements)}); só uma é permitida")

    tree = statements[0]
    if not isinstance(tree, _TOP_LEVEL_OK):
        return _block(f"só SELECT/WITH é permitido; recebido {_node_name(tree)}")

    for node in tree.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            return _block(f"instrução não permitida dentro da consulta: {_node_name(node)}")

    allowed = frozenset(t.lower() for t in (allowed_tables or DEFAULT_TABLES))
    cte_names = frozenset(cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE))

    for table in tree.find_all(exp.Table):
        reason = _check_table(table, allowed, cte_names)
        if reason:
            return _block(reason)

    for func in tree.find_all(exp.Func):
        name = _function_name(func)
        if _BLOCKED_FUNCTION_RE.match(name):
            return _block(f"função não permitida: {name}()")

    return GuardResult(ok=True, reason=None)
