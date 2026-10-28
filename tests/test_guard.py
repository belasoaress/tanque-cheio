"""Testes do guardrail de SQL (src/tanque/sql/guard.py, seção 5.5)."""

import pytest

from tanque.sql import guard
from tanque.sql.guard import GuardResult

BLOCKED = {
    "drop": "DROP TABLE coletas",
    "delete": "DELETE FROM coletas",
    "insert": "INSERT INTO coletas VALUES (1, 1, 'X', DATE '2026-01-01', 1.0)",
    "update": "UPDATE coletas SET valor_venda = 0",
    "create": "CREATE TABLE x AS SELECT 1",
    "create_view": "CREATE VIEW v AS SELECT * FROM coletas",
    "alter": "ALTER TABLE coletas ADD COLUMN x INTEGER",
    "attach": "ATTACH 'outro.db' AS outro",
    "detach": "DETACH outro",
    "copy_to": "COPY coletas TO 'saida.csv'",
    "copy_select_to": "COPY (SELECT * FROM coletas) TO 'saida.parquet' (FORMAT parquet)",
    "install_httpfs": "INSTALL httpfs",
    "load_httpfs": "LOAD httpfs",
    "pragma": "PRAGMA table_info('coletas')",
    "set": "SET threads = 1",
    "set_external_access": "SET enable_external_access = true",
    "export_database": "EXPORT DATABASE 'dump'",
    "call": "CALL pragma_version()",
    "describe": "DESCRIBE coletas",
    "show": "SHOW TABLES",
    "use": "USE outro",
    "begin": "BEGIN TRANSACTION",
    "two_statements": "SELECT 1; SELECT 2",
    "select_then_drop": "SELECT * FROM coletas; DROP TABLE coletas",
    "comment_hides_second": "SELECT 1 /* só um comentário */; DROP TABLE coletas",
    "line_comment_then_second": "SELECT 1; -- comentário\nDROP TABLE coletas",
    "string_with_semicolon_then_second": "SELECT ';'; DROP TABLE coletas",
    "read_csv": "SELECT * FROM read_csv('/etc/passwd')",
    "read_csv_auto": "SELECT * FROM read_csv_auto('dados.csv')",
    "read_parquet": "SELECT * FROM read_parquet('dados.parquet')",
    "read_json": "SELECT * FROM read_json('dados.json')",
    "read_json_auto": "SELECT * FROM read_json_auto('dados.json')",
    "read_text": "SELECT * FROM read_text('segredo.txt')",
    "glob": "SELECT * FROM glob('*')",
    "parquet_scan": "SELECT * FROM parquet_scan('dados.parquet')",
    "sniff_csv": "SELECT * FROM sniff_csv('dados.csv')",
    "query_function": "SELECT * FROM query('SELECT 1')",
    "duckdb_settings": "SELECT * FROM duckdb_settings()",
    "file_in_from_single_quotes": "SELECT * FROM 'dados.csv'",
    "file_in_from_double_quotes": 'SELECT * FROM "dados.parquet"',
    "url_in_from": "SELECT * FROM 'https://example.invalid/x.parquet'",
    "read_csv_in_join": "SELECT * FROM coletas c JOIN read_csv('x.csv') r ON true",
    "read_csv_in_subquery": "SELECT * FROM coletas WHERE posto_id IN (SELECT * FROM read_csv('x'))",
    "read_csv_in_cte": "WITH r AS (SELECT * FROM read_csv('x.csv')) SELECT * FROM r",
    "read_csv_scalar": "SELECT (SELECT count(*) FROM read_csv('x.csv'))",
    "unknown_table": "SELECT * FROM tabela_que_nao_existe",
    "other_catalog": "SELECT * FROM outro.main.coletas",
    "information_schema": "SELECT * FROM information_schema.tables",
    "getenv": "SELECT getenv('HOME')",
    "current_setting": "SELECT current_setting('temp_directory')",
    "getvariable": "SELECT getvariable('x')",
    "duckdb_scalar": "SELECT duckdb_api()",
    "pragma_scalar": "SELECT pragma_version()",
    "values_only": "VALUES (1), (2)",
    "summarize": "SUMMARIZE coletas",
    "invalid": "SELEC * FORM coletas",
    "unbalanced": "SELECT (1",
    "empty": "",
    "blank": "   \n\t ",
    "only_comment": "-- nada aqui",
    "only_semicolon": ";",
}

ALLOWED = {
    "simple": "SELECT * FROM coletas",
    "trailing_semicolon": "SELECT COUNT(*) FROM coletas;",
    "trailing_semicolon_and_comment": "SELECT COUNT(*) FROM coletas; -- fim",
    "join": (
        "SELECT m.uf, AVG(c.valor_venda) FROM coletas c JOIN postos p ON c.posto_id = p.posto_id "
        "JOIN municipios m ON p.municipio_id = m.municipio_id GROUP BY m.uf"
    ),
    "cte": (
        "WITH media AS (SELECT produto, AVG(valor_venda) AS v FROM coletas GROUP BY produto) "
        "SELECT * FROM media ORDER BY v DESC"
    ),
    "nested_cte_names": (
        "WITH a AS (SELECT 1 AS x), b AS (SELECT x + 1 AS y FROM a) SELECT * FROM a, b"
    ),
    "window": (
        "SELECT produto, data_coleta, valor_venda, "
        "RANK() OVER (PARTITION BY produto ORDER BY valor_venda DESC) AS r FROM coletas"
    ),
    "window_lag": (
        "SELECT data_coleta, valor_venda - LAG(valor_venda) OVER (ORDER BY data_coleta) "
        "FROM coletas"
    ),
    "union": "SELECT uf FROM municipios UNION SELECT 'XX'",
    "subquery_in_from": "SELECT * FROM (SELECT produto FROM coletas) s",
    "string_drop_literal": "SELECT * FROM coletas WHERE produto = 'DROP'",
    "string_read_csv_literal": "SELECT 'read_csv(''x'')' AS texto FROM coletas",
    "string_with_semicolon": "SELECT * FROM municipios WHERE nome = 'A; DROP TABLE coletas'",
    "comment_inside": "SELECT 1 /* DROP TABLE coletas */ FROM coletas",
    "column_named_like_keyword": 'SELECT c.produto AS "delete" FROM coletas c',
    "range": "SELECT * FROM range(10)",
    "generate_series": (
        "SELECT * FROM generate_series(DATE '2026-01-01', DATE '2026-03-01', INTERVAL 1 MONTH)"
    ),
    "unnest": "SELECT * FROM unnest([1, 2, 3])",
    "from_first": "FROM coletas SELECT produto",
    "main_schema": "SELECT * FROM main.coletas",
    "quoted_table": 'SELECT * FROM "coletas"',
    "upper_case_table": "SELECT * FROM COLETAS",
    "lowercase_select": "select produto from coletas limit 5",
    "date_functions": (
        "SELECT date_trunc('month', data_coleta) AS mes, AVG(valor_venda) FROM coletas "
        "WHERE data_coleta >= DATE '2026-01-01' - INTERVAL 30 DAY GROUP BY 1 ORDER BY 1"
    ),
}


@pytest.mark.parametrize("sql", BLOCKED.values(), ids=BLOCKED.keys())
def test_blocks_malicious_or_invalid(sql: str) -> None:
    result = guard.check(sql)
    assert isinstance(result, GuardResult)
    assert result.ok is False
    assert result.reason  # sempre explica o motivo


@pytest.mark.parametrize("sql", ALLOWED.values(), ids=ALLOWED.keys())
def test_allows_valid_select(sql: str) -> None:
    result = guard.check(sql)
    assert result == GuardResult(ok=True, reason=None)


def test_has_at_least_twenty_blocked_cases() -> None:
    assert len(BLOCKED) >= 20


def test_reason_mentions_statement_kind() -> None:
    assert "DROP" in guard.check("DROP TABLE coletas").reason.upper()
    assert "instru" in guard.check("SELECT 1; SELECT 2").reason


def test_reason_names_blocked_function() -> None:
    assert "read_csv" in guard.check("SELECT * FROM read_csv('x.csv')").reason


def test_custom_allowed_tables() -> None:
    assert guard.check("SELECT * FROM brinquedo", allowed_tables={"brinquedo"}).ok
    assert not guard.check("SELECT * FROM coletas", allowed_tables={"brinquedo"}).ok


def test_cte_name_cannot_smuggle_file_outside_scope() -> None:
    # O nome da CTE é permitido, mas um arquivo com outro nome continua bloqueado.
    sql = "WITH x AS (SELECT 1) SELECT * FROM x, 'dados.csv'"
    assert not guard.check(sql).ok


def test_non_string_input_is_blocked() -> None:
    assert not guard.check(None).ok  # type: ignore[arg-type]
