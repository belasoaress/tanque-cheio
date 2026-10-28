"""Testes da execução segura (src/tanque/sql/execute.py, seção 5.5).

Usam o banco de brinquedo de tests/conftest.py, nunca data/tanque.duckdb.
"""

from decimal import Decimal
from pathlib import Path

import duckdb

from tanque import config
from tanque.sql import execute
from tanque.sql.execute import ExecResult

# Consulta cara de propósito: ~10^12 combinações, impossível de terminar no teste.
EXPENSIVE_SQL = "SELECT SUM(a.range * b.range) FROM range(1000000) a, range(1000000) b"


def test_runs_select(m2_toy_db: Path) -> None:
    result = execute.run("SELECT COUNT(*) AS n FROM coletas", db_path=m2_toy_db)
    assert isinstance(result, ExecResult)
    assert result.error is None
    assert result.blocked is False
    assert result.columns == ["n"]
    assert result.rows == [(7,)]
    assert result.elapsed_ms >= 0


def test_keeps_native_types(m2_toy_db: Path) -> None:
    result = execute.run(
        "SELECT valor_venda, data_coleta FROM coletas WHERE coleta_id = 1", db_path=m2_toy_db
    )
    assert result.error is None
    value, day = result.rows[0]
    assert value == Decimal("6.190")
    assert day.isoformat() == "2026-08-03"


def test_trailing_semicolon_and_comment(m2_toy_db: Path) -> None:
    result = execute.run("SELECT COUNT(*) FROM postos; -- fim", db_path=m2_toy_db)
    assert result.error is None
    assert result.rows == [(4,)]


def test_guard_blocks_before_running(m2_toy_db: Path) -> None:
    result = execute.run("DROP TABLE coletas", db_path=m2_toy_db)
    assert result.blocked is True
    assert result.error and "DROP" in result.error.upper()
    assert result.rows == [] and result.columns == []
    # A tabela continua lá.
    assert execute.run("SELECT COUNT(*) FROM coletas", db_path=m2_toy_db).rows == [(7,)]


def test_guard_uses_tables_of_the_database(tmp_path: Path) -> None:
    path = tmp_path / "outro.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE brinquedo AS SELECT 1 AS x")
    con.close()
    assert execute.run("SELECT x FROM brinquedo", db_path=path).rows == [(1,)]
    assert execute.run("SELECT * FROM coletas", db_path=path).blocked is True


def test_write_fails_even_without_guard(m2_toy_db: Path) -> None:
    for sql in (
        "INSERT INTO municipios VALUES (99, 'X', 'XX', 'N')",
        "DELETE FROM coletas",
        "CREATE TABLE x AS SELECT 1",
        "DROP TABLE coletas",
    ):
        result = execute.run(sql, db_path=m2_toy_db, check_guard=False)
        assert result.error is not None, sql
        assert "read-only" in result.error, result.error
        assert result.blocked is False
    assert execute.run("SELECT COUNT(*) FROM coletas", db_path=m2_toy_db).rows == [(7,)]
    assert execute.run("SELECT COUNT(*) FROM municipios", db_path=m2_toy_db).rows == [(3,)]


def test_external_access_disabled_even_without_guard(m2_toy_db: Path, tmp_path: Path) -> None:
    csv = tmp_path / "segredo.csv"
    csv.write_text("a\n1\n", encoding="utf-8")
    sql = f"SELECT * FROM read_csv('{csv.as_posix()}')"
    for query in (sql, f"SELECT * FROM '{csv.as_posix()}'"):
        result = execute.run(query, db_path=m2_toy_db, check_guard=False)
        assert result.error is not None
        assert "disabled by configuration" in result.error, result.error
        assert result.rows == []


def test_timeout_interrupts_expensive_query(m2_toy_db: Path) -> None:
    result = execute.run(EXPENSIVE_SQL, db_path=m2_toy_db, timeout_s=0.5)
    assert result.error is not None
    assert "timeout" in result.error.lower()
    # Margem folgada para runners lentos (Windows no CI).
    assert result.elapsed_ms < 30_000


def test_connection_usable_after_timeout(m2_toy_db: Path) -> None:
    execute.run(EXPENSIVE_SQL, db_path=m2_toy_db, timeout_s=0.3)
    assert execute.run("SELECT COUNT(*) FROM postos", db_path=m2_toy_db).rows == [(4,)]


def test_max_rows_applied_from_outside(m2_toy_db: Path) -> None:
    result = execute.run("SELECT * FROM range(5000)", db_path=m2_toy_db, max_rows=10)
    assert result.error is None
    assert len(result.rows) == 10


def test_smaller_inner_limit_is_kept(m2_toy_db: Path) -> None:
    result = execute.run("SELECT * FROM range(5000) LIMIT 3", db_path=m2_toy_db, max_rows=10)
    assert len(result.rows) == 3


def test_larger_inner_limit_is_capped(m2_toy_db: Path) -> None:
    result = execute.run("SELECT * FROM range(5000) LIMIT 4000", db_path=m2_toy_db, max_rows=10)
    assert len(result.rows) == 10


def test_order_by_preserved_when_wrapping(m2_toy_db: Path) -> None:
    sql = "SELECT range AS n FROM range(5000) ORDER BY n DESC"
    result = execute.run(sql, db_path=m2_toy_db, max_rows=5)
    assert result.error is None
    assert result.columns == ["n"]
    assert result.rows == [(4999,), (4998,), (4997,), (4996,), (4995,)]


def test_order_by_preserved_on_real_tables(m2_toy_db: Path) -> None:
    sql = "SELECT coleta_id FROM coletas ORDER BY valor_venda DESC, coleta_id"
    result = execute.run(sql, db_path=m2_toy_db, max_rows=3)
    assert result.rows == [(4,), (7,), (1,)]


def test_wrapping_survives_trailing_line_comment(m2_toy_db: Path) -> None:
    sql = "SELECT * FROM range(50) -- comentário sem quebra de linha"
    result = execute.run(sql, db_path=m2_toy_db, max_rows=5)
    assert result.error is None
    assert len(result.rows) == 5


def test_cte_and_window(m2_toy_db: Path) -> None:
    sql = (
        "WITH x AS (SELECT produto, valor_venda FROM coletas) "
        "SELECT produto, RANK() OVER (PARTITION BY produto ORDER BY valor_venda) FROM x"
    )
    result = execute.run(sql, db_path=m2_toy_db)
    assert result.error is None
    assert len(result.rows) == 7


def test_sql_error_is_reported(m2_toy_db: Path) -> None:
    result = execute.run("SELECT coluna_inexistente FROM coletas", db_path=m2_toy_db)
    assert result.blocked is False
    assert result.error is not None
    assert result.rows == []


def test_missing_database_is_reported(tmp_path: Path) -> None:
    result = execute.run("SELECT 1 FROM coletas", db_path=tmp_path / "nao_existe.duckdb")
    assert result.error is not None


def test_defaults_come_from_settings() -> None:
    settings = config.load_settings()
    assert execute.default_timeout_s() == settings["execution"]["timeout_s"]
    assert execute.default_max_rows() == settings["execution"]["max_rows"]
    assert execute.default_db_path() == config.setting_path("db_path")
