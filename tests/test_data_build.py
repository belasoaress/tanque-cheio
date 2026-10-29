"""Pipeline ponta a ponta sobre as fixtures (sem rede) e teste de contrato do schema.

Os testes marcados `real_snapshot` rodam só onde `data/tanque.duckdb` existe (não no CI).
"""

import json
import re
import shutil
from pathlib import Path

import duckdb
import pytest
from typer.testing import CliRunner

from tanque import config
from tanque.cli import app
from tanque.data.build import (
    make_snapshot_meta,
    read_snapshot_meta,
    schema_sql,
    sha256_file,
    snapshot_meta_path,
    table_fingerprints,
    write_manifest,
)
from tanque.data.capitals import CAPITALS

FIXTURES = Path(__file__).parent / "fixtures"
runner = CliRunner()


def _build(raw_dir: Path, out_dir: Path) -> dict:
    result = runner.invoke(
        app, ["data", "build", "--raw-dir", str(raw_dir), "--out-dir", str(out_dir)]
    )
    assert result.exit_code == 0, result.output
    return json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("build")
    _build(FIXTURES, out)
    return out


def _query(db: Path, sql: str) -> list[tuple]:
    con = duckdb.connect(str(db), read_only=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


COLUMNS_SQL = (
    "SELECT table_name, column_name, ordinal_position, data_type, is_nullable, "
    "numeric_precision, numeric_scale FROM information_schema.columns "
    "WHERE table_schema = 'main' ORDER BY table_name, ordinal_position"
)
CONSTRAINTS_SQL = (
    "SELECT table_name, constraint_type, constraint_column_names::VARCHAR "
    "FROM duckdb_constraints() WHERE schema_name = 'main' ORDER BY ALL"
)


def _contract(con: duckdb.DuckDBPyConnection) -> tuple[list, list, list]:
    tables = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' "
        "ORDER BY table_name"
    ).fetchall()
    return tables, con.execute(COLUMNS_SQL).fetchall(), con.execute(CONSTRAINTS_SQL).fetchall()


def assert_schema_matches_contract(db: Path) -> None:
    ref = duckdb.connect(":memory:")
    ref.execute(schema_sql())
    expected = _contract(ref)
    ref.close()
    con = duckdb.connect(str(db), read_only=True)
    try:
        actual = _contract(con)
    finally:
        con.close()
    assert actual == expected


def test_schema_sql_has_contract_tables_and_capital_column():
    con = duckdb.connect(":memory:")
    con.execute(schema_sql())
    cols = con.execute(
        "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
        "WHERE table_name = 'municipios' ORDER BY ordinal_position"
    ).fetchall()
    con.close()
    assert cols[-1] == ("capital", "BOOLEAN", "NO")
    assert [c[0] for c in cols] == ["municipio_id", "nome", "uf", "regiao", "capital"]


def test_build_outputs_exist(built):
    for name in ("tanque.duckdb", "manifest.json", "schema.md", "parquet/coletas.parquet"):
        assert (built / name).is_file(), name
    assert not list(built.rglob("*.tmp"))


def test_schema_contract(built):
    assert_schema_matches_contract(built / "tanque.duckdb")


def test_no_invalid_valor_venda(built):
    db = built / "tanque.duckdb"
    n = _query(db, "SELECT count(*) FROM coletas WHERE valor_venda IS NULL OR valor_venda <= 0")
    assert n == [(0,)]


def test_manifest_counts(built):
    m = json.loads((built / "manifest.json").read_text(encoding="utf-8"))
    assert m["rows"]["read"] == 113
    assert m["rows"]["accepted"] == 103
    assert m["rows"]["rejected"] == {
        "data_invalida": 2,
        "linha_duplicada": 1,
        "produto_fora_da_lista": 4,
        "valor_venda_nao_positivo": 2,
        "valor_venda_vazio": 1,
    }
    assert m["rows"]["read"] == m["rows"]["accepted"] + m["rows"]["rejected_total"]
    assert m["tables"] == {"coletas": 103, "municipios": 12, "postos": 12}
    assert m["period"]["data_min"] == "2025-01-06"
    assert m["period"]["data_referencia"] == m["period"]["data_max"] == "2025-03-24"
    for entry, src in zip(m["raw_files"], sorted(FIXTURES.glob("*.csv")), strict=True):
        assert entry["name"] == src.name
        assert entry["bytes"] == src.stat().st_size
        assert len(entry["sha256"]) == 64


def test_manifest_has_no_time_outside_generated_at(built):
    m = json.loads((built / "manifest.json").read_text(encoding="utf-8"))
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", m.pop("generated_at"))
    assert not re.search(r"\d{2}:\d{2}", json.dumps(m))


def test_manifest_json_is_sorted_and_lf(built):
    raw = (built / "manifest.json").read_bytes()
    assert b"\r\n" not in raw and raw.endswith(b"\n")
    data = json.loads(raw.decode("utf-8"))
    assert list(data) == sorted(data)


def test_bandeira_from_most_recent_coleta(built):
    rows = _query(
        built / "tanque.duckdb",
        "SELECT cnpj, bandeira FROM postos WHERE cnpj IN "
        "('11.111.111/0001-11', '44.444.444/0001-44', '55.555.555/0001-55') ORDER BY cnpj",
    )
    # IPIRANGA em jan e fev, RAIZEN em mar -> RAIZEN; vazio -> BRANCA.
    assert rows == [
        ("11.111.111/0001-11", "RAIZEN"),
        ("44.444.444/0001-44", "BRANCA"),
        ("55.555.555/0001-55", "BRANCA"),
    ]


def test_municipios_regiao_and_capitals(built):
    db = built / "tanque.duckdb"
    rows = _query(db, "SELECT nome, uf, regiao, capital FROM municipios ORDER BY municipio_id")
    assert {r[2] for r in rows} <= {"N", "NE", "CO", "SE", "S"}
    assert ("SAO PAULO", "SP", "SE", True) in rows
    assert ("SAO CARLOS", "SP", "SE", False) in rows
    in_fixtures = {(n, u) for n, u, _, _ in rows} & CAPITALS
    assert {(n, u) for n, u, _, c in rows if c} == in_fixtures
    assert len(in_fixtures) == 6
    # IDs seguem a ordem estável (uf, nome), começando em 1.
    ids = _query(db, "SELECT municipio_id, uf, nome FROM municipios ORDER BY municipio_id")
    assert [i for i, _, _ in ids] == list(range(1, len(ids) + 1))
    assert [(u, n) for _, u, n in ids] == sorted((u, n) for _, u, n in ids)


def test_foreign_keys_resolve(built):
    db = built / "tanque.duckdb"
    assert _query(
        db,
        "SELECT count(*) FROM coletas c LEFT JOIN postos p USING (posto_id) "
        "WHERE p.posto_id IS NULL",
    ) == [(0,)]
    assert _query(db, "SELECT count(DISTINCT produto) FROM coletas") == [(6,)]


def test_rebuild_is_deterministic(built, tmp_path):
    """Duas execuções (com os CSVs listados em outra pasta) dão o mesmo banco e manifest."""
    raw = tmp_path / "raw"
    raw.mkdir()
    for src in FIXTURES.glob("*.csv"):
        shutil.copy2(src, raw / src.name)
    out = tmp_path / "out"
    second = _build(raw, out)
    first = json.loads((built / "manifest.json").read_text(encoding="utf-8"))
    assert first.pop("generated_at") and second.pop("generated_at")
    assert first == second
    for table, key in (("municipios", 1), ("postos", 1), ("coletas", 1)):
        sql = f"SELECT * FROM {table} ORDER BY {key}"
        assert _query(built / "tanque.duckdb", sql) == _query(out / "tanque.duckdb", sql)
    assert (built / "schema.md").read_text(encoding="utf-8") == (out / "schema.md").read_text(
        encoding="utf-8"
    )


def test_schema_doc_content(built):
    text = (built / "schema.md").read_text(encoding="utf-8")
    assert "Data de referência:** `2025-03-24`" in text
    for produto in ("`GASOLINA`", "`GASOLINA ADITIVADA`", "`ETANOL`", "`DIESEL S10`", "`GNV`"):
        assert produto in text
    for regiao in ("`N`", "`NE`", "`CO`", "`SE`", "`S`"):
        assert regiao in text
    assert "`RAIZEN`" in text and "`BRANCA`" in text
    assert "capital" in text and "CREATE TABLE municipios" in text


def test_stats_command(built):
    result = runner.invoke(app, ["data", "stats", "--db", str(built / "tanque.duckdb")])
    assert result.exit_code == 0, result.output
    assert "coletas" in result.output and "103" in result.output
    assert "ALERTA: só 6 de 27 UFs" in result.output
    assert "2025-03-24" in result.output
    assert "GNV" in result.output


def test_build_with_empty_raw_dir_fails_clearly(tmp_path):
    result = runner.invoke(
        app, ["data", "build", "--raw-dir", str(tmp_path), "--out-dir", str(tmp_path / "o")]
    )
    assert result.exit_code == 1
    assert "Nenhum arquivo" in result.output


# --- Snapshot real (só local: o CI não baixa os dados) ---

REAL_DB = config.setting_path("db_path")
real_snapshot = pytest.mark.skipif(not REAL_DB.is_file(), reason="snapshot real ausente")


@real_snapshot
def test_real_snapshot_contract_and_coverage():
    assert_schema_matches_contract(REAL_DB)
    assert _query(REAL_DB, "SELECT count(DISTINCT uf) FROM municipios") == [(27,)]
    assert _query(
        REAL_DB, "SELECT count(DISTINCT strftime(data_coleta, '%Y-%m')) FROM coletas"
    ) == [(24,)]
    assert _query(
        REAL_DB, "SELECT count(*) FROM coletas WHERE valor_venda IS NULL OR valor_venda <= 0"
    ) == [(0,)]


@real_snapshot
def test_real_snapshot_has_exactly_the_27_capitals():
    rows = set(_query(REAL_DB, "SELECT nome, uf FROM municipios WHERE capital"))
    assert rows == CAPITALS
    every = set(_query(REAL_DB, "SELECT nome, uf FROM municipios"))
    assert CAPITALS <= every


# --- Snapshot versionado (D25): banco a partir do Parquet ---

SNAPSHOT = config.resolve_path("snapshot/coletas-2026-09.parquet")


def _snapshot_copy(built: Path, tmp_path: Path) -> Path:
    """Cópia do Parquet das fixtures com o registro (JSON) ao lado, como em snapshot/."""
    parquet = tmp_path / "snap" / "coletas-teste.parquet"
    parquet.parent.mkdir()
    shutil.copy2(built / "parquet" / "coletas.parquet", parquet)
    meta = make_snapshot_meta(parquet, built / "tanque.duckdb")
    write_manifest(meta, snapshot_meta_path(parquet))
    return parquet


def _from_parquet(parquet: Path, out: Path):
    return runner.invoke(
        app, ["data", "build", "--from-parquet", str(parquet), "--out-dir", str(out)]
    )


def test_build_from_parquet_gives_the_same_database(built, tmp_path):
    parquet = _snapshot_copy(built, tmp_path)
    assert snapshot_meta_path(parquet).name == "coletas-teste.json"
    out = tmp_path / "rebuilt"
    result = _from_parquet(parquet, out)
    assert result.exit_code == 0, result.output
    assert "ok: SHA-256 e tabelas iguais" in result.output
    original, rebuilt = built / "tanque.duckdb", out / "tanque.duckdb"
    assert table_fingerprints(rebuilt) == table_fingerprints(original)
    assert table_fingerprints(rebuilt)["coletas"]["rows"] == 103
    for table in ("municipios", "postos", "coletas"):
        sql = f"SELECT * FROM {table} ORDER BY 1"
        assert _query(rebuilt, sql) == _query(original, sql)
    assert_schema_matches_contract(rebuilt)
    assert not (out / "manifest.json").exists() and not (out / "schema.md").exists()


def test_build_from_parquet_rejects_a_changed_file(built, tmp_path):
    parquet = _snapshot_copy(built, tmp_path)
    with parquet.open("ab") as f:
        f.write(b"\0")
    result = _from_parquet(parquet, tmp_path / "o")
    assert result.exit_code == 1
    assert "SHA-256 do Parquet diverge" in result.output


def test_build_from_parquet_rejects_a_different_database(built, tmp_path):
    parquet = _snapshot_copy(built, tmp_path)
    meta_path = snapshot_meta_path(parquet)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["tables"]["postos"]["md5"] = "0" * 32
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    out = tmp_path / "o"
    result = _from_parquet(parquet, out)
    assert result.exit_code == 1
    assert "diverge do snapshot" in result.output and "postos" in result.output
    assert not (out / "tanque.duckdb").exists()  # não deixa um banco errado no lugar


def test_table_fingerprint_changes_with_one_value(built, tmp_path):
    db = tmp_path / "copia.duckdb"
    shutil.copy2(built / "tanque.duckdb", db)
    before = table_fingerprints(db)
    con = duckdb.connect(str(db))
    try:
        con.execute("UPDATE coletas SET valor_compra = 1.234 WHERE coleta_id = 7")
    finally:
        con.close()
    after = table_fingerprints(db)
    assert after["coletas"]["rows"] == before["coletas"]["rows"]
    assert after["coletas"]["md5"] != before["coletas"]["md5"]
    assert after["postos"] == before["postos"]


def test_committed_snapshot_matches_its_sha256():
    meta = read_snapshot_meta(snapshot_meta_path(SNAPSHOT))
    assert meta["parquet"] == SNAPSHOT.name
    assert sha256_file(SNAPSHOT) == meta["sha256"]
    assert SNAPSHOT.stat().st_size == meta["bytes"]
    assert meta["period"]["data_referencia"] == "2026-09-30"
    manifest = json.loads(config.setting_path("manifest").read_text(encoding="utf-8"))
    assert {t: v["rows"] for t, v in meta["tables"].items()} == manifest["tables"]
    assert meta["period"] == manifest["period"]


@real_snapshot
def test_real_database_matches_the_committed_snapshot():
    meta = read_snapshot_meta(snapshot_meta_path(SNAPSHOT))
    assert table_fingerprints(REAL_DB) == meta["tables"]
