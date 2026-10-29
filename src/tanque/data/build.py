"""Build do banco DuckDB a partir do Parquet normalizado, e a pipeline completa.

IDs determinísticos: cada tabela é numerada por `row_number()` sobre uma ordenação
total (municípios por `(uf, nome)`, postos por `cnpj`, coletas por
`(data_coleta, cnpj, produto, valor_venda, valor_compra, unidade)`), então o mesmo
conjunto de linhas dá os mesmos IDs, qualquer que seja a ordem nos CSVs.

O banco e o Parquet são escritos em arquivo temporário e renomeados no fim; uma
falha no meio não deixa um banco pela metade no lugar do anterior.

Snapshot versionado (D25): `snapshot/coletas-2026-09.parquet` é o Parquet
normalizado do M1, commitado no repositório, e `snapshot/coletas-2026-09.json`
registra o SHA-256 dele e a contagem + MD5 de cada tabela do banco original.
`build_from_parquet` reconstrói o banco só a partir do Parquet (sem CSVs nem
download) e confere as duas coisas; é o que o CI usa.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

import duckdb

from tanque.data.capitals import CAPITALS
from tanque.data.download import RawFile
from tanque.data.normalize import NormalizeResult, normalize_files, write_parquet

TABLES: tuple[str, ...] = ("municipios", "postos", "coletas")
MANIFEST_VERSION = 1


def schema_sql() -> str:
    """DDL do contrato (src/tanque/data/schema.sql)."""
    return resources.files("tanque.data").joinpath("schema.sql").read_text(encoding="utf-8")


@dataclass(frozen=True)
class BuildPaths:
    parquet: Path
    db: Path
    manifest: Path
    schema_doc: Path


_BUILD_SQL = """
CREATE TEMP TABLE src AS SELECT * FROM read_parquet($parquet);

CREATE TEMP TABLE capitais (nome VARCHAR, uf VARCHAR);

-- Atributos do posto vêm da coleta mais recente; empates são desfeitos por todas as
-- colunas, para o resultado não depender da ordem de leitura.
CREATE TEMP TABLE posto_src AS
SELECT cnpj, razao_social, bandeira, bairro, cep, municipio, uf
FROM (
    SELECT *, row_number() OVER (
        PARTITION BY cnpj
        ORDER BY data_coleta DESC, razao_social DESC NULLS LAST, bandeira DESC,
                 bairro DESC NULLS LAST, cep DESC NULLS LAST, municipio DESC, uf DESC,
                 produto DESC, valor_venda DESC
    ) AS rn
    FROM src
)
WHERE rn = 1;

-- Região mais frequente do município no período (empate: menor sigla).
CREATE TEMP TABLE municipio_regiao AS
SELECT municipio, uf, regiao
FROM (
    SELECT municipio, uf, regiao, row_number() OVER (
        PARTITION BY municipio, uf ORDER BY count(*) DESC, regiao
    ) AS rn
    FROM src
    GROUP BY municipio, uf, regiao
)
WHERE rn = 1;
"""

_FILL_SQL = """
INSERT INTO municipios
SELECT row_number() OVER (ORDER BY m.uf, m.municipio) AS municipio_id,
       m.municipio, m.uf, r.regiao,
       EXISTS (SELECT 1 FROM capitais c WHERE c.nome = m.municipio AND c.uf = m.uf)
FROM (SELECT DISTINCT municipio, uf FROM posto_src) m
JOIN municipio_regiao r USING (municipio, uf)
ORDER BY municipio_id;

INSERT INTO postos
SELECT row_number() OVER (ORDER BY p.cnpj) AS posto_id,
       p.cnpj, p.razao_social, p.bandeira, p.bairro, p.cep, m.municipio_id
FROM posto_src p
JOIN municipios m ON m.nome = p.municipio AND m.uf = p.uf
ORDER BY posto_id;

INSERT INTO coletas
SELECT row_number() OVER (
           ORDER BY s.data_coleta, s.cnpj, s.produto, s.valor_venda,
                    s.valor_compra NULLS FIRST, s.unidade
       ) AS coleta_id,
       p.posto_id, s.produto, s.data_coleta, s.valor_venda, s.valor_compra, s.unidade
FROM src s
JOIN postos p USING (cnpj)
ORDER BY coleta_id;
"""


def build_database(parquet: Path, db_path: Path) -> dict[str, int]:
    """Cria `db_path` a partir do Parquet; devolve as contagens por tabela."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_name(db_path.name + ".tmp")
    for leftover in (tmp, tmp.with_name(tmp.name + ".wal")):
        leftover.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    try:
        con.execute(schema_sql())
        literal = "'" + str(Path(parquet).resolve()).replace("'", "''") + "'"
        con.execute(_BUILD_SQL.replace("$parquet", literal))
        con.executemany("INSERT INTO capitais VALUES (?, ?)", sorted(CAPITALS))
        con.execute(_FILL_SQL)
        counts = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in TABLES}
        con.execute("DROP TABLE src; DROP TABLE posto_src; DROP TABLE municipio_regiao;")
        con.execute("CHECKPOINT")
    finally:
        con.close()
    tmp.replace(db_path)
    return counts


def _db_summary(db_path: Path) -> dict[str, Any]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        dmin, dmax, months = con.execute(
            "SELECT min(data_coleta), max(data_coleta), "
            "count(DISTINCT strftime(data_coleta, '%Y-%m')) FROM coletas"
        ).fetchone()
        ufs_sql = "SELECT DISTINCT uf FROM municipios ORDER BY uf"
        ufs = [r[0] for r in con.execute(ufs_sql).fetchall()]
        produtos = dict(
            con.execute("SELECT produto, count(*) FROM coletas GROUP BY 1 ORDER BY 1").fetchall()
        )
        capitais = con.execute("SELECT count(*) FROM municipios WHERE capital").fetchone()[0]
        bad = con.execute("SELECT count(*) FROM coletas WHERE valor_venda <= 0").fetchone()[0]
    finally:
        con.close()
    return {
        "period": {
            "data_min": dmin.isoformat() if dmin else None,
            "data_max": dmax.isoformat() if dmax else None,
            "data_referencia": dmax.isoformat() if dmax else None,
            "months": months,
        },
        "ufs": ufs,
        "produtos": produtos,
        "capitais": capitais,
        "coletas_valor_venda_nao_positivo": bad,
    }


def make_manifest(
    raw_files: list[RawFile],
    result: NormalizeResult,
    counts: dict[str, int],
    db_path: Path,
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    reports = {f.name: f for f in result.files}
    files = []
    for raw in raw_files:
        entry = raw.manifest_entry()
        report = reports[raw.name]
        entry.update(
            encoding=report.encoding,
            rows_read=report.rows_read,
            rows_accepted=report.rows_accepted,
            rows_rejected=report.rejected,
        )
        files.append(entry)
    rejected = result.rejected
    summary = _db_summary(db_path)
    return {
        "manifest_version": MANIFEST_VERSION,
        "generated_at": generated_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "raw_files": files,
        "rows": {
            "read": result.rows_read,
            "accepted": result.frame.height,
            "rejected_total": sum(rejected.values()),
            "rejected": rejected,
        },
        "tables": counts,
        **summary,
    }


def write_manifest(manifest: dict[str, Any], path: Path) -> Path:
    """JSON com chaves ordenadas, UTF-8 e fim de linha LF em qualquer SO."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    tmp.replace(path)
    return path


def run_pipeline(
    raw_files: list[RawFile], raw_dir: Path, paths: BuildPaths, *, log=print
) -> dict[str, Any]:
    """normalize -> Parquet -> DuckDB -> manifest -> docs/schema.md."""
    from tanque.data.schema_doc import write_schema_doc

    raw_paths = [Path(raw_dir) / f.name for f in raw_files]
    log(f"[data] normalizando {len(raw_paths)} arquivo(s)")
    result = normalize_files(raw_paths)
    write_parquet(result.frame, paths.parquet)
    log(f"[data] {result.frame.height} coletas aceitas -> {paths.parquet}")
    counts = build_database(paths.parquet, paths.db)
    log(f"[data] banco -> {paths.db} {counts}")
    manifest = make_manifest(raw_files, result, counts, paths.db)
    write_manifest(manifest, paths.manifest)
    log(f"[data] manifest -> {paths.manifest}")
    write_schema_doc(paths.db, paths.schema_doc, manifest)
    log(f"[data] schema -> {paths.schema_doc}")
    return manifest


# --- snapshot versionado (D25) --------------------------------------------------

SNAPSHOT_META_VERSION = 1
NULL_MARK = "\\N"
_PRIMARY_KEYS = {"municipios": "municipio_id", "postos": "posto_id", "coletas": "coleta_id"}


class SnapshotError(RuntimeError):
    """O Parquet ou o banco reconstruído não bate com o registro do snapshot."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def table_fingerprints(db_path: Path) -> dict[str, dict[str, Any]]:
    """`{tabela: {rows, md5}}`: contagem e MD5 do conteúdo, na ordem da chave primária.

    O MD5 é de `string_agg(md5(linha), '' ORDER BY pk)`, em que `linha` junta as
    colunas (na ordem do schema) convertidas para texto, com `\\N` para NULL. Não
    depende da formatação de STRUCT do DuckDB, só do cast de cada tipo para VARCHAR.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        out: dict[str, dict[str, Any]] = {}
        for table in TABLES:
            cols = [
                r[0]
                for r in con.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = 'main' AND table_name = ? ORDER BY ordinal_position",
                    [table],
                ).fetchall()
            ]
            parts = ", ".join(f"coalesce(CAST({c} AS VARCHAR), '{NULL_MARK}')" for c in cols)
            rows, md5 = con.execute(
                f"SELECT count(*), md5(coalesce(string_agg(md5(concat_ws('|', {parts})), '' "
                f"ORDER BY {_PRIMARY_KEYS[table]}), '')) FROM {table}"
            ).fetchone()
            out[table] = {"rows": int(rows), "md5": md5}
    finally:
        con.close()
    return out


def snapshot_meta_path(parquet: Path) -> Path:
    """`snapshot/coletas-2026-09.parquet` -> `snapshot/coletas-2026-09.json`."""
    return Path(parquet).with_suffix(".json")


def make_snapshot_meta(parquet: Path, db_path: Path) -> dict[str, Any]:
    """Registro do snapshot: SHA-256 do Parquet e contagem + MD5 de cada tabela do banco."""
    parquet = Path(parquet)
    summary = _db_summary(Path(db_path))
    return {
        "snapshot_meta_version": SNAPSHOT_META_VERSION,
        "parquet": parquet.name,
        "bytes": parquet.stat().st_size,
        "sha256": sha256_file(parquet),
        "period": summary["period"],
        "tables": table_fingerprints(Path(db_path)),
    }


def read_snapshot_meta(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_from_parquet(
    parquet: Path, db_path: Path, meta: dict[str, Any] | None = None
) -> dict[str, dict[str, Any]]:
    """Constrói o banco a partir do Parquet do snapshot, sem CSVs nem download.

    Com `meta` (o JSON ao lado do Parquet), confere o SHA-256 do arquivo antes e
    a contagem e o MD5 de cada tabela depois; qualquer diferença levanta
    `SnapshotError` (e o banco reconstruído é apagado). Devolve as impressões
    digitais das tabelas.
    """
    parquet, db_path = Path(parquet), Path(db_path)
    if not parquet.is_file():
        raise SnapshotError(f"Parquet não encontrado: {parquet}")
    if meta is not None:
        actual = sha256_file(parquet)
        if actual != meta["sha256"]:
            raise SnapshotError(f"SHA-256 do Parquet diverge: {actual} (esperado {meta['sha256']})")
    build_database(parquet, db_path)
    prints = table_fingerprints(db_path)
    if meta is not None and prints != meta["tables"]:
        diff = {
            t: {"esperado": meta["tables"].get(t), "obtido": prints.get(t)}
            for t in TABLES
            if meta["tables"].get(t) != prints.get(t)
        }
        db_path.unlink(missing_ok=True)
        raise SnapshotError(f"banco reconstruído diverge do snapshot: {diff}")
    return prints
