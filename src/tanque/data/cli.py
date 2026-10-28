"""Comandos `tanque data`: download, normalização e build do banco DuckDB (M1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import duckdb
import typer

from tanque import config
from tanque.data.build import BuildPaths, run_pipeline
from tanque.data.capitals import CAPITALS
from tanque.data.download import DownloadError, download_all, local_files, sources_from_settings

app = typer.Typer(
    help="Dados da ANP: download, normalização e build do banco (M1).",
    no_args_is_help=True,
)

N_UFS = 27


@app.callback()
def main() -> None:
    """Dados da ANP: download, normalização e build do banco (M1)."""


def _default_paths(settings: dict) -> BuildPaths:
    return BuildPaths(
        parquet=config.setting_path("parquet_dir", settings) / "coletas.parquet",
        db=config.setting_path("db_path", settings),
        manifest=config.setting_path("manifest", settings),
        schema_doc=config.resolve_path(settings["data"].get("schema_doc", "docs/schema.md")),
    )


def _out_paths(out_dir: Path) -> BuildPaths:
    return BuildPaths(
        parquet=out_dir / "parquet" / "coletas.parquet",
        db=out_dir / "tanque.duckdb",
        manifest=out_dir / "manifest.json",
        schema_doc=out_dir / "schema.md",
    )


@app.command()
def download() -> None:
    """Baixa os CSVs de [[data.sources]] para data/raw/ (pula os que já existem)."""
    settings = config.load_settings()
    raw_dir = config.setting_path("raw_dir", settings)
    try:
        files = download_all(sources_from_settings(settings), raw_dir)
    except DownloadError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    total = sum(f.bytes for f in files)
    typer.echo(f"{len(files)} arquivo(s) em {raw_dir} ({total / 1e6:.1f} MB)")


@app.command()
def build(
    raw_dir: Annotated[
        Path | None,
        typer.Option(
            help="Usa os CSVs desta pasta, sem baixar nada (ex.: tests/fixtures). "
            "Sem a opção, baixa as fontes de settings.toml para data/raw/."
        ),
    ] = None,
    pattern: Annotated[str, typer.Option(help="Padrão dos CSVs em --raw-dir.")] = "*.csv",
    out_dir: Annotated[
        Path | None,
        typer.Option(
            help="Grava banco, Parquet, manifest.json e schema.md nesta pasta, em vez de "
            "data/ e docs/."
        ),
    ] = None,
    schema_doc: Annotated[
        Path | None, typer.Option(help="Caminho do schema.md (padrão: docs/schema.md).")
    ] = None,
) -> None:
    """Do zero: download -> normalize -> build -> manifest -> docs/schema.md."""
    settings = config.load_settings()
    try:
        if raw_dir is not None:
            raw_dir = raw_dir.resolve()
            files = local_files(raw_dir, pattern)
        else:
            raw_dir = config.setting_path("raw_dir", settings)
            files = download_all(sources_from_settings(settings), raw_dir)
    except DownloadError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    paths = _out_paths(out_dir.resolve()) if out_dir is not None else _default_paths(settings)
    if schema_doc is not None:
        paths = BuildPaths(paths.parquet, paths.db, paths.manifest, schema_doc.resolve())
    manifest = run_pipeline(files, raw_dir, paths, log=typer.echo)
    rows = manifest["rows"]
    typer.echo(
        f"ok: {rows['accepted']} coletas aceitas de {rows['read']} lidas "
        f"({rows['rejected_total']} rejeitadas); período "
        f"{manifest['period']['data_min']} a {manifest['period']['data_max']}"
    )


@app.command()
def stats(
    db: Annotated[Path | None, typer.Option(help="Banco (padrão: data/tanque.duckdb).")] = None,
    manifest: Annotated[
        Path | None, typer.Option(help="manifest.json (padrão: ao lado do banco ou data/).")
    ] = None,
) -> None:
    """Contagens, período, UFs, produtos e capitais do banco construído."""
    settings = config.load_settings()
    db = db.resolve() if db else config.setting_path("db_path", settings)
    if not db.is_file():
        typer.echo(f"Banco não encontrado: {db}. Rode `uv run tanque data build`.", err=True)
        raise typer.Exit(1)
    if manifest is None:
        beside = db.parent / "manifest.json"
        manifest = beside if beside.is_file() else config.setting_path("manifest", settings)
    con = duckdb.connect(str(db), read_only=True)
    try:
        for table in ("municipios", "postos", "coletas"):
            n = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            typer.echo(f"{table:<11} {n:>10}")
        dmin, dmax, months = con.execute(
            "SELECT min(data_coleta), max(data_coleta), "
            "count(DISTINCT strftime(data_coleta, '%Y-%m')) FROM coletas"
        ).fetchone()
        typer.echo(f"período     {dmin} a {dmax} ({months} meses); data de referência {dmax}")
        ufs_sql = "SELECT DISTINCT uf FROM municipios ORDER BY uf"
        ufs = [r[0] for r in con.execute(ufs_sql).fetchall()]
        typer.echo(f"UFs         {len(ufs)}: {' '.join(ufs)}")
        if len(ufs) < N_UFS:
            typer.echo(f"ALERTA: só {len(ufs)} de {N_UFS} UFs presentes.")
        typer.echo("produtos")
        for produto, unidade, n in con.execute(
            "SELECT produto, unidade, count(*) FROM coletas GROUP BY 1, 2 ORDER BY 1, 2"
        ).fetchall():
            typer.echo(f"  {produto:<20} {unidade:<11} {n:>10}")
        capitais = con.execute("SELECT count(*) FROM municipios WHERE capital").fetchone()[0]
        typer.echo(f"capitais    {capitais} de {len(CAPITALS)}")
        if capitais != len(CAPITALS):
            typer.echo(f"ALERTA: {capitais} capitais no banco; esperado {len(CAPITALS)}.")
        bad = con.execute(
            "SELECT count(*) FROM coletas WHERE valor_venda IS NULL OR valor_venda <= 0"
        ).fetchone()[0]
        typer.echo(f"valor_venda nulo ou <= 0: {bad}")
    finally:
        con.close()
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        rows = data["rows"]
        typer.echo(
            f"manifest    {len(data['raw_files'])} arquivo(s); {rows['read']} linhas lidas, "
            f"{rows['accepted']} aceitas, {rows['rejected_total']} rejeitadas"
        )
        for reason, n in rows["rejected"].items():
            typer.echo(f"  {reason:<26} {n:>10}")
