"""Comandos `tanque eval`: SQLs de referência, métricas, relatório e bake-off (M2, M3, M5, M7)."""

from pathlib import Path
from typing import Annotated

import typer

from tanque.eval import bakeoff as bakeoff_mod
from tanque.eval import report as report_mod
from tanque.eval import score as score_mod
from tanque.sql import execute

app = typer.Typer(
    help="Avaliação: SQLs de referência, métricas e relatório (M2, M3, M7).",
    no_args_is_help=True,
)

DbOption = Annotated[
    Path | None, typer.Option("--db", help="Banco DuckDB (padrão: [paths].db_path).")
]


@app.callback()
def main() -> None:
    """Avaliação: SQLs de referência, métricas e relatório (M2, M3, M7)."""


def _require_db(db: Path | None) -> Path:
    path = db if db is not None else execute.default_db_path()
    if not path.is_file():
        typer.echo(
            f"Banco não encontrado: {path}. Rode `tanque data build` ou passe --db.", err=True
        )
        raise typer.Exit(code=2)
    return path


@app.command("verify-gold")
def verify_gold(
    dataset: Annotated[Path, typer.Argument(help="Dataset JSONL (seção 5.2).")],
    db: DbOption = None,
) -> None:
    """Executa toda SQL de referência; falha se alguma der erro, for bloqueada ou vier vazia."""
    db_path = _require_db(db)
    items = score_mod.read_jsonl(dataset)
    problems, _ = score_mod.check_gold(items, db_path=db_path)
    if problems:
        typer.echo(f"Falhas em {len(problems)} de {len(items)} SQLs de referência:", err=True)
        for p in problems:
            typer.echo(f"  {p.id}: {p.reason}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"OK: {len(items)} SQLs de referência executam e retornam linhas ({dataset}).")


@app.command("score")
def score(
    dataset: Annotated[Path, typer.Argument(help="Dataset JSONL (seção 5.2).")],
    predictions: Annotated[Path, typer.Argument(help="Previsões JSONL (seção 5.3).")],
    db: DbOption = None,
    metrics: Annotated[
        Path | None,
        typer.Option(help="Arquivo de métricas (padrão: results/metrics.json)."),
    ] = None,
) -> None:
    """Calcula EX e demais métricas e grava em metrics.json na chave do sistema."""
    db_path = _require_db(db)
    try:
        result = score_mod.score(dataset, predictions, db_path=db_path)
    except (score_mod.GoldError, score_mod.PredictionsError, ValueError) as e:
        typer.echo(f"Erro: {e}", err=True)
        raise typer.Exit(code=1) from e
    out = score_mod.write_metrics(metrics or score_mod.default_metrics_path(), result)
    m = result.metrics
    typer.echo(f"{result.system} / {result.split}: EX = {m['ex']:.4f} (n = {m['n']})")
    for level, values in m["ex_by_difficulty"].items():
        ex = "—" if values["ex"] is None else f"{values['ex']:.4f}"
        typer.echo(f"  {level:<8} EX = {ex} (n = {values['n']})")
    typer.echo(
        f"  SQL válida = {m['valid_sql_rate']:.4f}  bloqueada = {m['blocked_rate']:.4f}  "
        f"sem previsão = {m['n_missing']}"
    )
    typer.echo(f"Métricas gravadas em {out}")


@app.command("report")
def report(
    metrics: Annotated[
        Path | None,
        typer.Option(help="Arquivo de métricas (padrão: results/metrics.json)."),
    ] = None,
    out: Annotated[
        Path | None, typer.Option(help="Relatório Markdown (padrão: results/report.md).")
    ] = None,
) -> None:
    """Gera results/report.md com a tabela principal a partir de metrics.json."""
    path = report_mod.write_report(
        metrics or score_mod.default_metrics_path(), out or report_mod.default_report_path()
    )
    typer.echo(f"Relatório gravado em {path}")


@app.command("bakeoff")
def bakeoff(
    metrics: Annotated[
        Path | None,
        typer.Option(help="Arquivo de métricas (padrão: results/metrics.json)."),
    ] = None,
    out: Annotated[
        Path | None, typer.Option(help="Documento do bake-off (padrão: results/bakeoff.md).")
    ] = None,
) -> None:
    """Atualiza a tabela dos modelos base no dev em results/bakeoff.md (M5)."""
    path = bakeoff_mod.write_bakeoff(
        metrics or score_mod.default_metrics_path(), out or bakeoff_mod.default_bakeoff_path()
    )
    typer.echo(f"Tabela do bake-off gravada em {path}")
