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
DatasetsOption = Annotated[
    Path | None,
    typer.Option(help="Pasta com dev.jsonl e test.jsonl (padrão: [paths].datasets_dir)."),
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
    db: DbOption = None,
    predictions_dir: Annotated[
        Path | None,
        typer.Option(help="Previsões (padrão: a pasta predictions/ ao lado do metrics.json)."),
    ] = None,
    annotations: Annotated[
        Path | None,
        typer.Option(help="Anotações manuais dos erros (padrão: results/error-annotations.json)."),
    ] = None,
    datasets_dir: DatasetsOption = None,
    check: Annotated[
        bool,
        typer.Option(
            "--check", help="Não grava: falha se o relatório no disco difere do gerado agora."
        ),
    ] = False,
) -> None:
    """Gera results/report.md (tabelas, base → ajustado, erros do ajustado, nota de método)."""
    from tanque.budget import Ledger

    metrics_path = metrics or score_mod.default_metrics_path()
    out_path = out or report_mod.default_report_path()
    try:
        text = report_mod.render_from_files(
            metrics_path,
            predictions_dir=predictions_dir,
            dataset_dir=datasets_dir,
            db_path=db if db is not None else execute.default_db_path(),
            annotations_path=annotations or report_mod.default_annotations_path(),
            spent_usd=float(Ledger().spent()),
        )
    except (report_mod.ReportInputError, score_mod.GoldError, ValueError) as e:
        typer.echo(f"Erro: {e}", err=True)
        raise typer.Exit(code=2) from e
    if check:
        current = out_path.read_text(encoding="utf-8") if out_path.is_file() else ""
        if current != text:
            typer.echo(
                f"{out_path} está desatualizado: rode `uv run tanque eval report` e commite.",
                err=True,
            )
            raise typer.Exit(code=1)
        typer.echo(f"OK: {out_path} confere com o gerado agora.")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    typer.echo(f"Relatório gravado em {out_path}")


@app.command("check-metrics")
def check_metrics(
    db: DbOption = None,
    metrics: Annotated[
        Path | None,
        typer.Option(help="Arquivo de métricas (padrão: results/metrics.json)."),
    ] = None,
    predictions_dir: Annotated[
        Path | None,
        typer.Option(help="Previsões (padrão: results/predictions)."),
    ] = None,
    datasets_dir: DatasetsOption = None,
    write: Annotated[
        bool, typer.Option("--write", help="Regrava metrics.json com o recalculado.")
    ] = False,
) -> None:
    """Recalcula as métricas de todas as previsões commitadas; falha se divergirem."""
    from tanque import config

    db_path = _require_db(db)
    metrics_path = metrics or score_mod.default_metrics_path()
    pred_dir = predictions_dir or config.setting_path("predictions_dir")
    files = score_mod.committed_predictions(pred_dir)
    try:
        actual = score_mod.recompute_all(
            pred_dir, datasets_dir or config.setting_path("datasets_dir"), db_path=db_path
        )
    except (score_mod.GoldError, score_mod.PredictionsError, ValueError) as e:
        typer.echo(f"Erro: {e}", err=True)
        raise typer.Exit(code=1) from e
    if write:
        score_mod.write_all_metrics(metrics_path, actual)
        typer.echo(f"{len(files)} arquivo(s) recalculado(s) -> {metrics_path}")
        return
    diffs = score_mod.diff_metrics(score_mod.load_metrics(metrics_path), actual)
    if diffs:
        typer.echo(f"{metrics_path} diverge do recalculado ({len(diffs)} diferença(s)):", err=True)
        for d in diffs:
            typer.echo(f"  {d}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"OK: {len(files)} arquivo(s) de previsões; métricas iguais a {metrics_path}.")


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
