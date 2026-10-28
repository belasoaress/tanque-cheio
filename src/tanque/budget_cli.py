"""Comandos `tanque budget`: gasto da API por marco, lido de results/spend.jsonl."""

from decimal import Decimal
from pathlib import Path
from typing import Annotated

import typer

from tanque.budget import Ledger

app = typer.Typer(
    help="Orçamento da API: gasto por marco e teto (results/spend.jsonl).",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Orçamento da API: gasto por marco e teto (results/spend.jsonl)."""


@app.command()
def status(
    ledger: Annotated[
        Path | None, typer.Option(help="Ledger JSONL (padrão: [paths].spend_ledger).")
    ] = None,
    budget_usd: Annotated[
        float | None, typer.Option(help="Teto em US$ (padrão: [budget].budget_usd).")
    ] = None,
) -> None:
    """Mostra o gasto por marco, o total e quanto resta do teto."""
    book = Ledger(
        ledger_path=ledger,
        budget_usd=Decimal(str(budget_usd)) if budget_usd is not None else None,
    )
    rows = book.spent_by_milestone()
    total = sum((cost for _, cost in rows.values()), Decimal(0))

    typer.echo(f"Ledger: {book.ledger_path}")
    typer.echo(f"{'Marco':<8}{'Chamadas':>10}{'US$':>12}")
    for milestone, (calls, cost) in rows.items():
        typer.echo(f"{milestone:<8}{calls:>10}{cost:>12.4f}")
    if not rows:
        typer.echo("(nenhum gasto registrado)")
    typer.echo(f"{'Total':<18}{total:>12.4f}")
    typer.echo(f"{'Teto':<18}{book.budget_usd:>12.2f}")
    typer.echo(f"{'Restante':<18}{book.budget_usd - total:>12.4f}")
