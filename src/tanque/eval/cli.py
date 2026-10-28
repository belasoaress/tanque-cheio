"""Comandos `tanque eval`: verificação das SQLs de referência, métricas e relatório (M2, M3, M7)."""

import typer

app = typer.Typer(
    help="Avaliação: SQLs de referência, métricas e relatório (M2, M3, M7).",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Avaliação: SQLs de referência, métricas e relatório (M2, M3, M7)."""
