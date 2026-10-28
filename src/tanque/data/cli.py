"""Comandos `tanque data`: download, normalização e build do banco DuckDB (M1)."""

import typer

app = typer.Typer(
    help="Dados da ANP: download, normalização e build do banco (M1).",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Dados da ANP: download, normalização e build do banco (M1)."""
