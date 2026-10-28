"""Comandos `tanque gen`: geração e validação do dataset de treino sintético (M4)."""

import typer

app = typer.Typer(
    help="Geração do dataset de treino sintético e checagem de vazamento (M4).",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Geração do dataset de treino sintético e checagem de vazamento (M4)."""
