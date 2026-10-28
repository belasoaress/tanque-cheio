"""Comandos `tanque infer`: previsões dos sistemas avaliados (M7)."""

import typer

app = typer.Typer(
    help="Inferência: previsões dos sistemas avaliados (M7).",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Inferência: previsões dos sistemas avaliados (M7)."""
