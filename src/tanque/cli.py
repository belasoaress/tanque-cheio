"""CLI raiz `tanque`. Só registra os sub-apps; cada marco edita o cli do seu pacote."""

import typer

from tanque.budget_cli import app as budget_app
from tanque.data.cli import app as data_app
from tanque.eval.cli import app as eval_app
from tanque.gen.cli import app as gen_app
from tanque.infer.cli import app as infer_app

app = typer.Typer(
    help="tanque-cheio: text-to-SQL em português sobre os preços de combustíveis da ANP.",
    no_args_is_help=True,
)
app.add_typer(data_app, name="data")
app.add_typer(eval_app, name="eval")
app.add_typer(gen_app, name="gen")
app.add_typer(infer_app, name="infer")
app.add_typer(budget_app, name="budget")
