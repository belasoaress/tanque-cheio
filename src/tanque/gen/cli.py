"""Comandos `tanque gen`: geração e validação do dataset de treino sintético (M4).

Caminho padrão, sem API paga (D23):

    uv run tanque gen templates --n 10000 --seed 42  # templates -> pipeline -> train
    uv run tanque gen check-leakage
    uv run tanque gen stats

Caminho opcional pela Batch API:

    uv run tanque gen pilot               # 20 requisições; espera, cobra e mostra amostra
    uv run tanque gen submit --requests 480 # resto da geração (pede --approved se > US$ 0,50)
    uv run tanque gen collect <batch_id>    # quando o lote terminar
    uv run tanque gen validate              # bruto -> datasets/train.jsonl
    uv run tanque gen check-leakage         # 0 colisões com dev/teste
    uv run tanque gen stats                 # results/gen-stats.md

`templates` não chama a API: grava os candidatos crus em data/gen/templates_raw.jsonl
(fora do controle de versão) e roda o mesmo pipeline do `validate`, com deduplicação
interna por pergunta ou SQL idêntica. Só `pilot`, `submit` e `collect` falam com a
API (precisam de ANTHROPIC_API_KEY no ambiente ou no `.env`). Os outros trabalham
com os arquivos locais.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from tanque.budget import BudgetExceededError, Ledger
from tanque.gen import batch as batch_mod
from tanque.gen import leakage, pipeline
from tanque.gen import templates as templates_mod
from tanque.infer import api
from tanque.sql import execute

app = typer.Typer(
    help="Geração do dataset de treino sintético e checagem de vazamento (M4).",
    no_args_is_help=True,
)

# Ponto de injeção para os testes (cliente falso).
make_client = api.make_client

DbOption = Annotated[
    Path | None, typer.Option("--db", help="Banco DuckDB (padrão: [paths].db_path).")
]
EffortOption = Annotated[
    str, typer.Option(help="Esforço de raciocínio do gerador (output_config.effort).")
]
MaxTokensOption = Annotated[int, typer.Option(help="max_tokens por requisição.")]
EstOutOption = Annotated[
    int, typer.Option(help="Saída esperada por requisição (tokens), para a estimativa.")
]
ApprovedOption = Annotated[
    bool, typer.Option("--approved", help="A Isabela aprovou gastar mais de US$ 0,50.")
]


@app.callback()
def main() -> None:
    """Geração do dataset de treino sintético e checagem de vazamento (M4)."""


def _client() -> Any:
    try:
        return make_client()
    except api.MissingApiKeyError as e:
        typer.echo(f"Erro: {e}", err=True)
        raise typer.Exit(code=2) from e


def _require_db(db: Path | None) -> Path:
    path = db if db is not None else execute.default_db_path()
    if not path.is_file():
        typer.echo(
            f"Banco não encontrado: {path}. Rode `tanque data build` ou passe --db.", err=True
        )
        raise typer.Exit(code=2)
    return path


def _prepare(
    client: Any,
    ledger: Ledger,
    registry: api.BatchRegistry,
    n_requests: int,
    db: Path,
    max_tokens: int,
    effort: str,
    est_output_tokens: int,
) -> tuple[list[batch_mod.GenRequest], batch_mod.Estimate, int]:
    price = batch_mod.cheap_price(ledger)
    texts = batch_mod.prefix_texts(batch_mod.load_municipios(db), batch_mod.default_system_prompt())
    system, prefix_tokens, cache = batch_mod.prepare_system(client, price, texts)
    offset = batch_mod.next_offset(registry)
    requests = batch_mod.build_requests(
        n_requests,
        model_id=price.id,
        system=system,
        offset=offset,
        max_tokens=max_tokens,
        effort=effort or None,
    )
    est = batch_mod.estimate(
        price, requests, prefix_tokens, cache, est_output_tokens=est_output_tokens
    )
    typer.echo(
        f"Modelo: chave `{price.key}` ({price.name}); {len(requests)} requisições x "
        f"{batch_mod.PAIRS_PER_REQUEST} pares; numeradas a partir de {offset}."
    )
    cache_note = "marcado" if cache else f"não marcado (mínimo: {api.CACHE_MIN_TOKENS})"
    typer.echo(f"Prefixo estável: {prefix_tokens} tokens; cache {cache_note}.")
    typer.echo(
        f"Estimativa (Batch, -50%): esperada US$ {est.expected_usd:.4f} "
        f"(saída de {est_output_tokens} tokens/req.); pior caso US$ {est.worst_usd:.4f} "
        f"(saída = max_tokens = {max_tokens})."
    )
    return requests, est, offset


def _submit(
    n_requests: int,
    db: Path | None,
    max_tokens: int,
    effort: str,
    est_output_tokens: int,
    approved: bool,
    label: str,
) -> tuple[Any, Ledger, api.BatchRegistry, str]:
    db_path = _require_db(db)
    ledger = Ledger()
    registry = api.BatchRegistry()
    client = _client()
    requests, est, offset = _prepare(
        client, ledger, registry, n_requests, db_path, max_tokens, effort, est_output_tokens
    )
    try:
        batch_id = batch_mod.submit(
            client,
            requests,
            est,
            ledger=ledger,
            registry=registry,
            offset=offset,
            approved=approved,
            label=label,
        )
    except (api.ApprovalRequiredError, BudgetExceededError) as e:
        typer.echo(f"Não enviado: {e}", err=True)
        raise typer.Exit(code=1) from e
    typer.echo(f"Lote enviado: {batch_id} (registrado em {registry.path})")
    return client, ledger, registry, batch_id


def _print_sample(pairs: list[dict[str, Any]], n: int = 10) -> None:
    for i, p in enumerate(batch_mod.sample_pairs(pairs, n), start=1):
        typer.echo(f"\n[{i}] ({p.get('intent')}) {p.get('question')}")
        typer.echo(f"    {p.get('sql')}")


def _collect_and_report(client: Any, ledger: Ledger, registry: api.BatchRegistry, batch_id: str):
    result = batch_mod.collect(client, batch_id, ledger=ledger, registry=registry)
    if result.already_collected:
        typer.echo(f"Lote {batch_id} já tinha sido coletado; nada foi cobrado de novo.")
    typer.echo(
        f"Lote {batch_id}: {result.succeeded} requisições ok, {result.errored} com erro; "
        f"{len(result.pairs)} pares crus; custo real US$ {result.cost_usd:.4f}."
    )
    typer.echo(f"Saída bruta: {result.raw_path}")
    return result


@app.command()
def pilot(
    db: DbOption = None,
    requests: Annotated[
        int, typer.Option(help="Requisições do piloto.")
    ] = batch_mod.PILOT_REQUESTS,
    max_tokens: MaxTokensOption = batch_mod.DEFAULT_MAX_TOKENS,
    effort: EffortOption = batch_mod.DEFAULT_EFFORT,
    est_output_tokens: EstOutOption = batch_mod.DEFAULT_EST_OUTPUT_TOKENS,
    poll_s: Annotated[float, typer.Option(help="Intervalo entre consultas ao lote (s).")] = 30.0,
    approved: ApprovedOption = False,
) -> None:
    """Lote piloto: envia 20 requisições, espera, cobra e mostra custo real e 10 pares."""
    client, ledger, registry, batch_id = _submit(
        requests, db, max_tokens, effort, est_output_tokens, approved, "pilot"
    )
    typer.echo("Esperando o lote terminar (Ctrl+C para sair; depois rode `tanque gen collect`).")
    status_batch = api.wait_for_batch(client, batch_id, poll_s=poll_s, echo=typer.echo)
    if api.get(status_batch, "processing_status") != "ended":
        typer.echo(f"Lote ainda não terminou. Rode depois: uv run tanque gen collect {batch_id}")
        raise typer.Exit(code=1)
    result = _collect_and_report(client, ledger, registry, batch_id)
    if result.succeeded:
        per_request = result.cost_usd / result.succeeded
        typer.echo(
            f"Custo por requisição: US$ {per_request:.6f}; projeção para 500 requisições: "
            f"US$ {per_request * 500:.4f}"
        )
    typer.echo("\nAmostra de 10 pares (ainda sem validação):")
    _print_sample(result.pairs)
    typer.echo("\nPARE AQUI: mostre o custo e a amostra à Isabela antes de `tanque gen submit`.")


@app.command()
def submit(
    requests: Annotated[int, typer.Option("--requests", help="Número de requisições.")],
    db: DbOption = None,
    max_tokens: MaxTokensOption = batch_mod.DEFAULT_MAX_TOKENS,
    effort: EffortOption = batch_mod.DEFAULT_EFFORT,
    est_output_tokens: EstOutOption = batch_mod.DEFAULT_EST_OUTPUT_TOKENS,
    approved: ApprovedOption = False,
    wait: Annotated[bool, typer.Option(help="Esperar o lote e coletar em seguida.")] = False,
    poll_s: Annotated[float, typer.Option(help="Intervalo entre consultas ao lote (s).")] = 60.0,
) -> None:
    """Envia um lote de N requisições (10 pares cada) na distribuição alvo."""
    client, ledger, registry, batch_id = _submit(
        requests, db, max_tokens, effort, est_output_tokens, approved, "submit"
    )
    if wait:
        api.wait_for_batch(client, batch_id, poll_s=poll_s, echo=typer.echo)
        _collect_and_report(client, ledger, registry, batch_id)
    else:
        typer.echo(f"Quando terminar: uv run tanque gen collect {batch_id}")


@app.command()
def collect(
    batch_id: Annotated[str, typer.Argument(help="Id do lote (msgbatch_...).")],
    wait: Annotated[bool, typer.Option(help="Esperar o lote terminar.")] = False,
    poll_s: Annotated[float, typer.Option(help="Intervalo entre consultas ao lote (s).")] = 60.0,
) -> None:
    """Coleta um lote encerrado: cobra o uso real no ledger e grava a saída bruta."""
    ledger = Ledger()
    registry = api.BatchRegistry()
    if registry.submitted(batch_id) is None:
        typer.echo(f"Lote {batch_id} não está em {registry.path}.", err=True)
        raise typer.Exit(code=2)
    if registry.is_collected(batch_id):
        _collect_and_report(None, ledger, registry, batch_id)
        return
    client = _client()
    if wait:
        status_batch = api.wait_for_batch(client, batch_id, poll_s=poll_s, echo=typer.echo)
    else:
        status_batch = client.messages.batches.retrieve(batch_id)
    status = api.get(status_batch, "processing_status")
    if status != "ended":
        typer.echo(f"Lote {batch_id} ainda está em `{status}`. Tente de novo mais tarde.")
        raise typer.Exit(code=1)
    _collect_and_report(client, ledger, registry, batch_id)


@app.command()
def validate(
    db: DbOption = None,
    timeout_s: Annotated[float | None, typer.Option(help="Timeout por SQL (s).")] = None,
) -> None:
    """Valida os pares crus, remove vazamento e duplicatas e grava datasets/train.jsonl."""
    db_path = _require_db(db)
    try:
        result = pipeline.build_train(db_path=db_path, timeout_s=timeout_s)
    except leakage.MissingEvalSetError as e:
        typer.echo(f"Erro: {e}. O treino só é gravado depois da checagem de vazamento.", err=True)
        raise typer.Exit(code=2) from e
    typer.echo(
        f"Pares crus: {result.raw_pairs}; válidos na execução: {result.valid}; "
        f"no treino: {result.final}"
    )
    for reason, n in result.reasons.items():
        typer.echo(f"  descartados por {reason}: {n}")
    typer.echo(f"Treino: {result.train_path}\nResumo: {result.summary_path}")


@app.command()
def templates(
    n: Annotated[int, typer.Option("--n", help="Candidatos crus a gerar.")] = 10000,
    seed: Annotated[int, typer.Option(help="Seed do sorteio dos parâmetros.")] = 42,
    db: DbOption = None,
    timeout_s: Annotated[float | None, typer.Option(help="Timeout por SQL (s).")] = None,
) -> None:
    """Gera o treino por templates (D23, sem API) e roda validação, vazamento e stats."""
    db_path = _require_db(db)
    try:
        leakage.load_reference()
    except leakage.MissingEvalSetError as e:
        typer.echo(f"Erro: {e}. O treino só é gravado depois da checagem de vazamento.", err=True)
        raise typer.Exit(code=2) from e
    typer.echo(f"Gerando {n} candidatos por templates (seed {seed})...")
    gen = templates_mod.generate(n, seed=seed, db_path=db_path)
    raw = templates_mod.write_raw(gen)
    typer.echo(f"Candidatos crus: {len(gen.candidates)} -> {raw}")
    if gen.exhausted:
        typer.echo("Intenções esgotadas (cota repartida no nível): " + ", ".join(gen.exhausted))
    result = pipeline.build_train(
        db_path=db_path,
        timeout_s=timeout_s,
        pairs=gen.candidates,
        dedupe_mode="exact",
        check_repeated=False,
        extra_summary=gen.summary(),
    )
    typer.echo(f"Válidos na execução: {result.valid}; no treino: {result.final}")
    for reason, k in result.reasons.items():
        typer.echo(f"  descartados por {reason}: {k}")
    out, _ = pipeline.write_stats()
    typer.echo(f"Treino: {result.train_path}\nResumo: {result.summary_path}\nStats: {out}")


@app.command("check-leakage")
def check_leakage(
    train: Annotated[
        Path | None, typer.Option(help="Treino a checar (padrão: datasets/train.jsonl).")
    ] = None,
    threshold: Annotated[
        float, typer.Option(help="Limiar do token_set_ratio.")
    ] = leakage.THRESHOLD,
) -> None:
    """Conta colisões do treino com dev/teste. Sai com 0 só se não houver nenhuma."""
    path = train or pipeline.train_path()
    if not path.is_file():
        typer.echo(f"Treino não encontrado: {path}", err=True)
        raise typer.Exit(code=2)
    try:
        reference, files = leakage.load_reference()
    except leakage.MissingEvalSetError as e:
        typer.echo(f"Erro: {e}", err=True)
        raise typer.Exit(code=2) from e
    items = pipeline.read_jsonl(path)
    collisions = leakage.find_collisions(items, reference, threshold)
    names = ", ".join(f.name for f in files)
    typer.echo(f"{len(items)} pares de treino x {len(reference)} itens de avaliação ({names})")
    if not collisions:
        typer.echo("0 colisões")
        return
    typer.echo(f"{len(collisions)} colisões", err=True)
    for c in collisions[:20]:
        train_id = items[c.index].get("id", c.index)
        typer.echo(f"  {train_id} x {c.other_id}: {c.kind} ({c.score:.0f})", err=True)
    raise typer.Exit(code=1)


@app.command()
def stats() -> None:
    """Gera results/gen-stats.md (níveis, intenções, descartes e custo do M4)."""
    out, text = pipeline.write_stats()
    typer.echo(text)
    typer.echo(f"Gravado em {out}")
