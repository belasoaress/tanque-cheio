"""Comandos `tanque infer`: previsões dos sistemas avaliados (M4/M7).

    uv run tanque infer claude datasets/dev.jsonl --model frontier --limit 10
    uv run tanque infer claude datasets/test.jsonl --model frontier --batch --approved
    uv run --group infer tanque infer local datasets/dev.jsonl --model <repo-hf> --limit 2 \
        --device cpu --dtype float32 --out-dir <pasta>

Em `claude`, o modelo é sempre uma chave de configs/pricing.toml (`cheap`, `frontier`).
Em `local`, é um repo do Hugging Face; precisa do grupo opcional `infer` (torch,
transformers). No Kaggle, os notebooks chamam `tanque.infer.local` diretamente.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from tanque.budget import BudgetExceededError, Ledger
from tanque.infer import api
from tanque.infer import claude as claude_mod
from tanque.infer import local as local_mod

app = typer.Typer(
    help="Inferência: previsões dos sistemas avaliados (M7).",
    no_args_is_help=True,
)

# Ponto de injeção para os testes (cliente falso).
make_client = api.make_client


@app.callback()
def main() -> None:
    """Inferência: previsões dos sistemas avaliados (M7)."""


def _client() -> Any:
    try:
        return make_client()
    except api.MissingApiKeyError as e:
        typer.echo(f"Erro: {e}", err=True)
        raise typer.Exit(code=2) from e


def _report(predictions: list[dict], out: Path) -> None:
    s = claude_mod.summarize(predictions)
    typer.echo(
        f"{s['n']} previsões ({s['errors']} com erro) em {out}; "
        f"custo real US$ {s['cost_usd']:.4f}; "
        f"saída média {s['mean_output_tokens']:.0f} tokens (máx. {s['max_output_tokens']})"
        + (f"; latência p50 {s['p50_latency_ms']} ms" if s["p50_latency_ms"] is not None else "")
    )


@app.command()
def claude(
    dataset: Annotated[Path, typer.Argument(help="Dataset JSONL (seção 5.2).")],
    model: Annotated[str, typer.Option(help="Chave do modelo em configs/pricing.toml.")],
    batch: Annotated[bool, typer.Option("--batch", help="Usar a Message Batches API.")] = False,
    limit: Annotated[
        int | None, typer.Option(help="Só os N primeiros itens (amostra para medir).")
    ] = None,
    effort: Annotated[str, typer.Option(help="Esforço (output_config.effort); '' omite.")] = (
        claude_mod.DEFAULT_EFFORT
    ),
    max_tokens: Annotated[int, typer.Option(help="max_tokens por pergunta.")] = (
        claude_mod.DEFAULT_MAX_TOKENS
    ),
    est_output_tokens: Annotated[
        int, typer.Option(help="Saída esperada por pergunta (tokens), para a estimativa.")
    ] = claude_mod.DEFAULT_EST_OUTPUT_TOKENS,
    milestone: Annotated[str, typer.Option(help="Marco no ledger.")] = claude_mod.DEFAULT_MILESTONE,
    out: Annotated[Path | None, typer.Option(help="Arquivo de saída.")] = None,
    approved: Annotated[
        bool, typer.Option("--approved", help="A Isabela aprovou gastar mais de US$ 0,50.")
    ] = False,
    poll_s: Annotated[float, typer.Option(help="Intervalo entre consultas ao lote (s).")] = 30.0,
) -> None:
    """Gera previsões (seção 5.3) de um modelo Claude, com reserva e ledger de gasto."""
    if not dataset.is_file():
        typer.echo(f"Dataset não encontrado: {dataset}", err=True)
        raise typer.Exit(code=2)
    ledger = Ledger()
    try:
        settings = claude_mod.settings_for(
            model, ledger, milestone=milestone, max_tokens=max_tokens, effort=effort or None
        )
    except KeyError as e:
        typer.echo(f"Erro: {e}", err=True)
        raise typer.Exit(code=2) from e
    items = claude_mod.read_dataset(dataset)
    if limit:
        items = items[:limit]
    split = claude_mod.split_of(items, dataset)
    out_path = out or claude_mod.default_output_path(settings.system, split, limit)

    client = _client()
    prefix = claude_mod.measure_prefix(client, settings)
    settings.cache = api.should_cache(prefix)
    plan = claude_mod.plan_cost(
        items, settings, prefix, batch=batch, est_output_tokens=est_output_tokens
    )
    typer.echo(
        f"{settings.system}: {plan.n} perguntas ({'batch' if batch else 'síncrono'}); "
        f"prefixo {prefix} tokens, cache {'marcado' if plan.cache else 'não marcado'}; "
        f"estimativa US$ {plan.expected_usd:.4f} (saída {est_output_tokens} tokens/pergunta), "
        f"pior caso US$ {plan.worst_usd:.4f} (saída = max_tokens = {max_tokens})"
    )
    try:
        api.check_approval(plan.expected_usd, approved)
        ledger.reserve(plan.expected_usd)
    except (api.ApprovalRequiredError, BudgetExceededError) as e:
        typer.echo(f"Não executado: {e}", err=True)
        raise typer.Exit(code=1) from e

    registry = api.BatchRegistry()
    if not batch:
        predictions = claude_mod.predict_sync(
            client, items, settings, ledger=ledger, echo=typer.echo
        )
    else:
        job = claude_mod.submit_batch(
            client, items, settings, plan, registry=registry, dataset=dataset, out_path=out_path
        )
        typer.echo(
            f"Lote enviado: {job.batch_id}. Esperando (Ctrl+C e depois `tanque infer collect`)."
        )
        status = api.wait_for_batch(client, job.batch_id, poll_s=poll_s, echo=typer.echo)
        if api.get(status, "processing_status") != "ended":
            typer.echo(f"Lote não terminou; rode: uv run tanque infer collect {job.batch_id}")
            raise typer.Exit(code=1)
        predictions = claude_mod.collect_batch(
            client, job, items, settings, ledger=ledger, registry=registry
        )
    claude_mod.write_predictions(out_path, predictions)
    _report(predictions, out_path)


@app.command()
def collect(
    batch_id: Annotated[str, typer.Argument(help="Id do lote enviado por `infer claude --batch`.")],
) -> None:
    """Coleta um lote de inferência já encerrado e grava as previsões."""
    registry = api.BatchRegistry()
    entry = registry.submitted(batch_id)
    if entry is None or entry.get("kind") != claude_mod.KIND:
        typer.echo(f"Lote de inferência {batch_id} não está em {registry.path}.", err=True)
        raise typer.Exit(code=2)
    if registry.is_collected(batch_id):
        typer.echo(f"Lote {batch_id} já foi coletado; previsões em {entry['out_path']}.")
        return
    ledger = Ledger()
    settings = claude_mod.settings_for(
        entry["model"],
        ledger,
        milestone=entry["milestone"],
        max_tokens=int(entry["max_tokens"]),
        effort=entry.get("effort"),
    )
    dataset = Path(entry["dataset"])
    wanted = set(entry["ids"].values())
    items = [i for i in claude_mod.read_dataset(dataset) if i["id"] in wanted]
    client = _client()
    status = client.messages.batches.retrieve(batch_id)
    if api.get(status, "processing_status") != "ended":
        typer.echo(f"Lote {batch_id} ainda está em `{api.get(status, 'processing_status')}`.")
        raise typer.Exit(code=1)
    job = claude_mod.BatchJob(batch_id, dict(entry["ids"]))
    predictions = claude_mod.collect_batch(
        client, job, items, settings, ledger=ledger, registry=registry
    )
    out_path = Path(entry["out_path"])
    claude_mod.write_predictions(out_path, predictions)
    _report(predictions, out_path)


@app.command()
def local(
    dataset: Annotated[Path, typer.Argument(help="Dataset JSONL (seção 5.2).")],
    model: Annotated[str, typer.Option(help="Repo do modelo no Hugging Face.")],
    out_dir: Annotated[Path, typer.Option(help="Pasta de saída (<system>__<split>.jsonl).")],
    adapter: Annotated[str | None, typer.Option(help="Adapter LoRA (repo HF ou pasta).")] = None,
    revision: Annotated[str | None, typer.Option(help="Commit/branch do modelo.")] = None,
    system: Annotated[str | None, typer.Option(help="Nome do sistema (padrão: base-/ft-).")] = None,
    limit: Annotated[int | None, typer.Option(help="Só os N primeiros itens.")] = None,
    dtype: Annotated[str, typer.Option(help="float16, bfloat16 ou float32.")] = "float16",
    device: Annotated[str, typer.Option(help="cuda ou cpu.")] = "cuda",
    max_new_tokens: Annotated[int, typer.Option(help="Limite de tokens gerados.")] = (
        local_mod.DEFAULT_MAX_NEW_TOKENS
    ),
    no_thinking: Annotated[
        bool, typer.Option("--no-thinking", help="Passa enable_thinking=False ao chat template.")
    ] = False,
    system_suffix: Annotated[
        str, typer.Option(help="Sinal ao template no fim do system (ex.: /system_override).")
    ] = "",
) -> None:
    """Previsões (seção 5.3) de um modelo open-source com transformers (teste local; M5/M6)."""
    if not dataset.is_file():
        typer.echo(f"Dataset não encontrado: {dataset}", err=True)
        raise typer.Exit(code=2)
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        typer.echo(
            "Instale as dependências opcionais: uv sync --group infer (torch, transformers).",
            err=True,
        )
        raise typer.Exit(code=2) from e
    info = local_mod.run_predictions(
        dataset,
        model,
        out_dir=out_dir,
        adapter_id=adapter,
        revision=revision,
        system=system,
        limit=limit,
        dtype=dtype,
        device=device,
        max_new_tokens=max_new_tokens,
        chat_template_kwargs={"enable_thinking": False} if no_thinking else None,
        system_suffix=system_suffix,
        echo=typer.echo,
    )
    typer.echo(json.dumps(info, ensure_ascii=False, indent=2))
