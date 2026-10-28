"""Previsões de um modelo Claude sobre um dataset, no formato da seção 5.3 (M4/M7).

Usa o prompt único (`prompting.build_prompt`), o modelo de uma chave de
`configs/pricing.toml` (nunca um ID escrito no código) e o ledger de gasto:
`reserve()` com a estimativa antes de chamar e `charge()` com o `usage` real de
cada resposta.

Dois modos:

- **síncrono** (padrão): uma chamada por pergunta; `latency_ms` é o tempo da
  chamada medido no cliente (inclui rede). Bom para medir latência numa amostra
  (`--limit`).
- **batch** (`--batch`): Message Batches API, 50% mais barata. Não existe
  latência por item no batch: `latency_ms` fica `null` (o `tanque eval score`
  ignora valores nulos no p50/p95). Para o relatório, meça a latência com o modo
  síncrono numa amostra pequena.

Raciocínio: o modelo de fronteira raciocina sempre e cobra o raciocínio como
saída. O esforço vai em `output_config.effort` (padrão `low`) e `max_tokens`
limita a saída por pergunta. Rode primeiro com `--limit 10` para medir a saída
média e só depois o conjunto inteiro, passando a média medida em
`--est-output-tokens` para a estimativa.

`system` das previsões = `claude-<id>`, onde `<id>` é o ID da API sem o prefixo
`claude-` repetido (ex.: o ID `claude-x-1` vira o sistema `claude-x-1`).

Cache: o prefixo do prompt (instrução, DDL, valores) tem por volta de 450-600
tokens, perto do mínimo de 512 do modelo. O prefixo é medido com
`count_tokens`; só leva `cache_control` se passar do mínimo (ver
`tanque.infer.api`).
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from tanque import config
from tanque.budget import Ledger, ModelPrice, find_model
from tanque.infer import api

KIND = "infer"
DEFAULT_MILESTONE = "M7"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_EFFORT = "low"
DEFAULT_EST_OUTPUT_TOKENS = 800
HARDWARE_SYNC = "anthropic-api"
HARDWARE_BATCH = "anthropic-batch"

BuildPrompt = Callable[[str], list[dict]]
ExtractSql = Callable[[str], str]


def _default_build_prompt(question: str) -> list[dict]:
    from tanque import prompting

    return prompting.build_prompt(question)


def _default_extract_sql(raw: str) -> str:
    from tanque import prompting

    return prompting.extract_sql(raw)


def system_name(price: ModelPrice) -> str:
    return price.id if price.id.startswith("claude-") else f"claude-{price.id}"


def split_of(items: list[dict[str, Any]], dataset_path: Path) -> str:
    splits = {str(i.get("split")) for i in items if i.get("split")}
    return splits.pop() if len(splits) == 1 else dataset_path.stem


def default_output_path(system: str, split: str, limit: int | None) -> Path:
    suffix = f"-sample{limit}" if limit else ""
    return config.setting_path("predictions_dir") / f"{system}__{split}{suffix}.jsonl"


def _custom_id(item_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", item_id)[:64]


@dataclass
class Settings:
    price: ModelPrice
    milestone: str = DEFAULT_MILESTONE
    max_tokens: int = DEFAULT_MAX_TOKENS
    effort: str | None = DEFAULT_EFFORT
    build_prompt: BuildPrompt = _default_build_prompt
    extract_sql: ExtractSql = _default_extract_sql
    cache: bool = False

    @property
    def system(self) -> str:
        return system_name(self.price)


def build_params(question: str, s: Settings) -> dict[str, Any]:
    system_text, messages = api.split_messages(s.build_prompt(question))
    params: dict[str, Any] = {
        "model": s.price.id,
        "max_tokens": s.max_tokens,
        "system": api.system_blocks([system_text], cache=s.cache),
        "messages": messages,
    }
    if s.effort:
        params["output_config"] = {"effort": s.effort}
    return params


def measure_prefix(client: Any, s: Settings) -> int:
    """Tokens do prefixo estável (system + pergunta mínima)."""
    system_text, _ = api.split_messages(s.build_prompt("."))
    measured = api.count_tokens(
        client,
        s.price.id,
        api.system_blocks([system_text], cache=False),
        [{"role": "user", "content": "."}],
    )
    return measured if measured is not None else api.heuristic_tokens(system_text)


@dataclass
class Plan:
    n: int
    prefix_tokens: int
    input_tokens_each: int
    cache: bool
    expected_usd: Decimal
    worst_usd: Decimal


def plan_cost(
    items: list[dict[str, Any]],
    s: Settings,
    prefix_tokens: int,
    *,
    batch: bool,
    est_output_tokens: int = DEFAULT_EST_OUTPUT_TOKENS,
) -> Plan:
    longest_q = max((api.heuristic_tokens(i.get("question", "")) for i in items), default=0)
    input_each = prefix_tokens + longest_q + 8
    cached = prefix_tokens if s.cache else 0
    n = len(items)
    expected = api.estimate_cost(
        s.price, n, input_each, est_output_tokens, batch=batch, cached_prefix_tokens=cached
    )
    worst = api.estimate_cost(
        s.price, n, input_each, s.max_tokens, batch=batch, cached_prefix_tokens=cached
    )
    return Plan(n, prefix_tokens, input_each, s.cache, expected, worst)


def _prediction(
    item_id: str,
    s: Settings,
    message: Any,
    *,
    ledger: Ledger,
    batch: bool,
    latency_ms: int | None,
) -> dict[str, Any]:
    counts = api.usage_counts(api.get(message, "usage"))
    cost = ledger.charge(s.price.key, batch=batch, milestone=s.milestone, **counts)
    raw = api.text_of(message)
    stop = api.get(message, "stop_reason")
    error = None
    if stop == "refusal":
        error = "recusa do modelo (stop_reason=refusal)"
    elif stop == "max_tokens":
        error = "saída cortada em max_tokens"
    return {
        "id": item_id,
        "system": s.system,
        "sql": s.extract_sql(raw) if raw else "",
        "raw_output": raw,
        "latency_ms": latency_ms,
        "input_tokens": counts["input_tokens"]
        + counts["cached_tokens"]
        + counts["cache_write_tokens"],
        "output_tokens": counts["output_tokens"],
        "cost_usd": float(cost),
        "hardware": HARDWARE_BATCH if batch else HARDWARE_SYNC,
        "error": error,
    }


def _failed(item_id: str, s: Settings, error: str, *, batch: bool) -> dict[str, Any]:
    return {
        "id": item_id,
        "system": s.system,
        "sql": "",
        "raw_output": "",
        "latency_ms": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "hardware": HARDWARE_BATCH if batch else HARDWARE_SYNC,
        "error": error,
    }


def predict_sync(
    client: Any,
    items: list[dict[str, Any]],
    s: Settings,
    *,
    ledger: Ledger,
    clock: Callable[[], float] = time.perf_counter,
    echo: Callable[[str], None] = lambda _: None,
) -> list[dict[str, Any]]:
    out = []
    for n, item in enumerate(items, start=1):
        params = build_params(item["question"], s)
        start = clock()
        try:
            message = client.messages.create(**params)
        except Exception as e:  # erro de API vira previsão com erro, sem parar o lote
            out.append(_failed(item["id"], s, f"{type(e).__name__}: {e}", batch=False))
            continue
        latency = round((clock() - start) * 1000)
        out.append(
            _prediction(item["id"], s, message, ledger=ledger, batch=False, latency_ms=latency)
        )
        echo(f"[{n}/{len(items)}] {item['id']}: {latency} ms")
    return out


@dataclass
class BatchJob:
    batch_id: str
    id_by_custom: dict[str, str] = field(default_factory=dict)


def submit_batch(
    client: Any,
    items: list[dict[str, Any]],
    s: Settings,
    plan: Plan,
    *,
    registry: api.BatchRegistry,
    dataset: Path,
    out_path: Path,
) -> BatchJob:
    id_by_custom = {_custom_id(i["id"]): i["id"] for i in items}
    if len(id_by_custom) != len(items):
        raise ValueError("ids do dataset colidem depois de sanitizados para custom_id")
    batch = client.messages.batches.create(
        requests=[
            {"custom_id": _custom_id(i["id"]), "params": build_params(i["question"], s)}
            for i in items
        ]
    )
    batch_id = str(api.get(batch, "id"))
    registry.append(
        "submitted",
        batch_id,
        kind=KIND,
        milestone=s.milestone,
        model=s.price.key,
        system=s.system,
        n_requests=len(items),
        dataset=str(dataset),
        out_path=str(out_path),
        effort=s.effort,
        max_tokens=s.max_tokens,
        prefix_tokens=plan.prefix_tokens,
        cache=plan.cache,
        estimate_usd=float(plan.expected_usd),
        worst_case_usd=float(plan.worst_usd),
        ids=id_by_custom,
    )
    return BatchJob(batch_id, id_by_custom)


def collect_batch(
    client: Any,
    job: BatchJob,
    items: list[dict[str, Any]],
    s: Settings,
    *,
    ledger: Ledger,
    registry: api.BatchRegistry,
) -> list[dict[str, Any]]:
    """Previsões na ordem do dataset; cobra cada resultado uma única vez."""
    if registry.is_collected(job.batch_id):
        raise RuntimeError(f"lote {job.batch_id} já foi coletado (e cobrado)")
    by_id: dict[str, dict[str, Any]] = {}
    for result in client.messages.batches.results(job.batch_id):
        cid = str(api.get(result, "custom_id"))
        item_id = job.id_by_custom.get(cid, cid)
        res = api.get(result, "result")
        rtype = api.get(res, "type")
        if rtype == "succeeded":
            by_id[item_id] = _prediction(
                item_id, s, api.get(res, "message"), ledger=ledger, batch=True, latency_ms=None
            )
        else:
            by_id[item_id] = _failed(item_id, s, f"lote: {rtype}", batch=True)
    registry.append("collected", job.batch_id, kind=KIND, n_results=len(by_id))
    return [
        by_id.get(i["id"]) or _failed(i["id"], s, "sem resultado no lote", batch=True)
        for i in items
    ]


def write_predictions(path: Path, predictions: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return path


def read_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [p for p in predictions if not p.get("error")]
    out_tokens = [p["output_tokens"] for p in predictions if p.get("output_tokens")]
    lat = sorted(p["latency_ms"] for p in predictions if p.get("latency_ms") is not None)
    return {
        "n": len(predictions),
        "errors": len(predictions) - len(ok),
        "cost_usd": sum(float(p.get("cost_usd") or 0) for p in predictions),
        "mean_output_tokens": (sum(out_tokens) / len(out_tokens)) if out_tokens else 0,
        "max_output_tokens": max(out_tokens) if out_tokens else 0,
        "p50_latency_ms": lat[len(lat) // 2] if lat else None,
    }


def settings_for(model_key: str, ledger: Ledger, **kwargs: Any) -> Settings:
    return Settings(price=find_model(ledger.pricing, model_key), **kwargs)
