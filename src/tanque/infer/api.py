"""Peças comuns às chamadas pagas da API da Anthropic (M4 e M7).

- `make_client()`: carrega `.env` (python-dotenv) e cria `anthropic.Anthropic()`.
  Todo código que chama a API recebe o cliente por parâmetro, para os testes
  usarem um cliente falso; nenhum teste chama a API.
- `split_messages()`: converte as mensagens de `prompting.build_prompt`
  (formato chat, com `role: system`) para o formato da API (`system` separado).
- `system_blocks()`: blocos de `system`, com `cache_control` no último quando o
  prefixo atinge o mínimo de tokens para cache.
- `usage_counts()`: contagens de `usage` na convenção de `tanque.budget`.
- `BatchRegistry`: registro append-only dos lotes enviados
  (`results/api-batches.jsonl`), que impede cobrar o mesmo lote duas vezes.

**Cache.** A documentação oficial
(https://platform.claude.com/docs/en/build-with-claude/prompt-caching, consultada
em 2026-10-28) diz que o prefixo mínimo para cache é de **512 tokens** para o
Claude Haiku 5.5 e o Claude Fable 5.1 (os modelos `cheap` e `frontier` do
pricing.toml). Abaixo disso o pedido roda sem cache e sem erro. Por isso o
código só marca `cache_control` quando o prefixo medido (pelo endpoint gratuito
`count_tokens`, ou por estimativa conservadora se ele falhar) passa do mínimo, e
a estimativa de custo considera a entrada sem cache nesse caso.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import time
from collections.abc import Callable, Iterable
from decimal import Decimal
from pathlib import Path
from typing import Any

from tanque import config
from tanque.budget import ModelPrice, cost_usd

# Mínimo de tokens do prefixo para o cache funcionar (ver docstring).
CACHE_MIN_TOKENS = 512
# Pela regra 5 do plano, execução paga acima disso exige aprovação da Isabela.
APPROVAL_LIMIT_USD = Decimal("0.50")
REGISTRY_RELPATH = Path("results") / "api-batches.jsonl"


class ApprovalRequiredError(RuntimeError):
    """A estimativa passa de US$ 0,50 e a execução não foi aprovada (`--approved`)."""


def load_env() -> None:
    """Carrega `.env` da raiz do repositório, se existir (não sobrescreve o ambiente)."""
    from dotenv import load_dotenv

    path = config.repo_root() / ".env"
    if path.is_file():
        load_dotenv(path, override=False)


class MissingApiKeyError(RuntimeError):
    """ANTHROPIC_API_KEY não está no ambiente nem no `.env`."""


def make_client() -> Any:
    """Cliente real da API. Usa ANTHROPIC_API_KEY do ambiente ou do `.env`."""
    load_env()
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise MissingApiKeyError(
            "ANTHROPIC_API_KEY não encontrada. Copie .env.example para .env na raiz do "
            "repositório e preencha a chave (o .env nunca vai para o git)."
        )
    import anthropic

    return anthropic.Anthropic()


def get(obj: Any, name: str, default: Any = None) -> Any:
    """Lê `name` de um objeto do SDK ou de um dict (os testes usam dicts)."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def split_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """(texto de system, mensagens sem system) a partir do formato chat."""
    system_parts = [str(m["content"]) for m in messages if m.get("role") == "system"]
    rest = [
        {"role": m["role"], "content": m["content"]} for m in messages if m.get("role") != "system"
    ]
    return "\n\n".join(system_parts), rest


def system_blocks(texts: Iterable[str], cache: bool) -> list[dict]:
    """Blocos de texto para `system`; `cache_control` só no último, se `cache`."""
    blocks = [{"type": "text", "text": t} for t in texts if t]
    if cache and blocks:
        blocks[-1]["cache_control"] = {"type": "ephemeral"}
    return blocks


def heuristic_tokens(text: str) -> int:
    """Estimativa conservadora (para cima) de tokens: 1 token a cada 3 caracteres."""
    return math.ceil(len(text) / 3)


def count_tokens(client: Any, model_id: str, system: Any, messages: list[dict]) -> int | None:
    """Tokens de entrada pelo endpoint `count_tokens` (gratuito); None se falhar."""
    counter = get(get(client, "messages"), "count_tokens")
    if counter is None:
        return None
    try:
        result = counter(model=model_id, system=system, messages=messages)
    except Exception:  # qualquer falha cai na estimativa local
        return None
    value = get(result, "input_tokens")
    return int(value) if value is not None else None


def should_cache(prefix_tokens: int, min_tokens: int = CACHE_MIN_TOKENS) -> bool:
    return prefix_tokens >= min_tokens


def usage_counts(usage: Any) -> dict[str, int]:
    """Contagens de tokens na convenção de `budget.charge`."""
    return {
        "input_tokens": int(get(usage, "input_tokens", 0) or 0),
        "output_tokens": int(get(usage, "output_tokens", 0) or 0),
        "cached_tokens": int(get(usage, "cache_read_input_tokens", 0) or 0),
        "cache_write_tokens": int(get(usage, "cache_creation_input_tokens", 0) or 0),
    }


def text_of(message: Any) -> str:
    """Concatena os blocos de texto da resposta (ignora blocos de raciocínio)."""
    parts = []
    for block in get(message, "content", []) or []:
        if get(block, "type") == "text":
            parts.append(str(get(block, "text", "")))
    return "".join(parts)


def estimate_cost(
    price: ModelPrice,
    n_requests: int,
    input_tokens_each: int,
    output_tokens_each: int,
    *,
    batch: bool,
    cached_prefix_tokens: int = 0,
) -> Decimal:
    """Estimativa do custo de `n_requests` requisições iguais.

    Com `cached_prefix_tokens > 0`, supõe o pior caso do cache em Batch (que é de
    melhor esforço): toda requisição grava o prefixo no cache (1,25x a entrada).
    """
    if n_requests <= 0:
        return Decimal(0)
    uncached = max(input_tokens_each - cached_prefix_tokens, 0)
    each = cost_usd(
        price,
        input_tokens=uncached,
        output_tokens=output_tokens_each,
        cache_write_tokens=cached_prefix_tokens,
        batch=batch,
    )
    return each * n_requests


def check_approval(estimate: Decimal, approved: bool) -> None:
    if estimate > APPROVAL_LIMIT_USD and not approved:
        raise ApprovalRequiredError(
            f"Estimativa de US$ {estimate:.4f} passa de US$ {APPROVAL_LIMIT_USD} (regra 5 do "
            "plano). Peça aprovação à Isabela e rode de novo com --approved."
        )


def wait_for_batch(
    client: Any,
    batch_id: str,
    *,
    poll_s: float = 30.0,
    timeout_s: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    echo: Callable[[str], None] = print,
) -> Any:
    """Consulta o lote até `processing_status == "ended"` (ou até `timeout_s`)."""
    start = time.monotonic()
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = get(batch, "processing_status")
        if status == "ended":
            return batch
        counts = get(batch, "request_counts")
        echo(
            f"Lote {batch_id}: {status} (processando: {get(counts, 'processing', '?')}, "
            f"ok: {get(counts, 'succeeded', '?')})"
        )
        if timeout_s is not None and time.monotonic() - start > timeout_s:
            return batch
        sleep(poll_s)


class BatchRegistry:
    """Eventos dos lotes enviados (`submitted`, `collected`), um JSON por linha."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = config.resolve_path(path or REGISTRY_RELPATH)

    def events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def append(self, event: str, batch_id: str, **fields: Any) -> dict[str, Any]:
        entry = {
            "ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "event": event,
            "batch_id": batch_id,
            **fields,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        return entry

    def submitted(self, batch_id: str) -> dict[str, Any] | None:
        for e in self.events():
            if e["event"] == "submitted" and e["batch_id"] == batch_id:
                return e
        return None

    def is_collected(self, batch_id: str) -> bool:
        return any(e["event"] == "collected" and e["batch_id"] == batch_id for e in self.events())

    def submitted_of_kind(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.events() if e["event"] == "submitted" and e.get("kind") == kind]
