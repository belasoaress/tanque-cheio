"""Ledger e teto de gasto da API.

Toda chamada paga passa por `charge()`, que calcula o custo a partir de
configs/pricing.toml e acrescenta uma linha em results/spend.jsonl. Antes de
cada lote, `reserve()` recusa a execução se `gasto + estimativa > budget_usd`
(exatamente no teto é permitido).

Convenção de tokens (igual ao campo `usage` da API):

- `input_tokens`: entrada não cacheada;
- `cached_tokens`: entrada lida do cache (`cache_read_input_tokens`);
- `cache_write_tokens`: entrada gravada no cache (`cache_creation_input_tokens`),
  cobrada pelo preço de escrita de 5 minutos;
- `output_tokens`: saída (inclui tokens de raciocínio).

O desconto da Batch API se aplica a todos os componentes. Valores em dinheiro
usam `Decimal` para que somas como "exatamente no teto" sejam exatas.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from tanque import config

MTOK = Decimal(1_000_000)


class BudgetExceededError(RuntimeError):
    """A estimativa somada ao gasto acumulado passaria do teto."""


@dataclass(frozen=True)
class LongContextPrice:
    threshold_tokens: int
    input_per_mtok: Decimal
    output_per_mtok: Decimal
    cache_write_per_mtok: Decimal
    cache_read_per_mtok: Decimal


@dataclass(frozen=True)
class ModelPrice:
    key: str
    id: str
    name: str
    input_per_mtok: Decimal
    output_per_mtok: Decimal
    cache_write_per_mtok: Decimal
    cache_read_per_mtok: Decimal
    batch_discount: Decimal
    source_url: str
    retrieved_on: dt.date
    long_context: LongContextPrice | None = None


def _dec(value: Any) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _parse_model(key: str, raw: Mapping[str, Any]) -> ModelPrice:
    long_raw = raw.get("long_context")
    long_context = None
    if long_raw is not None:
        long_context = LongContextPrice(
            threshold_tokens=int(long_raw["threshold_tokens"]),
            input_per_mtok=_dec(long_raw["input_per_mtok"]),
            output_per_mtok=_dec(long_raw["output_per_mtok"]),
            cache_write_per_mtok=_dec(long_raw["cache_write_per_mtok"]),
            cache_read_per_mtok=_dec(long_raw["cache_read_per_mtok"]),
        )
    return ModelPrice(
        key=key,
        id=str(raw["id"]),
        name=str(raw.get("name", raw["id"])),
        input_per_mtok=_dec(raw["input_per_mtok"]),
        output_per_mtok=_dec(raw["output_per_mtok"]),
        cache_write_per_mtok=_dec(raw["cache_write_per_mtok"]),
        cache_read_per_mtok=_dec(raw["cache_read_per_mtok"]),
        batch_discount=_dec(raw["batch_discount"]),
        source_url=str(raw["source_url"]),
        retrieved_on=raw["retrieved_on"],
        long_context=long_context,
    )


def load_pricing(path: str | Path | None = None) -> dict[str, ModelPrice]:
    """Modelos de pricing.toml, indexados pela chave curta (`[models.<chave>]`)."""
    raw = config.load_pricing_toml(path)
    return {key: _parse_model(key, value) for key, value in raw.get("models", {}).items()}


def find_model(pricing: Mapping[str, ModelPrice], model: str) -> ModelPrice:
    """Busca pela chave curta ou pelo ID da API."""
    if model in pricing:
        return pricing[model]
    for price in pricing.values():
        if price.id == model:
            return price
    raise KeyError(f"Modelo {model!r} não está em pricing.toml")


def cost_usd(
    price: ModelPrice,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
    batch: bool = False,
) -> Decimal:
    """Custo em US$ de uma requisição."""
    counts = (input_tokens, output_tokens, cached_tokens, cache_write_tokens)
    if any(n < 0 for n in counts):
        raise ValueError("Contagens de tokens não podem ser negativas")

    rates: ModelPrice | LongContextPrice = price
    prompt_tokens = input_tokens + cached_tokens + cache_write_tokens
    if price.long_context and prompt_tokens > price.long_context.threshold_tokens:
        rates = price.long_context

    total = (
        input_tokens * rates.input_per_mtok
        + output_tokens * rates.output_per_mtok
        + cached_tokens * rates.cache_read_per_mtok
        + cache_write_tokens * rates.cache_write_per_mtok
    ) / MTOK
    if batch:
        total *= 1 - price.batch_discount
    return total


class Ledger:
    """Ledger de gasto em JSONL com teto. Caminhos injetáveis para teste."""

    def __init__(
        self,
        ledger_path: str | Path | None = None,
        pricing_path: str | Path | None = None,
        budget_usd: Decimal | float | str | None = None,
    ) -> None:
        self.ledger_path = (
            config.resolve_path(ledger_path)
            if ledger_path is not None
            else config.setting_path("spend_ledger")
        )
        self.pricing_path = pricing_path
        self.budget_usd = _dec(budget_usd) if budget_usd is not None else config.budget_usd()
        self._pricing: dict[str, ModelPrice] | None = None

    @property
    def pricing(self) -> dict[str, ModelPrice]:
        if self._pricing is None:
            self._pricing = load_pricing(self.pricing_path)
        return self._pricing

    def entries(self) -> list[dict[str, Any]]:
        if not self.ledger_path.exists():
            return []
        with self.ledger_path.open(encoding="utf-8") as f:
            return [json.loads(line, parse_float=Decimal) for line in f if line.strip()]

    def spent(self) -> Decimal:
        return sum((_dec(e["cost_usd"]) for e in self.entries()), Decimal(0))

    def spent_by_milestone(self) -> dict[str, tuple[int, Decimal]]:
        """{marco: (número de chamadas, gasto em US$)}, em ordem de nome."""
        totals: dict[str, tuple[int, Decimal]] = {}
        for entry in self.entries():
            milestone = str(entry.get("milestone") or "?")
            calls, cost = totals.get(milestone, (0, Decimal(0)))
            totals[milestone] = (calls + 1, cost + _dec(entry["cost_usd"]))
        return dict(sorted(totals.items()))

    def reserve(self, estimated_usd: Decimal | float | str) -> None:
        """Levanta `BudgetExceededError` se gasto + estimativa > teto."""
        estimate = _dec(estimated_usd)
        if estimate < 0:
            raise ValueError("Estimativa não pode ser negativa")
        spent = self.spent()
        if spent + estimate > self.budget_usd:
            raise BudgetExceededError(
                f"Gasto acumulado US$ {spent:.4f} + estimativa US$ {estimate:.4f} "
                f"passa do teto de US$ {self.budget_usd}"
            )

    def charge(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        batch: bool = False,
        *,
        milestone: str,
        cache_write_tokens: int = 0,
    ) -> Decimal:
        """Calcula o custo, acrescenta uma linha no ledger e devolve o custo."""
        if not milestone:
            raise ValueError("milestone é obrigatório (ex.: 'M4')")
        price = find_model(self.pricing, model)
        cost = cost_usd(
            price,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
            batch=batch,
        )
        entry = {
            "ts": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "milestone": milestone,
            "model": price.key,
            "model_id": price.id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "cache_write_tokens": cache_write_tokens,
            "batch": batch,
            "cost_usd": float(cost),
        }
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return cost


def charge(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    batch: bool = False,
    *,
    milestone: str,
    cache_write_tokens: int = 0,
    ledger_path: str | Path | None = None,
    pricing_path: str | Path | None = None,
) -> Decimal:
    """Atalho para `Ledger(...).charge(...)` com os caminhos padrão."""
    ledger = Ledger(ledger_path=ledger_path, pricing_path=pricing_path, budget_usd=0)
    return ledger.charge(
        model,
        input_tokens,
        output_tokens,
        cached_tokens,
        batch,
        milestone=milestone,
        cache_write_tokens=cache_write_tokens,
    )


def reserve(
    estimated_usd: Decimal | float | str,
    *,
    ledger_path: str | Path | None = None,
    budget_usd: Decimal | float | str | None = None,
) -> None:
    """Atalho para `Ledger(...).reserve(...)` com os caminhos padrão."""
    Ledger(ledger_path=ledger_path, budget_usd=budget_usd).reserve(estimated_usd)
