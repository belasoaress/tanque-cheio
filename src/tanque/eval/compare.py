"""Comparação de resultados de execução (seção 5.6).

`results_match(gold, pred, order_matters)` decide se a previsão acertou.

Regras e decisões de borda (documentadas aqui e testadas em tests/test_compare.py):

- Nomes de coluna são ignorados; o **número** de colunas precisa bater.
- Qualquer erro (`error` preenchido) ou bloqueio pelo guardrail, no gold ou na
  previsão, é `False`.
- Números (`int`, `float`, `Decimal`) viram `Decimal` com 2 casas, e `Decimal` e
  `float` com o mesmo valor são iguais. Para não depender do ruído binário de
  float (`2.67499999999` vindo de um `AVG` em DOUBLE contra `2.675` em DECIMAL),
  o valor é primeiro arredondado a 9 casas (meio-para-par) e só então a 2 casas
  com **meio-para-cima** (`ROUND_HALF_UP`, ou seja, `0.125 -> 0.13` e
  `-0.125 -> -0.13`). `-0.0` e valores que arredondam para `-0.00` viram `0.00`.
- `bool` não é número: `True` não casa com `1`.
- `NaN` (float ou Decimal) casa com `NaN`; `None` (NULL) casa só com `None`.
- Strings: só `strip()` nas pontas; caixa e acentos importam; texto numérico
  (`'6.19'`) não casa com número.
- Datas: `date` casa com `datetime` à meia-noite (sem fuso), porque `date_trunc` e
  afins podem devolver TIMESTAMP ou DATE; fora isso, o tipo precisa bater
  (data não casa com a string `'2026-08-01'`).
- `order_matters=False`: compara multiconjuntos de linhas (duplicatas contam).
  `order_matters=True`: compara a sequência.
- Até 4 colunas, tenta todas as permutações de colunas da previsão (a mesma
  permutação para todas as linhas). Com 5 ou mais, só a ordem original.
- Vazio × vazio casa **desde que o número de colunas seja igual** (a regra do
  número de colunas vale também para resultados vazios). Na prática o gold
  nunca é vazio: `tanque eval verify-gold` recusa SQL de referência sem linhas.
"""

from __future__ import annotations

import datetime as dt
import decimal
import itertools
import math
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from tanque.sql.execute import ExecResult

MAX_PERMUTED_COLUMNS = 4

_PRE = Decimal("1e-9")
_CENTS = Decimal("0.01")
_ZERO = Decimal("0.00")
_CTX = decimal.Context(prec=60)


class _NaN:
    """Marcador único para NaN (NaN != NaN em Python, mas aqui casa)."""

    _instance: _NaN | None = None

    def __new__(cls) -> _NaN:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "NaN"


NAN = _NaN()


@dataclass(frozen=True)
class _Bool:
    """Booleano embrulhado para não casar com 0/1 (em Python, True == 1)."""

    value: bool


def _round_number(value: Decimal) -> Decimal | _NaN:
    if value.is_nan():
        return NAN
    if value.is_infinite():
        return value
    rounded = value.quantize(_PRE, rounding=decimal.ROUND_HALF_EVEN, context=_CTX)
    rounded = rounded.quantize(_CENTS, rounding=decimal.ROUND_HALF_UP, context=_CTX)
    return _ZERO if rounded.is_zero() else rounded


def normalize_value(value: Any) -> Any:
    """Forma canônica e hashable de um valor de célula."""
    if value is None:
        return None
    if isinstance(value, bool):
        return _Bool(value)
    if isinstance(value, Decimal):
        return _round_number(value)
    if isinstance(value, int):
        return _round_number(Decimal(value))
    if isinstance(value, float):
        if math.isnan(value):
            return NAN
        return _round_number(Decimal(repr(value)))
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dt.datetime):
        if value.tzinfo is None and value.time() == dt.time(0, 0):
            return value.date()
        return value
    if isinstance(value, (list, tuple)):
        return tuple(normalize_value(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((str(k), repr(normalize_value(v))) for k, v in value.items()))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _width(result: ExecResult) -> int:
    if result.columns:
        return len(result.columns)
    return len(result.rows[0]) if result.rows else 0


def _canonical(rows: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    return [tuple(normalize_value(v) for v in row) for row in rows]


def results_match(gold: ExecResult, pred: ExecResult, order_matters: bool) -> bool:
    """True se `pred` tem o mesmo resultado que `gold` pelas regras deste módulo."""
    if gold.error or pred.error or gold.blocked or pred.blocked:
        return False
    width = _width(gold)
    if width != _width(pred):
        return False
    if not gold.rows and not pred.rows:
        return True
    if len(gold.rows) != len(pred.rows):
        return False

    gold_rows = _canonical(gold.rows)
    pred_rows = _canonical(pred.rows)
    if any(len(r) != width for r in gold_rows) or any(len(r) != width for r in pred_rows):
        return False

    gold_bag = None if order_matters else Counter(gold_rows)
    identity = tuple(range(width))
    if width <= MAX_PERMUTED_COLUMNS:
        perms = itertools.permutations(identity)
    else:
        perms = iter([identity])

    for perm in perms:
        candidate = (
            pred_rows if perm == identity else [tuple(r[i] for i in perm) for r in pred_rows]
        )
        if order_matters:
            if candidate == gold_rows:
                return True
        elif Counter(candidate) == gold_bag:
            return True
    return False
