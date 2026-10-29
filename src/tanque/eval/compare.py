"""Comparação de resultados de execução (seção 5.6 da v1 e 5.6 da v2).

`results_match(gold, pred, order_matters)` decide se a previsão acertou (EX rígida,
a métrica principal). `results_match_relaxed` é a EX tolerante (D28), descrita no fim.

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

EX tolerante (`results_match_relaxed`, 5.6 da v2, D28):

- Se `results_match` é verdadeiro, `results_match_relaxed` também é (é testado
  primeiro, inclusive com mais de 6 colunas).
- Senão, a previsão pode ter colunas **a mais**: procura uma escolha injetiva de
  colunas da previsão, uma para cada coluna do gold, em qualquer ordem, **a mesma
  para todas as linhas**. Só quando a previsão tem até 6 colunas
  (`MAX_RELAXED_COLUMNS`); com 7 ou mais, vale só a comparação rígida.
- Colunas a menos nunca casam. Duas colunas iguais no gold precisam de duas colunas
  distintas na previsão; colunas repetidas na previsão podem ser qualquer uma.
- Depois da projeção valem as mesmas regras: valores normalizados, mesmo número
  de linhas, sequência (`order_matters`) ou multiconjunto, erros e bloqueios falsos.
- Vazio × vazio casa quando o gold tem no máximo tantas colunas quanto a previsão.
- Para não testar as 720 escolhas de 6 colunas à toa, cada coluna do gold só é
  ligada a colunas da previsão com a mesma sequência de valores (`order_matters`)
  ou o mesmo multiconjunto (sem ordem), condição necessária para casar.
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
MAX_RELAXED_COLUMNS = 6

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


def _assignments(options: list[list[int]], used: frozenset[int] = frozenset()):
    """Escolhas injetivas: uma coluna da previsão (de `options[j]`) por coluna do gold."""
    if not options:
        yield ()
        return
    for k in options[0]:
        if k not in used:
            for rest in _assignments(options[1:], used | {k}):
                yield (k, *rest)


def results_match_relaxed(gold: ExecResult, pred: ExecResult, order_matters: bool) -> bool:
    """EX tolerante (D28): como `results_match`, mas a previsão pode ter colunas a mais."""
    if gold.error or pred.error or gold.blocked or pred.blocked:
        return False
    if results_match(gold, pred, order_matters):
        return True
    gold_width, pred_width = _width(gold), _width(pred)
    if gold_width > pred_width or pred_width > MAX_RELAXED_COLUMNS:
        return False
    if not gold.rows and not pred.rows:
        return True
    if len(gold.rows) != len(pred.rows):
        return False

    gold_rows = _canonical(gold.rows)
    pred_rows = _canonical(pred.rows)
    if any(len(r) != gold_width for r in gold_rows) or any(len(r) != pred_width for r in pred_rows):
        return False

    def column(rows: list[tuple[Any, ...]], j: int) -> tuple[Any, ...]:
        return tuple(r[j] for r in rows)

    def key(col: tuple[Any, ...]) -> Any:
        return col if order_matters else Counter(col)

    pred_keys = [key(column(pred_rows, k)) for k in range(pred_width)]
    options = []
    for j in range(gold_width):
        gold_key = key(column(gold_rows, j))
        compatible = [k for k in range(pred_width) if pred_keys[k] == gold_key]
        if not compatible:
            return False
        options.append(compatible)

    gold_bag = None if order_matters else Counter(gold_rows)
    for chosen in _assignments(options):
        candidate = [tuple(r[k] for k in chosen) for r in pred_rows]
        if order_matters:
            if candidate == gold_rows:
                return True
        elif Counter(candidate) == gold_bag:
            return True
    return False
