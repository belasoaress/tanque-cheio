"""Estatística da avaliação (D29): IC bootstrap de 95% e teste de McNemar pareado.

- `bootstrap_ci(values)`: intervalo **percentil** da média de `values` (para a EX, um
  vetor de 0/1, um por pergunta). São `BOOTSTRAP_RESAMPLES = 10.000` reamostragens com
  reposição, do mesmo tamanho da amostra, com `random.Random(BOOTSTRAP_SEED = 42)`
  novo a cada chamada: o mesmo vetor dá sempre o mesmo intervalo, em qualquer
  plataforma (o Mersenne Twister e `Random.choices` são os mesmos do Python 3.11 ao
  3.13). Os limites são os percentis 2,5 e 97,5 das médias reamostradas, com
  interpolação linear (como o `numpy.percentile` padrão).
- `mcnemar_exact(a_only, b_only)`: p bilateral exato (binomial com p = 1/2) a partir
  dos pares discordantes: `a_only` = itens que só o sistema A acerta, `b_only` = só o
  B. `p = min(1, 2 · P[X ≤ min(a_only, b_only)])`, X ~ Binomial(a_only + b_only, 1/2);
  sem pares discordantes, `p = 1`.
- `paired_comparison(a, b)`: os dois vetores de acerto alinhados por pergunta viram
  `{n, a_only, b_only, delta_ex, mcnemar_p}`.

Sem numpy: em Python puro, 10.000 × 150 sorteios levam ~0,3 s por vetor, e o projeto
não precisa de outra dependência só para isso.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from fractions import Fraction

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42
CONFIDENCE = 0.95


def percentile(values: Iterable[float], q: float) -> float | None:
    """Percentil `q` (0–100) com interpolação linear entre vizinhos (como o numpy)."""
    data = sorted(values)
    if not data:
        return None
    pos = (len(data) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def bootstrap_ci(
    values: Sequence[float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = CONFIDENCE,
) -> tuple[float, float] | None:
    """IC percentil da média de `values` (None se a amostra for vazia)."""
    n = len(values)
    if n == 0:
        return None
    data = [float(v) for v in values]
    rng = random.Random(seed)
    choices = rng.choices
    means = [sum(choices(data, k=n)) / n for _ in range(resamples)]
    tail = (1 - confidence) / 2 * 100
    lo, hi = percentile(means, tail), percentile(means, 100 - tail)
    assert lo is not None and hi is not None
    return lo, hi


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """p bilateral exato do teste de McNemar (binomial) para os pares discordantes."""
    if a_only < 0 or b_only < 0:
        raise ValueError("contagens negativas")
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = Fraction(sum(math.comb(n, i) for i in range(k + 1)), 2**n)
    return float(min(Fraction(1), 2 * tail))


def paired_comparison(a: Sequence[bool], b: Sequence[bool]) -> dict[str, float | int]:
    """Compara dois sistemas nos mesmos itens (`a[i]` e `b[i]` = acertou o item i)."""
    if len(a) != len(b):
        raise ValueError(f"vetores de tamanhos diferentes: {len(a)} e {len(b)}")
    n = len(a)
    a_only = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    b_only = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    return {
        "n": n,
        "a_only": a_only,
        "b_only": b_only,
        "delta_ex": round((sum(map(bool, b)) - sum(map(bool, a))) / n, 4) if n else 0.0,
        "mcnemar_p": round_sig(mcnemar_exact(a_only, b_only)),
    }


def round_sig(value: float, digits: int = 4) -> float:
    """Arredonda a `digits` algarismos significativos (p muito pequeno não vira 0)."""
    return float(f"{value:.{digits}g}")
