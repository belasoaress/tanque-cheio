"""Testes de src/tanque/eval/stats.py: IC bootstrap e McNemar (D29)."""

import math
import time

import pytest

from tanque.eval import stats


def _binary(k: int, n: int) -> list[int]:
    return [1] * k + [0] * (n - k)


# ---------- bootstrap ----------


def test_bootstrap_constant_samples_give_degenerate_interval() -> None:
    assert stats.bootstrap_ci(_binary(150, 150)) == (1.0, 1.0)
    assert stats.bootstrap_ci(_binary(0, 150)) == (0.0, 0.0)
    assert stats.bootstrap_ci([0.5]) == (0.5, 0.5)


def test_bootstrap_empty_sample() -> None:
    assert stats.bootstrap_ci([]) is None


@pytest.mark.parametrize(("k", "n"), [(114, 150), (18, 150), (25, 50), (75, 150)])
def test_bootstrap_matches_the_normal_approximation(k: int, n: int) -> None:
    # Com n grande, o IC percentil da proporção fica perto de p ± 1,96·√(p(1−p)/n).
    p = k / n
    half = 1.96 * math.sqrt(p * (1 - p) / n)
    lo, hi = stats.bootstrap_ci(_binary(k, n))
    assert lo <= p <= hi
    assert lo == pytest.approx(p - half, abs=0.012)
    assert hi == pytest.approx(p + half, abs=0.012)


def test_bootstrap_76_percent_of_150_is_about_seven_points() -> None:
    # D29: com 150 perguntas, a margem da EX é de cerca de ±7 p.p.
    lo, hi = stats.bootstrap_ci(_binary(114, 150))
    assert 0.12 < hi - lo < 0.15


def test_bootstrap_is_deterministic_and_seeded() -> None:
    values = [(i * 37) % 101 / 100 for i in range(60)]
    first = stats.bootstrap_ci(values)
    assert stats.bootstrap_ci(values) == first
    assert stats.bootstrap_ci(values, seed=7) != first
    # Valores fixos: mudam só se o algoritmo (ou a seed) mudar, o que mudaria o metrics.json.
    assert tuple(round(x, 4) for x in first) == (0.4197, 0.568)
    ex = stats.bootstrap_ci(_binary(114, 150))
    assert tuple(round(x, 4) for x in ex) == (0.6933, 0.8267)


def test_bootstrap_is_fast_enough_for_ci() -> None:
    start = time.perf_counter()
    stats.bootstrap_ci(_binary(114, 150))
    assert time.perf_counter() - start < 3.0


def test_percentile_interpolates_like_numpy() -> None:
    assert stats.percentile([1, 2, 3, 4], 50) == 2.5
    assert stats.percentile([10], 95) == 10
    assert stats.percentile([], 50) is None
    assert stats.percentile([0, 10], 2.5) == pytest.approx(0.25)


# ---------- McNemar ----------


@pytest.mark.parametrize(
    ("a_only", "b_only", "expected"),
    [
        (0, 0, 1.0),  # sem discordância
        (1, 0, 1.0),  # 2 · 1/2, limitado a 1
        (3, 3, 1.0),
        (0, 5, 2 / 32),  # 2 · (1/2)^5
        (2, 8, 2 * (1 + 10 + 45) / 1024),  # 2 · P[X ≤ 2], X ~ Bin(10, 1/2)
        (8, 2, 2 * (1 + 10 + 45) / 1024),  # simétrico
        (1, 10, 2 * (1 + 11) / 2048),
    ],
)
def test_mcnemar_exact_known_values(a_only: int, b_only: int, expected: float) -> None:
    assert stats.mcnemar_exact(a_only, b_only) == pytest.approx(expected)


def test_mcnemar_tiny_p_is_not_rounded_to_zero() -> None:
    p = stats.mcnemar_exact(0, 96)
    assert p == pytest.approx(2.0**-95)
    assert stats.round_sig(p) > 0


def test_mcnemar_rejects_negative_counts() -> None:
    with pytest.raises(ValueError):
        stats.mcnemar_exact(-1, 2)


def test_paired_comparison() -> None:
    a = [True, True, False, False, True, False]
    b = [True, False, True, True, True, True]
    out = stats.paired_comparison(a, b)
    assert out == {
        "n": 6,
        "a_only": 1,
        "b_only": 3,
        "delta_ex": round(5 / 6 - 3 / 6, 4),
        "mcnemar_p": 0.625,  # 2 · P[X ≤ 1], X ~ Bin(4, 1/2) = 2 · 5/16
    }


def test_paired_comparison_needs_aligned_vectors() -> None:
    with pytest.raises(ValueError):
        stats.paired_comparison([True], [True, False])


def test_round_sig() -> None:
    assert stats.round_sig(0.123456) == 0.1235
    assert stats.round_sig(2.524354896707238e-29) == 2.524e-29
    assert stats.round_sig(1.0) == 1.0
