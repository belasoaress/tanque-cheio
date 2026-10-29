"""Testes do comparador de resultados (src/tanque/eval/compare.py, seção 5.6 v1 e v2)."""

import datetime as dt
from decimal import Decimal

import pytest

from tanque.eval.compare import normalize_value, results_match, results_match_relaxed
from tanque.sql.execute import ExecResult


def res(rows, columns=None, error=None, blocked=False) -> ExecResult:
    rows = [tuple(r) for r in rows]
    if columns is None:
        width = len(rows[0]) if rows else 1
        columns = [f"c{i}" for i in range(width)]
    return ExecResult(columns=list(columns), rows=rows, error=error, blocked=blocked)


# ---------- normalização de valores ----------


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (Decimal("6.190"), 6.19),
        (6.19, 6.1900001),
        (6, Decimal("6.00")),
        (6, 6.0),
        (Decimal("2.675"), 2.675),  # meio exato: arredonda para cima nos dois
        (2.674999999999, Decimal("2.675")),  # ruído de float não muda o resultado
        (-0.0, 0.0),
        (-0.001, 0),  # arredonda para -0.00, tratado como 0.00
        ("  SAO CARLOS ", "SAO CARLOS"),
        (None, None),
        (float("nan"), float("nan")),
        (dt.date(2026, 8, 1), dt.datetime(2026, 8, 1)),
        (True, True),
    ],
)
def test_values_equal(a, b) -> None:
    assert normalize_value(a) == normalize_value(b)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (6.19, 6.20),
        (Decimal("2.665"), Decimal("2.675")),
        (None, 0),
        (None, ""),
        (float("nan"), None),
        ("6.19", 6.19),  # texto não vira número
        ("sao carlos", "SAO CARLOS"),  # caixa importa
        (dt.date(2026, 8, 1), dt.datetime(2026, 8, 1, 12, 0)),
        (dt.date(2026, 8, 1), "2026-08-01"),
        (True, 1),
    ],
)
def test_values_differ(a, b) -> None:
    assert normalize_value(a) != normalize_value(b)


def test_rounding_is_half_up() -> None:
    assert normalize_value(Decimal("0.125")) == Decimal("0.13")
    assert normalize_value(Decimal("0.135")) == Decimal("0.14")
    assert normalize_value(Decimal("-0.125")) == Decimal("-0.13")


def test_nested_values() -> None:
    assert normalize_value([1.001, " a "]) == normalize_value((Decimal("1.00"), "a"))


# ---------- results_match ----------


def test_identical() -> None:
    a = res([("SP", 6.19), ("PE", 6.49)])
    assert results_match(a, a, order_matters=True)


def test_column_names_ignored() -> None:
    gold = res([("SP", 6.19)], columns=["uf", "media"])
    pred = res([("SP", 6.19)], columns=["estado", "avg(valor_venda)"])
    assert results_match(gold, pred, order_matters=False)


def test_floats_rounded_to_two_places() -> None:
    gold = res([("SP", Decimal("6.190"))])
    pred = res([("SP", 6.1904999)])
    assert results_match(gold, pred, order_matters=False)
    assert not results_match(gold, res([("SP", 6.1999)]), order_matters=False)


def test_strings_stripped() -> None:
    assert results_match(res([("SP ",)]), res([(" SP",)]), order_matters=False)


def test_order_ignored_when_not_required() -> None:
    gold = res([("SP", 1), ("PE", 2), ("PR", 3)])
    pred = res([("PR", 3), ("SP", 1), ("PE", 2)])
    assert results_match(gold, pred, order_matters=False)


def test_order_required() -> None:
    gold = res([("SP", 1), ("PE", 2)])
    pred = res([("PE", 2), ("SP", 1)])
    assert not results_match(gold, pred, order_matters=True)
    assert results_match(gold, res([("SP", 1), ("PE", 2)]), order_matters=True)


def test_multiset_counts_duplicates() -> None:
    gold = res([("SP",), ("SP",), ("PE",)])
    assert not results_match(gold, res([("SP",), ("PE",), ("PE",)]), order_matters=False)
    assert not results_match(gold, res([("SP",), ("PE",)]), order_matters=False)
    assert results_match(gold, res([("PE",), ("SP",), ("SP",)]), order_matters=False)


def test_different_row_count() -> None:
    assert not results_match(res([(1,), (2,)]), res([(1,)]), order_matters=False)


def test_different_column_count() -> None:
    gold = res([("SP", 6.19)])
    pred = res([("SP", 6.19, "extra")])
    assert not results_match(gold, pred, order_matters=False)
    assert not results_match(res([("SP",)]), res([("SP", 6.19)]), order_matters=False)


def test_permuted_columns_match() -> None:
    gold = res([("SP", 6.19, 10), ("PE", 6.49, 3)])
    pred = res([(10, "SP", 6.19), (3, "PE", 6.49)])
    assert results_match(gold, pred, order_matters=False)
    assert results_match(gold, pred, order_matters=True)


def test_permutation_must_be_consistent_across_rows() -> None:
    # Cada linha casa com alguma permutação, mas nenhuma serve para todas.
    gold = res([(1, 2), (3, 4)])
    pred = res([(2, 1), (3, 4)])
    assert not results_match(gold, pred, order_matters=True)
    assert not results_match(gold, pred, order_matters=False)


def test_permutation_up_to_four_columns_only() -> None:
    gold4 = res([(1, 2, 3, 4)])
    assert results_match(gold4, res([(4, 3, 2, 1)]), order_matters=False)
    gold5 = res([(1, 2, 3, 4, 5)])
    assert not results_match(gold5, res([(5, 4, 3, 2, 1)]), order_matters=False)
    assert results_match(gold5, res([(1, 2, 3, 4, 5)]), order_matters=False)


def test_permutation_with_order_required() -> None:
    gold = res([("SP", 1), ("PE", 2)])
    pred = res([(2, "PE"), (1, "SP")])
    assert not results_match(gold, pred, order_matters=True)
    assert results_match(gold, pred, order_matters=False)


def test_empty_vs_empty() -> None:
    assert results_match(res([], columns=["a"]), res([], columns=["b"]), order_matters=True)
    assert results_match(res([], columns=["a"]), res([], columns=["b"]), order_matters=False)


def test_empty_vs_empty_with_different_column_count() -> None:
    # Decisão documentada em compare.py: o número de colunas vale também para vazios.
    gold = res([], columns=["a"])
    pred = res([], columns=["a", "b"])
    assert not results_match(gold, pred, order_matters=False)


def test_empty_vs_non_empty() -> None:
    assert not results_match(res([], columns=["a"]), res([(1,)]), order_matters=False)
    assert not results_match(res([(1,)]), res([], columns=["a"]), order_matters=False)


def test_error_never_matches() -> None:
    gold = res([(1,)])
    assert not results_match(gold, res([], error="Binder Error"), order_matters=False)
    assert not results_match(gold, res([], error="bloqueada", blocked=True), order_matters=False)
    assert not results_match(res([], error="x"), res([], error="x"), order_matters=False)


def test_nulls_and_nan_in_rows() -> None:
    gold = res([(None, float("nan"))])
    assert results_match(gold, res([(None, float("nan"))]), order_matters=True)
    assert not results_match(gold, res([(None, None)]), order_matters=True)


def test_decimal_and_float_rows_match() -> None:
    gold = res([(Decimal("4.290"), dt.date(2026, 8, 3))])
    pred = res([(4.29, dt.date(2026, 8, 3))])
    assert results_match(gold, pred, order_matters=True)


def test_mixed_types_in_column_do_not_crash() -> None:
    gold = res([(1, "a"), (None, "b"), ("x", None)])
    pred = res([("x", None), (None, "b"), (1, "a")])
    assert results_match(gold, pred, order_matters=False)


# ---------- results_match_relaxed (EX tolerante, 5.6 v2, D28) ----------


def test_relaxed_accepts_extra_columns() -> None:
    gold = res([("SP",), ("PE",)], columns=["uf"])
    pred = res([("SP", 6.19), ("PE", 6.49)], columns=["uf", "media"])
    assert not results_match(gold, pred, order_matters=False)
    assert results_match_relaxed(gold, pred, order_matters=False)


def test_relaxed_extra_columns_in_any_position_and_permuted() -> None:
    gold = res([("SAO PAULO", "SP"), ("RECIFE", "PE")])
    pred = res([(6.19, "SP", 1, "SAO PAULO"), (6.49, "PE", 2, "RECIFE")])
    assert results_match_relaxed(gold, pred, order_matters=False)
    assert results_match_relaxed(gold, pred, order_matters=True)


def test_relaxed_rejects_missing_columns() -> None:
    gold = res([("SP", 6.19)])
    assert not results_match_relaxed(gold, res([("SP",)]), order_matters=False)
    assert not results_match_relaxed(gold, res([(6.19,)]), order_matters=False)


def test_relaxed_still_needs_same_rows() -> None:
    gold = res([("SP",), ("PE",)])
    # Uma linha a mais (ou a menos) continua errada, mesmo com as colunas certas.
    assert not results_match_relaxed(gold, res([("SP", 1), ("PE", 2), ("PR", 3)]), False)
    assert not results_match_relaxed(gold, res([("SP", 1)]), False)
    # Valores diferentes também.
    assert not results_match_relaxed(gold, res([("SP", 1), ("RJ", 2)]), False)


def test_relaxed_respects_order() -> None:
    gold = res([("SP",), ("PE",)])
    pred = res([("PE", 6.49), ("SP", 6.19)])
    assert not results_match_relaxed(gold, pred, order_matters=True)
    assert results_match_relaxed(gold, pred, order_matters=False)
    assert results_match_relaxed(gold, res([("SP", 6.19), ("PE", 6.49)]), order_matters=True)


def test_relaxed_same_permutation_for_all_rows() -> None:
    # A linha 1 casa pela coluna 0 e a linha 2 pela coluna 1, mas nenhuma coluna serve
    # para as duas linhas.
    gold = res([("SP",), ("PE",)])
    pred = res([("SP", "RJ"), ("MG", "PE")])
    assert not results_match_relaxed(gold, pred, order_matters=True)
    assert not results_match_relaxed(gold, pred, order_matters=False)


def test_relaxed_column_multisets_are_not_enough() -> None:
    # Cada coluna do gold tem o mesmo multiconjunto que uma coluna da previsão, mas os
    # pares (linha a linha) não batem.
    gold = res([("SP", 1), ("PE", 2)])
    pred = res([("SP", 2, "x"), ("PE", 1, "y")])
    assert not results_match_relaxed(gold, pred, order_matters=False)


def test_relaxed_up_to_six_prediction_columns() -> None:
    gold = res([("SP",)])
    six = res([(1, 2, 3, 4, 5, "SP")])
    seven = res([(1, 2, 3, 4, 5, 6, "SP")])
    assert results_match_relaxed(gold, six, order_matters=True)
    assert not results_match_relaxed(gold, seven, order_matters=True)


def test_relaxed_permutes_five_and_six_columns() -> None:
    # O rígido só permuta até 4 colunas; o tolerante procura entre até 6.
    gold = res([(1, 2, 3, 4, 5)])
    pred = res([(5, 4, 3, 2, 1, 0)])
    assert results_match_relaxed(gold, pred, order_matters=True)
    assert results_match_relaxed(res([(1, 2, 3, 4, 5, 6)]), res([(6, 5, 4, 3, 2, 1)]), True)


def test_relaxed_implied_by_rigid_even_beyond_six_columns() -> None:
    gold = res([(1, 2, 3, 4, 5, 6, 7)])
    assert results_match(gold, gold, order_matters=True)
    assert results_match_relaxed(gold, gold, order_matters=True)


def test_relaxed_duplicate_equal_columns() -> None:
    # Colunas repetidas na previsão: qualquer uma serve.
    gold = res([("SAO PAULO", "SP"), ("RECIFE", "PE")])
    pred = res([("SAO PAULO", "SAO PAULO", "SP"), ("RECIFE", "RECIFE", "PE")])
    assert results_match_relaxed(gold, pred, order_matters=True)
    # Duas colunas iguais no gold precisam de duas colunas distintas na previsão.
    gold2 = res([("SP", "SP"), ("PE", "PE")])
    assert not results_match_relaxed(gold2, res([("SP", 1), ("PE", 2)]), order_matters=False)
    assert results_match_relaxed(gold2, res([("SP", 1, "SP"), ("PE", 2, "PE")]), False)


def test_relaxed_empty_results() -> None:
    gold = res([], columns=["a"])
    assert results_match_relaxed(gold, res([], columns=["a", "b"]), order_matters=False)
    assert not results_match_relaxed(res([], columns=["a", "b"]), gold, order_matters=False)
    assert not results_match_relaxed(gold, res([(1, 2)]), order_matters=False)
    assert not results_match_relaxed(res([(1,)]), res([], columns=["a", "b"]), False)


def test_relaxed_errors_never_match() -> None:
    gold = res([(1,)])
    assert not results_match_relaxed(gold, res([], error="Binder Error"), order_matters=False)
    assert not results_match_relaxed(gold, res([], error="x", blocked=True), False)
    assert not results_match_relaxed(res([], error="x"), res([(1,)]), order_matters=False)


def test_relaxed_uses_same_value_rules() -> None:
    gold = res([("SP", Decimal("6.190"))])
    pred = res([(" SP ", 6.1904999, "extra")])
    assert results_match_relaxed(gold, pred, order_matters=True)
    assert not results_match_relaxed(gold, res([("SP", 6.1999, "extra")]), True)


@pytest.mark.parametrize(
    ("gold_rows", "pred_rows", "order_matters"),
    [
        ([("SP", 1), ("PE", 2)], [(2, "PE"), (1, "SP")], False),
        ([("SP", 1), ("PE", 2)], [("SP", 1), ("PE", 2)], True),
        ([(1, 2, 3, 4)], [(4, 3, 2, 1)], True),
        ([(None, float("nan"))], [(None, float("nan"))], True),
        ([("SP",), ("SP",), ("PE",)], [("PE",), ("SP",), ("SP",)], False),
    ],
)
def test_rigid_match_implies_relaxed(gold_rows, pred_rows, order_matters) -> None:
    gold, pred = res(gold_rows), res(pred_rows)
    assert results_match(gold, pred, order_matters)
    assert results_match_relaxed(gold, pred, order_matters)
