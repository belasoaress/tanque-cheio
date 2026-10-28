"""Testes do ledger e do teto de gasto (src/tanque/budget.py).

Usa um pricing.toml de fixture com modelos fictícios: nenhum ID real de modelo
aparece fora de configs/pricing.toml.
"""

import json
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tanque import budget
from tanque.budget import BudgetExceededError, Ledger
from tanque.cli import app

FIXTURE_PRICING = """
[models.barato]
id = "modelo-ficticio-barato"
name = "Modelo Fictício Barato"
input_per_mtok = 1.00
output_per_mtok = 5.00
cache_write_per_mtok = 1.25
cache_read_per_mtok = 0.10
batch_discount = 0.5
source_url = "https://example.invalid/pricing"
retrieved_on = 2026-10-28

[models.caro]
id = "modelo-ficticio-caro"
name = "Modelo Fictício Caro"
input_per_mtok = 10.00
output_per_mtok = 50.00
cache_write_per_mtok = 12.50
cache_read_per_mtok = 0.25
batch_discount = 0.5
source_url = "https://example.invalid/pricing"
retrieved_on = 2026-10-28

[models.escalonado]
id = "modelo-ficticio-escalonado"
name = "Modelo Fictício com preço por tamanho de prompt"
input_per_mtok = 1.00
output_per_mtok = 5.00
cache_write_per_mtok = 1.25
cache_read_per_mtok = 0.10
batch_discount = 0.5
source_url = "https://example.invalid/pricing"
retrieved_on = 2026-10-28

[models.escalonado.long_context]
threshold_tokens = 100000
input_per_mtok = 2.00
output_per_mtok = 10.00
cache_write_per_mtok = 2.50
cache_read_per_mtok = 0.20
"""

MTOK = 1_000_000


@pytest.fixture
def pricing_path(tmp_path: Path) -> Path:
    path = tmp_path / "pricing.toml"
    path.write_text(FIXTURE_PRICING, encoding="utf-8")
    return path


@pytest.fixture
def ledger(tmp_path: Path, pricing_path: Path) -> Ledger:
    return Ledger(
        ledger_path=tmp_path / "results" / "spend.jsonl",
        pricing_path=pricing_path,
        budget_usd=Decimal("10"),
    )


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --- cálculo de custo -------------------------------------------------------


def test_charge_computes_cost_from_pricing_and_appends_line(ledger: Ledger) -> None:
    cost = ledger.charge("barato", input_tokens=MTOK, output_tokens=MTOK, milestone="M4")

    assert cost == Decimal("6.00")  # 1.00 entrada + 5.00 saída
    lines = read_lines(ledger.ledger_path)
    assert len(lines) == 1
    line = lines[0]
    assert line["milestone"] == "M4"
    assert line["model"] == "barato"
    assert line["model_id"] == "modelo-ficticio-barato"
    assert line["input_tokens"] == MTOK
    assert line["output_tokens"] == MTOK
    assert line["batch"] is False
    assert Decimal(str(line["cost_usd"])) == Decimal("6.00")
    assert "ts" in line


def test_charge_accepts_api_id_as_model(ledger: Ledger) -> None:
    cost = ledger.charge("modelo-ficticio-caro", input_tokens=MTOK, output_tokens=0, milestone="M7")
    assert cost == Decimal("10.00")
    assert read_lines(ledger.ledger_path)[0]["model"] == "caro"


def test_charge_unknown_model_raises(ledger: Ledger) -> None:
    with pytest.raises(KeyError):
        ledger.charge("inexistente", input_tokens=1, output_tokens=1, milestone="M4")
    assert not ledger.ledger_path.exists()


def test_charge_appends_instead_of_overwriting(ledger: Ledger) -> None:
    ledger.charge("barato", input_tokens=MTOK, output_tokens=0, milestone="M4")
    ledger.charge("caro", input_tokens=0, output_tokens=MTOK, milestone="M7")

    assert len(read_lines(ledger.ledger_path)) == 2
    assert ledger.spent() == Decimal("51.00")


def test_batch_applies_discount_to_every_component(ledger: Ledger) -> None:
    cost = ledger.charge(
        "barato",
        input_tokens=MTOK,
        output_tokens=MTOK,
        cached_tokens=MTOK,
        cache_write_tokens=MTOK,
        batch=True,
        milestone="M4",
    )
    # (1.00 + 5.00 + 0.10 + 1.25) * 0.5
    assert cost == Decimal("3.675")
    assert read_lines(ledger.ledger_path)[0]["batch"] is True


def test_cache_read_tokens_use_cache_read_price(ledger: Ledger) -> None:
    cost = ledger.charge(
        "barato", input_tokens=0, output_tokens=0, cached_tokens=MTOK, milestone="M4"
    )
    assert cost == Decimal("0.10")
    assert read_lines(ledger.ledger_path)[0]["cached_tokens"] == MTOK


def test_cache_write_tokens_use_cache_write_price(ledger: Ledger) -> None:
    cost = ledger.charge(
        "barato", input_tokens=0, output_tokens=0, cache_write_tokens=MTOK, milestone="M4"
    )
    assert cost == Decimal("1.25")
    assert read_lines(ledger.ledger_path)[0]["cache_write_tokens"] == MTOK


def test_cache_mixed_with_fresh_input(ledger: Ledger) -> None:
    # 500 requisições: 200 tokens novos + 1.800 lidos do cache + 1.500 de saída
    cost = Decimal(0)
    for _ in range(500):
        cost += ledger.charge(
            "barato", input_tokens=200, output_tokens=1500, cached_tokens=1800, milestone="M4"
        )
    expected = (
        Decimal(500 * 200) * Decimal("1.00")
        + Decimal(500 * 1800) * Decimal("0.10")
        + Decimal(500 * 1500) * Decimal("5.00")
    ) / MTOK
    assert cost == expected
    assert ledger.spent() == expected


def test_long_context_prices_apply_above_threshold(ledger: Ledger) -> None:
    # 100.000 tokens de prompt ainda usam o preço normal
    at_threshold = budget.cost_usd(
        ledger.pricing["escalonado"], input_tokens=100_000, output_tokens=0
    )
    assert at_threshold == Decimal("0.10")
    # cache conta no tamanho do prompt: 60k novos + 50k em cache > 100k
    above = budget.cost_usd(
        ledger.pricing["escalonado"], input_tokens=60_000, output_tokens=1000, cached_tokens=50_000
    )
    assert (
        above
        == (Decimal(60_000) * 2 + Decimal(50_000) * Decimal("0.20") + Decimal(1000) * 10) / MTOK
    )


def test_negative_tokens_rejected(ledger: Ledger) -> None:
    with pytest.raises(ValueError):
        ledger.charge("barato", input_tokens=-1, output_tokens=0, milestone="M4")


def test_milestone_is_required_and_non_empty(ledger: Ledger) -> None:
    with pytest.raises(ValueError):
        ledger.charge("barato", input_tokens=1, output_tokens=1, milestone="")


# --- soma do ledger e teto --------------------------------------------------


def test_spent_is_zero_without_ledger(ledger: Ledger) -> None:
    assert not ledger.ledger_path.exists()
    assert ledger.spent() == Decimal(0)


def test_spent_by_milestone(ledger: Ledger) -> None:
    ledger.charge("barato", input_tokens=MTOK, output_tokens=0, milestone="M4")
    ledger.charge("barato", input_tokens=MTOK, output_tokens=0, milestone="M4")
    ledger.charge("caro", input_tokens=MTOK, output_tokens=0, milestone="M7")

    assert ledger.spent_by_milestone() == {
        "M4": (2, Decimal("2.00")),
        "M7": (1, Decimal("10.00")),
    }


def test_reserve_below_ceiling_is_allowed(ledger: Ledger) -> None:
    ledger.charge("barato", input_tokens=4 * MTOK, output_tokens=0, milestone="M4")  # 4.00
    ledger.reserve(Decimal("5.99"))  # 9.99 <= 10


def test_reserve_exactly_at_ceiling_is_allowed(ledger: Ledger) -> None:
    ledger.charge("barato", input_tokens=4 * MTOK, output_tokens=0, milestone="M4")  # 4.00
    ledger.reserve(Decimal("6.00"))  # 10.00 == 10
    ledger.reserve(6.0)  # float também é aceito


def test_reserve_exactly_at_ceiling_after_many_small_charges(ledger: Ledger) -> None:
    # 10 cobranças de 0.70 em float somariam 6.999999...; em Decimal é exato
    for _ in range(10):
        ledger.charge("barato", input_tokens=700_000, output_tokens=0, milestone="M4")
    assert ledger.spent() == Decimal("7.00")
    ledger.reserve(Decimal("3.00"))


def test_reserve_above_ceiling_raises(ledger: Ledger) -> None:
    ledger.charge("barato", input_tokens=4 * MTOK, output_tokens=0, milestone="M4")  # 4.00
    with pytest.raises(BudgetExceededError, match="10"):
        ledger.reserve(Decimal("6.01"))


def test_reserve_above_ceiling_with_empty_ledger_raises(ledger: Ledger) -> None:
    with pytest.raises(BudgetExceededError):
        ledger.reserve(Decimal("10.000001"))


def test_reserve_negative_estimate_rejected(ledger: Ledger) -> None:
    with pytest.raises(ValueError):
        ledger.reserve(Decimal("-1"))


def test_reserve_does_not_write_to_ledger(ledger: Ledger) -> None:
    ledger.reserve(Decimal("1"))
    assert not ledger.ledger_path.exists()


# --- funções de módulo com caminhos injetados ------------------------------


def test_module_functions_accept_injected_paths(tmp_path: Path, pricing_path: Path) -> None:
    ledger_path = tmp_path / "spend.jsonl"
    cost = budget.charge(
        "barato",
        MTOK,
        0,
        milestone="M4",
        ledger_path=ledger_path,
        pricing_path=pricing_path,
    )
    assert cost == Decimal("1.00")
    budget.reserve(Decimal("1"), ledger_path=ledger_path, budget_usd=2)
    with pytest.raises(BudgetExceededError):
        budget.reserve(Decimal("1.01"), ledger_path=ledger_path, budget_usd=2)


def test_repo_pricing_has_source_and_date_for_every_model() -> None:
    pricing = budget.load_pricing()
    assert pricing, "configs/pricing.toml sem modelos"
    for price in pricing.values():
        assert price.source_url.startswith("https://")
        assert price.retrieved_on is not None
        assert price.input_per_mtok > 0
        assert price.output_per_mtok > 0
        assert 0 < price.batch_discount < 1


# --- CLI --------------------------------------------------------------------


def test_budget_status_shows_spend_by_milestone(ledger: Ledger) -> None:
    ledger.charge("barato", input_tokens=MTOK, output_tokens=0, milestone="M4")
    ledger.charge("caro", input_tokens=MTOK, output_tokens=0, milestone="M7")

    result = CliRunner().invoke(
        app,
        ["budget", "status", "--ledger", str(ledger.ledger_path), "--budget-usd", "20"],
    )

    assert result.exit_code == 0, result.output
    assert "M4" in result.output
    assert "M7" in result.output
    assert "11.00" in result.output  # total
    assert "9.00" in result.output  # restante


def test_budget_status_without_ledger(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["budget", "status", "--ledger", str(tmp_path / "nada.jsonl")])
    assert result.exit_code == 0, result.output
    assert "0.00" in result.output


def test_root_help_lists_groups() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    for group in ("data", "eval", "gen", "infer", "budget"):
        assert group in result.output
