import json

import pytest
from typer.testing import CliRunner

from m4_fakes import FakeClient, make_toy_db, read_jsonl, write_jsonl
from tanque.budget import Ledger
from tanque.cli import app
from tanque.gen import batch, leakage, pipeline
from tanque.gen import cli as gen_cli
from tanque.infer import api

runner = CliRunner()
PAIRS = [
    {
        "question": f"Quantos postos há em São Paulo, versão {i}?",
        "sql": f"SELECT COUNT(*) + {i} FROM postos",
        "order_matters": False,
    }
    for i in range(10)
]


@pytest.fixture
def m4_cli(tmp_path, monkeypatch):
    client = FakeClient(lambda params: json.dumps(PAIRS), prefix_tokens=2000)
    ledger_path = tmp_path / "spend.jsonl"
    monkeypatch.setattr(gen_cli, "make_client", lambda: client)
    monkeypatch.setattr(gen_cli, "Ledger", lambda: Ledger(ledger_path=ledger_path, budget_usd=10))
    monkeypatch.setattr(api, "REGISTRY_RELPATH", tmp_path / "api-batches.jsonl")
    monkeypatch.setattr(batch, "RAW_RELDIR", tmp_path / "raw")
    monkeypatch.setattr(pipeline, "train_path", lambda: tmp_path / "train.jsonl")
    monkeypatch.setattr(pipeline, "SUMMARY_RELPATH", tmp_path / "summary.json")
    monkeypatch.setattr(pipeline, "WORK_RELDIR", tmp_path / "work")
    monkeypatch.setattr(leakage, "datasets_dir", lambda: tmp_path / "datasets")
    db = make_toy_db(tmp_path / "toy.duckdb")
    return {"client": client, "ledger": ledger_path, "db": db, "tmp": tmp_path}


def test_pilot_submits_20_waits_charges_and_shows_sample(m4_cli):
    result = runner.invoke(app, ["gen", "pilot", "--db", str(m4_cli["db"]), "--poll-s", "0"])
    assert result.exit_code == 0, result.output
    (requests,) = m4_cli["client"].messages.batches.created.values()
    assert len(requests) == 20
    assert "cache marcado" in result.output
    assert "custo real US$" in result.output
    assert result.output.count("\n[") == 10
    assert "PARE AQUI" in result.output
    entries = read_jsonl(m4_cli["ledger"])
    assert len(entries) == 20 and all(e["milestone"] == "M4" and e["batch"] for e in entries)
    assert len(list((m4_cli["tmp"] / "raw").glob("*.jsonl"))) == 1


def test_submit_then_collect_later(m4_cli):
    client = m4_cli["client"]
    client.messages.batches.status = "in_progress"
    result = runner.invoke(app, ["gen", "submit", "--requests", "3", "--db", str(m4_cli["db"])])
    assert result.exit_code == 0, result.output
    (bid,) = client.messages.batches.created
    assert f"tanque gen collect {bid}" in result.output

    pending = runner.invoke(app, ["gen", "collect", bid])
    assert pending.exit_code == 1 and "in_progress" in pending.output
    assert not m4_cli["ledger"].exists()

    client.messages.batches.status = "ended"
    done = runner.invoke(app, ["gen", "collect", bid])
    assert done.exit_code == 0, done.output
    assert len(read_jsonl(m4_cli["ledger"])) == 3
    again = runner.invoke(app, ["gen", "collect", bid])
    assert again.exit_code == 0 and "já tinha sido coletado" in again.output
    assert len(read_jsonl(m4_cli["ledger"])) == 3


def test_collect_unknown_batch(m4_cli):
    result = runner.invoke(app, ["gen", "collect", "msgbatch_nope"])
    assert result.exit_code == 2


def test_submit_over_approval_limit_is_refused(m4_cli, monkeypatch):
    monkeypatch.setattr(api, "APPROVAL_LIMIT_USD", api.Decimal("0.0000001"))
    result = runner.invoke(app, ["gen", "submit", "--requests", "2", "--db", str(m4_cli["db"])])
    assert result.exit_code == 1
    assert "--approved" in result.output
    assert m4_cli["client"].messages.batches.created == {}


def test_missing_api_key_gives_clear_error(m4_cli, monkeypatch):
    monkeypatch.setattr(gen_cli, "make_client", api.make_client)
    monkeypatch.setattr(api, "load_env", lambda: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    result = runner.invoke(app, ["gen", "pilot", "--db", str(m4_cli["db"])])
    assert result.exit_code == 2
    assert ".env" in result.output


def _eval_sets(tmp, test_question="Qual a bandeira com mais postos no Acre?"):
    write_jsonl(
        tmp / "datasets" / "dev.jsonl", [{"id": "dev-1", "question": "x y z", "sql": "SELECT 7"}]
    )
    write_jsonl(
        tmp / "datasets" / "test.jsonl",
        [{"id": "test-1", "question": test_question, "sql": "SELECT 8"}],
    )


def test_validate_and_check_leakage_clean(m4_cli):
    runner.invoke(app, ["gen", "pilot", "--db", str(m4_cli["db"]), "--poll-s", "0"])
    _eval_sets(m4_cli["tmp"])
    result = runner.invoke(app, ["gen", "validate", "--db", str(m4_cli["db"])])
    assert result.exit_code == 0, result.output
    train = read_jsonl(m4_cli["tmp"] / "train.jsonl")
    assert train  # pares válidos sobreviveram
    check = runner.invoke(app, ["gen", "check-leakage"])
    assert check.exit_code == 0, check.output
    assert "0 colisões" in check.output


def test_check_leakage_fails_on_collision(m4_cli):
    tmp = m4_cli["tmp"]
    _eval_sets(tmp, test_question="Quantos postos há em São Paulo?")
    write_jsonl(
        tmp / "train.jsonl",
        [{"id": "train-00001", "question": "quantos postos ha em sao paulo", "sql": "SELECT 1"}],
    )
    result = runner.invoke(app, ["gen", "check-leakage"])
    assert result.exit_code == 1
    assert "1 colisões" in result.output
    assert "train-00001 x test-1" in result.output
    # nada do texto da avaliação aparece na saída
    assert "Quantos postos há em São Paulo?" not in result.output


def test_check_leakage_requires_eval_sets(m4_cli):
    write_jsonl(m4_cli["tmp"] / "train.jsonl", [{"id": "t", "question": "q", "sql": "SELECT 1"}])
    result = runner.invoke(app, ["gen", "check-leakage"])
    assert result.exit_code == 2


def test_validate_requires_eval_sets(m4_cli):
    result = runner.invoke(app, ["gen", "validate", "--db", str(m4_cli["db"])])
    assert result.exit_code == 2
    assert not (m4_cli["tmp"] / "train.jsonl").exists()


def test_stats_command_writes_report(m4_cli, monkeypatch):
    out = m4_cli["tmp"] / "gen-stats.md"
    monkeypatch.setattr(pipeline, "STATS_RELPATH", out)
    monkeypatch.setattr(pipeline, "Ledger", lambda: Ledger(ledger_path=m4_cli["ledger"]))
    result = runner.invoke(app, ["gen", "stats"])
    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8").startswith("# Estatísticas do dataset de treino")
