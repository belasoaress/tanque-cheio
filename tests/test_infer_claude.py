import json
from decimal import Decimal
from itertools import count

import pytest
from typer.testing import CliRunner

from m4_fakes import FakeClient, read_jsonl, write_jsonl
from tanque.budget import Ledger, cost_usd
from tanque.cli import app
from tanque.infer import api
from tanque.infer import claude as claude_mod
from tanque.infer import cli as infer_cli

runner = CliRunner()

ITEMS = [
    {"id": "dev-0001", "question": "Pergunta um?", "split": "dev"},
    {"id": "dev-0002", "question": "Pergunta dois?", "split": "dev"},
    {"id": "dev-0003", "question": "Pergunta três?", "split": "dev"},
]


def stub_prompt(question):
    return [
        {"role": "system", "content": "PREFIXO ESTÁVEL"},
        {"role": "user", "content": f"Pergunta: {question}"},
    ]


def stub_extract(raw):
    return raw.replace("```sql", "").replace("```", "").strip()


def responder(params):
    q = params["messages"][-1]["content"]
    return f"```sql\nSELECT '{q}'\n```"


@pytest.fixture
def ledger(tmp_path):
    return Ledger(ledger_path=tmp_path / "spend.jsonl", budget_usd=10)


@pytest.fixture
def settings(ledger):
    return claude_mod.settings_for(
        "frontier", ledger, build_prompt=stub_prompt, extract_sql=stub_extract, milestone="M7"
    )


def test_system_name_uses_pricing_id(settings):
    assert settings.system.startswith("claude-")
    assert settings.system == settings.price.id or settings.system == f"claude-{settings.price.id}"


def test_build_params_splits_system_and_sets_effort(settings):
    params = claude_mod.build_params("Q?", settings)
    assert params["model"] == settings.price.id
    assert params["system"] == [{"type": "text", "text": "PREFIXO ESTÁVEL"}]
    assert params["messages"] == [{"role": "user", "content": "Pergunta: Q?"}]
    assert params["output_config"] == {"effort": "low"}
    assert "thinking" not in params
    settings.cache = True
    settings.effort = None
    params = claude_mod.build_params("Q?", settings)
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "output_config" not in params


def test_predict_sync_records_latency_tokens_and_cost(settings, ledger):
    client = FakeClient(responder)
    ticks = count(start=0, step=0.25)
    preds = claude_mod.predict_sync(
        client, ITEMS, settings, ledger=ledger, clock=lambda: next(ticks)
    )
    assert [p["id"] for p in preds] == ["dev-0001", "dev-0002", "dev-0003"]
    p = preds[0]
    assert set(p) == {
        "id", "system", "sql", "raw_output", "latency_ms", "input_tokens",
        "output_tokens", "cost_usd", "hardware", "error",
    }  # fmt: skip
    assert p["sql"] == "SELECT 'Pergunta: Pergunta um?'"
    assert p["latency_ms"] == 250
    assert p["input_tokens"] == 1100 and p["output_tokens"] == 50
    expected = cost_usd(settings.price, 100, 50, cached_tokens=1000, batch=False)
    assert Decimal(str(p["cost_usd"])) == expected
    assert p["error"] is None and p["hardware"] == "anthropic-api"
    entries = ledger.entries()
    assert len(entries) == 3 and all(e["milestone"] == "M7" and not e["batch"] for e in entries)


def test_predict_sync_api_error_and_refusal_become_error_predictions(settings, ledger):
    client = FakeClient(responder)
    client.messages.raise_for = {"dois"}
    preds = claude_mod.predict_sync(client, ITEMS, settings, ledger=ledger)
    assert preds[1]["error"].startswith("RuntimeError") and preds[1]["sql"] == ""
    assert len(ledger.entries()) == 2
    client.messages.raise_for = set()
    client.messages.stop_reason = "refusal"
    preds = claude_mod.predict_sync(client, ITEMS[:1], settings, ledger=ledger)
    assert "refusal" in preds[0]["error"]


def test_batch_submit_and_collect(tmp_path, settings, ledger):
    client = FakeClient(responder)
    registry = api.BatchRegistry(tmp_path / "reg.jsonl")
    plan = claude_mod.plan_cost(ITEMS, settings, 500, batch=True)
    job = claude_mod.submit_batch(
        client, ITEMS, settings, plan, registry=registry, dataset=tmp_path / "d.jsonl",
        out_path=tmp_path / "o.jsonl",
    )  # fmt: skip
    entry = registry.submitted(job.batch_id)
    assert entry["kind"] == "infer" and entry["ids"]["dev-0002"] == "dev-0002"
    client.messages.batches.drop_ids = {"dev-0003"}
    client.messages.batches.fail_ids = {"dev-0002"}
    preds = claude_mod.collect_batch(client, job, ITEMS, settings, ledger=ledger, registry=registry)
    assert [p["id"] for p in preds] == ["dev-0001", "dev-0002", "dev-0003"]
    assert preds[0]["latency_ms"] is None and preds[0]["hardware"] == "anthropic-batch"
    assert preds[0]["sql"] == "SELECT 'Pergunta: Pergunta um?'"
    assert preds[1]["error"] == "lote: errored"
    assert preds[2]["error"] == "sem resultado no lote"
    (charged,) = ledger.entries()
    assert charged["batch"] is True
    with pytest.raises(RuntimeError):
        claude_mod.collect_batch(client, job, ITEMS, settings, ledger=ledger, registry=registry)


def test_plan_cost_with_and_without_cache(settings):
    plan = claude_mod.plan_cost(ITEMS, settings, 450, batch=True, est_output_tokens=600)
    each = plan.input_tokens_each
    assert plan.expected_usd == 3 * cost_usd(settings.price, each, 600, batch=True)
    assert plan.worst_usd == 3 * cost_usd(settings.price, each, settings.max_tokens, batch=True)
    settings.cache = True
    cached = claude_mod.plan_cost(ITEMS, settings, 600, batch=True, est_output_tokens=600)
    assert cached.cache is True


def test_measure_prefix_uses_count_tokens_or_heuristic(settings):
    assert claude_mod.measure_prefix(FakeClient(prefix_tokens=455), settings) == 455
    assert claude_mod.measure_prefix(FakeClient(prefix_tokens=None), settings) == (
        api.heuristic_tokens("PREFIXO ESTÁVEL")
    )


def test_summarize():
    preds = [
        {"cost_usd": 0.1, "output_tokens": 100, "latency_ms": 10, "error": None},
        {"cost_usd": 0.2, "output_tokens": 300, "latency_ms": 30, "error": "x"},
    ]
    s = claude_mod.summarize(preds)
    assert s["n"] == 2 and s["errors"] == 1
    assert s["mean_output_tokens"] == 200 and s["max_output_tokens"] == 300
    assert s["p50_latency_ms"] == 30


# --- CLI --------------------------------------------------------------------


@pytest.fixture
def m4_infer(tmp_path, monkeypatch):
    client = FakeClient(responder, prefix_tokens=450)
    ledger_path = tmp_path / "spend.jsonl"
    monkeypatch.setattr(infer_cli, "make_client", lambda: client)
    monkeypatch.setattr(infer_cli, "Ledger", lambda: Ledger(ledger_path=ledger_path, budget_usd=10))
    monkeypatch.setattr(api, "REGISTRY_RELPATH", tmp_path / "reg.jsonl")
    # A CLI usa o prompt real (`tanque.prompting`), que não depende do banco.
    dataset = write_jsonl(tmp_path / "dev.jsonl", ITEMS)
    return {"client": client, "ledger": ledger_path, "dataset": dataset, "tmp": tmp_path}


def test_cli_sync_with_limit(m4_infer):
    out = m4_infer["tmp"] / "preds.jsonl"
    args = ["infer", "claude", str(m4_infer["dataset"]), "--model", "frontier"]
    result = runner.invoke(app, [*args, "--limit", "2", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "cache não marcado" in result.output  # 450 < 512
    preds = read_jsonl(out)
    assert [p["id"] for p in preds] == ["dev-0001", "dev-0002"]
    assert all(p["latency_ms"] is not None for p in preds)
    assert len(read_jsonl(m4_infer["ledger"])) == 2
    calls = m4_infer["client"].messages.create_calls
    assert all("cache_control" not in b for c in calls for b in c["system"])


def test_cli_batch_writes_default_path(m4_infer, monkeypatch):
    monkeypatch.setattr(
        claude_mod,
        "default_output_path",
        lambda system, split, limit: m4_infer["tmp"] / f"{system}__{split}.jsonl",
    )
    args = ["infer", "claude", str(m4_infer["dataset"]), "--model", "frontier", "--batch"]
    result = runner.invoke(app, [*args, "--poll-s", "0"])
    assert result.exit_code == 0, result.output
    (path,) = m4_infer["tmp"].glob("claude-*__dev.jsonl")
    preds = read_jsonl(path)
    assert len(preds) == 3 and all(p["latency_ms"] is None for p in preds)
    assert all(e["batch"] for e in read_jsonl(m4_infer["ledger"]))


def test_cli_refuses_without_approval(m4_infer, monkeypatch):
    monkeypatch.setattr(api, "APPROVAL_LIMIT_USD", Decimal("0.0000001"))
    args = ["infer", "claude", str(m4_infer["dataset"]), "--model", "frontier"]
    result = runner.invoke(app, args)
    assert result.exit_code == 1 and "--approved" in result.output
    assert m4_infer["client"].messages.create_calls == []


def test_cli_unknown_model_key(m4_infer):
    args = ["infer", "claude", str(m4_infer["dataset"]), "--model", "nao-existe"]
    assert runner.invoke(app, args).exit_code == 2


def test_cli_collect_resumes_batch(m4_infer):
    client = m4_infer["client"]
    client.messages.batches.status = "in_progress"
    out = m4_infer["tmp"] / "b.jsonl"
    args = ["infer", "claude", str(m4_infer["dataset"]), "--model", "frontier", "--batch"]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(api, "wait_for_batch", lambda *a, **k: {"processing_status": "in_progress"})
        first = runner.invoke(app, [*args, "--out", str(out)])
    assert first.exit_code == 1
    (bid,) = client.messages.batches.created
    client.messages.batches.status = "ended"
    done = runner.invoke(app, ["infer", "collect", bid])
    assert done.exit_code == 0, done.output
    assert len(read_jsonl(out)) == 3
    assert json.loads(out.read_text(encoding="utf-8").splitlines()[0])["id"] == "dev-0001"
