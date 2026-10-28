import json
import re
from decimal import Decimal

import pytest
from m4_fakes import FakeClient, make_toy_db, read_jsonl

from tanque.budget import BudgetExceededError, Ledger, cost_usd
from tanque.gen import batch
from tanque.infer import api

CUSTOM_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
PAIRS = [
    {"question": f"Pergunta {i}?", "sql": f"SELECT {i}", "order_matters": False} for i in range(10)
]


@pytest.fixture
def ledger(tmp_path):
    return Ledger(ledger_path=tmp_path / "spend.jsonl", budget_usd=10)


@pytest.fixture
def registry(tmp_path):
    return api.BatchRegistry(tmp_path / "api-batches.jsonl")


def _texts():
    return batch.prefix_texts([("SAO PAULO", "SP"), ("RECIFE", "PE")], "PROMPT 5.4")


# --- parse_pairs ------------------------------------------------------------


def test_parse_plain_array():
    assert batch.parse_pairs(json.dumps(PAIRS)) == PAIRS


def test_parse_fenced_with_text_around():
    text = "Aqui estão:\n```json\n" + json.dumps(PAIRS[:2]) + "\n```\nPronto."
    assert batch.parse_pairs(text) == PAIRS[:2]


def test_parse_object_wrapper():
    assert batch.parse_pairs(json.dumps({"pairs": PAIRS[:3]})) == PAIRS[:3]


def test_parse_truncated_array_recovers_complete_objects():
    text = json.dumps(PAIRS[:3])[:-30]
    got = batch.parse_pairs(text)
    assert got == PAIRS[:2]


def test_parse_normalizes_types_and_missing_fields():
    text = json.dumps([{"question": " Q? ", "sql": 3}, {"sql": "SELECT 1", "order_matters": "x"}])
    assert batch.parse_pairs(text) == [
        {"question": "Q?", "sql": "", "order_matters": None},
        {"question": "", "sql": "SELECT 1", "order_matters": None},
    ]


@pytest.mark.parametrize("text", ["", "   ", "não sei", "[1, 2]", "{bad json"])
def test_parse_garbage_returns_empty(text):
    assert batch.parse_pairs(text) == []


# --- prompt e requisições ---------------------------------------------------


def test_prefix_contains_conventions_target_prompt_and_municipalities():
    texts = _texts()
    joined = "\n".join(texts)
    assert "AVG(c.valor_venda)" in joined
    assert "<= 0.7" in joined and "'GASOLINA'" in joined
    assert "2026-09-30" in joined
    assert "m.capital = true" in joined
    assert "PROMPT 5.4" in joined
    assert "PE: RECIFE" in joined and "SP: SAO PAULO" in joined


def test_build_requests_uses_given_model_and_stable_cached_prefix(ledger):
    price = batch.cheap_price(ledger)
    system = api.system_blocks(_texts(), cache=True)
    reqs = batch.build_requests(20, model_id=price.id, system=system, offset=5, max_tokens=999)
    assert len(reqs) == 20
    ids = [r.custom_id for r in reqs]
    assert len(set(ids)) == 20 and all(CUSTOM_ID_RE.match(i) for i in ids)
    assert ids[0].startswith("g00005-")
    for r in reqs:
        assert r.params["model"] == price.id
        assert r.params["max_tokens"] == 999
        assert r.params["system"] == system
        assert r.params["system"][-1]["cache_control"] == {"type": "ephemeral"}
        assert all("cache_control" not in b for b in r.params["system"][:-1])
        assert r.params["output_config"] == {"effort": batch.DEFAULT_EFFORT}
        assert batch.intent_of_custom_id(r.custom_id) == r.intent_id
        assert r.intent_id in r.params["messages"][0]["content"]


def test_effort_can_be_omitted(ledger):
    reqs = batch.build_requests(1, model_id="m", system=[], effort=None)
    assert "output_config" not in reqs[0].params


def test_same_intent_gets_different_focus_per_request():
    from tanque.gen.intents import INTENTS

    intent = INTENTS[0]
    messages = {batch.user_message(intent, k) for k in range(30)}
    assert len(messages) > 10
    assert batch.user_message(intent, 3) == batch.user_message(intent, 3)


@pytest.mark.parametrize(("tokens", "expected"), [(900, True), (512, True), (400, False)])
def test_prepare_system_marks_cache_only_above_minimum(ledger, tokens, expected):
    client = FakeClient(prefix_tokens=tokens)
    system, measured, cache = batch.prepare_system(client, batch.cheap_price(ledger), _texts())
    assert measured == tokens and cache is expected
    assert ("cache_control" in system[-1]) is expected
    assert client.messages.count_calls[0]["model"] == batch.cheap_price(ledger).id


def test_prepare_system_falls_back_to_heuristic(ledger):
    client = FakeClient(prefix_tokens=None)
    _, measured, _ = batch.prepare_system(client, batch.cheap_price(ledger), _texts())
    assert measured == api.heuristic_tokens("".join(_texts()))


def test_estimate_expected_and_worst_case(ledger):
    price = batch.cheap_price(ledger)
    reqs = batch.build_requests(4, model_id=price.id, system=[], max_tokens=8000)
    est = batch.estimate(price, reqs, 1000, cache=False, est_output_tokens=2000)
    each_in = est.input_tokens_each
    assert est.expected_usd == 4 * cost_usd(price, each_in, 2000, batch=True)
    assert est.worst_usd == 4 * cost_usd(price, each_in, 8000, batch=True)
    cached = batch.estimate(price, reqs, 1000, cache=True, est_output_tokens=2000)
    assert cached.expected_usd == 4 * cost_usd(
        price, each_in - 1000, 2000, cache_write_tokens=1000, batch=True
    )


# --- envio e coleta ---------------------------------------------------------


def _submit(client, ledger, registry, n=3, approved=False):
    price = batch.cheap_price(ledger)
    reqs = batch.build_requests(n, model_id=price.id, system=[])
    est = batch.estimate(price, reqs, 1000, cache=False)
    bid = batch.submit(
        client, reqs, est, ledger=ledger, registry=registry, offset=0, approved=approved
    )
    return bid, reqs, est


def test_submit_registers_batch(ledger, registry):
    client = FakeClient()
    bid, reqs, est = _submit(client, ledger, registry)
    entry = registry.submitted(bid)
    assert entry["kind"] == "gen" and entry["n_requests"] == 3 and entry["model"] == "cheap"
    assert entry["intents"] == {r.custom_id: r.intent_id for r in reqs}
    assert batch.next_offset(registry) == 3
    assert ledger.entries() == []  # nada cobrado antes da coleta


def test_submit_refuses_over_budget(tmp_path, registry):
    tight = Ledger(ledger_path=tmp_path / "s.jsonl", budget_usd=Decimal("0.0000001"))
    client = FakeClient()
    with pytest.raises(BudgetExceededError):
        _submit(client, tight, registry)
    assert client.messages.batches.created == {}


def test_submit_requires_approval_above_50_cents(ledger, registry, monkeypatch):
    monkeypatch.setattr(api, "APPROVAL_LIMIT_USD", Decimal("0.0000001"))
    client = FakeClient()
    with pytest.raises(api.ApprovalRequiredError):
        _submit(client, ledger, registry)
    assert client.messages.batches.created == {}
    _submit(client, ledger, registry, approved=True)
    assert len(client.messages.batches.created) == 1


def test_collect_charges_real_usage_once_and_writes_raw(tmp_path, ledger, registry):
    client = FakeClient(lambda p: "```json\n" + json.dumps(PAIRS) + "\n```")
    bid, reqs, _ = _submit(client, ledger, registry)
    client.messages.batches.fail_ids = {reqs[2].custom_id}
    raw_dir = tmp_path / "raw"

    result = batch.collect(client, bid, ledger=ledger, registry=registry, raw_dir=raw_dir)
    assert (result.succeeded, result.errored) == (2, 1)
    assert len(result.pairs) == 20
    assert {p["intent"] for p in result.pairs} == {r.intent_id for r in reqs[:2]}

    entries = ledger.entries()
    assert len(entries) == 2
    assert all(e["batch"] is True and e["milestone"] == "M4" for e in entries)
    assert all(e["model"] == "cheap" and e["cached_tokens"] == 1000 for e in entries)
    assert result.cost_usd == sum(Decimal(str(e["cost_usd"])) for e in entries)

    rows = read_jsonl(raw_dir / f"{bid}.jsonl")
    assert [r["type"] for r in rows] == ["succeeded", "succeeded", "errored"]
    assert rows[0]["usage"]["cached_tokens"] == 1000
    assert registry.is_collected(bid)

    again = batch.collect(client, bid, ledger=ledger, registry=registry, raw_dir=raw_dir)
    assert again.already_collected and len(again.pairs) == 20
    assert len(ledger.entries()) == 2
    assert client.messages.batches.results_calls == 1


def test_raw_candidates_reads_all_raw_files(tmp_path, ledger, registry):
    client = FakeClient(lambda p: json.dumps(PAIRS[:2]))
    raw_dir = tmp_path / "raw"
    for _ in range(2):
        bid, _, _ = _submit(client, ledger, registry, n=2)
        batch.collect(client, bid, ledger=ledger, registry=registry, raw_dir=raw_dir)
    cands = batch.raw_candidates(raw_dir)
    assert len(cands) == 8
    assert all({"question", "sql", "intent", "custom_id"} <= set(c) for c in cands)


def test_load_municipios_reads_toy_db(tmp_path):
    db = make_toy_db(tmp_path / "toy.duckdb")
    assert sorted(batch.load_municipios(db)) == [
        ("CAMPINAS", "SP"),
        ("RECIFE", "PE"),
        ("SAO PAULO", "SP"),
    ]


def test_no_model_id_written_in_gen_code():
    from pathlib import Path

    import tanque.gen

    for path in Path(tanque.gen.__file__).parent.glob("*.py"):
        assert not re.search(r"claude-[a-z]+-\d", path.read_text(encoding="utf-8")), path
