import json

import pytest

from m4_fakes import make_toy_db, read_jsonl, write_jsonl
from tanque.budget import Ledger
from tanque.gen import leakage, pipeline
from tanque.infer import api

J = (
    "FROM coletas c JOIN postos p ON p.posto_id = c.posto_id "
    "JOIN municipios m ON m.municipio_id = p.municipio_id"
)


def raw_row(custom_id, pairs):
    return {
        "custom_id": custom_id,
        "intent": custom_id.split("-", 1)[1],
        "type": "succeeded",
        "pairs": pairs,
    }


@pytest.fixture
def m4_env(tmp_path):
    db = make_toy_db(tmp_path / "toy.duckdb")
    raw = tmp_path / "raw"
    good = [
        {
            "question": "Qual o preço médio da gasolina em SP?",
            "sql": f"SELECT AVG(c.valor_venda) {J} WHERE c.produto = 'GASOLINA' AND m.uf = 'SP'",
            "order_matters": False,
        },
        {
            "question": "Quanto custa o etanol, em média, na Campinas paulista?",
            "sql": f"SELECT AVG(c.valor_venda) {J} WHERE c.produto = 'ETANOL' "
            "AND m.nome = 'CAMPINAS' AND m.uf = 'SP'",
            "order_matters": False,
        },
        # vaza (mesma pergunta que um item do "teste" falso)
        {
            "question": "Qual o preço médio do diesel S10 em Recife?",
            "sql": f"SELECT AVG(c.valor_venda) {J} WHERE c.produto = 'DIESEL S10' "
            "AND m.nome = 'RECIFE'",
            "order_matters": False,
        },
        # SQL quebrada
        {"question": "Pergunta com SQL quebrada?", "sql": "SELEC 1", "order_matters": False},
    ]
    write_jsonl(raw / "msgbatch_a.jsonl", [raw_row("g00000-media_produto_municipio", good)])
    write_jsonl(
        raw / "msgbatch_b.jsonl",
        [
            raw_row(
                "g00001-media_produto_municipio",
                [{**good[0], "question": "qual o preco medio da gasolina em sp"}],
            ),
            {"custom_id": "g00002-x", "intent": "x", "type": "errored", "error": "api_error"},
        ],
    )
    ref = tmp_path / "datasets"
    write_jsonl(
        ref / "dev.jsonl",
        [{"id": "dev-0001", "question": "Ranking de bairros?", "sql": "SELECT 9"}],
    )
    write_jsonl(
        ref / "test.jsonl",
        [
            {
                "id": "test-0001",
                "question": "Qual o preço médio do diesel S10 em Recife?",
                "sql": "SELECT 8",
            }
        ],
    )
    return {"db": db, "raw": raw, "ref": ref, "tmp": tmp_path}


def _build(env):
    t = env["tmp"]
    return pipeline.build_train(
        db_path=env["db"],
        raw_dir=env["raw"],
        reference_dir=env["ref"],
        out_path=t / "train.jsonl",
        work_dir=t / "work",
        summary_path=t / "summary.json",
    )


def test_build_train_end_to_end(m4_env):
    result = _build(m4_env)
    t = m4_env["tmp"]
    train = read_jsonl(t / "train.jsonl")
    assert [x["id"] for x in train] == ["train-00001", "train-00002"]
    assert [x["question"] for x in train] == [
        "Qual o preço médio da gasolina em SP?",
        "Quanto custa o etanol, em média, na Campinas paulista?",
    ]
    assert all(x["split"] == "train" and x["source"] == "synthetic" for x in train)
    assert result.raw_pairs == 5 and result.final == 2
    assert result.reasons == {"parse": 1, "vazamento": 1, "duplicado": 1}

    summary = json.loads((t / "summary.json").read_text(encoding="utf-8"))
    assert summary["requests_with_pairs"] == 2
    assert summary["discarded_by_reason"] == result.reasons
    discards = read_jsonl(t / "work" / "discarded.jsonl")
    leak = next(d for d in discards if d["reason"] == "vazamento")
    assert "test-0001" in leak["detail"]

    # o treino gravado passa na checagem de vazamento
    reference, _ = leakage.load_reference(m4_env["ref"])
    assert leakage.find_collisions(train, reference) == []


def test_build_train_refuses_without_eval_sets(m4_env):
    (m4_env["ref"] / "test.jsonl").unlink()
    with pytest.raises(leakage.MissingEvalSetError):
        _build(m4_env)
    assert not (m4_env["tmp"] / "train.jsonl").exists()


def test_stats_report(m4_env, tmp_path):
    _build(m4_env)
    ledger = Ledger(ledger_path=tmp_path / "spend.jsonl", budget_usd=10)
    ledger.charge("cheap", 1000, 2000, 500, True, milestone="M4")
    ledger.charge("cheap", 1000, 2000, 500, True, milestone="M7")
    registry = api.BatchRegistry(tmp_path / "reg.jsonl")
    registry.append("submitted", "msgbatch_a", kind="gen", n_requests=1, estimate_usd=0.01)
    out, text = pipeline.write_stats(
        train_file=tmp_path / "train.jsonl",
        summary_path=tmp_path / "summary.json",
        out_path=tmp_path / "gen-stats.md",
        ledger=ledger,
        registry=registry,
    )
    assert out.read_text(encoding="utf-8") == text
    assert "**Pares no treino:** 2" in text
    assert "| facil | 2 | 100.0% | 28% | não |" in text  # alvo v2 (M15)
    assert "| `media_produto_municipio` | facil | 2 |" in text
    assert "| `vazamento` | 1 | 20.0% |" in text
    assert "Chamadas cobradas: 1" in text
    m4_cost = ledger.entries()[0]["cost_usd"]
    assert f"US$ {m4_cost:.4f}" in text


def test_stats_without_data(tmp_path):
    _, text = pipeline.write_stats(
        train_file=tmp_path / "nada.jsonl",
        summary_path=tmp_path / "nada.json",
        out_path=tmp_path / "s.md",
        ledger=Ledger(ledger_path=tmp_path / "l.jsonl", budget_usd=10),
        registry=api.BatchRegistry(tmp_path / "r.jsonl"),
    )
    assert "**Pares no treino:** 0" in text
    assert "rode `tanque gen validate`" in text
