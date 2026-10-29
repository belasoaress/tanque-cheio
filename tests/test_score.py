"""Testes do avaliador (src/tanque/eval/score.py, report.py e eval/cli.py).

Usam o banco e o dataset de brinquedo de tests/conftest.py.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tanque.cli import app
from tanque.eval import report, score
from tanque.eval.score import GoldError, PredictionsError

runner = CliRunner()


def prediction(item: dict, sql: str | None = None, **extra) -> dict:
    row = {
        "id": item["id"],
        "system": "base-brinquedo",
        "sql": item["sql"] if sql is None else sql,
        "raw_output": "",
        "latency_ms": 100,
        "input_tokens": 1000,
        "output_tokens": 50,
        "cost_usd": 0.0,
        "hardware": "cpu",
        "error": None,
    }
    row.update(extra)
    return row


@pytest.fixture
def gold_predictions(tmp_path, m2_toy_items, m2_write_jsonl) -> Path:
    rows = [prediction(item) for item in m2_toy_items]
    return m2_write_jsonl(tmp_path / "base-brinquedo__dev.jsonl", rows)


# ---------- score ----------


def test_predictions_equal_to_gold_give_ex_one(m2_toy_db, m2_dataset, gold_predictions) -> None:
    result = score.score(m2_dataset, gold_predictions, db_path=m2_toy_db)
    m = result.metrics
    assert result.system == "base-brinquedo"
    assert result.split == "dev"
    assert m["n"] == 4
    assert m["ex"] == 1.0
    assert m["valid_sql_rate"] == 1.0
    assert m["blocked_rate"] == 0.0
    assert m["n_missing"] == 0
    assert m["ex_by_difficulty"] == {
        "facil": {"n": 2, "ex": 1.0, "ex_relaxed": 1.0},
        "medio": {"n": 1, "ex": 1.0, "ex_relaxed": 1.0},
        "dificil": {"n": 1, "ex": 1.0, "ex_relaxed": 1.0},
    }
    assert m["ex_relaxed"] == 1.0
    assert m["ex_ci95"] == [1.0, 1.0]
    assert m["ex_relaxed_ci95"] == [1.0, 1.0]


def test_mixed_predictions(tmp_path, m2_toy_db, m2_dataset, m2_toy_items, m2_write_jsonl) -> None:
    facil1, facil2, medio, dificil = m2_toy_items
    rows = [
        # certo, mas com colunas renomeadas e permutadas
        prediction(facil2, "SELECT COUNT(*) AS n, produto AS p FROM coletas GROUP BY 2"),
        # executa, mas resultado errado
        prediction(medio, "SELECT uf, 1 FROM municipios ORDER BY uf", latency_ms=300),
        # bloqueado pelo guardrail
        prediction(dificil, "DROP TABLE coletas", latency_ms=200, cost_usd=0.002),
        # facil1 sem previsão
    ]
    preds = m2_write_jsonl(tmp_path / "p.jsonl", rows)
    m = score.score(m2_dataset, preds, db_path=m2_toy_db).metrics
    assert m["n"] == 4
    assert m["n_missing"] == 1
    assert m["ex"] == 0.25
    assert m["valid_sql_rate"] == 0.5  # facil2 e medio executaram
    assert m["blocked_rate"] == 0.25
    assert m["ex_by_difficulty"]["facil"] == {"n": 2, "ex": 0.5, "ex_relaxed": 0.5}
    assert m["ex_by_difficulty"]["medio"] == {"n": 1, "ex": 0.0, "ex_relaxed": 0.0}
    assert m["ex_by_difficulty"]["dificil"] == {"n": 1, "ex": 0.0, "ex_relaxed": 0.0}
    assert m["ex_relaxed"] == 0.25
    lo, hi = m["ex_ci95"]
    assert 0.0 <= lo <= 0.25 <= hi <= 1.0
    assert m["latency_ms"] == {"p50": 200.0, "p95": 290.0, "mean": 200.0}
    assert m["cost_per_1k_usd"] == pytest.approx(0.5)  # 0.002 / 4 itens * 1000


def test_relaxed_ex_accepts_extra_columns(
    tmp_path, m2_toy_db, m2_dataset, m2_toy_items, m2_write_jsonl
) -> None:
    facil1, facil2, medio, dificil = m2_toy_items
    rows = [
        prediction(facil1),
        # coluna a mais (o mínimo junto da contagem): errado na rígida, certo na tolerante
        prediction(
            facil2, "SELECT MIN(valor_venda), COUNT(*), produto FROM coletas GROUP BY produto"
        ),
        # coluna a mais e a ordem certa
        prediction(medio, medio["sql"].replace("SELECT m.uf,", "SELECT m.uf, COUNT(*),")),
        # coluna faltando: errado nas duas
        prediction(dificil, "SELECT 1"),
    ]
    preds = m2_write_jsonl(tmp_path / "p.jsonl", rows)
    result = score.score(m2_dataset, preds, db_path=m2_toy_db)
    m = result.metrics
    assert m["ex"] == 0.25
    assert m["ex_relaxed"] == 0.75
    assert m["ex_by_difficulty"]["facil"] == {"n": 2, "ex": 0.5, "ex_relaxed": 1.0}
    assert m["ex_by_difficulty"]["medio"] == {"n": 1, "ex": 0.0, "ex_relaxed": 1.0}
    assert m["ex_by_difficulty"]["dificil"] == {"n": 1, "ex": 0.0, "ex_relaxed": 0.0}
    assert [r.correct_relaxed for r in result.items] == [True, True, True, False]
    assert all(r.correct_relaxed for r in result.items if r.correct)
    assert m["ex_relaxed_ci95"][0] <= 0.75 <= m["ex_relaxed_ci95"][1]


def test_paired_comparisons_cover_every_pair_in_a_split() -> None:
    def fake(system: str, split: str, flags: list[bool]) -> score.ScoreResult:
        items = [
            score.ItemResult(f"i{k}", "facil", True, f, True, False, None, f)
            for k, f in enumerate(flags)
        ]
        return score.ScoreResult(system, split, {}, items)

    results = [
        fake("ft-m", "test", [True, True, True, True, False]),
        fake("base-m", "test", [False, False, True, False, False]),
        fake("claude-x", "test", [True, True, True, True, True]),
        fake("base-m", "dev", [True, False]),
    ]
    out = score.paired_comparisons(results)
    assert [(c["split"], c["a"], c["b"]) for c in out] == [
        ("test", "base-m", "ft-m"),
        ("test", "base-m", "claude-x"),
        ("test", "ft-m", "claude-x"),
    ]
    assert out[0] == {
        "a": "base-m",
        "b": "ft-m",
        "split": "test",
        "n": 5,
        "a_only": 0,
        "b_only": 3,
        "delta_ex": 0.6,
        "mcnemar_p": 0.25,
    }
    with pytest.raises(PredictionsError):
        score.paired_comparisons(
            [fake("base-m", "dev", [True]), fake("ft-m", "dev", [True, False])]
        )


def test_paired_comparisons_with_retry_files() -> None:
    def fake(system: str, flags: list[bool]) -> score.ScoreResult:
        items = [
            score.ItemResult(f"i{k}", "facil", True, f, True, False, None, f)
            for k, f in enumerate(flags)
        ]
        return score.ScoreResult(system, "test", {}, items)

    results = [
        fake("ft-m+retry", [True, True, True, True]),
        fake("base-m+retry", [True, False, False, False]),
        fake("ft-m", [True, True, False, False]),
        fake("base-m", [False, False, False, False]),
    ]
    pairs = [(c["a"], c["b"]) for c in score.paired_comparisons(results)]
    # todos os pares single-shot e cada <s> × <s>+retry; nada entre dois +retry
    assert pairs == [
        ("base-m", "base-m+retry"),
        ("base-m", "ft-m"),
        ("ft-m", "ft-m+retry"),
    ]


def test_system_sort_key_orders_by_prefix_family_size_and_retry() -> None:
    systems = [
        "claude-x",
        "ft2-qwen2.5-coder-1.5b-instruct",
        "base-qwen2.5-coder-32b-instruct",
        "base-qwen2.5-coder-7b-instruct+retry",
        "outro",
        "base-qwen2.5-coder-14b-instruct",
        "ft-qwen2.5-coder-1.5b-instruct+retry",
        "base-qwen2.5-coder-7b-instruct",
        "ft-qwen2.5-coder-1.5b-instruct",
        "base-qwen2.5-coder-1.5b-instruct",
    ]
    assert sorted(systems, key=score.system_sort_key) == [
        "base-qwen2.5-coder-1.5b-instruct",
        "base-qwen2.5-coder-7b-instruct",
        "base-qwen2.5-coder-7b-instruct+retry",
        "base-qwen2.5-coder-14b-instruct",
        "base-qwen2.5-coder-32b-instruct",
        "ft-qwen2.5-coder-1.5b-instruct",
        "ft-qwen2.5-coder-1.5b-instruct+retry",
        "ft2-qwen2.5-coder-1.5b-instruct",
        "claude-x",
        "outro",
    ]


def test_committed_predictions_include_dev2_and_retry(tmp_path) -> None:
    for name in ("a__dev.jsonl", "a__dev2.jsonl", "a+retry__dev2.jsonl", "a__test.jsonl"):
        (tmp_path / name).write_text("", encoding="utf-8")
    (tmp_path / "a__dev-sample2.jsonl").write_text("", encoding="utf-8")
    found = [(split, p.name) for split, p in score.committed_predictions(tmp_path)]
    assert found == [
        ("dev", "a__dev.jsonl"),
        ("dev2", "a+retry__dev2.jsonl"),
        ("dev2", "a__dev2.jsonl"),
        ("test", "a__test.jsonl"),
    ]


def test_system_metrics_drops_comparisons() -> None:
    data = {"base-m": {"dev": {}}, "comparisons": []}
    assert score.system_metrics(data) == {"base-m": {"dev": {}}}


def test_generation_error_counts_as_invalid(
    tmp_path, m2_toy_db, m2_dataset, m2_toy_items, m2_write_jsonl
) -> None:
    rows = [prediction(item) for item in m2_toy_items]
    rows[0] = prediction(m2_toy_items[0], "", error="timeout da API")
    preds = m2_write_jsonl(tmp_path / "p.jsonl", rows)
    m = score.score(m2_dataset, preds, db_path=m2_toy_db).metrics
    assert m["ex"] == 0.75
    assert m["valid_sql_rate"] == 0.75
    assert m["blocked_rate"] == 0.0


def test_execution_error_counts_as_invalid(
    tmp_path, m2_toy_db, m2_dataset, m2_toy_items, m2_write_jsonl
) -> None:
    rows = [prediction(item) for item in m2_toy_items]
    rows[1] = prediction(m2_toy_items[1], "SELECT coluna_que_nao_existe FROM coletas")
    preds = m2_write_jsonl(tmp_path / "p.jsonl", rows)
    result = score.score(m2_dataset, preds, db_path=m2_toy_db)
    assert result.metrics["ex"] == 0.75
    assert result.metrics["valid_sql_rate"] == 0.75
    failed = [r for r in result.items if not r.correct]
    assert [r.id for r in failed] == ["toy-0002"]
    assert failed[0].pred_error


def test_percentiles_and_cost(
    tmp_path, m2_toy_db, m2_dataset, m2_toy_items, m2_write_jsonl
) -> None:
    rows = [
        prediction(item, latency_ms=lat, cost_usd=0.001)
        for item, lat in zip(m2_toy_items, [400, 100, 300, 200], strict=True)
    ]
    preds = m2_write_jsonl(tmp_path / "p.jsonl", rows)
    m = score.score(m2_dataset, preds, db_path=m2_toy_db).metrics
    assert m["latency_ms"] == {"p50": 250.0, "p95": 385.0, "mean": 250.0}
    assert m["cost_per_1k_usd"] == pytest.approx(1.0)
    assert m["hardware"] == ["cpu"]


def test_gpus_and_peak_memory_only_when_present(
    tmp_path, m2_toy_db, m2_dataset, m2_toy_items, m2_write_jsonl
) -> None:
    plain = m2_write_jsonl(tmp_path / "a.jsonl", [prediction(i) for i in m2_toy_items])
    m = score.score(m2_dataset, plain, db_path=m2_toy_db).metrics
    assert "gpus" not in m and "peak_gpu_mem_gib" not in m  # métricas da v1 não mudam
    rows = [
        {**prediction(i), "gpus": 2, "peak_gpu_mem_gib": g}
        for i, g in zip(m2_toy_items, [18.5, 19.25, 18.0, 19.0], strict=True)
    ]
    m = score.score(
        m2_dataset, m2_write_jsonl(tmp_path / "b.jsonl", rows), db_path=m2_toy_db
    ).metrics
    assert m["gpus"] == 2
    assert m["peak_gpu_mem_gib"] == 19.25


def test_percentile_helper() -> None:
    assert score.percentile([5], 95) == 5
    assert score.percentile([1, 2, 3, 4, 5], 50) == 3
    assert score.percentile([], 50) is None


def test_duplicate_prediction_ids_rejected(
    tmp_path, m2_toy_db, m2_dataset, m2_toy_items, m2_write_jsonl
) -> None:
    item = m2_toy_items[0]
    preds = m2_write_jsonl(tmp_path / "p.jsonl", [prediction(item), prediction(item)])
    with pytest.raises(PredictionsError, match="toy-0001"):
        score.score(m2_dataset, preds, db_path=m2_toy_db)


def test_mixed_systems_rejected(
    tmp_path, m2_toy_db, m2_dataset, m2_toy_items, m2_write_jsonl
) -> None:
    rows = [prediction(m2_toy_items[0]), prediction(m2_toy_items[1], system="ft-outro")]
    preds = m2_write_jsonl(tmp_path / "p.jsonl", rows)
    with pytest.raises(PredictionsError, match="system"):
        score.score(m2_dataset, preds, db_path=m2_toy_db)


def test_broken_gold_aborts_scoring(tmp_path, m2_toy_db, m2_toy_items, m2_write_jsonl) -> None:
    items = [dict(m2_toy_items[0], sql="SELECT nada FROM coletas")]
    dataset = m2_write_jsonl(tmp_path / "d.jsonl", items)
    preds = m2_write_jsonl(tmp_path / "p.jsonl", [prediction(items[0])])
    with pytest.raises(GoldError, match="toy-0001"):
        score.score(dataset, preds, db_path=m2_toy_db)


# ---------- metrics.json ----------


def test_write_metrics_merges_and_is_deterministic(
    tmp_path, m2_toy_db, m2_dataset, gold_predictions
) -> None:
    metrics_path = tmp_path / "results" / "metrics.json"
    metrics_path.parent.mkdir()
    metrics_path.write_text(
        json.dumps({"claude-x": {"test": {"ex": 0.5}}}), encoding="utf-8", newline="\n"
    )

    result = score.score(m2_dataset, gold_predictions, db_path=m2_toy_db)
    score.write_metrics(metrics_path, result)
    first = metrics_path.read_bytes()
    score.write_metrics(metrics_path, result)
    assert metrics_path.read_bytes() == first
    assert b"\r\n" not in first

    data = json.loads(first.decode("utf-8"))
    assert list(data) == ["base-brinquedo", "claude-x"]  # chaves ordenadas
    assert data["claude-x"] == {"test": {"ex": 0.5}}
    assert data["base-brinquedo"]["dev"]["ex"] == 1.0
    assert "time" not in first.decode("utf-8").lower()


# ---------- CLI ----------


def test_cli_verify_gold_ok(m2_toy_db, m2_dataset) -> None:
    result = runner.invoke(app, ["eval", "verify-gold", str(m2_dataset), "--db", str(m2_toy_db)])
    assert result.exit_code == 0, result.output
    assert "4" in result.output


def test_cli_verify_gold_lists_bad_ids(tmp_path, m2_toy_db, m2_toy_items, m2_write_jsonl) -> None:
    items = [
        m2_toy_items[0],
        dict(m2_toy_items[1], id="ruim-erro", sql="SELECT nada FROM coletas"),
        dict(m2_toy_items[2], id="ruim-bloqueada", sql="SELECT * FROM read_csv('x.csv')"),
        dict(m2_toy_items[3], id="ruim-vazia", sql="SELECT * FROM coletas WHERE false"),
    ]
    dataset = m2_write_jsonl(tmp_path / "d.jsonl", items)
    result = runner.invoke(app, ["eval", "verify-gold", str(dataset), "--db", str(m2_toy_db)])
    assert result.exit_code != 0
    for bad in ("ruim-erro", "ruim-bloqueada", "ruim-vazia"):
        assert bad in result.output
    assert m2_toy_items[0]["id"] not in result.output.split("Falhas", 1)[-1]


def test_cli_verify_gold_missing_db(tmp_path, m2_dataset) -> None:
    missing = tmp_path / "nao_existe.duckdb"
    result = runner.invoke(app, ["eval", "verify-gold", str(m2_dataset), "--db", str(missing)])
    assert result.exit_code != 0
    assert "nao_existe.duckdb" in result.output


def test_cli_score_and_report(tmp_path, m2_toy_db, m2_dataset, gold_predictions) -> None:
    metrics_path = tmp_path / "metrics.json"
    report_path = tmp_path / "report.md"
    result = runner.invoke(
        app,
        [
            "eval",
            "score",
            str(m2_dataset),
            str(gold_predictions),
            "--db",
            str(m2_toy_db),
            "--metrics",
            str(metrics_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "EX" in result.output
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert data["base-brinquedo"]["dev"]["ex"] == 1.0

    result = runner.invoke(
        app, ["eval", "report", "--metrics", str(metrics_path), "--out", str(report_path)]
    )
    assert result.exit_code == 0, result.output
    text = report_path.read_text(encoding="utf-8")
    assert "base-brinquedo" in text
    assert "100,0%" in text


def test_report_table_is_deterministic() -> None:
    metrics = {
        "ft-b": {"test": {"n": 2, "ex": 0.5}},
        "base-a": {
            "test": {
                "n": 150,
                "ex": 0.4133,
                "ex_by_difficulty": {
                    "facil": {"n": 50, "ex": 0.7},
                    "medio": {"n": 60, "ex": 0.35},
                    "dificil": {"n": 40, "ex": 0.1},
                },
                "valid_sql_rate": 0.9,
                "blocked_rate": 0.0133,
                "latency_ms": {"p50": 812.0, "p95": 1500.5},
                "cost_per_1k_usd": 0.0,
            }
        },
    }
    text = report.render(metrics)
    assert text == report.render(metrics)
    lines = [line for line in text.splitlines() if line.startswith("| ")]
    assert "base-a" in lines[1] and "ft-b" in lines[2]  # ordem alfabética
    assert "41,3%" in lines[1]
    assert "—" in lines[2]  # métricas ausentes
