"""Relatório final (M7): classificação de erros, custo de GPU, tabelas e regressão do CI.

Tudo com dados falsos: o banco de brinquedo do conftest e métricas inventadas.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tanque.cli import app
from tanque.eval import errors, report, score
from tanque.sql.execute import ExecResult

runner = CliRunner()
EVAL = "eval"

GOLD_SP = (
    "SELECT AVG(c.valor_venda) FROM coletas c JOIN postos p USING (posto_id) "
    "JOIN municipios m USING (municipio_id) WHERE c.produto = 'GASOLINA' AND m.uf = 'SP'"
)


def ok(rows, columns=None) -> ExecResult:
    width = len(rows[0]) if rows else 1
    return ExecResult(columns=columns or [f"c{i}" for i in range(width)], rows=rows)


# --- classificação --------------------------------------------------------------


def classify(gold_sql, pred_sql, gold=None, pred=None, order_matters=False, gen_error=None):
    gold = gold or ok([(1.0,)])
    pred = pred if pred is not None or pred_sql is None else ok([(2.0,)])
    return errors.classify(
        gold_sql, pred_sql, gold, pred, order_matters=order_matters, generation_error=gen_error
    )


def test_classify_generation_blocked_and_invalid():
    assert classify(GOLD_SP, "", pred=None) == "geracao"
    assert classify(GOLD_SP, "SELECT 1", gen_error="OutOfMemoryError") == "geracao"
    blocked = ExecResult(error="só SELECT", blocked=True)
    assert classify(GOLD_SP, "DROP TABLE coletas", pred=blocked) == "bloqueada"
    invalid = ExecResult(error='Binder Error: column "uf" not found')
    assert classify(GOLD_SP, "SELECT uf FROM coletas", pred=invalid) == "sql_invalida"


def test_classify_shape_and_order():
    assert classify(GOLD_SP, "SELECT 1, 2", pred=ok([(1, 2)])) == "forma"
    gold = ok([("SP", 2.0), ("PE", 1.0)])
    pred = ok([("PE", 1.0), ("SP", 2.0)])
    sql = "SELECT uf, x FROM t ORDER BY x"
    assert classify(sql, sql + " DESC", gold=gold, pred=pred, order_matters=True) == "ordenacao"


def test_classify_by_sql_difference():
    no_join = "SELECT AVG(valor_venda) FROM coletas WHERE produto = 'GASOLINA'"
    assert classify(GOLD_SP, no_join) == "join"
    assert classify(GOLD_SP, GOLD_SP.replace("AVG", "MAX")) == "agregacao"
    dated = GOLD_SP + " AND c.data_coleta >= DATE '2026-09-01'"
    assert classify(dated, dated.replace("2026-09-01", "2026-08-01")) == "filtro_data"
    trunc = GOLD_SP.replace("AVG(c.valor_venda)", "date_trunc('month', c.data_coleta)")
    assert classify(trunc, trunc.replace("'month'", "'quarter'")) == "filtro_data"
    assert classify(GOLD_SP, GOLD_SP.replace("'SP'", "'PE'")) == "filtro_valor"
    assert classify(GOLD_SP, GOLD_SP.replace("=", "<>", 1)) == "logica"


def test_ctes_do_not_count_as_tables():
    gold = "WITH x AS (SELECT * FROM coletas) SELECT count(*) FROM x"
    pred = "WITH y AS (SELECT * FROM coletas) SELECT count(*) FROM y WHERE 1 = 1"
    assert classify(gold, pred) == "logica"


def test_annotations_override_and_validate(tmp_path):
    path = tmp_path / "ann.json"
    path.write_text(
        json.dumps({"annotations": {"t-2": {"type": "join", "note": "postos.uf"}}}),
        encoding="utf-8",
    )
    ann = errors.load_annotations(path)
    out = errors.apply_annotations({"t-1": "logica", "t-2": "logica"}, ann)
    assert [(e.id, e.type, e.source, e.note) for e in out] == [
        ("t-1", "logica", "heurística", None),
        ("t-2", "join", "manual", "postos.uf"),
    ]
    assert errors.load_annotations(tmp_path / "nao-existe.json") == {}
    path.write_text(json.dumps({"annotations": {"t-1": {"type": "outro"}}}), encoding="utf-8")
    with pytest.raises(errors.AnnotationsError, match="t-1"):
        errors.load_annotations(path)


def test_committed_annotations_file_is_valid():
    from tanque import config

    errors.load_annotations(config.resolve_path("results/error-annotations.json"))


def test_select_examples_rotates_types_deterministically():
    items = [errors.ClassifiedError(f"t-{i:02d}", "forma", "heurística") for i in range(8)]
    items += [errors.ClassifiedError("t-50", "join", "heurística")]
    items += [errors.ClassifiedError("t-60", "logica", "heurística")]
    items += [errors.ClassifiedError("t-07b", "forma", "manual")]
    chosen = errors.select_examples(items, k=5)
    assert [(e.id, e.type) for e in chosen] == [
        ("t-07b", "forma"),  # manual primeiro dentro do tipo
        ("t-50", "join"),
        ("t-60", "logica"),
        ("t-00", "forma"),
        ("t-01", "forma"),
    ]
    assert errors.select_examples(items, k=50) == errors.select_examples(items, k=50)
    assert len(errors.select_examples(items, k=50)) == len(items)
    assert errors.type_counts(items) == [("forma", 9), ("join", 1), ("logica", 1)]


# --- custo ----------------------------------------------------------------------

GPU = report.GpuPrice(0.35, "T4", False, "https://exemplo", "2026-10-29", ("kaggle-t4",))


def test_gpu_cost_per_1k():
    # 3,6 s por pergunta -> 1.000 perguntas = 1 h de GPU
    assert report.gpu_cost_per_1k(3600, 0.35) == pytest.approx(0.35)
    assert report.gpu_cost_per_1k(2836, 0.35) == pytest.approx(0.27572, rel=1e-4)


def test_cost_per_1k_api_estimate_or_missing():
    t4 = {"hardware": ["kaggle-t4"], "latency_ms": {"mean": 7200.0}, "cost_per_1k_usd": 0.0}
    assert report.cost_per_1k(t4, GPU) == (pytest.approx(0.7), True)
    api = {"hardware": ["api"], "latency_ms": {"mean": 900.0}, "cost_per_1k_usd": 4.5}
    assert report.cost_per_1k(api, GPU) == (4.5, False)
    cpu = {"hardware": ["cpu"], "latency_ms": {"mean": 900.0}, "cost_per_1k_usd": 0.0}
    assert report.cost_per_1k(cpu, GPU) == (None, False)
    old = {"latency_ms": {"p50": 1.0}, "cost_per_1k_usd": 0.0}  # métricas sem `mean`
    assert report.cost_per_1k(old, GPU) == (None, False)


def test_settings_has_a_sourced_gpu_price():
    gpu = report.load_gpu_price()
    assert gpu is not None and gpu.hour_usd > 0
    assert gpu.source_url.startswith("https://") and gpu.checked_on
    assert "kaggle-t4" in gpu.hardware


# --- documento ------------------------------------------------------------------


def metric(ex, p50, mean, *, cost=0.0, hardware=("kaggle-t4",), n=150):
    return {
        "n": n,
        "ex": ex,
        "ex_by_difficulty": {
            "facil": {"n": 50, "ex": min(1.0, ex + 0.2)},
            "medio": {"n": 60, "ex": ex},
            "dificil": {"n": 40, "ex": max(0.0, ex - 0.2)},
        },
        "valid_sql_rate": 0.9,
        "blocked_rate": 0.01,
        "latency_ms": {"p50": p50, "p95": p50 * 2, "mean": mean},
        "cost_per_1k_usd": cost,
        "hardware": list(hardware),
    }


FAKE = {
    "ft-m": {"test": metric(0.7, 3000.0, 3600.0), "dev": metric(0.74, 8000.0, 8000.0, n=50)},
    "base-m": {"test": metric(0.1, 2800.0, 3240.0)},
}


def test_render_main_table_comparison_and_d24_note():
    text = report.render(FAKE, report.ReportContext(gpu=GPU))
    assert text == report.render(FAKE, report.ReportContext(gpu=GPU))
    main = text.split("## Tabela principal", 1)[1].split("##", 1)[0]
    rows = [line for line in main.splitlines() if line.startswith("| `")]
    assert rows[0].startswith("| `base-m` | test | 150 | 10,0%")
    assert rows[1].startswith("| `ft-m` | test | 150 | 70,0%")
    assert rows[1].endswith("| 0,3500* |")  # 3,6 s × 1.000 = 1 h de T4
    comparison = text.split("## Base → ajustado", 1)[1].split("##", 1)[0]
    assert "| EX | 10,0% | 70,0% | +60,0 p.p. |" in comparison
    assert "| Latência p50 (ms) | 2.800 | 3.000 | +200 |" in comparison
    assert "| US$ / 1.000 | 0,3150* | 0,3500* | +0,0350 |" in comparison
    assert "Sem LLM de fronteira por enquanto (D24)" in text
    assert "_Pendente: precisa de `results/predictions/ft-*__test.jsonl`._" in text
    assert "## Referência: dev" in text and "`ft-m` | dev | 50 | 74,0%" in text


def test_third_system_enters_by_itself():
    metrics = {
        **FAKE,
        "claude-x": {"test": metric(0.8, 900.0, 950.0, cost=4.2, hardware=("api",))},
    }
    text = report.render(metrics, report.ReportContext(gpu=GPU))
    main = text.split("## Tabela principal", 1)[1].split("##", 1)[0]
    rows = [line for line in main.splitlines() if line.startswith("| `")]
    assert [r.split("`")[1] for r in rows] == ["base-m", "ft-m", "claude-x"]
    assert rows[2].endswith("| 4,2000 |")  # custo real da API, sem `*`
    assert "inclui a rede" in text and "D24" not in text


def test_main_table_shows_relaxed_ex_and_confidence_intervals():
    ft = {
        **metric(0.7, 3000.0, 3600.0),
        "ex_ci95": [0.6267, 0.7733],
        "ex_relaxed": 0.82,
        "ex_relaxed_ci95": [0.7533, 0.88],
    }
    text = report.render({**FAKE, "ft-m": {**FAKE["ft-m"], "test": ft}}, report.ReportContext())
    header = next(line for line in text.splitlines() if line.startswith("| Sistema"))
    assert "| EX | IC 95% EX | EX tolerante | IC 95% tolerante | EX fácil |" in header
    assert "| `ft-m` | test | 150 | 70,0% | 62,7–77,3% | 82,0% | 75,3–88,0% | 90,0% |" in text
    # métricas antigas, sem os campos novos: traço, sem quebrar
    assert "| `base-m` | test | 150 | 10,0% | — | — | — | 30,0% |" in text
    comparison = text.split("## Base → ajustado", 1)[1].split("##", 1)[0]
    assert "| EX tolerante | — | 82,0% | — |" in comparison


def test_paired_comparisons_table_and_method_note():
    metrics = {
        **FAKE,
        "comparisons": [
            {
                "a": "base-m",
                "b": "ft-m",
                "split": "test",
                "n": 150,
                "a_only": 2,
                "b_only": 92,
                "delta_ex": 0.6,
                "mcnemar_p": 2.4e-25,
            },
            {
                "a": "base-m",
                "b": "ft-m",
                "split": "dev",
                "n": 50,
                "a_only": 3,
                "b_only": 6,
                "delta_ex": 0.06,
                "mcnemar_p": 0.5078,
            },
        ],
    }
    text = report.render(metrics, report.ReportContext())
    paired = text.split("## Comparações pareadas (teste)", 1)[1].split("##", 1)[0]
    assert "| A | B | N | EX A | EX B | Δ EX (B − A) | Só A | Só B | p (McNemar) |" in paired
    assert "| `base-m` | `ft-m` | 150 | 10,0% | 70,0% | +60,0 p.p. | 2 | 92 | < 0,001 |" in paired
    dev = text.split("## Referência: dev", 1)[1]
    assert "| `base-m` | `ft-m` | 50 | — | 74,0% | +6,0 p.p. | 3 | 6 | 0,508 |" in dev
    note = text.split("## Nota de método", 1)[1]
    assert "**EX tolerante (secundária, D28" in note
    assert "10.000 reamostragens" in note and "seed 42" in note
    assert "McNemar exato" in note
    # a chave `comparisons` não vira sistema nem split
    assert "`comparisons`" not in text.split("## Nota de método", 1)[0]


def test_retry_rows_ft2_large_models_and_dev2_section():
    metrics = {
        "base-qwen2.5-coder-1.5b-instruct": {"test": metric(0.1, 2800.0, 3000.0)},
        "base-qwen2.5-coder-32b-instruct": {"test": metric(0.6, 9000.0, 9000.0)},
        "base-qwen2.5-coder-7b-instruct": {"test": metric(0.4, 5000.0, 5000.0)},
        "ft-qwen2.5-coder-1.5b-instruct": {
            "test": metric(0.7, 3000.0, 3600.0),
            "dev2": metric(0.6, 3000.0, 3000.0, n=113),
        },
        "ft-qwen2.5-coder-1.5b-instruct+retry": {
            "test": metric(0.75, 3100.0, 3700.0),
            "dev2": metric(0.65, 3100.0, 3100.0, n=113),
        },
        "ft2-qwen2.5-coder-1.5b-instruct": {"test": metric(0.8, 3000.0, 3600.0)},
        "comparisons": [
            {
                "a": "ft-qwen2.5-coder-1.5b-instruct",
                "b": "ft-qwen2.5-coder-1.5b-instruct+retry",
                "split": "test",
                "n": 150,
                "a_only": 0,
                "b_only": 8,
                "delta_ex": 0.05,
                "mcnemar_p": 0.0078,
            },
            {
                "a": "ft-qwen2.5-coder-1.5b-instruct",
                "b": "ft-qwen2.5-coder-1.5b-instruct+retry",
                "split": "dev2",
                "n": 113,
                "a_only": 0,
                "b_only": 6,
                "delta_ex": 0.05,
                "mcnemar_p": 0.0312,
            },
        ],
    }
    text = report.render(metrics, report.ReportContext())
    main = text.split("## Tabela principal", 1)[1].split("##", 1)[0]
    rows = [line.split("`")[1] for line in main.splitlines() if line.startswith("| `")]
    assert rows == [
        "base-qwen2.5-coder-1.5b-instruct",
        "base-qwen2.5-coder-7b-instruct",
        "base-qwen2.5-coder-32b-instruct",
        "ft-qwen2.5-coder-1.5b-instruct",
        "ft-qwen2.5-coder-1.5b-instruct+retry",
        "ft2-qwen2.5-coder-1.5b-instruct",
    ]
    comparison = text.split("## Base → ajustado", 1)[1].split("## Comparações", 1)[0]
    assert "`base-qwen2.5-coder-1.5b-instruct` → `ft2-qwen2.5-coder-1.5b-instruct`" in comparison
    paired = text.split("## Comparações pareadas (teste)", 1)[1].split("##", 1)[0]
    assert "| 70,0% | 75,0% | +5,0 p.p. | 0 | 8 | 0,008 |" in paired
    dev2 = text.split("## Referência: dev2 (113 perguntas)", 1)[1].split("## Nota", 1)[0]
    assert "| `ft-qwen2.5-coder-1.5b-instruct+retry` | dev2 | 113 | 65,0%" in dev2
    assert "Comparações pareadas no dev2:" in dev2 and "| 0,031 |" in dev2
    assert "## Split `dev2`" not in text
    assert "**Nova tentativa (D32):**" in text


def test_final_system_defaults_to_best_single_shot_ft():
    metrics = {
        "ft-a": {"test": {"ex": 0.7}},
        "ft-a+retry": {"test": {"ex": 0.9}},
        "ft2-a": {"test": {"ex": 0.8}},
        "ft-b": {"test": {"ex": 0.8}},
    }
    candidates = ["ft-a", "ft-a+retry", "ft-b", "ft2-a"]
    # empate entre ft-b e ft2-a: vale a ordem dos sistemas (ft antes de ft2); +retry fora
    assert report.final_system(metrics, candidates) == "ft-b"
    assert report.final_system(metrics, candidates, "ft-a") == "ft-a"
    assert report.final_system(metrics, ["ft-a+retry"]) is None
    with pytest.raises(report.ReportInputError):
        report.final_system(metrics, candidates, "ft-a+retry")


def test_paired_table_pending_without_comparisons():
    text = report.render(FAKE, report.ReportContext())
    paired = text.split("## Comparações pareadas (teste)", 1)[1].split("##", 1)[0]
    assert "_Pendente: precisa de 2 sistemas pontuados no teste" in paired


def test_render_without_test_metrics_says_pending():
    text = report.render({"base-m": {"dev": metric(0.1, 1.0, 1.0, n=50)}})
    assert "_Previsões do teste ainda não pontuadas" in text
    assert "_Pendente: precisa de `base-*` e `ft-*`" in text


# --- ponta a ponta com o banco de brinquedo -------------------------------------


def _pred(item, sql, system="ft-brinquedo", latency=1000):
    return {
        "id": item["id"],
        "system": system,
        "sql": sql,
        "raw_output": sql,
        "latency_ms": latency,
        "input_tokens": 800,
        "output_tokens": 40,
        "cost_usd": 0.0,
        "hardware": "kaggle-t4",
        "error": None,
    }


@pytest.fixture
def workspace(tmp_path, m2_toy_db, m2_toy_items, m2_write_jsonl):
    """results/ (metrics, predictions, anotações) e datasets/ falsos, com split `test`."""
    items = [{**i, "split": "test"} for i in m2_toy_items]
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    m2_write_jsonl(datasets / "test.jsonl", items)
    m2_write_jsonl(datasets / "dev.jsonl", [{**i, "split": "dev"} for i in m2_toy_items])
    preds = tmp_path / "results" / "predictions"
    preds.mkdir(parents=True)
    facil1, facil2, medio, dificil = items
    m2_write_jsonl(
        preds / "ft-brinquedo__test.jsonl",
        [
            _pred(facil1, facil1["sql"].replace("'SP'", "'PE'")),  # filtro de valor
            _pred(facil2, "SELECT produto FROM coletas GROUP BY produto"),  # forma
            _pred(medio, medio["sql"].replace("DESC", "ASC")),  # ordenação
            _pred(dificil, "SELECT * FROM nao_existe"),  # SQL inválida
        ],
    )
    m2_write_jsonl(
        preds / "base-brinquedo__test.jsonl",
        [_pred(i, "SELECT 1", system="base-brinquedo") for i in items],
    )
    m2_write_jsonl(
        preds / "base-brinquedo__dev.jsonl",
        [_pred(i, i["sql"], system="base-brinquedo") for i in items],
    )
    # amostra de ensaio: fica fora da regressão
    m2_write_jsonl(preds / "base-brinquedo__dev-sample2.jsonl", [_pred(items[0], "SELECT 1")])
    return tmp_path, m2_toy_db


def _cli(*args):
    return runner.invoke(app, [EVAL, *map(str, args)])


def test_check_metrics_write_then_check_then_detect_drift(workspace):
    root, db = workspace
    metrics = root / "results" / "metrics.json"
    common = (
        "--db",
        db,
        "--metrics",
        metrics,
        "--predictions-dir",
        root / "results" / "predictions",
        "--datasets-dir",
        root / "datasets",
    )
    result = _cli("check-metrics", *common, "--write")
    assert result.exit_code == 0, result.output
    data = json.loads(metrics.read_text(encoding="utf-8"))
    assert set(data) == {"base-brinquedo", "ft-brinquedo", "comparisons"}
    assert data["comparisons"] == [
        {
            "a": "base-brinquedo",
            "b": "ft-brinquedo",
            "split": "test",
            "n": 4,
            "a_only": 0,
            "b_only": 0,
            "delta_ex": 0.0,
            "mcnemar_p": 1.0,
        }
    ]
    assert data["ft-brinquedo"]["test"]["ex_relaxed"] == 0.0
    assert data["ft-brinquedo"]["test"]["ex_ci95"] == [0.0, 0.0]
    assert set(data["base-brinquedo"]) == {"dev", "test"}
    assert data["base-brinquedo"]["dev"]["ex"] == 1.0
    assert data["ft-brinquedo"]["test"]["ex"] == 0.0

    result = _cli("check-metrics", *common)
    assert result.exit_code == 0, result.output
    assert "OK: 3 arquivo(s)" in result.output

    data["ft-brinquedo"]["test"]["ex"] = 0.5
    data["fantasma"] = {"dev": {}}
    data["comparisons"][0]["mcnemar_p"] = 0.5
    metrics.write_text(score.dump_json(data), encoding="utf-8")
    result = _cli("check-metrics", *common)
    assert result.exit_code == 1
    assert "ft-brinquedo.test.ex" in result.output
    assert "fantasma: está em metrics.json, mas não foi recalculado" in result.output
    assert "comparisons" in result.output


def test_check_metrics_rejects_a_file_named_after_another_system(workspace):
    root, db = workspace
    preds = root / "results" / "predictions"
    shutil.copy2(preds / "ft-brinquedo__test.jsonl", preds / "ft-outro__test.jsonl")
    result = _cli(
        "check-metrics",
        "--db",
        db,
        "--metrics",
        root / "results" / "metrics.json",
        "--predictions-dir",
        preds,
        "--datasets-dir",
        root / "datasets",
    )
    assert result.exit_code == 1
    assert "não bate com o nome do arquivo" in result.output


def test_report_command_classifies_ft_errors_and_check_detects_edits(workspace):
    root, db = workspace
    results = root / "results"
    paths = (
        "--metrics",
        results / "metrics.json",
        "--predictions-dir",
        results / "predictions",
        "--datasets-dir",
        root / "datasets",
    )
    assert _cli("check-metrics", "--db", db, *paths, "--write").exit_code == 0
    ann = results / "error-annotations.json"
    ann.write_text(
        json.dumps({"annotations": {"toy-0004": {"type": "join", "note": "tabela inventada"}}}),
        encoding="utf-8",
    )
    out = results / "report.md"
    args = ("report", "--db", db, *paths, "--annotations", ann, "--out", out)
    result = _cli(*args)
    assert result.exit_code == 0, result.output
    text = out.read_text(encoding="utf-8")
    section = text.split("## Erros do modelo ajustado", 1)[1].split("## Referência", 1)[0]
    assert "4 erros em 4 perguntas" in section
    for label in ("Coluna/forma do resultado", "Ordenação", "Join errado", "Filtro de valor"):
        assert label in section
    assert "`toy-0004`** · dificil · Join errado (manual)" in section
    assert "Nota: tabela inventada" in section
    assert "SELECT * FROM nao_existe" in section
    assert "`base-brinquedo` → `ft-brinquedo`" in text
    assert "Modelo final: `ft-brinquedo` (o `ft*` sem nova tentativa de maior EX" in text

    assert _cli(*args, "--check").exit_code == 0
    out.write_text(text.replace("4 erros", "3 erros"), encoding="utf-8")
    result = _cli(*args, "--check")
    assert result.exit_code == 1 and "desatualizado" in result.output


def test_retry_files_do_not_duplicate_error_examples(workspace):
    root, db = workspace
    results = root / "results"
    preds = results / "predictions"
    rows = [
        {**json.loads(line), "system": "ft-brinquedo+retry"}
        for line in (preds / "ft-brinquedo__test.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    (preds / "ft-brinquedo+retry__test.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    paths = (
        "--metrics",
        results / "metrics.json",
        "--predictions-dir",
        preds,
        "--datasets-dir",
        root / "datasets",
    )
    result = _cli("check-metrics", "--db", db, *paths, "--write")
    assert result.exit_code == 0, result.output
    data = json.loads((results / "metrics.json").read_text(encoding="utf-8"))
    pairs = [(c["split"], c["a"], c["b"]) for c in data["comparisons"]]
    assert pairs == [
        ("test", "base-brinquedo", "ft-brinquedo"),
        ("test", "ft-brinquedo", "ft-brinquedo+retry"),
    ]
    out = results / "report.md"
    result = _cli("report", "--db", db, *paths, "--out", out)
    assert result.exit_code == 0, result.output
    text = out.read_text(encoding="utf-8")
    assert text.count("### `ft-brinquedo") == 1
    assert "| `ft-brinquedo+retry` | test |" in text
    result = _cli("report", "--db", db, *paths, "--out", out, "--final-system", "nao-existe")
    assert result.exit_code == 2 and "modelo final" in result.output


def test_report_needs_the_database_only_for_ft_test_predictions(workspace, tmp_path):
    root, _ = workspace
    result = _cli(
        "report",
        "--db",
        tmp_path / "nao-existe.duckdb",
        "--metrics",
        root / "results" / "metrics.json",
        "--predictions-dir",
        root / "results" / "predictions",
        "--datasets-dir",
        root / "datasets",
        "--out",
        tmp_path / "r.md",
    )
    assert result.exit_code == 2
    assert "Banco não encontrado" in result.output


def test_unmerged_adapter_latency_is_flagged(tmp_path):
    preds = tmp_path / "predictions"
    preds.mkdir()
    info = {"system": "ft-m", "split": "dev", "adapter_id": "/kaggle/working/adapter"}
    (preds / "run-info__ft-m__dev.json").write_text(json.dumps(info), encoding="utf-8")
    merged = {**info, "split": "test", "adapter_merged": True}
    (preds / "run-info__ft-m__test.json").write_text(json.dumps(merged), encoding="utf-8")
    assert report.unmerged_runs(preds) == {("ft-m", "dev")}
    ctx = report.ReportContext(gpu=GPU, unmerged=report.unmerged_runs(preds))
    text = report.render(FAKE, ctx)
    assert "| 8.000† | 16.000† |" in text and "† Latência medida com o adapter" in text
    assert "3.000† " not in text  # o teste, fundido, não ganha a marca


def test_committed_report_is_up_to_date_with_metrics():
    """Sem previsões do teste do ft, o relatório não precisa do banco: confere já no CI."""
    from tanque import config

    preds = config.setting_path("predictions_dir")
    if list(Path(preds).glob("ft-*__test.jsonl")) and not config.setting_path("db_path").is_file():
        pytest.skip("previsões do teste presentes: a conferência roda no job de regressão")
    result = _cli("report", "--check")
    assert result.exit_code == 0, result.output
