"""Tabela do bake-off (M5) gerada a partir de results/metrics.json."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from tanque.cli import app
from tanque.eval import bakeoff


def metrics_entry(ex, p50=900.0, valid=0.8, n=50):
    return {
        "dataset": "dev.jsonl",
        "n": n,
        "n_missing": 0,
        "ex": ex,
        "ex_by_difficulty": {
            "facil": {"n": 15, "ex": 0.6},
            "medio": {"n": 20, "ex": 0.35},
            "dificil": {"n": 15, "ex": None},
        },
        "valid_sql_rate": valid,
        "blocked_rate": 0.02,
        "latency_ms": {"p50": p50, "p95": 2100.5},
        "cost_per_1k_usd": 0.0,
    }


METRICS = {
    "base-qwen3-1.7b": {"dev": metrics_entry(0.42, p50=1234.0)},
    "base-smollm3-3b": {"dev": metrics_entry(0.5)},
    "base-old": {"test": metrics_entry(0.9)},  # sem dev: fica de fora
    "ft-qwen3-1.7b": {"dev": metrics_entry(0.7)},  # ajustado: fica de fora
    "claude-x": {"dev": metrics_entry(0.9)},
}


def test_rows_only_base_systems_on_dev_sorted_by_ex():
    rows = bakeoff.rows(METRICS)
    assert [r[0] for r in rows] == ["`base-smollm3-3b`", "`base-qwen3-1.7b`"]


def test_table_formats_like_the_report():
    table = bakeoff.render_table(METRICS)
    lines = table.splitlines()
    assert lines[0].startswith("| Sistema | N | EX | EX fácil")
    assert "| `base-qwen3-1.7b` | 50 | 42,0% | 60,0% | 35,0% | — | 80,0% | 2,0% | 1.234 |" in (
        table
    )


def test_table_without_metrics_says_so():
    assert "Nenhum `base-*` com métricas no dev" in bakeoff.render_table({})


def test_update_replaces_only_between_markers():
    doc = (
        "# Bake-off\n\nTexto da Isabela.\n\n"
        f"{bakeoff.BEGIN}\nvelho\n{bakeoff.END}\n\n## Recomendação\n\nEscolha X.\n"
    )
    out = bakeoff.update_document(doc, "| nova |\n")
    assert "velho" not in out
    assert "| nova |" in out
    assert out.startswith("# Bake-off\n\nTexto da Isabela.")
    assert out.endswith("## Recomendação\n\nEscolha X.\n")
    assert bakeoff.update_document(out, "| nova |\n") == out  # idempotente


def test_update_without_markers_appends_block():
    out = bakeoff.update_document("# Bake-off\n", "| t |\n")
    assert out == f"# Bake-off\n\n{bakeoff.BEGIN}\n| t |\n{bakeoff.END}\n"


def test_committed_skeleton_is_in_sync_with_the_generator():
    doc = bakeoff.default_bakeoff_path().read_text(encoding="utf-8")
    assert bakeoff.BEGIN in doc and bakeoff.END in doc
    if "| Sistema |" not in doc:  # ainda sem números: o gerador não muda nada
        assert bakeoff.update_document(doc, bakeoff.render_table({})) == doc


def test_cli_writes_file(tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps(METRICS), encoding="utf-8")
    out = tmp_path / "bakeoff.md"
    out.write_text(f"# B\n\n{bakeoff.BEGIN}\n{bakeoff.END}\n\nfim\n", encoding="utf-8")
    result = CliRunner().invoke(
        app, ["eval", "bakeoff", "--metrics", str(metrics), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    text = out.read_text(encoding="utf-8")
    assert "`base-smollm3-3b`" in text and text.endswith("fim\n")
