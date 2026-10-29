"""Gráfico EX × custo por 1.000 perguntas (M17, R22), com métricas falsas."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from tanque.cli import app
from tanque.eval import chart, report

runner = CliRunner()
GPU = report.GpuPrice(0.35, "T4", False, "https://exemplo", "2026-10-29", ("kaggle-t4",))


def m(ex, mean_ms, gpus=1, ci=None):
    return {
        "n": 150,
        "ex": ex,
        "ex_ci95": ci or [round(ex - 0.07, 4), round(ex + 0.07, 4)],
        "latency_ms": {"p50": mean_ms, "p95": mean_ms * 2, "mean": mean_ms},
        "cost_per_1k_usd": 0.0,
        "hardware": ["kaggle-t4"],
        "gpus": gpus,
    }


FAKE = {
    "base-qwen2.5-coder-1.5b-instruct": {"test": m(0.12, 3600.0)},
    "base-qwen2.5-coder-1.5b-instruct+retry": {"test": m(0.15, 5000.0)},
    "base-qwen2.5-coder-32b-instruct": {"test": m(0.72, 36000.0, gpus=2)},
    "ft-qwen2.5-coder-1.5b-instruct": {"test": m(0.76, 4000.0)},
    "ft2-qwen2.5-coder-1.5b-instruct": {"test": m(0.82, 4100.0)},
    "ft2-qwen2.5-coder-1.5b-instruct+retry": {"test": m(0.83, 4200.0)},
    "base-qwen3-1.7b": {"dev": m(0.05, 1000.0)},  # outro split: fora
    "claude-x": {"test": {**m(0.9, 900.0), "hardware": ["api"]}},  # sem custo: fora
    "comparisons": [],
}


def test_short_labels_name_the_size_and_the_training():
    assert chart.short_label("base-qwen2.5-coder-1.5b-instruct") == "1,5B base"
    assert chart.short_label("ft-qwen2.5-coder-1.5b-instruct") == "1,5B ft-v1"
    assert chart.short_label("ft2-qwen2.5-coder-1.5b-instruct+retry") == "1,5B ft-v2"
    assert chart.short_label("base-qwen2.5-coder-32b-instruct") == "32B base"
    assert chart.short_label("claude-x") == "claude-x"


def test_points_use_the_report_cost_and_skip_what_has_no_cost():
    pts = chart.points(FAKE, gpu=GPU)
    assert [p.system for p in pts] == [
        "base-qwen2.5-coder-1.5b-instruct",
        "base-qwen2.5-coder-1.5b-instruct+retry",
        "base-qwen2.5-coder-32b-instruct",
        "ft-qwen2.5-coder-1.5b-instruct",
        "ft2-qwen2.5-coder-1.5b-instruct",
        "ft2-qwen2.5-coder-1.5b-instruct+retry",
    ]
    first, big = pts[0], pts[2]
    assert first.cost == pytest.approx(0.35) and first.retry is False and first.group == "base"
    assert big.cost == pytest.approx(7.0)  # 10 s × 1.000 = 10 h × 2 T4 × 0,35 (D35)
    assert (big.ex, big.lo, big.hi) == (0.72, 0.65, 0.79)
    assert pts[1].retry is True and pts[5].group == "ft2"
    assert chart.points(FAKE, split="dev2", gpu=GPU) == []


def test_digest_changes_with_the_data_and_ignores_key_order():
    pts = chart.points(FAKE, gpu=GPU)
    reordered = dict(reversed(list(FAKE.items())))
    assert chart.data_digest(pts) == chart.data_digest(chart.points(reordered, gpu=GPU))
    other = {**FAKE, "ft-qwen2.5-coder-1.5b-instruct": {"test": m(0.70, 4000.0)}}
    assert chart.data_digest(pts) != chart.data_digest(chart.points(other, gpu=GPU))


def test_png_text_chunks_reads_what_was_written(tmp_path):
    path = tmp_path / "x.png"
    chart.write_png_with_text(path, {"tanque-data": "abc"})  # PNG mínimo, sem imagem
    assert chart.png_text_chunks(path) == {"tanque-data": "abc"}
    (tmp_path / "nao.png").write_bytes(b"isto nao e um png")
    assert chart.png_text_chunks(tmp_path / "nao.png") is None


def test_check_states(tmp_path):
    png = tmp_path / "ex-vs-cost.png"
    pts = chart.points(FAKE, gpu=GPU)
    ok, msg = chart.check(pts, png)
    assert not ok and "não existe" in msg
    chart.write_png_with_text(png, {chart.DIGEST_KEY: "outro"})
    ok, msg = chart.check(pts, png)
    assert not ok and "desatualizado" in msg
    chart.write_png_with_text(png, {chart.DIGEST_KEY: chart.data_digest(pts)})
    assert chart.check(pts, png)[0]
    # sem métricas do teste: nada a desenhar, e um PNG velho seria engano
    assert chart.check([], tmp_path / "nenhum.png")[0]
    ok, msg = chart.check([], png)
    assert not ok and "não há métricas" in msg


def test_render_writes_a_png_with_the_digest(tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "ex-vs-cost.png"
    pts = chart.points(FAKE, gpu=GPU)
    chart.render(pts, out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert chart.png_text_chunks(out)[chart.DIGEST_KEY] == chart.data_digest(pts)
    assert chart.check(pts, out)[0]
    # mesma entrada, mesmo arquivo (no mesmo ambiente)
    first = out.read_bytes()
    chart.render(pts, out)
    assert out.read_bytes() == first


def test_chart_command_writes_and_checks(tmp_path):
    pytest.importorskip("matplotlib")
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps(FAKE), encoding="utf-8")
    out = tmp_path / "ex-vs-cost.png"
    args = ["eval", "chart", "--metrics", str(metrics), "--out", str(out)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert out.is_file()
    assert runner.invoke(app, [*args, "--check"]).exit_code == 0
    changed = {**FAKE, "ft-qwen2.5-coder-1.5b-instruct": {"test": m(0.70, 4000.0)}}
    metrics.write_text(json.dumps(changed), encoding="utf-8")
    result = runner.invoke(app, [*args, "--check"])
    assert result.exit_code == 1 and "desatualizado" in result.output


def test_chart_check_without_test_metrics_passes(tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"base-m": {"dev": m(0.1, 1.0)}}), encoding="utf-8")
    out = tmp_path / "ex-vs-cost.png"
    args = ["eval", "chart", "--metrics", str(metrics), "--out", str(out)]
    result = runner.invoke(app, [*args, "--check"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, args)  # não grava nada sem pontos
    assert result.exit_code == 0 and not out.exists()


def test_committed_chart_matches_the_committed_metrics():
    result = runner.invoke(app, ["eval", "chart", "--check"])
    assert result.exit_code == 0, result.output
