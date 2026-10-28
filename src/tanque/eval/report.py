"""Relatório em Markdown a partir de results/metrics.json (versão inicial; o M7 expande).

Gera só a tabela principal: uma linha por (sistema, split), em ordem alfabética.
A saída é determinística (sem data/hora), para poder ser versionada e comparada.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tanque import config
from tanque.eval.score import load_metrics

REPORT_FILENAME = "report.md"
MISSING = "—"

COLUMNS = (
    "Sistema",
    "Split",
    "N",
    "EX",
    "EX fácil",
    "EX médio",
    "EX difícil",
    "SQL válida",
    "Bloqueada",
    "p50 (ms)",
    "p95 (ms)",
    "US$ / 1.000",
)


def default_report_path() -> Path:
    return config.setting_path("results_dir") / REPORT_FILENAME


def _pct(value: Any) -> str:
    if value is None:
        return MISSING
    return f"{float(value) * 100:.1f}%".replace(".", ",")


def _num(value: Any, places: int) -> str:
    if value is None:
        return MISSING
    text = f"{float(value):,.{places}f}"
    # Formato brasileiro: ponto para milhar, vírgula para decimal.
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def _row(system: str, split: str, m: dict[str, Any]) -> list[str]:
    levels = m.get("ex_by_difficulty") or {}
    latency = m.get("latency_ms") or {}
    return [
        f"`{system}`",
        split,
        str(m["n"]) if m.get("n") is not None else MISSING,
        _pct(m.get("ex")),
        _pct((levels.get("facil") or {}).get("ex")),
        _pct((levels.get("medio") or {}).get("ex")),
        _pct((levels.get("dificil") or {}).get("ex")),
        _pct(m.get("valid_sql_rate")),
        _pct(m.get("blocked_rate")),
        _num(latency.get("p50"), 0),
        _num(latency.get("p95"), 0),
        _num(m.get("cost_per_1k_usd"), 4),
    ]


def render(metrics: dict[str, Any]) -> str:
    lines = [
        "# Resultados",
        "",
        "Gerado por `tanque eval report` a partir de `results/metrics.json`.",
        "EX = execution accuracy (seção 5.6 do plano); percentuais sobre todos os itens do split.",
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "|".join("---" for _ in COLUMNS) + "|",
    ]
    for system in sorted(metrics):
        for split in sorted(metrics[system]):
            lines.append("| " + " | ".join(_row(system, split, metrics[system][split])) + " |")
    if len(lines) == 7:
        lines.append("")
        lines.append("_Nenhuma métrica registrada ainda. Rode `tanque eval score`._")
    return "\n".join(lines) + "\n"


def write_report(metrics_path: str | Path, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        f.write(render(load_metrics(metrics_path)))
    return out
