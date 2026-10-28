"""Tabela do bake-off dos modelos base (M5) em results/bakeoff.md.

Lê `results/metrics.json` (gravado por `tanque eval score`) e monta uma linha
por sistema `base-*` com métricas no split `dev`, em ordem decrescente de EX.
Só o trecho entre os marcadores `BEGIN`/`END` é reescrito: o texto em volta
(contexto, recomendação, escolha da Isabela) é preservado. Saída
determinística, sem data/hora.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tanque import config
from tanque.eval.report import MISSING, _num, _pct
from tanque.eval.score import load_metrics

BAKEOFF_FILENAME = "bakeoff.md"
SPLIT = "dev"
PREFIX = "base-"
BEGIN = "<!-- bakeoff:begin (gerado por `tanque eval bakeoff`; não edite à mão) -->"
END = "<!-- bakeoff:end -->"

COLUMNS = (
    "Sistema",
    "N",
    "EX",
    "EX fácil",
    "EX médio",
    "EX difícil",
    "SQL válida",
    "Bloqueada",
    "p50 (ms)",
    "p95 (ms)",
)


def default_bakeoff_path() -> Path:
    return config.setting_path("results_dir") / BAKEOFF_FILENAME


def _candidates(metrics: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    found = [
        (system, splits[SPLIT])
        for system, splits in metrics.items()
        if system.startswith(PREFIX) and SPLIT in splits
    ]
    return sorted(found, key=lambda sm: (-(sm[1].get("ex") or 0.0), sm[0]))


def rows(metrics: dict[str, Any]) -> list[list[str]]:
    out = []
    for system, m in _candidates(metrics):
        levels = m.get("ex_by_difficulty") or {}
        latency = m.get("latency_ms") or {}
        out.append(
            [
                f"`{system}`",
                str(m["n"]) if m.get("n") is not None else MISSING,
                _pct(m.get("ex")),
                _pct((levels.get("facil") or {}).get("ex")),
                _pct((levels.get("medio") or {}).get("ex")),
                _pct((levels.get("dificil") or {}).get("ex")),
                _pct(m.get("valid_sql_rate")),
                _pct(m.get("blocked_rate")),
                _num(latency.get("p50"), 0),
                _num(latency.get("p95"), 0),
            ]
        )
    return out


def render_table(metrics: dict[str, Any]) -> str:
    body = rows(metrics)
    if not body:
        return (
            "_Nenhum `base-*` com métricas no dev ainda. Rode `tanque eval score` com as "
            "previsões do Kaggle e depois `tanque eval bakeoff`._\n"
        )
    lines = [
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "|".join("---" for _ in COLUMNS) + "|",
        *("| " + " | ".join(r) + " |" for r in body),
    ]
    return "\n".join(lines) + "\n"


def update_document(document: str, table: str) -> str:
    """Troca o trecho entre BEGIN e END por `table`; sem marcadores, acrescenta o bloco."""
    block = f"{BEGIN}\n{table}{END}"
    start, end = document.find(BEGIN), document.find(END)
    if start == -1 or end == -1 or end < start:
        return document.rstrip("\n") + f"\n\n{block}\n"
    return document[:start] + block + document[end + len(END) :]


def write_bakeoff(metrics_path: str | Path, out_path: str | Path) -> Path:
    out = Path(out_path)
    current = out.read_text(encoding="utf-8") if out.is_file() else "# Bake-off dos modelos base\n"
    text = update_document(current, render_table(load_metrics(metrics_path)))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return out
