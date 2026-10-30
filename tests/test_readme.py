"""Os números do README e do card v2 batem com results/report.md (M18, R23).

O relatório é gerado e conferido pelo CI (`tanque eval report --check`). O README e o
card do adapter v2 copiam números dele à mão; este teste garante que a cópia não
desatualizou: a tabela de resultados do README tem de ser igual, célula a célula, à
tabela principal do relatório, e todo percentual com uma casa decimal citado nos dois
textos tem de aparecer no relatório.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
CARD_V2 = ROOT / "hf" / "model-v2" / "README.md"
REPORT = ROOT / "results" / "report.md"

SYSTEM = re.compile(r"`([^`]+)`")
PERCENT = re.compile(r"\d+[,.]\d%")


def _cells(line: str) -> list[str]:
    return [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]


def _table_after(text: str, marker: str) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Primeira tabela markdown depois de `marker`: cabeçalho e linhas por sistema."""
    lines = text[text.index(marker) :].splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("| Sistema"))
    header = _cells(lines[start])
    rows: dict[str, dict[str, str]] = {}
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        cells = _cells(line)
        name = SYSTEM.search(cells[0])
        assert name, f"linha sem sistema entre crases: {line}"
        rows[name.group(1)] = dict(zip(header, cells, strict=True))
    return header, rows


def test_readme_results_table_matches_report():
    report_header, report_rows = _table_after(
        REPORT.read_text(encoding="utf-8"), "## Tabela principal: teste"
    )
    header, rows = _table_after(README.read_text(encoding="utf-8"), "## Resultados")
    assert rows, "README sem tabela de resultados"
    for col in header:
        assert col in report_header, f"coluna {col!r} do README não existe no relatório"
    for system, row in rows.items():
        assert system in report_rows, f"{system} não está no relatório"
        for col in header[1:]:
            assert row[col] == report_rows[system][col], (system, col)


@pytest.mark.parametrize("path", [README, CARD_V2], ids=["readme", "card-v2"])
def test_percentages_come_from_report(path):
    report = REPORT.read_text(encoding="utf-8")
    text = path.read_text(encoding="utf-8")
    missing = sorted({p for p in PERCENT.findall(text) if p.replace(".", ",") not in report})
    assert not missing, f"{path.name}: percentuais fora de results/report.md: {missing}"
