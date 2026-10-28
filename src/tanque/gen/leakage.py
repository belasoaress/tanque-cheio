"""Vazamento entre treino e dev/teste, e deduplicação interna do treino (M4, R6).

Um par de treino **colide** com um item de avaliação quando:

- a pergunta normalizada tem `rapidfuzz.fuzz.token_set_ratio >= 85` com a
  pergunta normalizada do item; ou
- a SQL normalizada é idêntica.

Normalização da pergunta: minúsculas, sem acento, pontuação vira espaço,
espaços colapsados. Normalização da SQL: o sqlglot reescreve a consulta no
dialeto duckdb (`normalize=True`, identificadores em minúsculas); depois tudo vai
para minúsculas e todos os espaços saem. Se a SQL não faz parse, usa o texto
original com a mesma limpeza.

A deduplicação interna do caminho Batch usa o mesmo critério (`dedupe`): na
ordem do arquivo, o primeiro par fica e os seguintes que colidem com algum par já
mantido saem. No treino gerado por templates (D23), a deduplicação interna é
`dedupe_exact`: sai só o par com pergunta normalizada **idêntica** ou SQL
normalizada idêntica à de um par já mantido. O fuzzy trataria "gasolina em SP em
2025" e "gasolina em MG em 2025" como duplicatas, apesar de serem exemplos
distintos. O filtro contra dev/teste não muda.

Atenção ao `token_set_ratio`: quando o conjunto de palavras de uma pergunta está
contido no da outra, a nota é 100. Perguntas do mesmo molde com um parâmetro
trocado ("... em SP em 2025" x "... em MG em 2025") também passam de 85. É o
critério do plano; o efeito no rendimento aparece em `tanque gen stats`.

**Isolamento.** Este é o único módulo que lê `datasets/dev.jsonl`,
`datasets/test.jsonl` e `datasets/candidates_reserve.jsonl` (reserva do M3, que
pode virar dev/teste), sempre em tempo de execução. Nada do conteúdo deles é
impresso: os relatórios mostram só ids e notas.

Desempenho: `rapidfuzz.process.extractOne` com `score_cutoff` (laço em C++, sem
numpy): 5 mil x 200 comparações contra a avaliação e ~5 mil x 5 mil no pior caso
da deduplicação levam segundos.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sqlglot
from rapidfuzz import fuzz, process
from sqlglot.errors import SqlglotError

from tanque import config

THRESHOLD = 85
EVAL_FILES = ("dev.jsonl", "test.jsonl", "candidates_reserve.jsonl")
# Arquivos que precisam existir para a checagem valer.
REQUIRED_EVAL_FILES = ("dev.jsonl", "test.jsonl")

_PUNCT_RE = re.compile(r"[^0-9a-z]+")


def normalize_question(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    no_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _PUNCT_RE.sub(" ", no_accents).strip()


def normalize_sql(sql: str) -> str:
    text = sql.strip().rstrip(";")
    try:
        trees = [t for t in sqlglot.parse(text, read="duckdb") if t is not None]
        if len(trees) == 1:
            text = trees[0].sql(dialect="duckdb", normalize=True)
    except SqlglotError:
        pass
    return re.sub(r"\s+", "", text.lower())


@dataclass(frozen=True)
class Collision:
    index: int  # posição no treino
    kind: str  # "pergunta" ou "sql"
    other_index: int  # posição no conjunto de referência
    other_id: str | None
    score: float


def _question_matches(
    queries: Sequence[str], choices: Sequence[str], threshold: float
) -> dict[int, tuple[int, float]]:
    """{i: (j, nota)} do melhor casamento acima do limiar."""
    found: dict[int, tuple[int, float]] = {}
    if not choices:
        return found
    for i, q in enumerate(queries):
        if not q:
            continue
        best = process.extractOne(
            q, choices, scorer=fuzz.token_set_ratio, processor=None, score_cutoff=threshold
        )
        if best is not None:
            _, score, j = best
            found[i] = (j, float(score))
    return found


def find_collisions(
    train: Sequence[dict[str, Any]],
    reference: Sequence[dict[str, Any]],
    threshold: float = THRESHOLD,
) -> list[Collision]:
    """Colisões de cada par de treino com os itens de referência (dev/teste)."""
    ref_q = [normalize_question(r.get("question", "")) for r in reference]
    ref_sql: dict[str, int] = {}
    for j, r in enumerate(reference):
        ref_sql.setdefault(normalize_sql(r.get("sql", "")), j)
    train_q = [normalize_question(t.get("question", "")) for t in train]
    by_question = _question_matches(train_q, ref_q, threshold)

    out = []
    for i, t in enumerate(train):
        if i in by_question:
            j, score = by_question[i]
            out.append(Collision(i, "pergunta", j, reference[j].get("id"), score))
            continue
        norm = normalize_sql(t.get("sql", ""))
        if norm and norm in ref_sql:
            j = ref_sql[norm]
            out.append(Collision(i, "sql", j, reference[j].get("id"), 100.0))
    return out


def remove_leaks(
    train: Sequence[dict[str, Any]],
    reference: Sequence[dict[str, Any]],
    threshold: float = THRESHOLD,
) -> tuple[list[dict[str, Any]], list[Collision]]:
    collisions = find_collisions(train, reference, threshold)
    bad = {c.index for c in collisions}
    return [t for i, t in enumerate(train) if i not in bad], collisions


def dedupe(
    items: Sequence[dict[str, Any]], threshold: float = THRESHOLD
) -> tuple[list[dict[str, Any]], list[Collision]]:
    """Mantém o primeiro de cada grupo de pares que colidem entre si."""
    kept: list[dict[str, Any]] = []
    kept_q: list[str] = []
    kept_pos: list[int] = []
    kept_sql: dict[str, int] = {}
    removed: list[Collision] = []
    for i, item in enumerate(items):
        q = normalize_question(item.get("question", ""))
        s = normalize_sql(item.get("sql", ""))
        if s and s in kept_sql:
            removed.append(Collision(i, "sql", kept_sql[s], items[kept_sql[s]].get("id"), 100.0))
            continue
        best = (
            process.extractOne(
                q, kept_q, scorer=fuzz.token_set_ratio, processor=None, score_cutoff=threshold
            )
            if q and kept_q
            else None
        )
        if best is not None:
            _, score, j = best
            pos = kept_pos[j]
            removed.append(Collision(i, "pergunta", pos, items[pos].get("id"), float(score)))
            continue
        kept.append(item)
        kept_q.append(q)
        kept_pos.append(i)
        if s:
            kept_sql[s] = i
    return kept, removed


def dedupe_exact(
    items: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[Collision]]:
    """Mantém o primeiro de cada grupo com pergunta ou SQL normalizada idêntica (D23)."""
    kept: list[dict[str, Any]] = []
    kept_q: dict[str, int] = {}
    kept_sql: dict[str, int] = {}
    removed: list[Collision] = []
    for i, item in enumerate(items):
        q = normalize_question(item.get("question", ""))
        s = normalize_sql(item.get("sql", ""))
        if q and q in kept_q:
            j = kept_q[q]
            removed.append(Collision(i, "pergunta", j, items[j].get("id"), 100.0))
            continue
        if s and s in kept_sql:
            j = kept_sql[s]
            removed.append(Collision(i, "sql", j, items[j].get("id"), 100.0))
            continue
        kept.append(item)
        if q:
            kept_q[q] = i
        if s:
            kept_sql[s] = i
    return kept, removed


# --- arquivos de avaliação --------------------------------------------------


def datasets_dir() -> Path:
    return config.setting_path("datasets_dir")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class MissingEvalSetError(FileNotFoundError):
    """dev.jsonl ou test.jsonl não existe: a checagem de vazamento não vale."""


def load_reference(
    directory: str | Path | None = None, names: Iterable[str] = EVAL_FILES
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Itens de dev/teste (e reserva, se existir) e os arquivos lidos."""
    base = Path(directory) if directory is not None else datasets_dir()
    missing = [n for n in REQUIRED_EVAL_FILES if not (base / n).is_file()]
    if missing:
        raise MissingEvalSetError(f"faltam conjuntos de avaliação em {base}: {', '.join(missing)}")
    items: list[dict[str, Any]] = []
    read: list[Path] = []
    for name in names:
        path = base / name
        if path.is_file():
            items.extend(_read_jsonl(path))
            read.append(path)
    return items, read
