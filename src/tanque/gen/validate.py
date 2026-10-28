"""Validação por execução dos pares gerados (M4).

Passos, na ordem; o primeiro que falha descarta o par com o motivo indicado:

| motivo | regra |
|---|---|
| `campos_vazios` | pergunta ou SQL vazia |
| `intencao_desconhecida` | intenção fora do catálogo |
| `parse` | a SQL não faz parse no sqlglot (dialeto duckdb) |
| `guardrail` | `tanque.sql.guard.check` recusa a SQL |
| `timeout` | a execução passou do timeout |
| `erro_execucao` | o DuckDB devolveu erro |
| `resultado_vazio` | a consulta não devolve linhas |
| `resultado_nulo` | todas as células do resultado são NULL (ex.: AVG sobre nada) |
| `resultado_repetido` | outra pergunta **diferente** da mesma intenção já tem o mesmo resultado |

`resultado_repetido` é o sinal de SQL genérica, que ignora os filtros da pergunta.

Sobre `resultado_repetido`: o primeiro par de cada resultado fica; os seguintes
com pergunta normalizada diferente saem. O resultado é comparado como
multiconjunto de linhas, com os valores normalizados como no comparador
(`tanque.eval.compare.normalize_value`). Isso também remove paráfrases da mesma
pergunta, o que a deduplicação faria de qualquer forma.

Os pares aprovados têm os espaços da SQL normalizados (fora de literais de
texto) e viram exemplos no formato da seção 5.2 (`split = "train"`,
`source = "synthetic"`, `reviewed = false`). A intenção vai em `tags` como
`intent:<id>`, para não mudar o contrato. `uses_window` é detectado na SQL;
`order_matters` vem do gerador e, se faltar, do `ORDER BY` de topo.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from tanque.eval.compare import normalize_value
from tanque.gen import leakage
from tanque.gen.intents import INTENTS_BY_ID
from tanque.sql import execute, guard
from tanque.sql.execute import ExecResult

REASONS = (
    "campos_vazios",
    "intencao_desconhecida",
    "parse",
    "guardrail",
    "timeout",
    "erro_execucao",
    "resultado_vazio",
    "resultado_nulo",
    "resultado_repetido",
)

RunFn = Callable[..., ExecResult]


@dataclass
class Discard:
    pair: dict[str, Any]
    reason: str
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {**self.pair, "reason": self.reason, "detail": self.detail}


@dataclass
class ValidationResult:
    valid: list[dict[str, Any]] = field(default_factory=list)
    discarded: list[Discard] = field(default_factory=list)

    def reasons(self) -> dict[str, int]:
        counts = Counter(d.reason for d in self.discarded)
        return {r: counts[r] for r in REASONS if counts[r]} | {
            r: c for r, c in counts.items() if r not in REASONS
        }


_QUOTED_RE = re.compile(r"('(?:[^']|'')*'|\"(?:[^\"]|\"\")*\")")


def normalize_spaces(sql: str) -> str:
    """Colapsa espaços em branco fora de literais e tira `;` e espaços das pontas."""
    parts = _QUOTED_RE.split(sql.strip())
    out = []
    for i, part in enumerate(parts):
        out.append(part if i % 2 else re.sub(r"\s+", " ", part))
    text = "".join(out).strip()
    while text.endswith(";"):
        text = text[:-1].rstrip()
    return text


def _parse(sql: str) -> exp.Expression | None:
    trees = [t for t in sqlglot.parse(sql, read=guard.DIALECT) if t is not None]
    return trees[0] if len(trees) == 1 else None


def uses_window(tree: exp.Expression) -> bool:
    return tree.find(exp.Window) is not None


def has_top_level_order(tree: exp.Expression) -> bool:
    return tree.args.get("order") is not None


def result_fingerprint(result: ExecResult) -> str:
    """Hash do resultado como multiconjunto de linhas normalizadas."""
    rows = sorted(repr(tuple(normalize_value(v) for v in row)) for row in result.rows)
    width = len(result.columns) if result.columns else (len(result.rows[0]) if result.rows else 0)
    payload = f"{width}|" + "\n".join(rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _all_null(result: ExecResult) -> bool:
    return all(v is None for row in result.rows for v in row)


def validate_pairs(
    pairs: Iterable[dict[str, Any]],
    *,
    db_path: str | Path | None = None,
    timeout_s: float | None = None,
    run: RunFn = execute.run,
) -> ValidationResult:
    """Valida cada par `{question, sql, intent, order_matters?, ...}` (ver docstring)."""
    out = ValidationResult()
    seen: dict[tuple[str, str], str] = {}  # (intenção, resultado) -> pergunta normalizada
    for pair in pairs:
        question = (pair.get("question") or "").strip()
        sql = normalize_spaces(pair.get("sql") or "")
        intent_id = pair.get("intent") or ""
        if not question or not sql:
            out.discarded.append(Discard(pair, "campos_vazios"))
            continue
        intent = INTENTS_BY_ID.get(intent_id)
        if intent is None:
            out.discarded.append(Discard(pair, "intencao_desconhecida", intent_id))
            continue
        try:
            tree = _parse(sql)
        except SqlglotError as e:
            out.discarded.append(Discard(pair, "parse", str(e).splitlines()[0][:200]))
            continue
        if tree is None:
            out.discarded.append(Discard(pair, "parse", "não é exatamente uma instrução"))
            continue
        verdict = guard.check(sql)
        if not verdict.ok:
            out.discarded.append(Discard(pair, "guardrail", verdict.reason))
            continue
        result = run(sql, timeout_s=timeout_s, db_path=db_path)
        if result.error:
            reason = "timeout" if result.error.startswith("timeout") else "erro_execucao"
            if result.blocked:
                reason = "guardrail"
            out.discarded.append(Discard(pair, reason, result.error[:300]))
            continue
        if not result.rows:
            out.discarded.append(Discard(pair, "resultado_vazio"))
            continue
        if _all_null(result):
            out.discarded.append(Discard(pair, "resultado_nulo"))
            continue
        key = (intent_id, result_fingerprint(result))
        norm_q = leakage.normalize_question(question)
        if key in seen and seen[key] != norm_q:
            out.discarded.append(Discard(pair, "resultado_repetido", seen[key]))
            continue
        seen.setdefault(key, norm_q)

        order = pair.get("order_matters")
        if not isinstance(order, bool):
            order = has_top_level_order(tree)
        out.valid.append(
            {
                "question": question,
                "sql": sql,
                "difficulty": intent.level,
                "uses_window": uses_window(tree),
                "order_matters": order,
                "split": "train",
                "source": "synthetic",
                "reviewed": False,
                "tags": [f"intent:{intent_id}", *intent.tags],
            }
        )
    return out


def assign_ids(items: list[dict[str, Any]], prefix: str = "train") -> list[dict[str, Any]]:
    """Ids `train-00001`... na ordem dada, com `id` como primeira chave."""
    return [{"id": f"{prefix}-{i:05d}", **item} for i, item in enumerate(items, start=1)]


def intent_of(item: dict[str, Any]) -> str | None:
    for tag in item.get("tags", []):
        if isinstance(tag, str) and tag.startswith("intent:"):
            return tag.split(":", 1)[1]
    return None
