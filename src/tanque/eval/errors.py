"""Classificação dos erros de um sistema, para os exemplos do relatório final (M7).

Cada item errado recebe um tipo de `ERROR_TYPES`. A classificação é heurística,
por código, e pode ser sobrescrita à mão em `results/error-annotations.json`
(versionado), que o relatório lê:

```json
{"annotations": {"test-0042": {"type": "join", "note": "usou postos.uf"}}}
```

Heurística, na ordem (o primeiro que se aplica vence):

1. `geracao`: sem previsão, erro na geração ou SQL vazia;
2. `bloqueada`: o guardrail recusou a SQL;
3. `sql_invalida`: a SQL não executou (binder, parse no DuckDB, timeout...);
4. `forma`: número de colunas diferente do gold;
5. `ordenacao`: `order_matters` e as linhas casam como multiconjunto, só a ordem difere;
6. `join`: conjunto de tabelas diferente do gold (CTEs não contam como tabela);
7. `agregacao`: multiconjunto de funções de agregação diferente;
8. `filtro_data`: literais de data, `INTERVAL` ou funções de data diferentes;
9. `filtro_valor`: demais literais de texto diferentes (produto, UF, bandeira...);
10. `logica`: o resto (mesma "forma" de SQL, resultado diferente).

A seleção de exemplos é determinística: percorre os tipos na ordem acima, um
item por tipo por rodada (anotados à mão primeiro, depois por id), até `k`.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp

from tanque.eval.compare import results_match
from tanque.sql.execute import ExecResult

ERROR_TYPES: dict[str, str] = {
    "geracao": "Sem SQL / erro de geração",
    "bloqueada": "Bloqueada pelo guardrail",
    "sql_invalida": "SQL inválida",
    "forma": "Coluna/forma do resultado",
    "ordenacao": "Ordenação",
    "join": "Join errado",
    "agregacao": "Agregação",
    "filtro_data": "Filtro de data",
    "filtro_valor": "Filtro de valor",
    "logica": "Lógica",
}
ANNOTATIONS_FILENAME = "error-annotations.json"

_DATE_LITERAL = re.compile(r"^\d{4}-\d{2}(-\d{2})?")
_DATE_FUNCS = {
    "date_trunc",
    "datetrunc",
    "timestamptrunc",
    "year",
    "month",
    "quarter",
    "week",
    "day",
    "extract",
    "strftime",
    "timetostr",
    "date_part",
    "datepart",
    "current_date",
    "currentdate",
    "date_diff",
    "datediff",
    "date_add",
    "dateadd",
    "date_sub",
    "datesub",
    "last_day",
    "lastday",
    "make_date",
}


class AnnotationsError(ValueError):
    """Arquivo de anotações com tipo desconhecido ou formato inválido."""


@dataclass(frozen=True)
class ClassifiedError:
    id: str
    type: str
    source: str  # "heurística" ou "manual"
    note: str | None = None


# --- traços da SQL ----------------------------------------------------------------


def _parse(sql: str) -> exp.Expression | None:
    try:
        return sqlglot.parse_one(sql, read="duckdb")
    except Exception:  # SQL que o sqlglot não entende: sem traços
        return None


def _tables(tree: exp.Expression) -> set[str]:
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    return {t.name.lower() for t in tree.find_all(exp.Table) if t.name.lower() not in ctes}


def _aggregates(tree: exp.Expression) -> Counter:
    return Counter(type(a).__name__.lower() for a in tree.find_all(exp.AggFunc))


def _func_name(node: exp.Expression) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.name).lower()
    return type(node).__name__.lower()


def _date_traits(tree: exp.Expression) -> Counter:
    traits: Counter = Counter()
    for lit in tree.find_all(exp.Literal):
        if lit.is_string and _DATE_LITERAL.match(lit.this):
            traits[f"lit:{lit.this}"] += 1
    for node in tree.find_all(exp.Interval):
        traits[f"interval:{node.sql(dialect='duckdb').lower()}"] += 1
    for node in tree.find_all(exp.Func):
        name = _func_name(node)
        if name in _DATE_FUNCS:
            traits[f"func:{node.sql(dialect='duckdb').lower()}"] += 1
    return traits


def _value_literals(tree: exp.Expression) -> Counter:
    return Counter(
        lit.this.strip().upper()
        for lit in tree.find_all(exp.Literal)
        if lit.is_string and not _DATE_LITERAL.match(lit.this)
    )


def _width(res: ExecResult) -> int:
    if res.columns:
        return len(res.columns)
    return len(res.rows[0]) if res.rows else 0


def classify(
    gold_sql: str,
    pred_sql: str | None,
    gold: ExecResult,
    pred: ExecResult | None,
    *,
    order_matters: bool,
    generation_error: str | None = None,
) -> str:
    """Tipo do erro (chave de `ERROR_TYPES`) de um item que o sistema errou."""
    if generation_error or not (pred_sql or "").strip() or pred is None:
        return "geracao"
    if pred.blocked:
        return "bloqueada"
    if pred.error:
        return "sql_invalida"
    if _width(gold) != _width(pred):
        return "forma"
    if order_matters and results_match(gold, pred, order_matters=False):
        return "ordenacao"
    g, p = _parse(gold_sql), _parse(pred_sql or "")
    if g is None or p is None:
        return "logica"
    if _tables(g) != _tables(p):
        return "join"
    if _aggregates(g) != _aggregates(p):
        return "agregacao"
    if _date_traits(g) != _date_traits(p):
        return "filtro_data"
    if _value_literals(g) != _value_literals(p):
        return "filtro_valor"
    return "logica"


# --- anotações manuais ------------------------------------------------------------


def load_annotations(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """`{id: {type, note}}` de `results/error-annotations.json` (vazio se não existir)."""
    if path is None or not Path(path).is_file():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    annotations = data.get("annotations", {})
    if not isinstance(annotations, dict):
        raise AnnotationsError("`annotations` deve ser um objeto {id: {type, note}}")
    for item_id, ann in annotations.items():
        if not isinstance(ann, dict) or ann.get("type") not in ERROR_TYPES:
            raise AnnotationsError(
                f"{item_id}: `type` deve ser um de {sorted(ERROR_TYPES)}, veio {ann!r}"
            )
    return annotations


def apply_annotations(
    heuristic: dict[str, str], annotations: dict[str, dict[str, Any]]
) -> list[ClassifiedError]:
    """Junta a heurística e as anotações (a anotação vence); ignora ids que não erraram."""
    out = []
    for item_id in sorted(heuristic):
        ann = annotations.get(item_id)
        if ann:
            out.append(ClassifiedError(item_id, ann["type"], "manual", ann.get("note")))
        else:
            out.append(ClassifiedError(item_id, heuristic[item_id], "heurística"))
    return out


def select_examples(errors: list[ClassifiedError], k: int = 10) -> list[ClassifiedError]:
    """Até `k` exemplos, variando o tipo (rodízio na ordem de `ERROR_TYPES`)."""
    queues = {
        t: sorted((e for e in errors if e.type == t), key=lambda e: (e.source != "manual", e.id))
        for t in ERROR_TYPES
    }
    chosen: list[ClassifiedError] = []
    while len(chosen) < k and any(queues.values()):
        for t in ERROR_TYPES:
            if queues[t] and len(chosen) < k:
                chosen.append(queues[t].pop(0))
    return chosen


def type_counts(errors: list[ClassifiedError]) -> list[tuple[str, int]]:
    """(tipo, n) na ordem de `ERROR_TYPES`, só os tipos presentes."""
    counts = Counter(e.type for e in errors)
    return [(t, counts[t]) for t in ERROR_TYPES if counts[t]]
