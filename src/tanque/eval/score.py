"""Cálculo das métricas de um sistema sobre um dataset (seções 5.2, 5.3 e 5.6).

Para cada item do dataset, executa a SQL de referência (gold) e a SQL prevista
com `tanque.sql.execute.run` (guardrail ligado) e compara com
`tanque.eval.compare.results_match`.

Métricas (todas sobre os `n` itens do dataset):

- `ex`: fração de acertos; item sem previsão, bloqueado ou com erro conta como errado;
- `ex_by_difficulty`: `{nivel: {n, ex}}` para `facil`, `medio` e `dificil`;
- `valid_sql_rate`: fração com SQL que passou no guardrail e executou sem erro
  (item sem previsão ou com `error` de geração conta como inválido);
- `blocked_rate`: fração bloqueada pelo guardrail;
- `latency_ms`: p50 e p95 dos `latency_ms` das previsões (interpolação linear);
- `cost_per_1k_usd`: `soma(cost_usd) / n * 1000`.

`write_metrics` grava em `results/metrics.json` como `{system: {split: métricas}}`,
com chaves ordenadas e sem horário, para o CI poder comparar o arquivo byte a byte.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tanque import config
from tanque.eval.compare import results_match
from tanque.sql import execute
from tanque.sql.execute import ExecResult

DIFFICULTIES = ("facil", "medio", "dificil")
METRICS_FILENAME = "metrics.json"


class GoldError(RuntimeError):
    """Alguma SQL de referência falhou; rode `tanque eval verify-gold`."""


class PredictionsError(ValueError):
    """Arquivo de previsões inconsistente (ids repetidos, sistemas misturados...)."""


@dataclass
class ItemResult:
    id: str
    difficulty: str
    has_prediction: bool
    correct: bool
    valid: bool
    blocked: bool
    pred_error: str | None


@dataclass
class ScoreResult:
    system: str
    split: str
    metrics: dict[str, Any]
    items: list[ItemResult]


@dataclass
class GoldProblem:
    id: str
    reason: str


def default_metrics_path() -> Path:
    return config.setting_path("results_dir") / METRICS_FILENAME


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: JSON inválido: {e}") from e
    return rows


def percentile(values: Iterable[float], q: float) -> float | None:
    """Percentil `q` (0–100) com interpolação linear entre vizinhos (como o numpy)."""
    data = sorted(values)
    if not data:
        return None
    pos = (len(data) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _run(sql: str, db_path: Path | None, timeout_s: float | None, max_rows: int | None):
    return execute.run(sql, timeout_s=timeout_s, max_rows=max_rows, db_path=db_path)


def check_gold(
    items: list[dict[str, Any]],
    db_path: str | Path | None = None,
    timeout_s: float | None = None,
    max_rows: int | None = None,
) -> tuple[list[GoldProblem], dict[str, ExecResult]]:
    """Executa cada SQL de referência; devolve os problemas e os resultados por id.

    Problema = id repetido, SQL bloqueada pelo guardrail, erro de execução ou
    resultado sem linhas.
    """
    db = Path(db_path) if db_path is not None else None
    problems: list[GoldProblem] = []
    results: dict[str, ExecResult] = {}
    for item in items:
        item_id = str(item.get("id"))
        if item_id in results:
            problems.append(GoldProblem(item_id, "id repetido no dataset"))
            continue
        res = _run(str(item.get("sql") or ""), db, timeout_s, max_rows)
        results[item_id] = res
        if res.blocked:
            problems.append(GoldProblem(item_id, res.error or "bloqueada pelo guardrail"))
        elif res.error:
            problems.append(GoldProblem(item_id, f"erro: {res.error}"))
        elif not res.rows:
            problems.append(GoldProblem(item_id, "resultado vazio"))
    return problems, results


def _index_predictions(predictions: list[dict[str, Any]]) -> tuple[str, dict[str, dict]]:
    by_id: dict[str, dict] = {}
    duplicates = []
    for pred in predictions:
        pid = str(pred.get("id"))
        if pid in by_id:
            duplicates.append(pid)
        by_id[pid] = pred
    if duplicates:
        raise PredictionsError(f"ids repetidos nas previsões: {', '.join(sorted(set(duplicates)))}")
    systems = sorted({str(p.get("system")) for p in predictions})
    if len(systems) != 1:
        raise PredictionsError(f"o arquivo deve ter um único `system`; encontrei {systems}")
    return systems[0], by_id


def _split_of(items: list[dict[str, Any]], dataset_path: Path) -> str:
    splits = {str(i["split"]) for i in items if i.get("split")}
    if len(splits) == 1:
        return splits.pop()
    return dataset_path.stem


def score(
    dataset_path: str | Path,
    predictions_path: str | Path,
    db_path: str | Path | None = None,
    timeout_s: float | None = None,
    max_rows: int | None = None,
) -> ScoreResult:
    """Calcula as métricas das previsões em `predictions_path` sobre `dataset_path`."""
    dataset_path = Path(dataset_path)
    items = read_jsonl(dataset_path)
    predictions = read_jsonl(predictions_path)
    if not items:
        raise ValueError(f"dataset vazio: {dataset_path}")
    if not predictions:
        raise PredictionsError(f"arquivo de previsões vazio: {predictions_path}")
    system, preds = _index_predictions(predictions)

    problems, gold_results = check_gold(items, db_path, timeout_s, max_rows)
    if problems:
        listed = "; ".join(f"{p.id}: {p.reason}" for p in problems)
        raise GoldError(
            f"SQL de referência com problema ({listed}). Rode `tanque eval verify-gold`."
        )

    db = Path(db_path) if db_path is not None else None
    results: list[ItemResult] = []
    for item in items:
        item_id = str(item["id"])
        difficulty = str(item.get("difficulty"))
        pred = preds.get(item_id)
        if pred is None:
            results.append(
                ItemResult(item_id, difficulty, False, False, False, False, "sem previsão")
            )
            continue
        sql = pred.get("sql") or ""
        if pred.get("error") or not str(sql).strip():
            reason = f"erro na geração: {pred.get('error')}" if pred.get("error") else "SQL vazia"
            results.append(ItemResult(item_id, difficulty, True, False, False, False, reason))
            continue
        res = _run(str(sql), db, timeout_s, max_rows)
        valid = not res.blocked and res.error is None
        correct = valid and results_match(
            gold_results[item_id], res, order_matters=bool(item.get("order_matters"))
        )
        results.append(
            ItemResult(item_id, difficulty, True, correct, valid, res.blocked, res.error)
        )

    n = len(results)
    by_level = {}
    for level in DIFFICULTIES:
        level_items = [r for r in results if r.difficulty == level]
        by_level[level] = {
            "n": len(level_items),
            "ex": _rate(sum(r.correct for r in level_items), len(level_items)),
        }
    unknown = sorted({r.difficulty for r in results} - set(DIFFICULTIES))
    if unknown:
        raise ValueError(f"dificuldade fora de {DIFFICULTIES}: {unknown}")

    present = [preds[r.id] for r in results if r.has_prediction]
    latencies = [float(p["latency_ms"]) for p in present if p.get("latency_ms") is not None]
    p50, p95 = percentile(latencies, 50), percentile(latencies, 95)
    total_cost = sum(float(p.get("cost_usd") or 0.0) for p in present)

    metrics = {
        "dataset": dataset_path.name,
        "n": n,
        "n_missing": sum(not r.has_prediction for r in results),
        "ex": _rate(sum(r.correct for r in results), n),
        "ex_by_difficulty": by_level,
        "valid_sql_rate": _rate(sum(r.valid for r in results), n),
        "blocked_rate": _rate(sum(r.blocked for r in results), n),
        "latency_ms": {
            "p50": None if p50 is None else round(p50, 1),
            "p95": None if p95 is None else round(p95, 1),
        },
        "cost_per_1k_usd": round(total_cost / n * 1000, 6),
    }
    return ScoreResult(system, _split_of(items, dataset_path), metrics, results)


def load_metrics(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    return json.loads(text) if text.strip() else {}


def dump_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_metrics(path: str | Path, result: ScoreResult) -> Path:
    """Grava `result` em `path` na chave `[system][split]`, preservando o resto."""
    path = Path(path)
    data = load_metrics(path)
    data.setdefault(result.system, {})[result.split] = result.metrics
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(dump_json(data))
    return path
