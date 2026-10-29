"""Cálculo das métricas de um sistema sobre um dataset (seções 5.2, 5.3, 5.6 e 5.7 v2).

Para cada item do dataset, executa a SQL de referência (gold) e a SQL prevista
com `tanque.sql.execute.run` (guardrail ligado) e compara com
`tanque.eval.compare.results_match` (EX rígida) e `results_match_relaxed` (EX
tolerante, D28).

Métricas (todas sobre os `n` itens do dataset):

- `ex`: fração de acertos; item sem previsão, bloqueado ou com erro conta como errado;
- `ex_relaxed`: o mesmo com a EX tolerante (a previsão pode ter colunas a mais);
- `ex_ci95`, `ex_relaxed_ci95`: `[lo, hi]`, IC bootstrap percentil de 95% (D29;
  10.000 reamostragens, seed 42; ver `stats.py`);
- `ex_by_difficulty`: `{nivel: {n, ex, ex_relaxed}}` para `facil`, `medio` e `dificil`;
- `valid_sql_rate`: fração com SQL que passou no guardrail e executou sem erro
  (item sem previsão ou com `error` de geração conta como inválido);
- `blocked_rate`: fração bloqueada pelo guardrail;
- `latency_ms`: p50 e p95 dos `latency_ms` das previsões (interpolação linear) e a
  média (`mean`), usada na estimativa de custo de GPU do relatório (M7);
- `cost_per_1k_usd`: `soma(cost_usd) / n * 1000` (custo de API; 0 para os modelos
  locais, cujo custo de GPU é estimado só no relatório);
- `hardware`: lista ordenada dos `hardware` das previsões (ex.: `["kaggle-t4"]`).

`write_metrics` grava em `results/metrics.json` como `{system: {split: métricas}}`,
com chaves ordenadas e sem horário, para o CI poder comparar o arquivo byte a byte.

`recompute_all` recalcula as métricas de todas as previsões commitadas
(`results/predictions/<sistema>__{dev,test}.jsonl`; amostras `-sampleN` ficam de
fora) e `diff_metrics` lista as diferenças contra `metrics.json`: é a regressão
do CI (`tanque eval check-metrics`, M7).

**Comparações pareadas** (`comparisons`, 5.7 v2): a chave reservada `comparisons` de
`metrics.json` é uma lista com **todo par de sistemas pontuados no mesmo split**,
`{a, b, split, n, a_only, b_only, delta_ex, mcnemar_p}`, com `delta_ex = ex(b) − ex(a)`
e o McNemar exato sobre a EX rígida, pergunta a pergunta (`a_only`/`b_only` = itens que
só `a`/só `b` acertam). `a` vem antes de `b` na ordem dos sistemas (`base-*`, `ft-*`,
`ft2-*`, `claude-*`, outros; alfabética dentro de cada grupo), então base → ajustado
sai sempre nesse sentido. A lista é ordenada por split (`dev`, `test`), `a` e `b`.
Como precisa do acerto item a item de dois sistemas, só `recompute_all` (`tanque eval
check-metrics --write`) a calcula; `tanque eval score` grava um sistema e deixa a
lista como está, e o CI acusa se ela ficar desatualizada.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tanque import config
from tanque.eval import stats
from tanque.eval.compare import results_match, results_match_relaxed
from tanque.sql import execute
from tanque.sql.execute import ExecResult

DIFFICULTIES = ("facil", "medio", "dificil")
METRICS_FILENAME = "metrics.json"
COMPARISONS_KEY = "comparisons"
SYSTEM_ORDER = ("base-", "ft-", "ft2-", "claude-")


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
    correct_relaxed: bool = False


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
    return stats.percentile(values, q)


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _ci95(flags: list[bool]) -> list[float] | None:
    ci = stats.bootstrap_ci([1 if f else 0 for f in flags])
    return None if ci is None else [round(ci[0], 4), round(ci[1], 4)]


def system_sort_key(system: str) -> tuple[int, str]:
    """Ordem dos sistemas em tabelas e comparações: base, ft, ft2, claude, outros."""
    for rank, prefix in enumerate(SYSTEM_ORDER):
        if system.startswith(prefix):
            return rank, system
    return len(SYSTEM_ORDER), system


def system_metrics(data: dict[str, Any]) -> dict[str, Any]:
    """`{system: {split: métricas}}` de um metrics.json, sem a chave `comparisons`."""
    return {k: v for k, v in data.items() if k != COMPARISONS_KEY}


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
        order_matters = bool(item.get("order_matters"))
        correct = valid and results_match(gold_results[item_id], res, order_matters)
        relaxed = valid and results_match_relaxed(gold_results[item_id], res, order_matters)
        results.append(
            ItemResult(item_id, difficulty, True, correct, valid, res.blocked, res.error, relaxed)
        )

    n = len(results)
    by_level = {}
    for level in DIFFICULTIES:
        level_items = [r for r in results if r.difficulty == level]
        by_level[level] = {
            "n": len(level_items),
            "ex": _rate(sum(r.correct for r in level_items), len(level_items)),
            "ex_relaxed": _rate(sum(r.correct_relaxed for r in level_items), len(level_items)),
        }
    unknown = sorted({r.difficulty for r in results} - set(DIFFICULTIES))
    if unknown:
        raise ValueError(f"dificuldade fora de {DIFFICULTIES}: {unknown}")

    present = [preds[r.id] for r in results if r.has_prediction]
    latencies = [float(p["latency_ms"]) for p in present if p.get("latency_ms") is not None]
    p50, p95 = percentile(latencies, 50), percentile(latencies, 95)
    mean = sum(latencies) / len(latencies) if latencies else None
    total_cost = sum(float(p.get("cost_usd") or 0.0) for p in present)

    metrics = {
        "dataset": dataset_path.name,
        "n": n,
        "n_missing": sum(not r.has_prediction for r in results),
        "ex": _rate(sum(r.correct for r in results), n),
        "ex_relaxed": _rate(sum(r.correct_relaxed for r in results), n),
        "ex_ci95": _ci95([r.correct for r in results]),
        "ex_relaxed_ci95": _ci95([r.correct_relaxed for r in results]),
        "ex_by_difficulty": by_level,
        "valid_sql_rate": _rate(sum(r.valid for r in results), n),
        "blocked_rate": _rate(sum(r.blocked for r in results), n),
        "latency_ms": {
            "p50": None if p50 is None else round(p50, 1),
            "p95": None if p95 is None else round(p95, 1),
            "mean": None if mean is None else round(mean, 1),
        },
        "cost_per_1k_usd": round(total_cost / n * 1000, 6),
        "hardware": sorted({str(p["hardware"]) for p in present if p.get("hardware")}),
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
    """Grava `result` em `path` na chave `[system][split]`, preservando o resto.

    `comparisons` fica como está (só `recompute_all` a recalcula).
    """
    path = Path(path)
    data = load_metrics(path)
    _reject_reserved(result.system)
    data.setdefault(result.system, {})[result.split] = result.metrics
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(dump_json(data))
    return path


PREDICTION_SPLITS = ("dev", "test")


def committed_predictions(predictions_dir: str | Path) -> list[tuple[str, Path]]:
    """(split, arquivo) de `<sistema>__dev.jsonl` e `<sistema>__test.jsonl`, ordenados."""
    out = []
    for split in PREDICTION_SPLITS:
        for path in sorted(Path(predictions_dir).glob(f"*__{split}.jsonl")):
            out.append((split, path))
    return out


def recompute_all(
    predictions_dir: str | Path,
    dataset_dir: str | Path,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """`{system: {split: métricas}, "comparisons": [...]}` de todas as previsões commitadas."""
    data: dict[str, Any] = {}
    scored: list[ScoreResult] = []
    for split, path in committed_predictions(predictions_dir):
        result = score(Path(dataset_dir) / f"{split}.jsonl", path, db_path=db_path)
        expected_system = path.name[: -len(f"__{split}.jsonl")]
        if result.system != expected_system:
            raise PredictionsError(
                f"{path.name}: `system` = {result.system!r} não bate com o nome do arquivo"
            )
        _reject_reserved(result.system)
        data.setdefault(result.system, {})[result.split] = result.metrics
        scored.append(result)
    comparisons = paired_comparisons(scored)
    if comparisons:
        data[COMPARISONS_KEY] = comparisons
    return data


def _reject_reserved(system: str) -> None:
    if system == COMPARISONS_KEY:
        raise PredictionsError(f"`{COMPARISONS_KEY}` é reservado e não pode ser nome de sistema")


def paired_comparisons(results: list[ScoreResult]) -> list[dict[str, Any]]:
    """Todo par de sistemas no mesmo split (ver o docstring do módulo), com McNemar."""
    by_split: dict[str, list[ScoreResult]] = {}
    for r in results:
        by_split.setdefault(r.split, []).append(r)
    split_rank = {s: i for i, s in enumerate(PREDICTION_SPLITS)}
    out = []
    for split in sorted(by_split, key=lambda s: (split_rank.get(s, len(split_rank)), s)):
        systems = sorted(by_split[split], key=lambda r: system_sort_key(r.system))
        for i, a in enumerate(systems):
            for b in systems[i + 1 :]:
                a_items = {it.id: it.correct for it in a.items}
                b_items = {it.id: it.correct for it in b.items}
                if set(a_items) != set(b_items):
                    raise PredictionsError(
                        f"{a.system} e {b.system} no split {split} não têm os mesmos itens"
                    )
                ids = sorted(a_items)
                cmp = stats.paired_comparison([a_items[k] for k in ids], [b_items[k] for k in ids])
                out.append({"a": a.system, "b": b.system, "split": split, **cmp})
    return out


def diff_metrics(expected: Any, actual: Any, path: str = "") -> list[str]:
    """Diferenças entre dois `metrics.json` (chaves ausentes, sobrando ou com outro valor)."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        out = []
        for key in sorted(set(expected) | set(actual)):
            where = f"{path}.{key}" if path else str(key)
            if key not in actual:
                out.append(f"{where}: está em metrics.json, mas não foi recalculado")
            elif key not in expected:
                out.append(f"{where}: recalculado, mas ausente de metrics.json")
            else:
                out.extend(diff_metrics(expected[key], actual[key], where))
        return out
    if expected != actual:
        return [f"{path}: metrics.json = {expected!r}, recalculado = {actual!r}"]
    return []


def write_all_metrics(path: str | Path, data: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(dump_json(data))
    return path
