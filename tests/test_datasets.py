"""Contrato dos conjuntos de dev e teste (R5, seção 5.2, M3).

Não precisa do banco: a execução das SQLs é checada por `tanque eval verify-gold`.
"""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"

QUOTAS = {
    "dev": {"facil": 15, "medio": 20, "dificil": 15},
    "test": {"facil": 50, "medio": 60, "dificil": 40},
}
MAX_WINDOW_TEST = 8
REQUIRED = {
    "id": str,
    "question": str,
    "sql": str,
    "difficulty": str,
    "uses_window": bool,
    "order_matters": bool,
    "split": str,
    "source": str,
    "reviewed": bool,
    "tags": list,
}
WINDOW_RE = re.compile(r"\bOVER\s*\(", re.IGNORECASE)


def _load(name: str) -> list[dict]:
    path = DATASETS / f"{name}.jsonl"
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _norm(q: str) -> str:
    q = unicodedata.normalize("NFKD", q.lower())
    q = "".join(ch for ch in q if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^\w\s]", " ", q).split())


@pytest.fixture(scope="module", params=["dev", "test"])
def split(request):
    return request.param, _load(request.param)


def test_exact_quotas(split):
    name, items = split
    assert len(items) == sum(QUOTAS[name].values())
    assert Counter(i["difficulty"] for i in items) == Counter(QUOTAS[name])


def test_window_cap_in_test():
    items = _load("test")
    assert sum(i["uses_window"] for i in items) <= MAX_WINDOW_TEST


def test_window_at_most_20_percent_of_hard(split):
    _, items = split
    hard = [i for i in items if i["difficulty"] == "dificil"]
    assert sum(i["uses_window"] for i in hard) <= 0.2 * len(hard)
    assert not any(i["uses_window"] for i in items if i["difficulty"] != "dificil")


def test_required_fields_and_values(split):
    name, items = split
    for it in items:
        assert set(it) == set(REQUIRED), it.get("id")
        for key, typ in REQUIRED.items():
            assert isinstance(it[key], typ), (it["id"], key)
        assert it["question"].strip() and it["sql"].strip(), it["id"]
        assert it["difficulty"] in {"facil", "medio", "dificil"}, it["id"]
        assert it["source"] == "human_reviewed", it["id"]
        assert it["split"] == name, it["id"]
        assert re.fullmatch(rf"{name}-\d{{4}}", it["id"]), it["id"]
        assert it["tags"] and all(isinstance(t, str) and t for t in it["tags"]), it["id"]


def test_uses_window_matches_sql(split):
    _, items = split
    for it in items:
        assert it["uses_window"] == bool(WINDOW_RE.search(it["sql"])), it["id"]


def test_order_matters_has_order_by(split):
    _, items = split
    for it in items:
        if it["order_matters"]:
            assert re.search(r"\bORDER\s+BY\b", it["sql"], re.IGNORECASE), it["id"]


def test_unique_ids_and_questions(split):
    _, items = split
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids))
    questions = [_norm(i["question"]) for i in items]
    assert len(questions) == len(set(questions))


def test_no_question_shared_between_dev_and_test():
    dev = {_norm(i["question"]) for i in _load("dev")}
    test = {_norm(i["question"]) for i in _load("test")}
    assert not dev & test


def test_reserve_is_disjoint_and_marked():
    reserve = _load("candidates_reserve")
    assert all(i["split"] == "reserve" for i in reserve)
    others = {_norm(i["question"]) for n in ("dev", "test") for i in _load(n)}
    assert not {_norm(i["question"]) for i in reserve} & others


def test_all_reviewed(split):
    _, items = split
    assert all(i["reviewed"] for i in items)
