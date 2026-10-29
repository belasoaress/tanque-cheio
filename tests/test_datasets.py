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


# --- dev2: dev ampliado (M13, D30) --------------------------------------------
#
# dev2 = os 50 do dev (já revisados) + os 63 da reserva, nessa ordem, com ids
# `dev2-0001…`, `split: "dev2"` e `origin_id` com o id original. É o único arquivo
# com um campo além da seção 5.2; os carregadores (verify-gold, score, review.py,
# leakage) ignoram campos extras. Os descartes da revisão não são repostos, então o
# total pode cair abaixo de 113 e não há cota exata por nível.

DEV2_MAX = 113


@pytest.fixture(scope="module")
def dev2():
    return _load("dev2")


def test_dev2_counts_by_level(dev2):
    counts = Counter(i["difficulty"] for i in dev2)
    print(
        f"\ndev2: {len(dev2)} itens; "
        + ", ".join(f"{lvl} {counts[lvl]}" for lvl in ("facil", "medio", "dificil"))
        + f"; revisados {sum(i['reviewed'] for i in dev2)}"
    )
    assert 50 < len(dev2) <= DEV2_MAX
    assert set(counts) <= {"facil", "medio", "dificil"}


def test_dev2_fields_and_split(dev2):
    required = {**REQUIRED, "origin_id": str}
    for it in dev2:
        assert set(it) == set(required), it.get("id")
        for key, typ in required.items():
            assert isinstance(it[key], typ), (it["id"], key)
        assert it["question"].strip() and it["sql"].strip(), it["id"]
        assert it["difficulty"] in {"facil", "medio", "dificil"}, it["id"]
        assert it["source"] == "human_reviewed", it["id"]
        assert it["split"] == "dev2", it["id"]
        assert re.fullmatch(r"(dev|res)-\d{4}", it["origin_id"]), it["id"]
        assert it["tags"] and all(isinstance(t, str) and t for t in it["tags"]), it["id"]
        assert it["uses_window"] == bool(WINDOW_RE.search(it["sql"])), it["id"]
        if it["order_matters"]:
            assert re.search(r"\bORDER\s+BY\b", it["sql"], re.IGNORECASE), it["id"]


def test_dev2_ids_unique_and_stable(dev2):
    ids = [i["id"] for i in dev2]
    assert all(re.fullmatch(r"dev2-\d{4}", i) for i in ids)
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)  # descartes deixam buracos, mas a ordem não muda
    origins = [i["origin_id"] for i in dev2]
    assert len(origins) == len(set(origins))
    questions = [_norm(i["question"]) for i in dev2]
    assert len(questions) == len(set(questions))


def test_dev2_contains_all_dev_first_and_unchanged(dev2):
    dev = _load("dev")
    head = dev2[: len(dev)]
    assert [i["origin_id"] for i in head] == [i["id"] for i in dev]
    assert [i["id"] for i in head] == [f"dev2-{k:04d}" for k in range(1, len(dev) + 1)]
    for new, old in zip(head, dev, strict=True):
        same = {k: v for k, v in new.items() if k not in {"id", "origin_id", "split"}}
        assert same == {k: v for k, v in old.items() if k not in {"id", "split"}}, new["id"]
        assert new["reviewed"] is True, new["id"]


def test_dev2_rest_comes_from_reserve(dev2):
    reserve = {i["id"] for i in _load("candidates_reserve")}
    rest = dev2[len(_load("dev")) :]
    assert rest and all(i["origin_id"] in reserve for i in rest)


def test_dev2_window_at_most_20_percent_of_hard(dev2):
    hard = [i for i in dev2 if i["difficulty"] == "dificil"]
    assert sum(i["uses_window"] for i in hard) <= 0.2 * len(hard)
    assert not any(i["uses_window"] for i in dev2 if i["difficulty"] != "dificil")


def test_no_question_shared_between_dev2_and_test(dev2):
    test = {_norm(i["question"]) for i in _load("test")}
    assert not {_norm(i["question"]) for i in dev2} & test


def test_dev2_all_reviewed(dev2):
    assert all(i["reviewed"] for i in dev2)
