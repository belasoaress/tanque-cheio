"""Testes das funções puras de scripts/review.py (seleção, gravação atômica, auditoria)."""

import importlib.util
import json
from pathlib import Path

import pytest

from tanque.sql.execute import ExecResult

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("review", ROOT / "scripts" / "review.py")
review = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(review)


def _item(n: int, reviewed: bool = False, **kw) -> dict:
    base = {
        "id": f"dev-{n:04d}",
        "question": f"Pergunta {n}?",
        "sql": "SELECT 1",
        "difficulty": "facil",
        "uses_window": False,
        "order_matters": False,
        "split": "dev",
        "source": "human_reviewed",
        "reviewed": reviewed,
        "tags": ["x"],
    }
    base.update(kw)
    return base


def test_pending_skips_reviewed_and_skipped():
    items = [_item(1, True), _item(2), _item(3), _item(4, True), _item(5)]
    assert review.pending_indices(items) == [1, 2, 4]
    assert review.pending_indices(items, skipped=["dev-0003"]) == [1, 4]
    assert review.pending_indices([_item(1, True)]) == []


def test_approve_and_discard_do_not_mutate_input():
    items = [_item(1), _item(2)]
    approved = review.approve(items, "dev-0002")
    assert [i["reviewed"] for i in approved] == [False, True]
    assert items[1]["reviewed"] is False

    rest, removed = review.discard(approved, "dev-0001")
    assert [i["id"] for i in rest] == ["dev-0002"]
    assert removed["id"] == "dev-0001"
    with pytest.raises(KeyError):
        review.discard(rest, "dev-9999")


def test_replace_item_keeps_order():
    items = [_item(1), _item(2), _item(3)]
    new = _item(2, question="Editada?")
    out = review.replace_item(items, "dev-0002", new)
    assert [i["question"] for i in out] == ["Pergunta 1?", "Editada?", "Pergunta 3?"]


def test_write_jsonl_atomic_roundtrip_and_no_leftovers(tmp_path):
    path = tmp_path / "dev.jsonl"
    path.write_text("velho\n", encoding="utf-8")
    items = [_item(1, question="Preço médio do etanol em São Paulo?"), _item(2)]
    review.write_jsonl_atomic(path, items)
    assert review.load_jsonl(path) == items
    # UTF-8 sem escapar acentos, uma linha por item, LF
    raw = path.read_bytes()
    assert "São Paulo".encode() in raw
    assert raw.count(b"\n") == 2 and b"\r\n" not in raw
    assert [p.name for p in tmp_path.iterdir()] == ["dev.jsonl"]


def test_write_jsonl_atomic_keeps_old_file_on_error(tmp_path):
    path = tmp_path / "dev.jsonl"
    review.write_jsonl_atomic(path, [_item(1)])
    before = path.read_bytes()

    class Boom:
        def __iter__(self):
            yield _item(2)
            raise RuntimeError("falha no meio")

    with pytest.raises(RuntimeError):
        review.write_jsonl_atomic(path, Boom())
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["dev.jsonl"]


def test_audit_sample_is_reproducible_and_bounded():
    items = [_item(i) for i in range(1, 101)]
    a = review.audit_sample(items, 10, seed=7)
    b = review.audit_sample(items, 10, seed=7)
    c = review.audit_sample(items, 10, seed=8)
    assert a == b
    assert a != c
    assert len({i["id"] for i in a}) == 10
    assert len(review.audit_sample(items[:3], 10, seed=1)) == 3
    with pytest.raises(ValueError):
        review.audit_sample(items, 0, seed=1)


def test_audit_summary_and_path(tmp_path):
    recs = [{"id": "a", "verdict": "aprovado"}, {"id": "b", "verdict": "reprovado"}] * 2
    recs.append({"id": "c", "verdict": "aprovado"})
    total, rejected, pct = review.audit_summary(recs)
    assert (total, rejected) == (5, 2)
    assert pct == pytest.approx(40.0)
    assert review.audit_summary([]) == (0, 0, 0.0)
    assert review.audit_path(Path("datasets/train.jsonl"), tmp_path) == tmp_path / (
        "audit-train.jsonl"
    )


def test_schema_problems():
    assert review.schema_problems(_item(1), expected_split="dev") == []
    bad = _item(1, difficulty="trivial", order_matters="sim", split="test")
    problems = review.schema_problems(bad, expected_split="dev")
    assert any("difficulty" in p for p in problems)
    assert any("order_matters" in p for p in problems)
    assert any("split" in p for p in problems)
    missing = _item(1)
    del missing["tags"]
    assert review.schema_problems(missing) == ["campo ausente: tags"]


def test_exec_problems():
    assert review.exec_problems(ExecResult(columns=["x"], rows=[(1,)])) == []
    assert review.exec_problems(ExecResult(columns=["x"], rows=[])) == ["a SQL não retornou linhas"]
    assert review.exec_problems(ExecResult(error="boom")) == ["boom"]


def test_editor_command():
    assert review.editor_command({"EDITOR": "code --wait"}, "linux") == ["code", "--wait"]
    assert review.editor_command({"VISUAL": "vim", "EDITOR": "nano"}, "linux") == ["vim"]
    assert review.editor_command({}, "win32") == ["notepad"]
    assert review.editor_command({}, "linux")[0] in {"nano", "vi"}


def test_format_result_shows_at_most_five_rows():
    res = ExecResult(columns=["uf", "p"], rows=[(f"U{i}", i) for i in range(8)], elapsed_ms=3)
    text = review.format_result(res)
    assert "U4" in text and "U5" not in text
    assert "8 linha(s)" in text and "+3 linhas" in text


def test_ask_retries_until_valid(capsys):
    answers = iter(["x", "", "Aprovar"])
    assert review.ask("> ", "aedsq", read=lambda _: next(answers)) == "a"
    assert "Opção inválida" in capsys.readouterr().out


def test_discarded_log_is_jsonl(tmp_path):
    log = tmp_path / "sub" / "discarded.jsonl"
    review.append_jsonl(log, {"from": "dev.jsonl", "reason": None, "item": _item(1)})
    review.append_jsonl(log, {"from": "dev.jsonl", "reason": "ambígua", "item": _item(2)})
    lines = log.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["item"]["id"] for line in lines] == ["dev-0001", "dev-0002"]
