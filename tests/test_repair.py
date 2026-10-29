"""Autocorreção com 1 nova tentativa (M16; plan.md v2, D32 e contratos 5.3/5.4 v2).

Gerador e executor falsos; o banco, quando há, é o de brinquedo do conftest.
O caminho real com o modelo minúsculo está em `test_infer_local_hf.py`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest

from tanque import prompting
from tanque.eval import report
from tanque.eval.score import read_jsonl, recompute_all, score
from tanque.infer import local, repair

PRED_KEYS = {
    "id",
    "system",
    "sql",
    "raw_output",
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "hardware",
    "error",
}
RETRY_KEYS = PRED_KEYS | {"attempts", "first_sql", "first_error"}

# SHA-256 do system do prompt único na main antes do M16 (5.4 v2: não muda).
SYSTEM_PROMPT_SHA256 = "5cf987a8efe46b5aec5dd9163f175226192d46eaf7ce00a3e779ae4836331218"

QUESTION = "Qual o preço médio da gasolina em SP?"
GOOD_SQL = "SELECT AVG(valor_venda) FROM coletas WHERE produto = 'GASOLINA'"
BAD_SQL = "SELECT AVG(preco) FROM coletas"
EVIL_SQL = "DROP TABLE coletas"


# --- prompt de correção (5.4 v2) --------------------------------------------------


def test_build_prompt_is_unchanged():
    messages = prompting.build_prompt(QUESTION)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1]["content"] == f"Pergunta: {QUESTION}"
    digest = hashlib.sha256(prompting.system_prompt().encode("utf-8")).hexdigest()
    assert digest == SYSTEM_PROMPT_SHA256


def test_repair_prompt_extends_the_single_prompt():
    messages = prompting.build_repair_prompt(QUESTION, BAD_SQL, "BinderException: coluna preco")
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[:2] == prompting.build_prompt(QUESTION)
    assert messages[0]["content"] == prompting.system_prompt()
    assert messages[1]["content"] == f"Pergunta: {QUESTION}"
    assert messages[2]["content"] == BAD_SQL
    assert messages[3]["content"] == (
        "A consulta falhou: BinderException: coluna preco. Responda apenas com a SQL corrigida."
    )


def test_repair_prompt_truncates_error_to_300_chars():
    error = "x" * 299 + "YZ" + "w" * 500
    content = prompting.build_repair_prompt(QUESTION, BAD_SQL, error)[-1]["content"]
    prefix, suffix = "A consulta falhou: ", ". Responda apenas com a SQL corrigida."
    assert content.startswith(prefix) and content.endswith(suffix)
    shown = content[len(prefix) : -len(suffix)]
    assert shown == "x" * 299 + "Y"
    assert len(shown) == prompting.REPAIR_ERROR_MAX_CHARS == 300


def test_repair_prompt_keeps_short_error_whole_and_strips_it():
    content = prompting.build_repair_prompt(QUESTION, "", "  erro curto \n")[-1]["content"]
    assert content == "A consulta falhou: erro curto. Responda apenas com a SQL corrigida."


# --- falsos -----------------------------------------------------------------------


@dataclass
class FakeExec:
    error: str | None = None
    blocked: bool = False
    rows: tuple = ()


class FakeExecutor:
    """SQL -> resultado pronto; o padrão é rodar sem erro (e sem linhas)."""

    def __init__(self, results: dict[str, FakeExec] | None = None):
        self.results = results or {}
        self.calls: list[str] = []

    def __call__(self, sql: str) -> FakeExec:
        self.calls.append(sql)
        return self.results.get(sql, FakeExec())


class FakeGenerate:
    """Mensagens -> saídas na ordem; uma saída `Exception` é levantada."""

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls: list[list[dict]] = []

    def __call__(self, messages: list[dict]):
        self.calls.append(messages)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def fake_prompt(question: str) -> list[dict]:
    return [{"role": "system", "content": "S"}, {"role": "user", "content": question}]


def fake_repair(question: str, sql: str, error: str) -> list[dict]:
    return [
        *fake_prompt(question),
        {"role": "assistant", "content": sql},
        {"role": "user", "content": error},
    ]


def ticking_clock(step_s: float = 0.25):
    t = {"now": 100.0}

    def clock() -> float:
        t["now"] += step_s
        return t["now"]

    return clock


ITEM = {"id": "dev-0001", "question": QUESTION}


def run_item(generate, executor, **kw):
    return repair.predict_item(
        ITEM,
        generate,
        executor,
        system="base-x",
        hardware="cpu",
        build_prompt=fake_prompt,
        build_repair_prompt=fake_repair,
        clock=ticking_clock(),
        **kw,
    )


# --- uma pergunta -------------------------------------------------------------------


def test_success_on_first_attempt_does_not_retry():
    gen = FakeGenerate(local.Generation(f"```sql\n{GOOD_SQL};\n```", 100, 20))
    executor = FakeExecutor()
    single, retried = run_item(gen, executor)
    assert len(gen.calls) == 1 and executor.calls == [GOOD_SQL]
    assert set(single) == PRED_KEYS and set(retried) == RETRY_KEYS
    assert single["system"] == "base-x" and retried["system"] == "base-x+retry"
    assert single["sql"] == GOOD_SQL and single["latency_ms"] == 250
    assert (single["input_tokens"], single["output_tokens"]) == (100, 20)
    assert retried["attempts"] == 1
    assert retried["first_sql"] is None and retried["first_error"] is None
    assert {k: retried[k] for k in PRED_KEYS - {"system"}} == {
        k: single[k] for k in PRED_KEYS - {"system"}
    }


def test_empty_result_is_not_a_failure():
    # Sem gold, o sistema não sabe se o resultado está "errado": vazio não dispara.
    gen = FakeGenerate(GOOD_SQL)
    single, retried = run_item(gen, FakeExecutor({GOOD_SQL: FakeExec(rows=())}))
    assert retried["attempts"] == 1 and len(gen.calls) == 1


@pytest.mark.parametrize(
    "outcome",
    [
        FakeExec(error="bloqueada pelo guardrail: instrução DROP", blocked=True),
        FakeExec(error='BinderException: Binder Error: column "preco" not found'),
        FakeExec(error="timeout após 10s"),
    ],
)
def test_blocked_or_execution_error_triggers_one_retry(outcome):
    gen = FakeGenerate(
        local.Generation(BAD_SQL, 100, 10), local.Generation(f"```sql\n{GOOD_SQL}\n```", 130, 12)
    )
    executor = FakeExecutor({BAD_SQL: outcome})
    single, retried = run_item(gen, executor)

    assert len(gen.calls) == 2
    assert gen.calls[1] == fake_repair(QUESTION, BAD_SQL, outcome.error)
    assert executor.calls == [BAD_SQL]  # a 2ª SQL não é executada aqui (o score executa)

    assert single["sql"] == BAD_SQL and single["latency_ms"] == 250
    assert (single["input_tokens"], single["output_tokens"]) == (100, 10)
    assert "attempts" not in single  # o single-shot fica no formato da v1

    assert retried["attempts"] == 2
    assert retried["first_sql"] == BAD_SQL and retried["first_error"] == outcome.error
    assert retried["sql"] == GOOD_SQL and retried["raw_output"].startswith("```sql")
    assert retried["latency_ms"] == 500  # soma das duas gerações
    assert (retried["input_tokens"], retried["output_tokens"]) == (230, 22)
    assert retried["error"] is None


def test_retry_happens_only_once_even_if_it_fails_again():
    gen = FakeGenerate(BAD_SQL, BAD_SQL, GOOD_SQL)
    executor = FakeExecutor({BAD_SQL: FakeExec(error="BinderException: x")})
    _, retried = run_item(gen, executor)
    assert len(gen.calls) == 2 and retried["attempts"] == 2 and retried["sql"] == BAD_SQL


def test_empty_sql_retries_without_touching_the_database():
    gen = FakeGenerate("Não sei.", GOOD_SQL)
    executor = FakeExecutor()
    single, retried = run_item(gen, executor, extract_sql=lambda raw: "")
    assert executor.calls == []
    assert retried["attempts"] == 2 and retried["first_error"] == "SQL vazia"


def test_string_generator_counts_no_tokens():
    single, _ = run_item(FakeGenerate(GOOD_SQL), FakeExecutor())
    assert single["raw_output"] == GOOD_SQL
    assert (single["input_tokens"], single["output_tokens"]) == (0, 0)


def test_generation_error_on_first_attempt_is_not_retried():
    gen = FakeGenerate(RuntimeError("CUDA out of memory"))
    executor = FakeExecutor()
    single, retried = run_item(gen, executor)
    assert executor.calls == [] and len(gen.calls) == 1
    for rec in (single, retried):
        assert rec["error"] == "RuntimeError: CUDA out of memory"
        assert rec["sql"] == "" and rec["latency_ms"] is None
    assert retried["attempts"] == 1 and retried["system"] == "base-x+retry"


def test_generation_error_on_second_attempt_is_recorded():
    gen = FakeGenerate(local.Generation(BAD_SQL, 100, 10), RuntimeError("OOM"))
    single, retried = run_item(gen, FakeExecutor({BAD_SQL: FakeExec(error="ParserException")}))
    assert single["error"] is None and single["sql"] == BAD_SQL
    assert retried["attempts"] == 2 and retried["error"] == "RuntimeError: OOM"
    assert retried["sql"] == "" and retried["latency_ms"] is None
    assert retried["first_sql"] == BAD_SQL and retried["first_error"] == "ParserException"


def test_default_prompts_are_the_contract_ones():
    gen = FakeGenerate(BAD_SQL, GOOD_SQL)
    repair.predict_item(
        ITEM,
        gen,
        FakeExecutor({BAD_SQL: FakeExec(error="BinderException: x")}),
        system="s",
        hardware="cpu",
    )
    assert gen.calls[0] == prompting.build_prompt(QUESTION)
    assert gen.calls[1] == prompting.build_repair_prompt(QUESTION, BAD_SQL, "BinderException: x")


# --- laço e resumo ----------------------------------------------------------------


def test_predict_with_retry_keeps_order_and_summarizes():
    items = [{"id": f"dev-{i}", "question": f"q{i}"} for i in range(3)]
    gen = FakeGenerate(GOOD_SQL, EVIL_SQL, GOOD_SQL, BAD_SQL, GOOD_SQL)
    executor = FakeExecutor(
        {
            EVIL_SQL: FakeExec(error="bloqueada pelo guardrail: DROP", blocked=True),
            BAD_SQL: FakeExec(error="BinderException: x"),
        }
    )
    lines: list[str] = []
    singles, retried = repair.predict_with_retry(
        items,
        gen,
        executor,
        system="base-x",
        hardware="cpu",
        build_prompt=fake_prompt,
        build_repair_prompt=fake_repair,
        echo=lines.append,
    )
    assert [p["id"] for p in singles] == [p["id"] for p in retried] == ["dev-0", "dev-1", "dev-2"]
    assert [p["attempts"] for p in retried] == [1, 2, 2]
    assert len(lines) == 3 and "nova tentativa" in lines[1]
    assert repair.summarize_retry(retried) == {
        "n": 3,
        "retried": 2,
        "first_blocked": 1,
        "first_exec_error": 1,
        "retry_generation_errors": 0,
    }
    s = local.retry_summary(retried)
    assert s["retried"] == 2 and s["n"] == 3 and "hit_max_new_tokens" not in s


# --- executor real (guardrail + execute.run no banco de brinquedo) ------------------


def test_make_executor_requires_the_database(tmp_path):
    with pytest.raises(FileNotFoundError, match="from-parquet"):
        repair.make_executor(tmp_path / "nao-existe.duckdb")


def test_real_executor_classifies_failures(m2_toy_db):
    run = repair.make_executor(m2_toy_db)
    assert repair.failure_of(GOOD_SQL, run) is None
    assert repair.failure_of("SELECT * FROM coletas WHERE 1 = 0", run) is None  # vazio: ok
    blocked = repair.failure_of(EVIL_SQL, run)
    assert blocked is not None and blocked.startswith("bloqueada pelo guardrail")
    error = repair.failure_of(BAD_SQL, run)
    assert error is not None and "preco" in error


# --- de ponta a ponta: arquivos, score, check-metrics e relatório ----------------------


def _two_files(tmp_path: Path, m2_toy_db: Path, m2_toy_items: list[dict]):
    """Previsões falsas dos itens de brinquedo: 1º certo; 2º bloqueado e corrigido;
    3º com erro e corrigido; 4º com erro e errado de novo."""
    gold = {i["id"]: i["sql"] for i in m2_toy_items}
    ids = [i["id"] for i in m2_toy_items]
    outputs = [
        gold[ids[0]],
        EVIL_SQL,
        gold[ids[1]],
        BAD_SQL,
        gold[ids[2]],
        BAD_SQL,
        BAD_SQL,
    ]
    singles, retried = repair.predict_with_retry(
        m2_toy_items,
        FakeGenerate(*outputs),
        repair.make_executor(m2_toy_db),
        system="base-toy",
        hardware="cpu",
        build_prompt=fake_prompt,
        build_repair_prompt=fake_repair,
    )
    out = tmp_path / "predictions"
    a = local.write_predictions(local.output_path(out, "base-toy", "dev"), singles)
    b = local.write_predictions(
        local.output_path(out, repair.retry_system("base-toy"), "dev"), retried
    )
    return a, b


def test_score_accepts_retry_files(tmp_path, m2_toy_db, m2_toy_items, m2_dataset):
    single_path, retry_path = _two_files(tmp_path, m2_toy_db, m2_toy_items)
    assert retry_path.name == "base-toy+retry__dev.jsonl"
    assert [r["attempts"] for r in read_jsonl(retry_path)] == [1, 2, 2, 2]

    single = score(m2_dataset, single_path, db_path=m2_toy_db)
    retried = score(m2_dataset, retry_path, db_path=m2_toy_db)
    assert single.system == "base-toy" and retried.system == "base-toy+retry"
    assert single.metrics["ex"] == 0.25 and retried.metrics["ex"] == 0.75
    assert single.metrics["blocked_rate"] == 0.25 and retried.metrics["blocked_rate"] == 0.0
    assert retried.metrics["valid_sql_rate"] == 0.75

    # check-metrics recalcula de todos os `<sistema>__<split>.jsonl`, inclusive o `+retry`.
    data = recompute_all(single_path.parent, m2_dataset.parent, db_path=m2_toy_db)
    assert set(data) == {"base-toy", "base-toy+retry"}
    assert data["base-toy+retry"]["dev"] == retried.metrics


def test_report_tables_list_retry_systems(tmp_path, m2_toy_db, m2_toy_items, m2_dataset):
    single_path, _ = _two_files(tmp_path, m2_toy_db, m2_toy_items)
    data = recompute_all(single_path.parent, m2_dataset.parent, db_path=m2_toy_db)
    assert report.systems_with(data, "dev") == ["base-toy", "base-toy+retry"]
    table = "\n".join(report.main_table(data, "dev"))
    assert "`base-toy+retry`" in table and "`base-toy`" in table


# --- integração em local.run_predictions -------------------------------------------


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")


class ScriptedHFGenerator:
    """Substitui `HFGenerator`: 1ª tentativa = SQL inválida no 2º item; correção = gold."""

    instances: list[ScriptedHFGenerator] = []

    def __init__(self, model, tokenizer, max_new_tokens):
        self.max_new_tokens = max_new_tokens
        self.prompts: list[str] = []
        ScriptedHFGenerator.instances.append(self)

    def sync(self) -> None:
        pass

    def generate(self, prompt: str) -> local.Generation:
        self.prompts.append(prompt)
        if "A consulta falhou:" in prompt:
            return local.Generation("SELECT produto, COUNT(*) FROM coletas GROUP BY produto", 9, 9)
        if "Quantas coletas" in prompt:
            return local.Generation(BAD_SQL, 5, 5)
        return local.Generation(GOOD_SQL, 5, 5)


@pytest.fixture
def fake_hf(monkeypatch):
    ScriptedHFGenerator.instances = []
    monkeypatch.setattr(local, "load_model", lambda *a, **k: (object(), FakeTokenizer()))
    monkeypatch.setattr(local, "HFGenerator", ScriptedHFGenerator)
    monkeypatch.setattr(local, "_device_name", lambda device: None)
    monkeypatch.setattr(local, "_free_cuda", lambda: None)
    monkeypatch.setattr(local, "_resolved_revision", lambda model_id, rev: ("abc123", None))
    return ScriptedHFGenerator


def test_run_predictions_with_retry_writes_both_files(fake_hf, tmp_path, m2_toy_db, m2_dataset):
    info = local.run_predictions(
        m2_dataset,
        "org/Modelo-X",
        out_dir=tmp_path / "out",
        device="cpu",
        retry=True,
        db_path=m2_toy_db,
        limit=2,
        echo=lambda _: None,
    )
    single_path = tmp_path / "out" / "base-modelo-x__dev-sample2.jsonl"
    retry_path = tmp_path / "out" / "base-modelo-x+retry__dev-sample2.jsonl"
    assert info["path"] == str(single_path) and info["retry_path"] == str(retry_path)
    assert info["retry"] is True and info["retry_system"] == "base-modelo-x+retry"
    assert info["max_new_tokens"] == 384 == fake_hf.instances[0].max_new_tokens
    assert info["db_path"] == str(m2_toy_db)
    assert info["retry_summary"]["retried"] == 1

    singles, retried = read_jsonl(single_path), read_jsonl(retry_path)
    assert [r["sql"] for r in singles] == [GOOD_SQL, BAD_SQL]
    assert [r["attempts"] for r in retried] == [1, 2]
    assert retried[1]["sql"].startswith("SELECT produto") and retried[1]["first_sql"] == BAD_SQL
    assert retried[1]["input_tokens"] == 14
    # o prompt de correção passou pelo chat template, com a SQL anterior como assistant
    repair_prompt = fake_hf.instances[0].prompts[-1]
    assert f"<assistant>{BAD_SQL}<user>A consulta falhou:" in repair_prompt

    run_info = local.retry_run_info(info)
    assert run_info["system"] == "base-modelo-x+retry" and run_info["path"] == str(retry_path)
    assert run_info["single_shot_system"] == "base-modelo-x"
    assert run_info["summary"] == info["retry_summary"]
    assert not any(k.startswith("retry_") for k in run_info)
    assert run_info["model_revision"] == "abc123" and run_info["max_new_tokens"] == 384


def test_run_predictions_with_retry_fails_fast_without_database(fake_hf, tmp_path, m2_dataset):
    with pytest.raises(FileNotFoundError):
        local.run_predictions(
            m2_dataset,
            "org/Modelo-X",
            out_dir=tmp_path,
            device="cpu",
            retry=True,
            db_path=tmp_path / "sem-banco.duckdb",
            echo=lambda _: None,
        )
    assert fake_hf.instances == []  # nem carregou o modelo


def test_run_predictions_without_retry_is_single_shot(fake_hf, tmp_path, m2_dataset):
    info = local.run_predictions(
        m2_dataset,
        "org/Modelo-X",
        out_dir=tmp_path,
        device="cpu",
        max_new_tokens=128,
        echo=lambda _: None,
    )
    assert info["retry"] is False and "retry_path" not in info
    assert info["max_new_tokens"] == 128 == fake_hf.instances[0].max_new_tokens
    assert sorted(p.name for p in tmp_path.glob("*.jsonl")) == [
        "base-modelo-x__dev.jsonl",
        "dev.jsonl",
    ]
    with pytest.raises(ValueError):
        local.retry_run_info(info)


def test_default_max_new_tokens_is_384():
    assert local.DEFAULT_MAX_NEW_TOKENS == 384
