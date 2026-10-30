"""Demo do HF Spaces (M19): a lógica do app sem o llama.cpp (gerador e executor falsos).

O executor real (`tanque.sql.execute.run`, com o guardrail) roda no banco de
brinquedo do conftest. A interface só é montada se o gradio estiver instalado (o CI
não tem: o teste é pulado).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import re
import sys
from decimal import Decimal
from pathlib import Path
from types import ModuleType

import pytest

from tanque import config, prompting
from tanque.sql import execute
from tanque.sql.execute import ExecResult

APP_DIR = config.resolve_path("app")

GOOD_SQL = (
    "SELECT AVG(c.valor_venda) AS preco FROM coletas c JOIN postos p USING (posto_id) "
    "JOIN municipios m USING (municipio_id) WHERE c.produto = 'GASOLINA' AND m.uf = 'SP'"
)
BAD_SQL = "SELECT AVG(preco_inexistente) FROM coletas"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


td = load_module("tanque_demo", APP_DIR / "tanque_demo.py")


class FakeGenerator:
    """Devolve as respostas na ordem e guarda as mensagens recebidas."""

    def __init__(self, *outputs: str | Exception):
        self.outputs = list(outputs)
        self.calls: list[list[dict]] = []

    def __call__(self, messages: list[dict]):
        self.calls.append(messages)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return td.Generation(f"```sql\n{out}\n```", 800, 40)


class Clock:
    def __init__(self, step: float = 1.5):
        self.t, self.step = 0.0, step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


@pytest.fixture
def toy_execute(m2_toy_db):
    return lambda sql: execute.run(sql, timeout_s=5, max_rows=1000, db_path=m2_toy_db)


# --- textos fixos ---------------------------------------------------------------


def test_disclaimer_and_examples():
    assert td.DISCLAIMER == (
        "Dados da ANP de 2024-10 a 2026-09; data de referência 30/09/2026; "
        "'último mês' = setembro de 2026."
    )
    assert len(td.EXAMPLES) == 6 and len(set(td.EXAMPLES)) == 6
    assert "O etanol compensou em São Carlos (SP) no último mês?" in td.EXAMPLES
    assert all(len(q) <= td.MAX_QUESTION_CHARS for q in td.EXAMPLES)


def test_decoding_parameters_follow_the_evaluation():
    assert td.MAX_TOKENS == 384  # D31
    assert td.N_CTX == 2048 and td.N_THREADS == 2
    assert td.MAX_TABLE_ROWS == 100


# --- uma pergunta -----------------------------------------------------------------


def test_answer_runs_the_sql_and_reports_time(toy_execute):
    gen = FakeGenerator(GOOD_SQL)
    ans = td.answer_question("Preço médio da gasolina em SP?", gen, toy_execute, clock=Clock())
    assert ans.ok and not ans.retried and ans.message is None
    assert ans.sql == GOOD_SQL
    assert ans.columns == ["preco"] and len(ans.rows) == 1
    assert ans.rows[0][0] == pytest.approx((6.190 + 6.090) / 2)
    assert ans.seconds > 0
    # o prompt é o prompt único da avaliação (5.4)
    assert gen.calls == [prompting.build_prompt("Preço médio da gasolina em SP?")]


def test_guard_block_gets_one_retry_and_a_friendly_message(toy_execute):
    gen = FakeGenerator("DROP TABLE postos", "DELETE FROM coletas")
    ans = td.answer_question("Apague tudo", gen, toy_execute, clock=Clock())
    assert not ans.ok and ans.retried and len(gen.calls) == 2
    assert ans.first_sql == "DROP TABLE postos"
    assert ans.first_error.startswith("bloqueada pelo guardrail")
    assert ans.sql == "DELETE FROM coletas"
    assert ans.message.startswith("A SQL gerada foi recusada pelo guardrail de segurança")
    assert "bloqueada pelo guardrail:" not in ans.message  # sem o prefixo técnico
    assert ans.rows == [] and td.table_value(ans) == {"headers": ["resultado"], "data": []}


def test_execution_error_is_repaired_with_the_repair_prompt(toy_execute):
    gen = FakeGenerator(BAD_SQL, GOOD_SQL)
    q = "Preço médio da gasolina em SP?"
    ans = td.answer_question(q, gen, toy_execute, clock=Clock())
    assert ans.ok and ans.retried and ans.sql == GOOD_SQL
    assert ans.first_sql == BAD_SQL and "Binder" in ans.first_error
    # D32: a 2ª chamada é exatamente o prompt de correção, com o erro da execução
    assert gen.calls[1] == prompting.build_repair_prompt(q, BAD_SQL, ans.first_error)
    status = td.status_markdown(ans)
    assert "nova tentativa" in status and "1 linha(s)" in status
    detail = td.first_attempt_markdown(ans)
    assert BAD_SQL in detail and "1ª tentativa" in detail


def test_successful_sql_is_executed_once():
    calls = []

    def fake_execute(sql):
        calls.append(sql)
        return ExecResult(columns=["x"], rows=[(1,)])

    ans = td.answer_question("Quantos?", FakeGenerator("SELECT 1 AS x"), fake_execute)
    assert ans.ok and calls == ["SELECT 1 AS x"]


def test_generation_failure_has_no_retry():
    gen = FakeGenerator(RuntimeError("sem memória"))
    ans = td.answer_question("Qual o preço?", gen, lambda sql: pytest.fail("não executa"))
    assert not ans.ok and not ans.retried and len(gen.calls) == 1
    assert ans.message == "O modelo não conseguiu gerar uma resposta agora. Tente de novo."


@pytest.mark.parametrize("question", ["", "   ", "x" * 301])
def test_invalid_question_does_not_call_the_model(question):
    gen = FakeGenerator()
    ans = td.answer_question(question, gen, lambda sql: pytest.fail("não executa"))
    assert not ans.ok and ans.message and gen.calls == []


def test_table_is_cut_at_100_rows():
    rows = [(i, Decimal("1.234"), dt.date(2026, 9, 1), None) for i in range(150)]

    def fake_execute(sql):
        return ExecResult(columns=["i", "v", "d", "n"], rows=rows)

    ans = td.answer_question("Liste", FakeGenerator("SELECT 1"), fake_execute)
    assert ans.ok and len(ans.rows) == 100 and ans.total_rows == 150 and ans.truncated
    assert ans.rows[0] == [0, 1.234, "2026-09-01", ""]
    assert "mostrando 100 de 150 linhas" in td.status_markdown(ans)


def test_midnight_timestamps_show_as_dates():
    assert td.cell_value(dt.datetime(2026, 9, 1)) == "2026-09-01"
    assert td.cell_value(dt.datetime(2026, 9, 1, 12, 30)) == "2026-09-01T12:30:00"


def test_timeout_message():
    msg = td.friendly_error("timeout após 10.0s", blocked=False)
    assert "limite de tempo" in msg


def test_respond_returns_the_four_outputs(toy_execute):
    sql, table, status, first = td.respond("Preço?", FakeGenerator(GOOD_SQL), toy_execute)
    assert sql == GOOD_SQL
    assert table["headers"] == ["preco"] and len(table["data"]) == 1
    assert "Tempo de resposta" in status and first == ""


# --- gerador do llama.cpp -------------------------------------------------------------


def test_llama_generator_is_greedy_and_reads_usage():
    class FakeLlama:
        def __init__(self):
            self.kwargs = None

        def create_chat_completion(self, **kwargs):
            self.kwargs = kwargs
            return {
                "choices": [{"message": {"role": "assistant", "content": "SELECT 1"}}],
                "usage": {"prompt_tokens": 812, "completion_tokens": 5},
            }

    llm = FakeLlama()
    msgs = prompting.build_prompt("Qual o preço?")
    gen = td.llama_generator(llm)(msgs)
    assert gen == td.Generation("SELECT 1", 812, 5)
    assert llm.kwargs["messages"] == msgs
    assert llm.kwargs["temperature"] == 0.0 and llm.kwargs["max_tokens"] == 384
    assert llm.kwargs["repeat_penalty"] == 1.0


# --- montagem do banco ----------------------------------------------------------------


def _fake_download(files: dict[str, bytes]):
    def download(url: str, dest: Path) -> None:
        rel = url.split(f"/{td.TANQUE_COMMIT}/", 1)[1]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(files[rel])

    return download


def test_fetch_root_downloads_the_pinned_files_and_checks_the_hash(tmp_path):
    parquet = b"PAR1 fake"
    meta = json.dumps({"sha256": hashlib.sha256(parquet).hexdigest()}).encode()
    files = {
        "configs/settings.toml": b"[paths]\n",
        "data/manifest.json": b"{}",
        td.SNAPSHOT_META: meta,
        td.SNAPSHOT: parquet,
    }
    root = td.fetch_root(tmp_path, download=_fake_download(files))
    for rel in td.ROOT_FILES:
        assert (root / rel).read_bytes() == files[rel]


def test_fetch_root_rejects_a_corrupted_snapshot(tmp_path):
    files = {
        "configs/settings.toml": b"",
        "data/manifest.json": b"{}",
        td.SNAPSHOT_META: json.dumps({"sha256": "0" * 64}).encode(),
        td.SNAPSHOT: b"outro",
    }
    with pytest.raises(RuntimeError, match="snapshot corrompido"):
        td.fetch_root(tmp_path, download=_fake_download(files))
    assert not (tmp_path / td.SNAPSHOT).exists()


def test_pinned_snapshot_matches_the_repository():
    # o snapshot do commit fixado é o mesmo versionado aqui (D25)
    meta = json.loads(config.resolve_path(td.SNAPSHOT_META).read_text(encoding="utf-8"))
    assert td.sha256_of(config.resolve_path(td.SNAPSHOT)) == meta["sha256"]
    assert td.raw_url(td.SNAPSHOT).endswith(f"/{td.TANQUE_COMMIT}/{td.SNAPSHOT}")


# --- arquivos do Space ------------------------------------------------------------------


def requirements() -> list[str]:
    text = (APP_DIR / "requirements.txt").read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip() and line[0] != "#"]


def front_matter() -> dict[str, str]:
    text = (APP_DIR / "README.md").read_text(encoding="utf-8")
    block = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
    assert block, "app/README.md sem front-matter"
    return dict(re.findall(r"^(\w+): *(.+)$", block.group(1), re.MULTILINE))


def test_requirements_are_pinned_and_use_the_same_commit():
    reqs = requirements()
    packages = [r for r in reqs if not r.startswith("--")]
    for req in packages:
        assert "==" in req or f"@{td.TANQUE_COMMIT}" in req, req
    assert f"tanque @ git+https://github.com/belasoaress/tanque-cheio@{td.TANQUE_COMMIT}" in reqs
    assert re.fullmatch(r"[0-9a-f]{40}", td.TANQUE_COMMIT)
    # wheel CPU pré-compilado do índice oficial: o Space não compila o llama.cpp
    assert "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu" in reqs
    assert "--only-binary=llama-cpp-python" in reqs


def test_requirements_match_the_lock_where_shared():
    lock = config.resolve_path("uv.lock").read_text(encoding="utf-8")
    locked = dict(re.findall(r'\[\[package\]\]\nname = "([^"]+)"\nversion = "([^"]+)"', lock))
    pins = dict(r.split("==") for r in requirements() if "==" in r)
    for name in ("duckdb", "sqlglot", "polars", "pyarrow", "huggingface-hub"):
        assert pins[name] == locked[name], name


def test_space_front_matter():
    meta = front_matter()
    assert meta["sdk"] == "gradio" and meta["app_file"] == "app.py"
    assert meta["license"] == "mit"
    pins = dict(r.split("==") for r in requirements() if "==" in r)
    assert meta["sdk_version"] == pins["gradio"]
    assert meta["python_version"].strip("\"'") == "3.11"
    readme = (APP_DIR / "README.md").read_text(encoding="utf-8")
    assert f"  - {td.GGUF_REPO}" in readme


# --- interface ----------------------------------------------------------------------------


def test_ui_builds_with_fake_runtime(monkeypatch, toy_execute):
    gr = pytest.importorskip("gradio")
    monkeypatch.syspath_prepend(str(APP_DIR))
    app = load_module("tanque_demo_app", APP_DIR / "app.py")
    ui = app.build_ui(FakeGenerator(GOOD_SQL), toy_execute)
    assert isinstance(ui, gr.Blocks)
