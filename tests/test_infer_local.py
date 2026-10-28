"""Inferência local (M5): laço de previsão, chat template e formato 5.3, sem torch.

O modelo e o tokenizer são falsos. O caminho real (transformers) está em
`test_infer_local_hf.py`, que só roda com o grupo `infer` instalado.
"""

from __future__ import annotations

import json

import pytest

from tanque.infer import local

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


class FakeTokenizer:
    """Chat template mínimo: `<role>content` por mensagem, mais `<assistant>` no fim."""

    def __init__(self, *, system_support: str = "ok") -> None:
        # "ok": aceita system; "raise": recusa system; "drop": ignora system em silêncio.
        self.system_support = system_support
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is False
        self.calls.append({"messages": messages, "add_generation_prompt": add_generation_prompt})
        self.calls[-1].update(kwargs)
        parts = []
        for m in messages:
            if m["role"] == "system":
                if self.system_support == "raise":
                    raise ValueError("System role not supported")
                if self.system_support == "drop":
                    continue
            parts.append(f"<{m['role']}>{m['content']}")
        if add_generation_prompt:
            parts.append("<assistant>")
        if kwargs.get("enable_thinking") is False:
            parts.append("<no-think>")
        return "".join(parts)


class FakeGenerator:
    """Devolve respostas prontas; conta "tokens" como palavras."""

    def __init__(self, outputs: dict[str, str] | None = None, fail_on: str | None = None):
        self.outputs = outputs or {}
        self.fail_on = fail_on
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> local.Generation:
        self.prompts.append(prompt)
        if self.fail_on and self.fail_on in prompt:
            raise RuntimeError("CUDA out of memory")
        for question, raw in self.outputs.items():
            if question in prompt:
                return local.Generation(raw, len(prompt.split()), len(raw.split()))
        return local.Generation("SELECT 1", len(prompt.split()), 2)


def fake_prompt(question: str) -> list[dict]:
    return [
        {"role": "system", "content": "SISTEMA"},
        {"role": "user", "content": f"Pergunta: {question}"},
    ]


ITEMS = [
    {"id": "dev-0001", "question": "Qual o preço médio da gasolina em SP?"},
    {"id": "dev-0002", "question": "Quantos postos há em Recife?"},
]


def ticking_clock(step_s: float = 0.25):
    t = {"now": 100.0}

    def clock() -> float:
        t["now"] += step_s
        return t["now"]

    return clock


# --- nomes ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_id", "slug"),
    [
        ("Qwen/Qwen2.5-Coder-1.5B-Instruct", "qwen2.5-coder-1.5b-instruct"),
        ("HuggingFaceTB/SmolLM3-3B", "smollm3-3b"),
        ("org/Modelo_Estranho 7B", "modelo-estranho-7b"),
    ],
)
def test_model_slug(model_id, slug):
    assert local.model_slug(model_id) == slug


def test_system_name_base_and_finetuned():
    assert local.system_name("Qwen/Qwen3-1.7B") == "base-qwen3-1.7b"
    assert local.system_name("Qwen/Qwen3-1.7B", adapter_id="belasoaress/x") == "ft-qwen3-1.7b"


def test_output_path(tmp_path):
    assert local.output_path(tmp_path, "base-qwen3-1.7b", "dev") == (
        tmp_path / "base-qwen3-1.7b__dev.jsonl"
    )
    assert local.output_path(tmp_path, "base-x", "test", limit=5) == (
        tmp_path / "base-x__test-sample5.jsonl"
    )


@pytest.mark.parametrize(
    ("device", "label"),
    [
        ("Tesla T4", "kaggle-t4"),
        ("NVIDIA L4", "kaggle-nvidia-l4"),
        (None, "cpu"),
    ],
)
def test_hardware_label(device, label):
    assert local.hardware_label(device) == label


# --- chat template --------------------------------------------------------------


def test_render_prompt_keeps_system_when_supported():
    tok = FakeTokenizer()
    rendered = local.render_prompt(tok, fake_prompt("Q?"))
    assert rendered.text == "<system>SISTEMA<user>Pergunta: Q?<assistant>"
    assert rendered.system_merged is False
    assert tok.calls[0]["add_generation_prompt"] is True


@pytest.mark.parametrize("support", ["raise", "drop"])
def test_render_prompt_merges_system_into_first_user(support):
    tok = FakeTokenizer(system_support=support)
    rendered = local.render_prompt(tok, fake_prompt("Q?"))
    assert rendered.system_merged is True
    assert rendered.text == "<user>SISTEMA\n\nPergunta: Q?<assistant>"


def test_render_prompt_forwards_template_kwargs():
    tok = FakeTokenizer()
    rendered = local.render_prompt(tok, fake_prompt("Q?"), {"enable_thinking": False})
    assert rendered.text.endswith("<no-think>")
    assert tok.calls[0]["enable_thinking"] is False


def test_render_prompt_system_suffix_reaches_template_only():
    tok = FakeTokenizer()
    rendered = local.render_prompt(tok, fake_prompt("Q?"), system_suffix="/flag")
    assert rendered.system_merged is False
    assert rendered.text.startswith("<system>SISTEMA/flag<user>")


def test_render_prompt_system_suffix_dropped_when_merging():
    tok = FakeTokenizer(system_support="raise")
    rendered = local.render_prompt(tok, fake_prompt("Q?"), system_suffix="/flag")
    assert rendered.system_merged is True
    assert "/flag" not in rendered.text


def test_merge_system_without_system_is_identity():
    messages = [{"role": "user", "content": "oi"}]
    assert local.merge_system(messages) == messages


# --- laço de previsão -----------------------------------------------------------


def test_predict_writes_section_5_3_records():
    gen = FakeGenerator({"gasolina": "```sql\nSELECT AVG(valor_venda) FROM coletas;\n```\nPronto."})
    preds = local.predict(
        ITEMS,
        gen,
        FakeTokenizer(),
        system="base-fake",
        hardware="kaggle-t4",
        build_prompt=fake_prompt,
        clock=ticking_clock(0.25),
    )
    assert [p["id"] for p in preds] == ["dev-0001", "dev-0002"]
    first = preds[0]
    assert set(first) == PRED_KEYS
    assert first["system"] == "base-fake"
    assert first["sql"] == "SELECT AVG(valor_venda) FROM coletas"
    assert first["raw_output"].startswith("```sql")
    assert first["latency_ms"] == 250
    assert first["input_tokens"] > 0 and first["output_tokens"] == 7
    assert first["cost_usd"] == 0.0
    assert first["hardware"] == "kaggle-t4"
    assert first["error"] is None
    # cada registro é JSON puro
    json.dumps(preds)


def test_predict_uses_the_single_prompt_by_default():
    gen = FakeGenerator()
    tok = FakeTokenizer()
    local.predict(ITEMS[:1], gen, tok, system="base-fake", hardware="cpu")
    messages = tok.calls[0]["messages"]
    assert messages[-1]["content"] == f"Pergunta: {ITEMS[0]['question']}"
    assert "Responda apenas com uma consulta SQL para DuckDB." in gen.prompts[0]


def test_predict_synchronizes_around_each_item():
    events: list[str] = []
    local.predict(
        ITEMS,
        FakeGenerator(),
        FakeTokenizer(),
        system="base-fake",
        hardware="cpu",
        build_prompt=fake_prompt,
        sync=lambda: events.append("sync"),
    )
    assert events == ["sync"] * 4


def test_predict_turns_exceptions_into_error_records_and_continues():
    gen = FakeGenerator(fail_on="Recife")
    preds = local.predict(
        ITEMS, gen, FakeTokenizer(), system="base-fake", hardware="cpu", build_prompt=fake_prompt
    )
    assert preds[0]["error"] is None
    bad = preds[1]
    assert set(bad) == PRED_KEYS
    assert bad["error"] == "RuntimeError: CUDA out of memory"
    assert bad["sql"] == "" and bad["raw_output"] == ""
    assert bad["latency_ms"] is None
    assert bad["cost_usd"] == 0.0


def test_predict_empty_output_gives_empty_sql():
    gen = FakeGenerator({"gasolina": ""})
    preds = local.predict(
        ITEMS[:1],
        gen,
        FakeTokenizer(),
        system="base-fake",
        hardware="cpu",
        build_prompt=fake_prompt,
    )
    assert preds[0]["sql"] == "" and preds[0]["error"] is None


def test_predict_echoes_progress():
    lines: list[str] = []
    local.predict(
        ITEMS,
        FakeGenerator(),
        FakeTokenizer(),
        system="base-fake",
        hardware="cpu",
        build_prompt=fake_prompt,
        echo=lines.append,
    )
    assert len(lines) == 2 and lines[0].startswith("[1/2] dev-0001")


# --- arquivo e resumo -----------------------------------------------------------


def test_write_and_summarize(tmp_path):
    preds = local.predict(
        ITEMS,
        FakeGenerator(fail_on="Recife"),
        FakeTokenizer(),
        system="base-fake",
        hardware="cpu",
        build_prompt=fake_prompt,
        clock=ticking_clock(0.1),
    )
    path = local.write_predictions(tmp_path / "out" / "base-fake__dev.jsonl", preds)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["dev-0001", "dev-0002"]

    s = local.summarize(preds, max_new_tokens=2)
    assert s["n"] == 2
    assert s["errors"] == 1
    assert s["empty_sql"] == 1  # o item com erro
    assert s["hit_max_new_tokens"] == 1  # "SELECT 1" = 2 "tokens"
    assert s["p50_latency_ms"] == 100


def test_select_items_limit_and_ids():
    items = [{"id": f"dev-{i:04d}"} for i in range(1, 6)]
    assert [i["id"] for i in local.select_items(items, limit=2)] == ["dev-0001", "dev-0002"]
    assert local.select_items(items, limit=None) == items
    assert local.select_items(items, limit=0) == items
