"""Formatação dos exemplos de treino (M6), sem torch: tokenizer falso.

O caminho com um tokenizer de verdade (chat template do Qwen2.5) está em
`test_train_format_hf.py`, que só roda com o grupo `infer` instalado.
"""

from __future__ import annotations

from collections import Counter

import pytest

from tanque.infer import local
from tanque.train import format as fmt
from test_infer_local import FakeTokenizer, fake_prompt


class EndTokenizer(FakeTokenizer):
    """Tokenizer falso que fecha cada mensagem com `<end>` e tokeniza por caractere."""

    eos_token = "<eos>"
    all_special_tokens = ["<eos>", "<end>"]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        text = super().apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=False, **kwargs
        )
        # fecha cada mensagem: <role>conteúdo<end>
        parts = text.split("<")[1:]
        out = "".join(f"<{p}<end>" for p in parts)
        return out + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            for special in self.all_special_tokens:
                if text.startswith(special, i):
                    ids.append(special)
                    i += len(special)
                    break
            else:
                ids.append(text[i])
                i += 1
        return {"input_ids": ids}

    def convert_ids_to_tokens(self, token_id):
        return token_id


ITEM = {"id": "train-00001", "question": "Qual o preço médio da gasolina em SP?", "sql": "SELECT 1"}


def test_prompt_is_the_inference_prompt_and_completion_is_the_sql():
    tok = FakeTokenizer()
    ex = fmt.to_prompt_completion(ITEM, tok, build_prompt=fake_prompt)
    expected = local.render_prompt(tok, fake_prompt(ITEM["question"])).text
    assert ex == {"id": "train-00001", "prompt": expected, "completion": "SELECT 1"}


def test_prompt_follows_the_same_template_options_as_inference():
    tok = FakeTokenizer(system_support="raise")
    ex = fmt.to_prompt_completion(
        ITEM,
        tok,
        chat_template_kwargs={"enable_thinking": False},
        system_suffix="/flag",
        build_prompt=fake_prompt,
    )
    rendered = local.render_prompt(
        tok, fake_prompt(ITEM["question"]), {"enable_thinking": False}, "/flag"
    )
    assert rendered.system_merged is True
    assert ex["prompt"] == rendered.text


def test_completion_keeps_the_sql_verbatim():
    item = {**ITEM, "sql": "SELECT m.nome\nFROM municipios m"}
    ex = fmt.to_prompt_completion(item, FakeTokenizer(), build_prompt=fake_prompt)
    assert ex["completion"] == "SELECT m.nome\nFROM municipios m"


def test_turn_end_token_is_the_token_that_closes_the_assistant_turn():
    assert fmt.turn_end_token(EndTokenizer()) == "<end>"


def test_turn_end_token_falls_back_to_eos():
    tok = EndTokenizer()
    tok.all_special_tokens = ["<eos>"]  # `<end>` vira texto comum
    assert fmt.turn_end_token(tok) == "<eos>"


# --- seleção e divisão ----------------------------------------------------------


def make_items(n_by_stratum: dict[tuple[str, str], int]) -> list[dict]:
    items = []
    for (difficulty, intent), n in n_by_stratum.items():
        for _ in range(n):
            items.append(
                {
                    "id": f"train-{len(items) + 1:05d}",
                    "question": f"q{len(items)}",
                    "sql": "SELECT 1",
                    "difficulty": difficulty,
                    "tags": [f"intent:{intent}", "x"],
                }
            )
    return items


def test_split_train_val_is_deterministic_disjoint_and_sized():
    items = make_items({("facil", "a"): 100})
    train, val = fmt.split_train_val(items, 0.05, seed=42)
    assert len(val) == 5 and len(train) == 95
    assert {i["id"] for i in train}.isdisjoint({i["id"] for i in val})
    again = fmt.split_train_val(items, 0.05, seed=42)
    assert (train, val) == again
    other = fmt.split_train_val(items, 0.05, seed=7)
    assert {i["id"] for i in other[1]} != {i["id"] for i in val}


def test_split_train_val_zero_fraction():
    items = make_items({("facil", "a"): 10})
    assert fmt.split_train_val(items, 0.0, seed=42) == (items, [])


def test_subsample_keeps_the_distribution():
    items = make_items({("facil", "a"): 300, ("medio", "b"): 500, ("dificil", "c"): 200})
    sample = fmt.subsample(items, 100, seed=42)
    assert len(sample) == 100
    counts = Counter(i["difficulty"] for i in sample)
    assert counts == {"facil": 30, "medio": 50, "dificil": 20}
    # mantém a ordem original e é determinístico
    positions = [items.index(i) for i in sample]
    assert positions == sorted(positions)
    assert sample == fmt.subsample(items, 100, seed=42)


def test_subsample_stratifies_by_intent_inside_a_level():
    items = make_items({("medio", "b"): 90, ("medio", "c"): 10})
    sample = fmt.subsample(items, 20, seed=1)
    assert Counter(fmt.stratum(i) for i in sample) == {("medio", "b"): 18, ("medio", "c"): 2}


@pytest.mark.parametrize("n", [None, 0, 10_000])
def test_subsample_without_limit_returns_everything(n):
    items = make_items({("facil", "a"): 7})
    assert fmt.subsample(items, n, seed=42) == items


def test_select_examples_drops_too_long_sequences():
    tok = EndTokenizer()
    examples = [
        {"id": "a", "prompt": "x" * 5, "completion": "y" * 3},
        {"id": "b", "prompt": "x" * 5, "completion": "y" * 10},
    ]
    kept, dropped = fmt.select_by_length(examples, tok, max_seq_len=10, end_token="<end>")
    # a: 5 + 3 + 1 (token de fim) = 9 <= 10; b: 16 > 10
    assert [e["id"] for e in kept] == ["a"]
    assert dropped == [{"id": "b", "tokens": 16}]
