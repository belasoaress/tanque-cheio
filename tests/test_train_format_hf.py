"""Formatação dos exemplos de treino com um tokenizer de verdade (chat template do Qwen2.5).

Só roda com o grupo opcional `infer` (`uv sync --group infer`) e com acesso ao
Hugging Face; o CI principal não instala o grupo, então este arquivo é pulado lá.
Prova que o prompt de treino é, token a token, o prompt de inferência e que a
parte treinada (completion) é exatamente a SQL mais o token de fim de turno.
"""

from __future__ import annotations

import pytest

pytest.importorskip("transformers")

from tanque import config, prompting  # noqa: E402
from tanque.eval.score import read_jsonl  # noqa: E402
from tanque.infer import local  # noqa: E402
from tanque.train import format as fmt  # noqa: E402

# Mesmo modelo minúsculo de `test_infer_local_hf.py`: tokenizer real do Qwen2.5.
TINY_MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(TINY_MODEL)
    except OSError as e:  # sem rede / Hugging Face fora do ar
        pytest.skip(f"tokenizer de teste indisponível: {e}")


@pytest.fixture(scope="module")
def train_items():
    return read_jsonl(config.resolve_path("datasets/train.jsonl"))[:5]


def test_training_prompt_equals_inference_prompt(tokenizer, train_items):
    for item in train_items:
        ex = fmt.to_prompt_completion(item, tokenizer)
        inference = local.render_prompt(tokenizer, prompting.build_prompt(item["question"]))
        assert ex["prompt"] == inference.text
        assert ex["prompt"].endswith("<|im_start|>assistant\n")
        assert ex["completion"] == item["sql"]


def test_prompt_plus_completion_matches_the_chat_template_of_the_answer(tokenizer, train_items):
    """prompt + SQL + fim de turno = conversa completa renderizada pelo template."""
    end = fmt.turn_end_token(tokenizer)
    assert end == "<|im_end|>"
    for item in train_items:
        ex = fmt.to_prompt_completion(item, tokenizer)
        messages = prompting.build_prompt(item["question"])
        full = tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": item["sql"]}], tokenize=False
        )
        assert full.startswith(ex["prompt"] + ex["completion"] + end)


def test_tokenized_prompt_is_a_prefix_of_prompt_plus_completion(tokenizer, train_items):
    """Sem fusão de tokens na fronteira: a máscara da perda começa exatamente na SQL."""
    for item in train_items:
        ex = fmt.to_prompt_completion(item, tokenizer)
        prompt_ids = tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(ex["prompt"] + ex["completion"], add_special_tokens=False)["input_ids"]
        assert full_ids[: len(prompt_ids)] == prompt_ids
        assert tokenizer.decode(full_ids[len(prompt_ids) :]) == item["sql"]
