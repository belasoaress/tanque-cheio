"""Caminho real da inferência local (transformers) com um modelo minúsculo, em CPU.

Só roda com o grupo opcional `infer` (`uv sync --group infer`) e com acesso ao
Hugging Face; o CI principal não instala o grupo, então este arquivo é pulado lá.
O modelo é aleatório: as SQLs saem lixo; o teste só prova o pipeline e o formato 5.3.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from tanque import config  # noqa: E402
from tanque.eval.score import read_jsonl  # noqa: E402
from tanque.infer import local  # noqa: E402

# Qwen2 aleatório com ~2,4 M de parâmetros e o chat template real do Qwen2.5.
TINY_MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


@pytest.fixture(scope="module")
def tiny():
    try:
        return local.load_model(TINY_MODEL, dtype="float32", device="cpu")
    except OSError as e:  # sem rede / Hugging Face fora do ar
        pytest.skip(f"modelo de teste indisponível: {e}")


def test_tiny_model_end_to_end_on_two_dev_items(tiny, tmp_path):
    model, tokenizer = tiny
    items = local.select_items(read_jsonl(config.resolve_path("datasets/dev.jsonl")), limit=2)
    gen = local.HFGenerator(model, tokenizer, max_new_tokens=8)
    preds = local.predict(
        items,
        gen,
        tokenizer,
        system=local.system_name(TINY_MODEL),
        hardware=local.hardware_label(None),
        sync=gen.sync,
    )
    path = local.write_predictions(local.output_path(tmp_path, preds[0]["system"], "dev"), preds)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows] == [i["id"] for i in items]
    for r in rows:
        assert r["error"] is None
        assert r["system"] == "base-tiny-qwen2forcausallm-2.5"
        assert r["hardware"] == "cpu"
        assert r["cost_usd"] == 0.0
        assert r["input_tokens"] > 500  # o prompt único inteiro passou pelo template
        assert 1 <= r["output_tokens"] <= 8
        assert isinstance(r["latency_ms"], int)
        assert isinstance(r["raw_output"], str) and isinstance(r["sql"], str)


def test_greedy_decoding_is_deterministic(tiny):
    model, tokenizer = tiny
    gen = local.HFGenerator(model, tokenizer, max_new_tokens=6)
    prompt = local.render_prompt(tokenizer, [{"role": "user", "content": "Pergunta: oi"}]).text
    assert gen.generate(prompt) == gen.generate(prompt)


def test_generation_overrides_sampling_defaults_of_the_checkpoint(tiny):
    model, tokenizer = tiny
    gen = local.HFGenerator(model, tokenizer, max_new_tokens=4)
    kwargs = gen.generate_kwargs
    assert kwargs["do_sample"] is False
    assert kwargs["num_beams"] == 1
    assert kwargs["repetition_penalty"] == 1.0
    assert kwargs["max_new_tokens"] == 4
    assert kwargs["temperature"] is None and kwargs["top_p"] is None and kwargs["top_k"] is None


def test_run_predictions_is_what_the_notebooks_call(tmp_path):
    try:
        info = local.run_predictions(
            config.resolve_path("datasets/dev.jsonl"),
            TINY_MODEL,
            out_dir=tmp_path,
            limit=2,
            dtype="float32",
            device="cpu",
            max_new_tokens=4,
            echo=lambda _: None,
        )
    except OSError as e:
        pytest.skip(f"modelo de teste indisponível: {e}")
    path = tmp_path / "base-tiny-qwen2forcausallm-2.5__dev-sample2.jsonl"
    assert info["path"] == str(path) and path.is_file()
    assert info["split"] == "dev" and info["summary"]["n"] == 2
    assert info["system_merged_into_user"] is False
    assert info["versions"]["transformers"]
    rows = read_jsonl(path)
    assert {r["system"] for r in rows} == {"base-tiny-qwen2forcausallm-2.5"}


# --- adapter fundido (M7) ---------------------------------------------------------


@pytest.fixture(scope="module")
def random_adapter(tmp_path_factory, tiny):  # `tiny` pula o teste sem rede
    """Adapter LoRA aleatório (B != 0) sobre o modelo minúsculo, salvo em disco."""
    peft = pytest.importorskip("peft")
    cfg = peft.LoraConfig(
        r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], init_lora_weights=False
    )
    path = tmp_path_factory.mktemp("adapter") / "adapter"
    wrapped = peft.get_peft_model(
        local.load_model(TINY_MODEL, dtype="float32", device="cpu")[0], cfg
    )
    wrapped.save_pretrained(str(path))
    return path


def _logits(model, tokenizer):
    import torch

    enc = tokenizer("SELECT avg(valor_venda) FROM coletas", return_tensors="pt")
    with torch.inference_mode():
        return model(**enc).logits


def test_adapter_is_merged_by_default_and_gives_the_same_logits(random_adapter):
    import torch
    from peft import PeftModel

    base, tokenizer = local.load_model(TINY_MODEL, dtype="float32", device="cpu")
    merged, _ = local.load_model(TINY_MODEL, str(random_adapter), dtype="float32", device="cpu")
    unmerged, _ = local.load_model(
        TINY_MODEL, str(random_adapter), dtype="float32", device="cpu", merge_adapter=False
    )
    assert not isinstance(merged, PeftModel)
    assert isinstance(unmerged, PeftModel)
    lm, lu, lb = (_logits(m, tokenizer) for m in (merged, unmerged, base))
    assert torch.allclose(lm, lu, atol=1e-4)
    assert not torch.allclose(lm, lb, atol=1e-4)  # o adapter muda a saída de fato


@pytest.mark.parametrize("merge", [True, False])
def test_run_predictions_records_whether_the_adapter_was_merged(random_adapter, tmp_path, merge):
    info = local.run_predictions(
        config.resolve_path("datasets/test.jsonl"),
        TINY_MODEL,
        out_dir=tmp_path,
        adapter_id=str(random_adapter),
        merge_adapter=merge,
        limit=2,
        dtype="float32",
        device="cpu",
        max_new_tokens=4,
        echo=lambda _: None,
    )
    assert info["adapter_merged"] is merge
    assert info["adapter_revision"] is None  # pasta local: sem commit do HF
    assert info["split"] == "test"
    rows = read_jsonl(tmp_path / "ft-tiny-qwen2forcausallm-2.5__test-sample2.jsonl")
    assert {r["system"] for r in rows} == {"ft-tiny-qwen2forcausallm-2.5"}
    assert all(r["error"] is None for r in rows)
