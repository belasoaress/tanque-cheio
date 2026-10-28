"""Ensaio de fumaça do fine-tuning (M6) em CPU: treino -> adapter -> previsões 5.3.

Só roda com os grupos opcionais `infer` e `train`
(`uv sync --group infer --group train`) e com acesso ao Hugging Face; o CI
principal não os instala, então este arquivo é pulado lá. Usa um Qwen2
aleatório minúsculo, sem 4-bit (o bitsandbytes não roda em CPU), por 2 passos.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("trl")

from tanque import config  # noqa: E402
from tanque.eval.score import read_jsonl  # noqa: E402
from tanque.infer import local  # noqa: E402
from tanque.train import config as train_config  # noqa: E402
from tanque.train import finetune  # noqa: E402

TINY_MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("ft")
    cfg = train_config.with_overrides(
        train_config.load_train_config(),
        {
            "model": {"base_model": TINY_MODEL},
            "data": {"max_train_examples": 24},
            "training": {
                "per_device_train_batch_size": 2,
                "per_device_eval_batch_size": 2,
                "gradient_accumulation_steps": 2,
                "logging_steps": 1,
                "eval_steps": 1,
            },
        },
    )
    try:
        info = finetune.run_finetune(cfg, out_dir=out, device="cpu", max_steps=2, echo=print)
    except OSError as e:  # sem rede / Hugging Face fora do ar
        pytest.skip(f"modelo de teste indisponível: {e}")
    return out, info


def test_training_writes_adapter_and_log(trained):
    out, info = trained
    adapter = out / "adapter"
    assert info["adapter_dir"] == str(adapter)
    assert (adapter / "adapter_config.json").is_file()
    assert (adapter / "adapter_model.safetensors").is_file()
    adapter_cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    assert adapter_cfg["r"] == 16 and adapter_cfg["lora_alpha"] == 32
    assert set(adapter_cfg["target_modules"]) >= {"q_proj", "down_proj"}
    assert info["system"] == "ft-tiny-qwen2forcausallm-2.5"
    assert info["quantized"] is False and info["hardware"] == "cpu"
    assert info["global_step"] == 2
    assert info["n_train"] + info["n_val"] + len(info["dropped_too_long"]) == 24
    log = json.loads((out / "train-log__ft-tiny-qwen2forcausallm-2.5.json").read_text())
    assert any("loss" in row for row in log["log_history"])
    assert any("eval_loss" in row for row in log["log_history"])
    assert log["train_seconds"] > 0
    assert not (out / "checkpoints").exists()  # só o adapter final fica na saída


def test_loss_is_only_on_the_completion(trained):
    _, info = trained
    # a fração de tokens com perda é a da SQL (+ fim de turno), bem menor que o prompt
    assert 0 < info["completion_token_fraction"] < 0.35


def test_predictions_with_the_trained_adapter(trained):
    out, info = trained
    run = local.run_predictions(
        config.resolve_path("datasets/dev.jsonl"),
        TINY_MODEL,
        out_dir=out,
        adapter_id=info["adapter_dir"],
        limit=2,
        dtype="float32",
        device="cpu",
        max_new_tokens=4,
        echo=lambda _: None,
    )
    path = out / "ft-tiny-qwen2forcausallm-2.5__dev-sample2.jsonl"
    assert run["path"] == str(path)
    rows = read_jsonl(path)
    assert len(rows) == 2
    for r in rows:
        assert r["system"] == "ft-tiny-qwen2forcausallm-2.5"
        assert r["error"] is None and r["hardware"] == "cpu"
        assert set(r) == {
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


def test_loss_table_renders_markdown(trained):
    _, info = trained
    table = finetune.loss_table(info["log_history"])
    lines = table.splitlines()
    assert lines[0].startswith("| Passo | Época | Perda (treino) | Perda (val)")
    assert len(lines) >= 3
