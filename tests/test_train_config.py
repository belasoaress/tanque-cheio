"""configs/train.toml (M6): valores do plano e estimativa de passos, sem torch."""

from __future__ import annotations

import pytest

from tanque.train import config as train_config


@pytest.fixture(scope="module")
def cfg():
    return train_config.load_train_config()


def test_plan_hyperparameters(cfg):
    assert cfg["model"]["base_model"] == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    q = cfg["quantization"]
    assert q["load_in_4bit"] is True and q["quant_type"] == "nf4"
    assert q["compute_dtype"] == "float16"  # a T4 não tem bf16
    lora = cfg["lora"]
    assert (lora["r"], lora["alpha"], lora["dropout"]) == (16, 32, 0.05)
    assert set(lora["target_modules"]) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    t = cfg["training"]
    assert t["learning_rate"] == 2e-4 and t["lr_scheduler"] == "cosine"
    assert t["warmup_ratio"] == 0.03 and t["epochs"] == 2
    assert t["fp16"] is True and t["bf16"] is False
    assert t["gradient_checkpointing"] is True
    assert train_config.effective_batch_size(cfg) == 16
    d = cfg["data"]
    assert d["seed"] == 42 and d["val_fraction"] == 0.05
    assert d["max_train_examples"] == 0  # 0 = todos os pares
    assert d["max_seq_len"] >= 1024
    assert t["completion_only_loss"] is True


def test_estimate_steps_for_the_full_training_set(cfg):
    est = train_config.estimate_steps(cfg, n_items=8217)
    assert est["n_val"] == 411 and est["n_train"] == 7806
    assert est["steps_per_epoch"] == 488  # ceil(7806 / 16)
    assert est["total_steps"] == 976


def test_estimate_steps_with_subsample_and_one_epoch(cfg):
    cfg = train_config.with_overrides(
        cfg, {"data": {"max_train_examples": 4000}, "training": {"epochs": 1}}
    )
    est = train_config.estimate_steps(cfg, n_items=8217)
    assert est["n_val"] == 200 and est["n_train"] == 3800
    assert est["total_steps"] == 238


def test_with_overrides_does_not_mutate(cfg):
    new = train_config.with_overrides(cfg, {"lora": {"r": 8}})
    assert new["lora"]["r"] == 8 and cfg["lora"]["r"] == 16
    assert new["lora"]["alpha"] == 32


def test_unknown_override_key_is_an_error(cfg):
    with pytest.raises(KeyError):
        train_config.with_overrides(cfg, {"lora": {"rank": 8}})
