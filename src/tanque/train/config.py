"""Leitura de configs/train.toml e estimativa do tamanho da rodada (M6).

Sem dependências de torch: o CI testa os valores do plano e a conta de passos.
"""

from __future__ import annotations

import copy
import math
import tomllib
from pathlib import Path
from typing import Any

from tanque import config

TRAIN_RELPATH = Path("configs") / "train.toml"


def load_train_config(path: str | Path | None = None) -> dict[str, Any]:
    """Conteúdo de configs/train.toml (ou do arquivo indicado)."""
    with config.resolve_path(path or TRAIN_RELPATH).open("rb") as f:
        return tomllib.load(f)


def with_overrides(cfg: dict[str, Any], overrides: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Cópia de `cfg` com `{seção: {chave: valor}}` trocados; chave desconhecida é erro."""
    new = copy.deepcopy(cfg)
    for section, values in overrides.items():
        for key, value in values.items():
            if key not in new[section]:
                raise KeyError(f"chave desconhecida em [{section}]: {key}")
            new[section][key] = value
    return new


def system_name(cfg: dict[str, Any]) -> str:
    """Nome do sistema treinado: `<system_prefix>-<modelo>` (5.3 v2).

    `[model].system_prefix` vale `ft` por omissão (v1, `configs/train.toml`) e `ft2` no
    treino v2 (`configs/train-v2.toml`, M15): `ft2-qwen2.5-coder-1.5b-instruct`.
    """
    from tanque.infer.local import model_slug

    prefix = cfg["model"].get("system_prefix") or "ft"
    if prefix not in ("ft", "ft2"):
        raise ValueError(f"system_prefix deve ser 'ft' ou 'ft2' (5.3 v2), não {prefix!r}")
    return f"{prefix}-{model_slug(cfg['model']['base_model'])}"


def effective_batch_size(cfg: dict[str, Any], n_devices: int = 1) -> int:
    t = cfg["training"]
    return t["per_device_train_batch_size"] * t["gradient_accumulation_steps"] * n_devices


def n_val(n_items: int, val_fraction: float) -> int:
    """Tamanho da validação: `round(n * fração)`, ao menos 1 se a fração for positiva."""
    if val_fraction <= 0 or n_items < 2:
        return 0
    return min(n_items - 1, max(1, round(n_items * val_fraction)))


def estimate_steps(cfg: dict[str, Any], n_items: int) -> dict[str, int]:
    """Pares de treino/validação e passos de otimização, contados como no `Trainer`.

    Não desconta os pares maiores que `max_seq_len` (descartados no notebook).
    """
    d, t = cfg["data"], cfg["training"]
    limit = d.get("max_train_examples") or 0
    pool = min(n_items, limit) if limit > 0 else n_items
    val = n_val(pool, d["val_fraction"])
    train = pool - val
    batches = math.ceil(train / t["per_device_train_batch_size"])
    steps_per_epoch = math.ceil(batches / t["gradient_accumulation_steps"])
    return {
        "n_items": pool,
        "n_train": train,
        "n_val": val,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": steps_per_epoch * t["epochs"],
    }
