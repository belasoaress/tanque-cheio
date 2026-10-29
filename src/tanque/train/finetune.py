"""Fine-tuning QLoRA com transformers + peft + trl (M6), chamado pelo `02_finetune.ipynb`.

O código fica no pacote, e não no notebook, para ser testado em CPU com um modelo
minúsculo (`tests/test_train_smoke.py`). Decisões:

- **Dados:** `configs/train.toml` -> `datasets/train.jsonl`, subamostra opcional
  estratificada (`max_train_examples`), 5% para a perda de validação (por seed),
  pares no formato prompt/completion de `tanque.train.format` (prompt idêntico
  ao da inferência; completion = SQL). Pares maiores que `max_seq_len` são
  descartados, não truncados, e contados no log.
- **Perda:** só na completion (`SFTConfig(completion_only_loss=True)`, trl 1.16).
  O token de fim de turno do chat template (`<|im_end|>` no Qwen2.5) é passado
  como `eos_token`, e o trl o acrescenta à completion: o modelo aprende a parar.
- **Quantização:** 4-bit nf4 (bitsandbytes) só com GPU. Em CPU (ensaio de
  fumaça) o modelo carrega em fp32 sem quantização.
- **Precisão:** fp16 na T4 (sem bf16). O trl converte os pesos LoRA de um modelo
  quantizado para bf16 (recomendação do artigo do QLoRA); com fp16 e
  `GradScaler`, os pesos treináveis voltam para fp32 aqui, o padrão de AMP.
- **Memória:** gradient checkpointing; o modelo vai inteiro para a GPU 0
  (`device_map={"": 0}`), mesmo se o Kaggle der 2 T4.
- **Saída:** `<out_dir>/adapter/` (só o adapter LoRA) e
  `<out_dir>/train-log__<system>.json` (config, log_history, tempo de GPU,
  memória, versões). Os checkpoints intermediários são apagados no fim.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tanque import config
from tanque.eval.score import read_jsonl
from tanque.infer import local
from tanque.train import config as train_config
from tanque.train import format as fmt

VERSION_PACKAGES = (
    "torch",
    "transformers",
    "peft",
    "trl",
    "accelerate",
    "bitsandbytes",
    "datasets",
)


def versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in VERSION_PACKAGES:
        try:
            out[name] = str(__import__(name).__version__)
        except Exception:  # pacote ausente (ex.: bitsandbytes em CPU)
            out[name] = None
    return out


def load_tokenizer(cfg: dict[str, Any]) -> Any:
    from transformers import AutoTokenizer

    m = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(m["base_model"], revision=m["revision"] or None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(cfg: dict[str, Any], device: str) -> tuple[Any, bool]:
    """(modelo base, quantizado?): 4-bit nf4 na GPU; fp32 sem quantização em CPU."""
    import torch
    from transformers import AutoModelForCausalLM

    m, q = cfg["model"], cfg["quantization"]
    kwargs: dict[str, Any] = {"revision": m["revision"] or None}
    quantized = bool(q["load_in_4bit"]) and device.startswith("cuda")
    if quantized:
        from transformers import BitsAndBytesConfig

        compute = getattr(torch, q["compute_dtype"])
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=q["quant_type"],
            bnb_4bit_use_double_quant=q["double_quant"],
            bnb_4bit_compute_dtype=compute,
        )
        kwargs["dtype"] = compute
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["dtype"] = torch.float32
        kwargs["device_map"] = {"": device}
    model = AutoModelForCausalLM.from_pretrained(m["base_model"], **kwargs)
    model.config.use_cache = False  # incompatível com gradient checkpointing
    return model, quantized


def lora_config(cfg: dict[str, Any]) -> Any:
    from peft import LoraConfig

    lora = cfg["lora"]
    return LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        bias=lora["bias"],
        target_modules=list(lora["target_modules"]),
        task_type="CAUSAL_LM",
    )


def sft_config(
    cfg: dict[str, Any],
    *,
    output_dir: str | Path,
    device: str,
    end_token: str,
    has_eval: bool,
    max_steps: int | None = None,
) -> Any:
    from trl import SFTConfig

    t, d = cfg["training"], cfg["data"]
    on_gpu = device.startswith("cuda")
    return SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=t["epochs"],
        max_steps=max_steps or -1,
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler"],
        warmup_steps=t["warmup_ratio"],  # float em [0, 1) = fração dos passos (transformers 5)
        optim=t["optim"],
        weight_decay=t["weight_decay"],
        max_grad_norm=t["max_grad_norm"],
        fp16=bool(t["fp16"]) and on_gpu,
        bf16=bool(t["bf16"]) and on_gpu,
        use_cpu=not on_gpu,
        gradient_checkpointing=t["gradient_checkpointing"],
        max_length=d["max_seq_len"],
        completion_only_loss=t["completion_only_loss"],
        eos_token=end_token,
        logging_steps=t["logging_steps"],
        eval_strategy="steps" if has_eval else "no",
        eval_steps=t["eval_steps"],
        save_strategy="steps",
        save_steps=t["save_steps"],
        save_total_limit=1,
        seed=d["seed"],
        data_seed=d["seed"],
        report_to="none",
    )


def build_datasets(
    cfg: dict[str, Any], tokenizer: Any, end_token: str, echo: Callable[[str], None]
) -> dict[str, Any]:
    from datasets import Dataset

    d, m = cfg["data"], cfg["model"]
    items = read_jsonl(config.resolve_path(d["train_path"]))
    pool = fmt.subsample(items, d.get("max_train_examples") or 0, seed=d["seed"])
    train_items, val_items = fmt.split_train_val(pool, d["val_fraction"], seed=d["seed"])

    def convert(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
        return [
            fmt.to_prompt_completion(
                r,
                tokenizer,
                chat_template_kwargs=m["chat_template_kwargs"],
                system_suffix=m["system_suffix"],
            )
            for r in rows
        ]

    train, dropped_train = fmt.select_by_length(
        convert(train_items), tokenizer, d["max_seq_len"], end_token
    )
    val, dropped_val = fmt.select_by_length(
        convert(val_items), tokenizer, d["max_seq_len"], end_token
    )
    dropped = dropped_train + dropped_val
    echo(
        f"pares: {len(items)} no arquivo, {len(pool)} usados; treino {len(train)}, "
        f"validação {len(val)}, descartados por tamanho {len(dropped)}"
    )
    return {
        "train": Dataset.from_list(train),
        "val": Dataset.from_list(val) if val else None,
        "n_items": len(items),
        "dropped": dropped,
    }


def completion_fraction(dataset: Any) -> float:
    """Fração dos tokens do treino que entram na perda (completion), após o trl tokenizar."""
    masked = total = 0
    for row in dataset:
        labels = row.get("labels")
        mask = row.get("completion_mask")
        if mask is not None:
            masked += sum(mask)
        elif labels is not None:
            masked += sum(1 for x in labels if x != -100)
        total += len(row["input_ids"])
    return masked / total if total else 0.0


def _cast_trainable_to_fp32(model: Any) -> None:
    import torch

    for param in model.parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)


def run_finetune(
    cfg: dict[str, Any],
    *,
    out_dir: str | Path,
    device: str = "cuda",
    max_steps: int | None = None,
    echo: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Treina o adapter LoRA, grava `adapter/` e `train-log__<system>.json` em `out_dir`."""
    import torch
    from trl import SFTTrainer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = out_dir / "adapter"
    ckpt_dir = out_dir / "checkpoints"
    base_model = cfg["model"]["base_model"]
    system = train_config.system_name(cfg)  # ft-<modelo> (v1) ou ft2-<modelo> (v2)
    started = time.time()

    tokenizer = load_tokenizer(cfg)
    end_token = fmt.turn_end_token(tokenizer, cfg["model"]["chat_template_kwargs"])
    data = build_datasets(cfg, tokenizer, end_token, echo)
    model, quantized = load_base_model(cfg, device)
    hardware = local.hardware_label(local._device_name(device))
    args = sft_config(
        cfg,
        output_dir=ckpt_dir,
        device=device,
        end_token=end_token,
        has_eval=data["val"] is not None,
        max_steps=max_steps,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["val"],
        processing_class=tokenizer,
        peft_config=lora_config(cfg),
    )
    if args.fp16:
        _cast_trainable_to_fp32(trainer.model)
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    frac = completion_fraction(trainer.train_dataset)
    echo(
        f"{system}: {hardware}, 4-bit={quantized}, fim de turno {end_token!r}, "
        f"{trainable:,} parâmetros treináveis, {frac:.1%} dos tokens na perda"
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    result = trainer.train()
    train_seconds = time.perf_counter() - t0
    final_eval = trainer.evaluate() if data["val"] is not None else {}

    trainer.save_model(str(adapter_dir))
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    log_history = list(trainer.state.log_history)
    peak_mem = (
        round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None
    )
    info = {
        "system": system,
        "base_model": base_model,
        "base_model_revision": local._resolved_revision(
            base_model, cfg["model"]["revision"] or None
        )[0],
        "adapter_dir": str(adapter_dir),
        "hardware": hardware,
        "quantized": quantized,
        "end_token": end_token,
        "n_items_file": data["n_items"],
        "n_train": len(data["train"]),
        "n_val": len(data["val"]) if data["val"] is not None else 0,
        "dropped_too_long": data["dropped"],
        "completion_token_fraction": round(frac, 4),
        "trainable_params": trainable,
        "global_step": trainer.state.global_step,
        "max_steps_override": max_steps,
        "train_seconds": round(train_seconds, 1),
        "seconds_per_step": round(train_seconds / max(1, trainer.state.global_step), 2),
        "total_seconds": round(time.time() - started, 1),
        "peak_gpu_mem_gib": peak_mem,
        "train_loss": result.training_loss,
        "final_eval_loss": final_eval.get("eval_loss"),
        "config": cfg,
        "versions": versions(),
        "log_history": log_history,
    }
    log_path = out_dir / f"train-log__{system}.json"
    log_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    echo(
        f"{system}: {info['global_step']} passos em {info['train_seconds']} s "
        f"({info['seconds_per_step']} s/passo); perda final de validação "
        f"{info['final_eval_loss']}; adapter em {adapter_dir}; log em {log_path}"
    )
    del trainer, model
    local._free_cuda()
    return info


# --- relatório ------------------------------------------------------------------


def _fmt_loss(x: Any) -> str:
    return f"{x:.4f}" if isinstance(x, int | float) and math.isfinite(x) else ""


def loss_table(log_history: list[dict[str, Any]]) -> str:
    """Tabela Markdown das curvas de perda (treino e validação) por passo."""
    rows: dict[int, dict[str, Any]] = {}
    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        row = rows.setdefault(step, {"epoch": entry.get("epoch")})
        if "loss" in entry:
            row["loss"] = entry["loss"]
        if "eval_loss" in entry:
            row["eval_loss"] = entry["eval_loss"]
    lines = ["| Passo | Época | Perda (treino) | Perda (val) |", "|---|---|---|---|"]
    for step in sorted(rows):
        r = rows[step]
        if "loss" not in r and "eval_loss" not in r:
            continue
        epoch = f"{r['epoch']:.2f}" if isinstance(r.get("epoch"), int | float) else ""
        lines.append(
            f"| {step} | {epoch} | {_fmt_loss(r.get('loss'))} | {_fmt_loss(r.get('eval_loss'))} |"
        )
    return "\n".join(lines)


def model_card(info: dict[str, Any], dev_ex: str = "_(pendente)_") -> str:
    """Model card provisório do adapter (README.md do repo no HF)."""
    cfg = info["config"]
    lora, t = cfg["lora"], cfg["training"]
    config_file = info.get("config_file", "configs/train.toml")
    return f"""---
base_model: {info["base_model"]}
library_name: peft
license: apache-2.0
language: [pt]
tags: [text-to-sql, duckdb, qlora, lora, anp]
---

# tanque-cheio: adapter LoRA de {info["base_model"]}

Text-to-SQL em português sobre os preços de combustíveis da ANP (DuckDB).
Projeto: https://github.com/belasoaress/tanque-cheio. **Card provisório (M6).**

- Base: `{info["base_model"]}` (commit `{info["base_model_revision"]}`), QLoRA 4-bit nf4.
- LoRA: r={lora["r"]}, alpha={lora["alpha"]}, dropout={lora["dropout"]},
  alvos {", ".join(lora["target_modules"])}.
- Treino: {info["n_train"]} pares sintéticos validados por execução, {t["epochs"]} épocas,
  lr {t["learning_rate"]} ({t["lr_scheduler"]}), batch efetivo
  {t["per_device_train_batch_size"] * t["gradient_accumulation_steps"]}, seed
  {cfg["data"]["seed"]}; perda só na SQL. Hardware: {info["hardware"]},
  {round(info["train_seconds"] / 3600, 2)} h.
- EX no {info.get("eval_split", "dev (50 perguntas)")}: {dev_ex}.
- Código: commit `{info.get("repo_commit", "?")}` do repositório; `{config_file}`.

Uso: o prompt é o de `tanque.prompting.build_prompt` pelo chat template do modelo base;
a resposta é só a SQL. Dados: série histórica de preços de combustíveis da ANP.
"""
