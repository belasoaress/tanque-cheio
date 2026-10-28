"""Exemplos de treino no formato prompt/completion do `trl` (M6).

Regra central: o modelo treina com **o mesmo texto de prompt que vê na
inferência**. Por isso o prompt de cada par é `tanque.infer.local.render_prompt`
aplicado a `prompting.build_prompt(pergunta)`, com as mesmas opções de template
(`chat_template_kwargs`, `system_suffix`) e o mesmo fallback de system no user,
e já termina no início da resposta do assistente (`add_generation_prompt=True`).
A completion é exatamente a SQL do par; o `SFTTrainer` acrescenta o token de fim
de turno (`turn_end_token`) e calcula a perda só na completion
(`completion_only_loss=True`).

Também aqui: divisão treino/validação por seed e subamostragem estratificada por
(nível, intenção), usada quando `max_train_examples > 0`.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from tanque.infer import local

BuildPrompt = Callable[[str], list[dict]]

_PROBE = "SELECT 1"


def to_prompt_completion(
    item: dict[str, Any],
    tokenizer: Any,
    *,
    chat_template_kwargs: dict[str, Any] | None = None,
    system_suffix: str = "",
    build_prompt: BuildPrompt = local._default_build_prompt,
) -> dict[str, str]:
    """`{"id", "prompt", "completion"}`: prompt de inferência renderizado + a SQL do par."""
    messages = build_prompt(item["question"])
    rendered = local.render_prompt(tokenizer, messages, chat_template_kwargs, system_suffix)
    return {"id": str(item["id"]), "prompt": rendered.text, "completion": item["sql"]}


def turn_end_token(tokenizer: Any, chat_template_kwargs: dict[str, Any] | None = None) -> str:
    """Token especial que o chat template põe logo depois da resposta do assistente.

    No Qwen2.5 é `<|im_end|>` (que também é o `eos_token` do tokenizer e para a
    geração). Se o template não fecha a resposta com um token especial, usa o
    `eos_token`.
    """
    messages = [
        {"role": "user", "content": "Pergunta: ?"},
        {"role": "assistant", "content": _PROBE},
    ]
    try:
        full = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, **(chat_template_kwargs or {})
        )
    except Exception:  # template sem suporte a essa conversa: fica o eos
        return tokenizer.eos_token
    idx = full.rfind(_PROBE)
    if idx >= 0:
        ids = tokenizer(full[idx + len(_PROBE) :], add_special_tokens=False)["input_ids"]
        if ids:
            first = tokenizer.convert_ids_to_tokens(ids[0])
            if first in tokenizer.all_special_tokens:
                return first
    return tokenizer.eos_token


def select_by_length(
    examples: list[dict[str, str]], tokenizer: Any, max_seq_len: int, end_token: str
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """(pares que cabem em `max_seq_len` tokens com o token de fim, descartados com o tamanho).

    Os descartados não são truncados: truncar cortaria o fim da SQL e o token de fim.
    """
    kept, dropped = [], []
    for ex in examples:
        text = ex["prompt"] + ex["completion"] + end_token
        n = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if n <= max_seq_len:
            kept.append(ex)
        else:
            dropped.append({"id": ex["id"], "tokens": n})
    return kept, dropped


# --- seleção e divisão ----------------------------------------------------------


def stratum(item: dict[str, Any]) -> tuple[str, str]:
    """(nível, intenção) do par; a intenção vem da tag `intent:<nome>` do gerador (M4)."""
    intent = next(
        (t.split(":", 1)[1] for t in item.get("tags") or [] if str(t).startswith("intent:")), ""
    )
    return str(item.get("difficulty", "")), intent


def subsample(items: list[dict[str, Any]], n: int | None, seed: int) -> list[dict[str, Any]]:
    """`n` pares sorteados mantendo a proporção de cada (nível, intenção), na ordem original.

    Cotas por maior resto. `n` vazio, zero ou >= len(items) devolve tudo.
    """
    if not n or n >= len(items):
        return list(items)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, item in enumerate(items):
        groups[stratum(item)].append(i)
    keys = sorted(groups)
    exact = {k: len(groups[k]) * n / len(items) for k in keys}
    quota = {k: int(exact[k]) for k in keys}
    by_remainder = sorted(keys, key=lambda k: (-(exact[k] - quota[k]), k))
    for k in by_remainder[: n - sum(quota.values())]:
        quota[k] += 1
    rng = random.Random(seed)
    chosen: list[int] = []
    for k in keys:
        chosen.extend(rng.sample(groups[k], quota[k]))
    return [items[i] for i in sorted(chosen)]


def split_train_val(
    items: list[dict[str, Any]], val_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(treino, validação), sorteados por `seed`; cada parte na ordem original."""
    from tanque.train.config import n_val

    k = n_val(len(items), val_fraction)
    if k == 0:
        return list(items), []
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    val_idx = set(order[:k])
    train = [x for i, x in enumerate(items) if i not in val_idx]
    val = [x for i, x in enumerate(items) if i in val_idx]
    return train, val
