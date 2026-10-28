"""Previsões de um modelo open-source local (transformers), no formato da seção 5.3 (M5/M6).

Usado pelos notebooks `03_predict.ipynb` (um modelo, um split) e
`01_bakeoff.ipynb` (vários candidatos no dev), que rodam no Kaggle. O código
fica no pacote, e não nos notebooks, para os dois usarem a mesma lógica e para
ela ser testada sem GPU (`tests/test_infer_local.py`, com modelo e tokenizer
falsos; `tests/test_infer_local_hf.py`, com um modelo minúsculo em CPU).

Decisões:

- **Prompt:** `prompting.build_prompt` (o mesmo de todos os sistemas) passado
  pelo chat template do tokenizer com `add_generation_prompt=True`. Se o
  template recusa a mensagem `system` (levanta erro) ou a descarta em silêncio
  (o texto do system não aparece no prompt renderizado), o system vai no começo
  da primeira mensagem `user`, separado por uma linha em branco
  (`render_prompt` informa isso em `system_merged`).
- **Raciocínio:** modelos híbridos (Qwen3, SmolLM3) recebem
  `chat_template_kwargs={"enable_thinking": False}` do notebook. O laço não
  remove blocos `<think>`: `sql = extract_sql(raw_output)`, como nos outros
  sistemas.
- **Data do dia:** o template do SmolLM3 põe "Today Date: <hoje>" no system, o
  que muda o prompt conforme o dia da execução e contradiz a data de referência
  (D14). O notebook passa `system_suffix="/system_override"`, sinal que o
  template remove e que desliga esse cabeçalho; o resto do prompt fica igual.
- **Decodificação gulosa pura:** `do_sample=False`, `num_beams=1` e
  `repetition_penalty=1.0`, passados como argumentos de `generate` (prioridade
  máxima no transformers 4.x e 5.x), para ignorar os padrões de amostragem do
  `generation_config.json` do checkpoint (ex.: Qwen2.5 traz
  `repetition_penalty=1.05`). `max_new_tokens=256`.
- **Latência:** por item, do template à decodificação, com
  `torch.cuda.synchronize()` antes e depois (`sync`). Lote de 1 pergunta.
- **Tokens:** `input_tokens` = tamanho do prompt tokenizado; `output_tokens` =
  tokens gerados (inclui o token de fim, se houver). Saída cortada em
  `max_new_tokens` **não** vira `error`: a SQL extraída é avaliada como está;
  `summarize` conta esses casos.
- **Precisão:** fp16 por padrão (a T4 não tem bf16 nativo). Sem 4-bit na
  inferência base, para não depender de bitsandbytes e para medir o modelo
  como ele é; o QLoRA do M6 quantiza só no treino.
- **Adapter (M6):** `adapter_id` (repo do HF ou pasta local, ex.:
  `/kaggle/working/adapter`) carrega o LoRA com `PeftModel.from_pretrained`
  sobre o base em fp16, sem fundir; o sistema vira `ft-<modelo>`. Testado em
  CPU com um adapter treinado na hora (`tests/test_train_smoke.py`).
- **Erros:** exceção num item (ex.: falta de memória) vira previsão com
  `error` e o laço continua.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MAX_NEW_TOKENS = 256
COST_USD = 0.0
SYSTEM_SEPARATOR = "\n\n"

BuildPrompt = Callable[[str], list[dict]]
ExtractSql = Callable[[str], str]


def _default_build_prompt(question: str) -> list[dict]:
    from tanque import prompting

    return prompting.build_prompt(question)


def _default_extract_sql(raw: str) -> str:
    from tanque import prompting

    return prompting.extract_sql(raw)


# --- nomes ----------------------------------------------------------------------


def model_slug(model_id: str) -> str:
    """Nome curto do modelo para `system`: o nome do repo, minúsculo, sem o dono."""
    name = model_id.rstrip("/").split("/")[-1].lower()
    return re.sub(r"[^a-z0-9.]+", "-", name).strip("-")


def system_name(model_id: str, adapter_id: str | None = None) -> str:
    """`base-<modelo>` sem adapter; `ft-<modelo>` com adapter (seção 5.3)."""
    prefix = "ft" if adapter_id else "base"
    return f"{prefix}-{model_slug(model_id)}"


def output_path(out_dir: str | Path, system: str, split: str, limit: int | None = None) -> Path:
    """`<out_dir>/<system>__<split>.jsonl` (`...-sample<N>.jsonl` com limite, como no Claude)."""
    suffix = f"-sample{limit}" if limit else ""
    return Path(out_dir) / f"{system}__{split}{suffix}.jsonl"


def hardware_label(device_name: str | None, platform: str = "kaggle") -> str:
    """`kaggle-t4` para a T4 do Kaggle; `kaggle-<gpu>` para outra GPU; `cpu` sem GPU."""
    if not device_name:
        return "cpu"
    if re.search(r"\bT4\b", device_name):
        return f"{platform}-t4"
    return f"{platform}-{re.sub(r'[^a-z0-9]+', '-', device_name.lower()).strip('-')}"


def select_items(items: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """Os `limit` primeiros itens (todos se `limit` for None ou 0)."""
    return items[:limit] if limit else items


# --- chat template --------------------------------------------------------------


@dataclass
class RenderedPrompt:
    text: str
    system_merged: bool


def merge_system(messages: list[dict]) -> list[dict]:
    """Junta o(s) `system` no começo da primeira mensagem `user`."""
    system = SYSTEM_SEPARATOR.join(str(m["content"]) for m in messages if m["role"] == "system")
    rest = [dict(m) for m in messages if m["role"] != "system"]
    if not system:
        return rest
    for m in rest:
        if m["role"] == "user":
            m["content"] = f"{system}{SYSTEM_SEPARATOR}{m['content']}"
            return rest
    return [{"role": "user", "content": system}, *rest]


def _apply(tokenizer: Any, messages: list[dict], kwargs: dict[str, Any]) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs
    )


def render_prompt(
    tokenizer: Any,
    messages: list[dict],
    chat_template_kwargs: dict[str, Any] | None = None,
    system_suffix: str = "",
) -> RenderedPrompt:
    """Prompt em texto pelo chat template; junta o system no user se o template não o aceita.

    `system_suffix` é acrescentado ao system só para sinalizar algo ao template
    (ex.: `/system_override` do SmolLM3, que o template remove e que desliga o
    cabeçalho com a data do dia). Não entra quando o system é juntado ao user.
    """
    kwargs = dict(chat_template_kwargs or {})
    systems = [str(m["content"]) for m in messages if m["role"] == "system"]
    if systems:
        flagged = [
            {**m, "content": f"{m['content']}{system_suffix}"} if m["role"] == "system" else m
            for m in messages
        ]
        try:
            text = _apply(tokenizer, flagged, kwargs)
        except Exception:  # template sem suporte a system (ex.: TemplateError do jinja)
            text = None
        if text is not None and all(s in text for s in systems):
            return RenderedPrompt(text, system_merged=False)
        return RenderedPrompt(_apply(tokenizer, merge_system(messages), kwargs), True)
    return RenderedPrompt(_apply(tokenizer, messages, kwargs), system_merged=False)


# --- geração --------------------------------------------------------------------


@dataclass(frozen=True)
class Generation:
    text: str
    input_tokens: int
    output_tokens: int


class HFGenerator:
    """Geração gulosa com um modelo `transformers` (lote de 1)."""

    def __init__(self, model: Any, tokenizer: Any, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS):
        self.model = model
        self.tokenizer = tokenizer
        pad = tokenizer.pad_token_id
        if pad is None:
            pad = tokenizer.eos_token_id
        self.generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "repetition_penalty": 1.0,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "pad_token_id": pad,
        }

    def sync(self) -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def generate(self, prompt: str) -> Generation:
        import torch

        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        device = self.model.device
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.inference_mode():
            out = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **self.generate_kwargs
            )
        new_ids = out[0, input_ids.shape[1] :].tolist()
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return Generation(text, int(input_ids.shape[1]), len(new_ids))


def load_model(
    model_id: str,
    adapter_id: str | None = None,
    *,
    dtype: str = "float16",
    device: str = "cuda",
    revision: str | None = None,
) -> tuple[Any, Any]:
    """(modelo, tokenizer) do Hugging Face, em `dtype`, com o adapter LoRA opcional."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    # `dtype=` existe a partir do transformers 4.56; antes era `torch_dtype=`.
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dtype_kw = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, device_map=device, **{dtype_kw: torch_dtype}
    )
    if adapter_id:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_id)
    model.eval()
    return model, tokenizer


# --- laço de previsão -----------------------------------------------------------


def _record(
    item_id: str,
    *,
    system: str,
    hardware: str,
    raw: str,
    sql: str,
    latency_ms: int | None,
    input_tokens: int,
    output_tokens: int,
    error: str | None,
) -> dict[str, Any]:
    return {
        "id": item_id,
        "system": system,
        "sql": sql,
        "raw_output": raw,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": COST_USD,
        "hardware": hardware,
        "error": error,
    }


def predict(
    items: Iterable[dict[str, Any]],
    generator: Any,
    tokenizer: Any,
    *,
    system: str,
    hardware: str,
    chat_template_kwargs: dict[str, Any] | None = None,
    system_suffix: str = "",
    build_prompt: BuildPrompt = _default_build_prompt,
    extract_sql: ExtractSql = _default_extract_sql,
    sync: Callable[[], None] = lambda: None,
    clock: Callable[[], float] = time.perf_counter,
    echo: Callable[[str], None] = lambda _: None,
) -> list[dict[str, Any]]:
    """Uma previsão (seção 5.3) por item, na ordem dos itens."""
    items = list(items)
    out: list[dict[str, Any]] = []
    for n, item in enumerate(items, start=1):
        item_id = str(item["id"])
        try:
            sync()
            start = clock()
            messages = build_prompt(item["question"])
            rendered = render_prompt(tokenizer, messages, chat_template_kwargs, system_suffix)
            gen = generator.generate(rendered.text)
            sync()
            latency = round((clock() - start) * 1000)
        except Exception as e:  # um item com erro não para o lote
            out.append(
                _record(
                    item_id,
                    system=system,
                    hardware=hardware,
                    raw="",
                    sql="",
                    latency_ms=None,
                    input_tokens=0,
                    output_tokens=0,
                    error=f"{type(e).__name__}: {e}",
                )
            )
            echo(f"[{n}/{len(items)}] {item_id}: erro {type(e).__name__}")
            continue
        out.append(
            _record(
                item_id,
                system=system,
                hardware=hardware,
                raw=gen.text,
                sql=extract_sql(gen.text) if gen.text else "",
                latency_ms=latency,
                input_tokens=gen.input_tokens,
                output_tokens=gen.output_tokens,
                error=None,
            )
        )
        echo(f"[{n}/{len(items)}] {item_id}: {latency} ms, {gen.output_tokens} tokens")
    return out


def write_predictions(path: str | Path, predictions: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return path


def summarize(
    predictions: list[dict[str, Any]], max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
) -> dict[str, Any]:
    """Contagens rápidas para o log do notebook (as métricas vêm do `tanque eval score`)."""
    lat = sorted(p["latency_ms"] for p in predictions if p.get("latency_ms") is not None)
    return {
        "n": len(predictions),
        "errors": sum(1 for p in predictions if p.get("error")),
        "empty_sql": sum(1 for p in predictions if not str(p.get("sql") or "").strip()),
        "hit_max_new_tokens": sum(
            1 for p in predictions if (p.get("output_tokens") or 0) >= max_new_tokens
        ),
        "p50_latency_ms": lat[len(lat) // 2] if lat else None,
        "mean_input_tokens": (
            sum(p["input_tokens"] for p in predictions) / len(predictions) if predictions else 0
        ),
    }


# --- execução completa (notebooks e `tanque infer local`) -----------------------


def _versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in ("torch", "transformers", "peft", "accelerate"):
        try:
            module = __import__(name)
            out[name] = str(module.__version__)
        except Exception:  # pacote ausente ou quebrado: só não registra a versão
            out[name] = None
    return out


def _resolved_revision(model_id: str, revision: str | None) -> tuple[str | None, str | None]:
    """(commit do checkpoint no Hugging Face, erro); sem rede, devolve o pedido e o erro."""
    try:
        from huggingface_hub import model_info

        return model_info(model_id, revision=revision).sha, None
    except Exception as e:  # sem rede, limite da API ou repo local
        return revision, f"{type(e).__name__}: {e}"


def _device_name(device: str) -> str | None:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return None


def _free_cuda() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_predictions(
    dataset_path: str | Path,
    model_id: str,
    *,
    out_dir: str | Path,
    adapter_id: str | None = None,
    revision: str | None = None,
    system: str | None = None,
    limit: int | None = None,
    dtype: str = "float16",
    device: str = "cuda",
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    chat_template_kwargs: dict[str, Any] | None = None,
    system_suffix: str = "",
    echo: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Carrega o modelo, gera as previsões do dataset, grava o JSONL e libera a memória.

    Devolve um resumo da execução (arquivo, contagens, versões, commit do
    checkpoint), que os notebooks imprimem e gravam em `run-info__<...>.json`.
    """
    from tanque.eval.score import read_jsonl

    dataset_path = Path(dataset_path)
    all_items = read_jsonl(dataset_path)
    splits = {str(i["split"]) for i in all_items if i.get("split")}
    split = splits.pop() if len(splits) == 1 else dataset_path.stem
    items = select_items(all_items, limit)
    if not items:
        raise ValueError(f"dataset vazio: {dataset_path}")
    system = system or system_name(model_id, adapter_id)
    path = output_path(out_dir, system, split, limit)

    started = time.time()
    model, tokenizer = load_model(
        model_id, adapter_id, dtype=dtype, device=device, revision=revision
    )
    load_s = round(time.time() - started, 1)
    hardware = hardware_label(_device_name(device))
    first_messages = _default_build_prompt(items[0]["question"])
    first = render_prompt(tokenizer, first_messages, chat_template_kwargs, system_suffix)
    echo(
        f"{system}: {len(items)} itens de {dataset_path.name}, {hardware}, {dtype}; "
        f"system {'junto ao user' if first.system_merged else 'como role system'}; "
        f"modelo carregado em {load_s} s"
    )
    gen = HFGenerator(model, tokenizer, max_new_tokens=max_new_tokens)
    try:
        predictions = predict(
            items,
            gen,
            tokenizer,
            system=system,
            hardware=hardware,
            chat_template_kwargs=chat_template_kwargs,
            system_suffix=system_suffix,
            sync=gen.sync,
            echo=echo,
        )
    finally:
        del gen, model
        _free_cuda()
    write_predictions(path, predictions)
    summary = summarize(predictions, max_new_tokens=max_new_tokens)
    model_revision, revision_error = _resolved_revision(model_id, revision)
    if revision_error:
        echo(f"Aviso: commit do checkpoint não resolvido ({revision_error})")
    echo(f"{system}: {summary} -> {path}")
    return {
        "system": system,
        "split": split,
        "path": str(path),
        "model_id": model_id,
        "model_revision": model_revision,
        "model_revision_error": revision_error,
        "adapter_id": adapter_id,
        "dtype": dtype,
        "hardware": hardware,
        "max_new_tokens": max_new_tokens,
        "chat_template_kwargs": dict(chat_template_kwargs or {}),
        "system_suffix": system_suffix,
        "system_merged_into_user": first.system_merged,
        "limit": limit,
        "load_s": load_s,
        "total_s": round(time.time() - started, 1),
        "versions": _versions(),
        "summary": summary,
    }
