"""Previsões de um modelo open-source local (transformers), no formato da seção 5.3 (M5/M6).

Usado pelos notebooks `03_predict.ipynb` (um modelo, um split) e
`01_bakeoff.ipynb` (vários candidatos no dev), que rodam no Kaggle, e pelos
seguintes (04 no teste; 05, os modelos maiores no dev2, via
`tanque.infer.large`). O código
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
  `repetition_penalty=1.05`). `max_new_tokens=384` por padrão na v2 (D31; na
  v1 era 256), configurável e registrado no run-info.
- **Latência:** por item, do template à decodificação, com
  `torch.cuda.synchronize()` antes e depois (`sync`). Lote de 1 pergunta.
- **Tokens:** `input_tokens` = tamanho do prompt tokenizado; `output_tokens` =
  tokens gerados (inclui o token de fim, se houver). Saída cortada em
  `max_new_tokens` **não** vira `error`: a SQL extraída é avaliada como está;
  `summarize` conta esses casos.
- **Precisão:** fp16 por padrão (a T4 não tem bf16 nativo). O 1,5B (base e
  ft) roda assim, sem 4-bit, para não depender de bitsandbytes e para medir o
  modelo como ele é; o QLoRA do M6 quantiza só no treino.
- **Modelos maiores (v2, M14, D27):** `quantization="nf4"` carrega o modelo em
  4 bits com `BitsAndBytesConfig` (nf4, quantização dupla, computação em fp16;
  `bnb_config_kwargs`). Precisa de bitsandbytes e de GPU (o notebook
  `05_large_baselines` instala). `device` vira o `device_map` do
  `from_pretrained`: `"cuda"` (padrão, igual à v1), `{"": 0}` (tudo na GPU 0)
  ou `"auto"` (o accelerate reparte as camadas entre as GPUs visíveis, com
  `max_memory` opcional por GPU). As GPUs usadas saem do `hf_device_map` do
  modelo (`gpu_indices`).
- **Memória e GPUs (5.3 v2):** antes de carregar, o pico de memória de cada GPU
  visível é zerado; no fim, `peak_gpu_mem_gib` = o maior
  `torch.cuda.max_memory_allocated` entre as GPUs usadas, em GiB (2 casas).
  Cada previsão (single-shot e `+retry`) e o run-info ganham `quantization`
  (`"nf4"` ou `null` sem quantização; o `dtype` fica no run-info), `gpus` (0 em
  CPU) e `peak_gpu_mem_gib` (`null` em CPU), o pico da execução inteira. O custo
  (D35) é `gpus × gpu_hour_usd` de `configs/settings.toml`, calculado no
  relatório.
- **Checkpoint por item e limite de tempo (M14, D37):** cada previsão é gravada
  no JSONL assim que sai (append + flush). Com `resume=True`, os itens que já
  estão nos arquivos (no single-shot e, com `retry`, também no `+retry`) são
  pulados; uma linha final cortada é descartada. Com `time_budget_s`, o laço
  para antes de começar um item quando o tempo desde o início de
  `run_predictions` (download e carga inclusos) passa do limite; o run-info
  registra `stopped_reason = "time_budget"` e `completed` < `n_items`. No fim,
  os arquivos são reescritos com o pico de memória final em todas as linhas.
- **Adapter (M6/M7):** `adapter_id` (repo do HF ou pasta local, ex.:
  `/kaggle/working/adapter`) carrega o LoRA com `PeftModel.from_pretrained`
  sobre o base em fp16; o sistema vira `ft-<modelo>`. Por padrão
  (`merge_adapter=True`) os pesos do LoRA são **fundidos** no base
  (`merge_and_unload()`) antes de gerar: a saída é a mesma (a menos de
  arredondamento em fp16), mas sem as camadas extras do PEFT a cada passo, o
  que deixa a latência do `ft` comparável à do `base` (no M6, sem fundir, o p50
  no dev foi 8,3 s contra 2,8 s). O run-info registra `adapter_merged`.
  Testado em CPU (`tests/test_infer_local_hf.py`, `tests/test_train_smoke.py`).
- **Erros:** exceção num item (ex.: falta de memória) vira previsão com
  `error` e o laço continua.
- **Autocorreção (v2, D32):** `run_predictions(..., retry=True)` usa
  `tanque.infer.repair`: executa cada SQL no banco (guardrail +
  `execute.run`) e, se ela foi bloqueada ou deu erro, gera 1 nova tentativa
  com `prompting.build_repair_prompt`. Numa só execução grava
  `<sistema>__<split>.jsonl` (single-shot, igual ao sem `retry`) e
  `<sistema>+retry__<split>.jsonl` (5.3 v2). O banco vem do snapshot: no
  Kaggle, o notebook roda antes `tanque data build --from-parquet
  snapshot/coletas-2026-09.parquet` (~6 s), que grava `data/tanque.duckdb`, o
  `db_path` padrão. Sem o banco, `run_predictions` falha antes de carregar o
  modelo.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MAX_NEW_TOKENS = 384  # D31 (v1: 256)
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


QUANTIZATIONS = ("nf4",)

DeviceMap = str | dict[str, Any]


def bnb_config_kwargs(quantization: str, compute_dtype: str = "float16") -> dict[str, Any]:
    """Argumentos do `BitsAndBytesConfig` para `quantization` (D27: nf4 em 4 bits).

    nf4 com quantização dupla (as constantes de escala também em 8 bits) e
    computação em fp16, porque a T4 não tem bf16 nativo.
    """
    if quantization not in QUANTIZATIONS:
        raise ValueError(f"quantização desconhecida: {quantization!r} (aceitas: {QUANTIZATIONS})")
    return {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": compute_dtype,
    }


def make_quantization_config(quantization: str | None, compute_dtype: str = "float16") -> Any:
    """`BitsAndBytesConfig` para `quantization`, ou None sem quantização."""
    if quantization is None:
        return None
    import torch
    from transformers import BitsAndBytesConfig

    kwargs = bnb_config_kwargs(quantization, compute_dtype)
    kwargs["bnb_4bit_compute_dtype"] = getattr(torch, kwargs["bnb_4bit_compute_dtype"])
    return BitsAndBytesConfig(**kwargs)


def load_model(
    model_id: str,
    adapter_id: str | None = None,
    *,
    dtype: str = "float16",
    device: DeviceMap = "cuda",
    revision: str | None = None,
    adapter_revision: str | None = None,
    merge_adapter: bool = True,
    quantization: str | None = None,
    max_memory: dict[Any, str] | None = None,
) -> tuple[Any, Any]:
    """(modelo, tokenizer) do Hugging Face, em `dtype`, com o adapter LoRA opcional.

    Com adapter e `merge_adapter=True`, devolve o modelo base com o LoRA fundido
    (um `PreTrainedModel` comum, sem `PeftModel` em volta).

    `device` é o `device_map` (`"cuda"`, `"cpu"`, `{"": 0}`, `"auto"`...);
    `quantization="nf4"` carrega em 4 bits (precisa de bitsandbytes e GPU);
    `max_memory` limita a memória por GPU com `device_map="auto"`.
    """
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if quantization is not None and adapter_id:
        raise ValueError("adapter sobre modelo quantizado não é suportado (fundir exige fp16)")
    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    # `dtype=` existe a partir do transformers 4.56; antes era `torch_dtype=`.
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dtype_kw = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    extra: dict[str, Any] = {}
    if quantization is not None:
        extra["quantization_config"] = make_quantization_config(quantization, dtype)
    if max_memory is not None:
        extra["max_memory"] = max_memory
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, device_map=device, **{dtype_kw: torch_dtype}, **extra
    )
    if adapter_id:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_id, revision=adapter_revision)
        if merge_adapter:
            model = model.merge_and_unload()
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


def _uses_cuda(device: DeviceMap) -> bool:
    if isinstance(device, dict):
        return bool(gpu_indices(device.values()))
    return device == "auto" or str(device).startswith("cuda")


def _device_name(device: DeviceMap) -> str | None:
    """Nome da GPU 0 (a mesma placa em todas as GPUs do Kaggle), ou None sem GPU."""
    import torch

    if _uses_cuda(device) and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return None


# --- GPUs e memória (5.3 v2) ------------------------------------------------------


def _gpu_index(value: Any) -> int | None:
    """Índice da GPU de um valor do `device_map` (0, "cuda", "cuda:1", torch.device)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value)
    if text == "cuda":
        return 0
    m = re.fullmatch(r"cuda:(\d+)", text)
    return int(m.group(1)) if m else None  # "cpu", "disk", "mps"...


def gpu_indices(devices: Iterable[Any]) -> list[int]:
    """GPUs distintas (ordenadas) entre os valores de um `device_map`."""
    found = {_gpu_index(v) for v in devices}
    return sorted(i for i in found if i is not None)


def model_gpus(model: Any) -> list[int]:
    """GPUs onde o modelo ficou: `hf_device_map` (accelerate) ou `model.device`."""
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        return gpu_indices(device_map.values())
    return gpu_indices([getattr(model, "device", "cpu")])


def peak_gpu_mem_gib(
    indices: Iterable[int], max_allocated: Callable[[int], int] | None = None
) -> float | None:
    """Maior pico de memória alocada entre as GPUs `indices`, em GiB (None sem GPU)."""
    indices = list(indices)
    if not indices:
        return None
    if max_allocated is None:
        import torch

        max_allocated = torch.cuda.max_memory_allocated
    return round(max(max_allocated(i) for i in indices) / 2**30, 2)


def _reset_peak_memory() -> None:
    import torch

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)


# --- checkpoint por item (M14) ----------------------------------------------------


def read_partial(path: str | Path) -> list[dict[str, Any]]:
    """Previsões já gravadas em `path` (vazio se não existe); ignora uma linha cortada."""
    path = Path(path)
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:  # a sessão caiu no meio da escrita
            break
    return out


def done_records(
    items: list[dict[str, Any]], singles: list[dict[str, Any]], retried: list[dict[str, Any]] | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(single-shot, +retry) já prontos, na ordem dos itens.

    Um item conta como pronto se está no single-shot e, quando há `+retry`
    (`retried` não é None), também nele.
    """
    by_single = {str(p["id"]): p for p in singles}
    by_retry = {str(p["id"]): p for p in retried or []}
    s_out: list[dict[str, Any]] = []
    r_out: list[dict[str, Any]] = []
    for item in items:
        item_id = str(item["id"])
        if item_id not in by_single or (retried is not None and item_id not in by_retry):
            continue
        s_out.append(by_single[item_id])
        if retried is not None:
            r_out.append(by_retry[item_id])
    return s_out, r_out


def _append(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


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
    adapter_revision: str | None = None,
    merge_adapter: bool = True,
    system: str | None = None,
    limit: int | None = None,
    dtype: str = "float16",
    device: DeviceMap = "cuda",
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    chat_template_kwargs: dict[str, Any] | None = None,
    system_suffix: str = "",
    retry: bool = False,
    db_path: str | Path | None = None,
    quantization: str | None = None,
    max_memory: dict[Any, str] | None = None,
    resume: bool = False,
    time_budget_s: float | None = None,
    echo: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Carrega o modelo, gera as previsões do dataset, grava o JSONL e libera a memória.

    Devolve um resumo da execução (arquivo, contagens, versões, commit do
    checkpoint), que os notebooks imprimem e gravam em `run-info__<...>.json`.

    Com `retry=True` (D32), executa cada SQL no banco `db_path` (padrão:
    `data/tanque.duckdb`, montado do snapshot) e grava também
    `<sistema>+retry__<split>.jsonl`; o resumo ganha `retry_system`,
    `retry_path` e `retry_summary` (`retry_run_info` monta o run-info desse
    arquivo).

    M14: `quantization`, `device` (device_map) e `max_memory` vão para
    `load_model`; `resume` pula os itens já gravados; `time_budget_s` para o
    laço quando o tempo acaba (ver o docstring do módulo).
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
    executor = None
    retry_name: str | None = None
    retry_path: Path | None = None
    if retry:
        from tanque.infer import repair

        # Antes de carregar o modelo: sem banco, falha em segundos, não em minutos.
        executor = repair.make_executor(db_path)
        retry_name = repair.retry_system(system)
        retry_path = output_path(out_dir, retry_name, split, limit)

    # Checkpoint por item: o que já está gravado (resume) fica; o resto é refeito.
    if resume:
        prev_single, prev_retry = done_records(
            items,
            read_partial(path),
            read_partial(retry_path) if retry_path is not None else None,
        )
    else:
        prev_single, prev_retry = [], []
    done_ids = {str(p["id"]) for p in prev_single}
    todo = [i for i in items if str(i["id"]) not in done_ids]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_predictions(path, prev_single)  # descarta linhas cortadas ou de itens incompletos
    if retry_path is not None:
        write_predictions(retry_path, prev_retry)
    if prev_single:
        echo(f"{system}: {len(prev_single)} itens já prontos (resume), {len(todo)} a fazer")

    started = time.time()
    gpus_used: list[int] = []
    hardware = str(prev_single[0].get("hardware")) if prev_single else hardware_label(None)
    system_merged = False
    load_s = 0.0
    stopped_reason: str | None = None
    new_single: list[dict[str, Any]] = []
    new_retry: list[dict[str, Any]] = []
    adapter_merged = bool(adapter_id) and merge_adapter
    if todo:
        if _uses_cuda(device):
            _reset_peak_memory()
        model, tokenizer = load_model(
            model_id,
            adapter_id,
            dtype=dtype,
            device=device,
            revision=revision,
            adapter_revision=adapter_revision,
            merge_adapter=merge_adapter,
            quantization=quantization,
            max_memory=max_memory,
        )
        load_s = round(time.time() - started, 1)
        gpus_used = model_gpus(model) if _uses_cuda(device) else []
        hardware = hardware_label(_device_name(device))
        first_messages = _default_build_prompt(todo[0]["question"])
        first = render_prompt(tokenizer, first_messages, chat_template_kwargs, system_suffix)
        system_merged = first.system_merged
        echo(
            f"{system}: {len(todo)} itens de {dataset_path.name}, {hardware}, {dtype}"
            + (f", {quantization}" if quantization else "")
            + f", GPUs {gpus_used or '-'}; "
            f"system {'junto ao user' if first.system_merged else 'como role system'}; "
            + (f"adapter {'fundido' if adapter_merged else 'sem fundir'}; " if adapter_id else "")
            + f"modelo carregado em {load_s} s"
        )
        gen = HFGenerator(model, tokenizer, max_new_tokens=max_new_tokens)
        common = {"quantization": quantization, "gpus": len(gpus_used)}
        total = len(items)
        try:
            for n, item in enumerate(todo, start=len(prev_single) + 1):
                if time_budget_s is not None and time.time() - started > time_budget_s:
                    stopped_reason = "time_budget"
                    echo(
                        f"{system}: limite de tempo ({time_budget_s / 3600:.2f} h) atingido; "
                        f"parado com {n - 1}/{total} itens"
                    )
                    break
                if executor is None:
                    single = predict(
                        [item],
                        gen,
                        tokenizer,
                        system=system,
                        hardware=hardware,
                        chat_template_kwargs=chat_template_kwargs,
                        system_suffix=system_suffix,
                        sync=gen.sync,
                    )[0]
                    retried = None
                else:
                    single, retried = repair.predict_item(
                        item,
                        chat_generate(gen, tokenizer, chat_template_kwargs, system_suffix),
                        executor,
                        system=system,
                        hardware=hardware,
                        sync=gen.sync,
                    )
                peak = {"peak_gpu_mem_gib": peak_gpu_mem_gib(gpus_used)}
                single.update(common, **peak)
                _append(path, single)
                new_single.append(single)
                if retried is not None and retry_path is not None:
                    retried.update(common, **peak)
                    _append(retry_path, retried)
                    new_retry.append(retried)
                if single["error"]:
                    status = f"erro {single['error'].split(':')[0]}"
                elif retried is not None:
                    status = "nova tentativa" if retried["attempts"] == 2 else "1 tentativa"
                else:
                    status = f"{single['output_tokens']} tokens"
                latency = (retried or single)["latency_ms"]
                echo(f"[{n}/{total}] {item['id']}: {status}, {latency} ms")
        finally:
            del gen, model
            final_peak = peak_gpu_mem_gib(gpus_used)
            _free_cuda()
    else:
        final_peak = None
        echo(f"{system}: todos os {len(items)} itens já estavam prontos; modelo não carregado")

    # Pico da execução inteira em todas as linhas (as de uma sessão anterior guardam o delas).
    old_peaks = [p.get("peak_gpu_mem_gib") for p in prev_single + prev_retry]
    peaks = [v for v in [final_peak, *old_peaks] if v is not None]
    run_peak = max(peaks) if peaks else None
    run_gpus = len(gpus_used) if todo else int(prev_single[0].get("gpus") or 0)
    predictions = prev_single + new_single
    for p in predictions:
        p.update(quantization=quantization, gpus=run_gpus, peak_gpu_mem_gib=run_peak)
    write_predictions(path, predictions)
    summary = summarize(predictions, max_new_tokens=max_new_tokens)
    retry_info: dict[str, Any] = {}
    if executor is not None and retry_path is not None:
        retried_all = prev_retry + new_retry
        for p in retried_all:
            p.update(quantization=quantization, gpus=run_gpus, peak_gpu_mem_gib=run_peak)
        write_predictions(retry_path, retried_all)
        retry_info = {
            "retry_system": retry_name,
            "retry_path": str(retry_path),
            "retry_summary": retry_summary(retried_all),
        }
        echo(f"{retry_name}: {retry_info['retry_summary']} -> {retry_path}")
    model_revision, revision_error = _resolved_revision(model_id, revision)
    if revision_error:
        echo(f"Aviso: commit do checkpoint não resolvido ({revision_error})")
    adapter_commit, adapter_error = None, None
    if adapter_id and not Path(adapter_id).is_dir():
        adapter_commit, adapter_error = _resolved_revision(adapter_id, adapter_revision)
        if adapter_error:
            echo(f"Aviso: commit do adapter não resolvido ({adapter_error})")
    echo(f"{system}: {summary} -> {path}")
    return {
        "system": system,
        "split": split,
        "path": str(path),
        "model_id": model_id,
        "model_revision": model_revision,
        "model_revision_error": revision_error,
        "adapter_id": adapter_id,
        "adapter_revision": adapter_commit,
        "adapter_revision_error": adapter_error,
        "adapter_merged": adapter_merged,
        "dtype": dtype,
        "quantization": quantization,
        "device_map": device,
        "max_memory": {str(k): v for k, v in max_memory.items()} if max_memory else None,
        "gpus": run_gpus,
        "gpu_indices": gpus_used,
        "peak_gpu_mem_gib": run_peak,
        "hardware": hardware,
        "max_new_tokens": max_new_tokens,
        "chat_template_kwargs": dict(chat_template_kwargs or {}),
        "system_suffix": system_suffix,
        "system_merged_into_user": system_merged,
        "retry": retry,
        "db_path": str(db_path) if db_path is not None else None,
        "limit": limit,
        "n_items": len(items),
        "completed": len(predictions),
        "resumed": len(prev_single),
        "time_budget_s": time_budget_s,
        "stopped_reason": stopped_reason,
        "load_s": load_s,
        "total_s": round(time.time() - started, 1),
        "versions": _versions(),
        "summary": summary,
        **retry_info,
    }


def chat_generate(
    generator: Any,
    tokenizer: Any,
    chat_template_kwargs: dict[str, Any] | None = None,
    system_suffix: str = "",
) -> Callable[[list[dict]], Generation]:
    """Gerador "mensagens -> geração" para `repair`: chat template + `generator.generate`."""

    def generate(messages: list[dict]) -> Generation:
        rendered = render_prompt(tokenizer, messages, chat_template_kwargs, system_suffix)
        return generator.generate(rendered.text)

    return generate


def retry_summary(retried: list[dict[str, Any]]) -> dict[str, Any]:
    """Resumo do arquivo `+retry`: contagens da nova tentativa, erros e latência.

    Sem `hit_max_new_tokens`: lá `output_tokens` soma as duas gerações.
    """
    from tanque.infer import repair

    base = summarize(retried)
    base.pop("hit_max_new_tokens")
    return {**base, **repair.summarize_retry(retried)}


def retry_run_info(info: dict[str, Any]) -> dict[str, Any]:
    """Run-info do arquivo `+retry` a partir do resumo de `run_predictions(retry=True)`.

    Mesmo modelo, hardware e versões; `system`, `path` e `summary` do `+retry`.
    """
    if not info.get("retry_system"):
        raise ValueError("a execução não teve retry=True")
    out = {k: v for k, v in info.items() if not k.startswith("retry_")}
    out.update(
        system=info["retry_system"],
        path=info["retry_path"],
        summary=info["retry_summary"],
        single_shot_system=info["system"],
    )
    return out
