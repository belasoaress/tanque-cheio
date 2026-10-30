"""Modelos maiores no dev2 (M14) e teste final v2 (M17; D26, D27, D35, D37).

Lógica dos notebooks `05_large_baselines.ipynb` (dev2) e `06_final_v2.ipynb`
(teste), aqui para ser testada sem GPU (`tests/test_infer_large.py`). O notebook
monta a lista de modelos (`MODELS`, na 1ª célula) e chama `run_spec` para cada
um, em ordem, com o dataset do notebook (`DATASET`); cada chamada usa
`local.run_predictions` (prompt único, gulosa, `max_new_tokens = 384`,
autocorreção com 1 nova tentativa) e grava os run-info.

Cada modelo é um dicionário (`spec`):

- `model_id`, `revision` (commit fixado no Hugging Face);
- `adapter_id`, `adapter_revision` (opcionais: o ft-v1 ou o ft-v2, fundido);
- `system` (opcional): nome do sistema; sem ele, `base-<modelo>` ou
  `ft-<modelo>` (o ft-v2 usa `ft2-<modelo>`, seção 5.3 da v2);
- `quantization`: `None` (fp16) ou `"nf4"` (D27);
- `gpus`: 1 (`device_map={"": 0}`) ou 2 (`device_map="auto"`, com `max_memory`
  opcional por GPU);
- `part`: `"a"` ou `"b"`, para dividir a execução em 2 versões do kernel
  (`PART` no notebook; `"all"` roda tudo);
- `device`, `dtype` (opcionais): sobrepõem o `device_map` e o `dtype` (o
  ensaio em CPU usa `"cpu"` e `"float32"`, com `gpus = 0`);
- `time_budget_h` (opcional): limite de tempo do modelo (D37, o 32B);
- `fallback` (opcional): `{"model_id", "revision"}` de um checkpoint **já
  quantizado em nf4** do mesmo modelo, usado só se o disco da sessão não
  comporta o checkpoint original (o 32B tem 61 GiB em bf16; o nf4 tem 18 GiB).
  O run-info registra `model_source = "fallback"` e o id usado.

Se o modelo não pode rodar (GPUs a menos, disco cheio, falta de memória ao
carregar ou qualquer exceção), `run_spec` não para a sessão: devolve um
run-info com `skipped = true` e `skip_reason` (D37), e o notebook segue para o
próximo modelo. Depois de cada modelo, o cache do Hugging Face daquele
checkpoint é apagado, para o próximo caber no disco.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from tanque.infer import local

PARTS = ("all", "a", "b")
GIB = 2**30
DISK_MARGIN_BYTES = 3 * GIB  # folga para o tokenizer, o cache e as saídas


def device_map_for(spec: dict[str, Any]) -> local.DeviceMap:
    """`{"": 0}` para 1 GPU; `"auto"` (accelerate reparte as camadas) para 2 ou mais.

    `spec["device"]`, se houver, tem prioridade (ex.: `"cpu"` no ensaio local).
    """
    if "device" in spec:
        return spec["device"]
    return "auto" if int(spec.get("gpus", 1)) > 1 else {"": 0}


def select_models(models: Iterable[dict[str, Any]], part: str = "all") -> list[dict[str, Any]]:
    """Os modelos da parte `part` (`"all"`: todos), na ordem dada."""
    if part not in PARTS:
        raise ValueError(f"PART desconhecida: {part!r} (aceitas: {PARTS})")
    models = list(models)
    if part == "all":
        return models
    return [m for m in models if m.get("part") == part]


def system_of(spec: dict[str, Any]) -> str:
    """`spec["system"]`, ou o nome pelo modelo pedido (não pelo checkpoint reserva)."""
    if spec.get("system"):
        return str(spec["system"])
    return local.system_name(spec["model_id"], spec.get("adapter_id"))


COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def unpinned(specs: Iterable[dict[str, Any]]) -> list[str]:
    """Problemas de fixação: `revision` (e `adapter_revision`, com adapter) que não são commits.

    Regra 0.8: no teste, todo checkpoint e todo adapter vão fixados por commit
    (40 caracteres hexadecimais). Um placeholder (ex.: `"PREENCHER"`) ou `None`
    entra na lista.
    """
    problems = []
    for spec in specs:
        keys = ["revision"] + (["adapter_revision"] if spec.get("adapter_id") else [])
        for key in keys:
            value = spec.get(key)
            if not (isinstance(value, str) and COMMIT_RE.fullmatch(value)):
                repo = spec["adapter_id"] if key == "adapter_revision" else spec["model_id"]
                problems.append(f"{system_of(spec)}: {key} de {repo} = {value!r}")
    return problems


def require_pinned(specs: Iterable[dict[str, Any]]) -> None:
    """Levanta `ValueError` se algum modelo selecionado não tem commit fixado."""
    problems = unpinned(specs)
    if problems:
        raise ValueError(
            "Todo checkpoint e adapter do teste precisa de commit fixado (40 caracteres "
            "hexadecimais): " + "; ".join(problems)
        )


def checkpoint_bytes(
    model_id: str, revision: str | None, info_fn: Callable[..., Any] | None = None
) -> int | None:
    """Tamanho dos pesos (`*.safetensors`) do checkpoint no HF; None se não der para saber."""
    try:
        if info_fn is None:
            from huggingface_hub import model_info as info_fn
        info = info_fn(model_id, revision=revision, files_metadata=True)
        sizes = [
            s.size or 0 for s in (info.siblings or []) if str(s.rfilename).endswith(".safetensors")
        ]
        return sum(sizes) or None
    except Exception:  # sem rede, limite da API: tenta baixar mesmo assim
        return None


def choose_source(
    spec: dict[str, Any],
    free_bytes: int,
    size_of: Callable[[str, str | None], int | None],
    margin_bytes: int = DISK_MARGIN_BYTES,
) -> tuple[dict[str, Any] | None, str | None]:
    """(checkpoint a usar, motivo do pulo): o original se cabe no disco, senão o reserva.

    O checkpoint é `{"model_id", "revision", "source"}` com `source` =
    `"official"` ou `"fallback"`. Tamanho desconhecido conta como "cabe".
    """
    candidates = [("official", spec["model_id"], spec.get("revision"))]
    if spec.get("fallback"):
        fb = spec["fallback"]
        candidates.append(("fallback", fb["model_id"], fb.get("revision")))
    needs = []
    for source, model_id, revision in candidates:
        size = size_of(model_id, revision)
        if size is None or size + margin_bytes <= free_bytes:
            return {"model_id": model_id, "revision": revision, "source": source}, None
        needs.append(f"{model_id}: {size / GIB:.1f} GiB")
    return None, (
        f"disco insuficiente: livres {free_bytes / GIB:.1f} GiB; precisa de " + "; ".join(needs)
    )


def clear_hf_cache(repo_ids: Iterable[str], scan: Callable[[], Any] | None = None) -> int:
    """Apaga do cache do HF todas as revisões dos repos `repo_ids`; devolve os bytes liberados."""
    wanted = set(repo_ids)
    try:
        if scan is None:
            from huggingface_hub import scan_cache_dir as scan
        cache = scan()
        hashes = [
            rev.commit_hash
            for repo in cache.repos
            if repo.repo_id in wanted
            for rev in repo.revisions
        ]
        if not hashes:
            return 0
        strategy = cache.delete_revisions(*hashes)
        strategy.execute()
        return int(strategy.expected_freed_size)
    except Exception:  # cache vazio ou ilegível: nada a apagar
        return 0


def free_disk_bytes(path: str | Path | None = None) -> int:
    """Espaço livre no disco do cache do HF (ou de `path`)."""
    if path is None:
        from huggingface_hub import constants

        path = constants.HF_HUB_CACHE
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free


def _gpu_count() -> int:
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def skipped_info(
    spec: dict[str, Any],
    *,
    split: str,
    out_dir: str | Path,
    limit: int | None,
    reason: str,
    retry: bool = True,
) -> dict[str, Any]:
    """Run-info de um modelo que não rodou (D37)."""
    system = system_of(spec)
    return {
        "system": system,
        "split": split,
        "path": str(local.output_path(out_dir, system, split, limit)),
        "model_id": spec["model_id"],
        "model_revision": spec.get("revision"),
        "adapter_id": spec.get("adapter_id"),
        "quantization": spec.get("quantization"),
        "gpus_requested": int(spec.get("gpus", 1)),
        "limit": limit,
        "retry": retry,
        "skipped": True,
        "skip_reason": reason,
    }


def run_spec(
    spec: dict[str, Any],
    *,
    dataset_path: str | Path,
    split: str,
    out_dir: str | Path,
    limit: int | None = None,
    max_new_tokens: int = local.DEFAULT_MAX_NEW_TOKENS,
    retry: bool = True,
    dtype: str = "float16",
    db_path: str | Path | None = None,
    run: Callable[..., dict[str, Any]] | None = None,
    gpu_count: Callable[[], int] = _gpu_count,
    free_disk: Callable[[], int] = free_disk_bytes,
    size_of: Callable[[str, str | None], int | None] = checkpoint_bytes,
    clear_cache: Callable[[Iterable[str]], int] = clear_hf_cache,
    echo: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Roda um modelo da lista (ou registra por que não rodou) e devolve o run-info."""
    run = run or local.run_predictions
    system = system_of(spec)
    need_gpus = int(spec.get("gpus", 1))
    have = gpu_count()
    if have < need_gpus:
        reason = f"precisa de {need_gpus} GPU(s); a sessão tem {have}"
        echo(f"{system}: pulado ({reason})")
        return skipped_info(
            spec, split=split, out_dir=out_dir, limit=limit, reason=reason, retry=retry
        )

    source, reason = choose_source(spec, free_disk(), size_of)
    if source is None:
        echo(f"{system}: pulado ({reason})")
        return skipped_info(
            spec, split=split, out_dir=out_dir, limit=limit, reason=str(reason), retry=retry
        )

    budget_h = spec.get("time_budget_h")
    started = time.time()
    try:
        info = run(
            dataset_path,
            source["model_id"],
            out_dir=out_dir,
            revision=source["revision"],
            adapter_id=spec.get("adapter_id"),
            adapter_revision=spec.get("adapter_revision"),
            merge_adapter=True,
            system=system,
            limit=limit,
            dtype=spec.get("dtype", dtype),
            device=device_map_for(spec),
            max_new_tokens=max_new_tokens,
            retry=retry,
            db_path=db_path,
            quantization=spec.get("quantization"),
            max_memory=spec.get("max_memory"),
            resume=True,
            time_budget_s=float(budget_h) * 3600 if budget_h is not None else None,
            echo=echo,
        )
    except Exception as e:  # falta de memória ao carregar, download etc.: segue (D37)
        reason = f"{type(e).__name__}: {str(e)[:500]}"
        echo(f"{system}: pulado ({reason})")
        info = skipped_info(
            spec, split=split, out_dir=out_dir, limit=limit, reason=reason, retry=retry
        )
        info["elapsed_s"] = round(time.time() - started, 1)
    else:
        info["skipped"] = False
        info["gpus_requested"] = need_gpus
    finally:
        local._free_cuda()
        freed = clear_cache([source["model_id"]])
        if freed:
            echo(f"cache do HF: {freed / GIB:.1f} GiB liberados ({source['model_id']})")
    info["model_source"] = source["source"]
    info["requested_model_id"] = spec["model_id"]
    info["requested_revision"] = spec.get("revision")
    info["time_budget_h"] = budget_h
    return info


def run_info_paths(info: dict[str, Any], out_dir: str | Path) -> list[tuple[Path, dict]]:
    """[(caminho, conteúdo)] dos run-info: o single-shot e, com retry, o `+retry`."""
    out_dir = Path(out_dir)
    stem = Path(info["path"]).stem
    out = [(out_dir / f"run-info__{stem}.json", info)]
    if info.get("retry_system"):
        retry = local.retry_run_info(info)
        out.append((out_dir / f"run-info__{Path(retry['path']).stem}.json", retry))
    elif info.get("skipped") and info.get("retry"):
        retry_system = f"{info['system']}+retry"
        skipped = {**info, "system": retry_system, "single_shot_system": info["system"]}
        name = Path(info["path"]).name.replace(info["system"], retry_system, 1)
        skipped["path"] = str(Path(info["path"]).with_name(name))
        out.append((out_dir / f"run-info__{Path(name).stem}.json", skipped))
    return out


def write_run_infos(info: dict[str, Any], out_dir: str | Path, **extra: Any) -> list[Path]:
    """Grava os run-info (com os campos `extra`, ex.: commit do repositório)."""
    written = []
    for path, content in run_info_paths(info, out_dir):
        content = {**content, **extra}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return written
