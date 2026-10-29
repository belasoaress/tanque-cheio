"""Monta a pasta de upload do dataset no Hugging Face (M8).

A pasta sai com o layout que o card (`hf/dataset/README.md`) declara nos `configs`:

    README.md                         <- hf/dataset/README.md (dataset card)
    data/{train,dev,test}.jsonl       <- datasets/{train,dev,test}.jsonl
    snapshot/<snapshot>.parquet       <- [paths].snapshot (D25)
    snapshot/<snapshot>.json          <- registro do snapshot (SHA-256, tabelas)
    docs/schema.md                    <- docs/schema.md
    manifest.json                     <- data/manifest.json (CSVs da ANP e seus hashes)

Só copia: não publica nada. O upload é um passo da Isabela (`hf upload`, regra 5).
`datasets/candidates_reserve.jsonl` (rascunhos descartados do M3) fica de fora.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from tanque.data.build import sha256_file, snapshot_meta_path

CARD = Path("hf") / "dataset" / "README.md"
SPLITS = ("train", "dev", "test")


class BundleError(RuntimeError):
    """Arquivo faltando, snapshot divergente ou pasta de saída inválida."""


def bundle_plan(root: Path, snapshot: str | Path) -> dict[str, Path]:
    """Destino na pasta de upload (posix) -> arquivo de origem no repositório."""
    parquet = root / snapshot
    plan = {"README.md": root / CARD}
    for split in SPLITS:
        plan[f"data/{split}.jsonl"] = root / "datasets" / f"{split}.jsonl"
    plan[f"snapshot/{parquet.name}"] = parquet
    meta = snapshot_meta_path(parquet)
    plan[f"snapshot/{meta.name}"] = meta
    plan["docs/schema.md"] = root / "docs" / "schema.md"
    plan["manifest.json"] = root / "data" / "manifest.json"
    return plan


def _check_sources(plan: dict[str, Path], root: Path, parquet: Path) -> None:
    missing = [src for src in plan.values() if not src.is_file()]
    if missing:
        names = ", ".join(_rel(src, root) for src in missing)
        raise BundleError(f"Arquivo(s) não encontrado(s): {names}")
    meta = json.loads(snapshot_meta_path(parquet).read_text(encoding="utf-8"))
    actual = sha256_file(parquet)
    if actual != meta.get("sha256"):
        raise BundleError(
            f"SHA-256 de {parquet.name} diverge do registro: {actual} "
            f"(esperado {meta.get('sha256')})"
        )


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _prepare_out(out: Path, root: Path, force: bool) -> None:
    out_r, root_r = out.resolve(), root.resolve()
    # Com --force a pasta é apagada: nunca pode ser a raiz, um ancestral dela ou uma
    # pasta de origem (ou algo dentro de uma).
    if out_r == root_r or out_r in root_r.parents:
        raise BundleError(f"A pasta de saída {out} contém o repositório (dentro dela).")
    for protected in ("datasets", "snapshot", "data", "hf", "docs"):
        p = (root / protected).resolve()
        if out_r == p or p in out_r.parents:
            raise BundleError(f"A pasta de saída {out} fica dentro de {protected}/.")
    if out.exists() and any(out.iterdir()):
        if not force:
            raise BundleError(f"{out} não está vazia. Use --force para apagar e montar de novo.")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)


def build_bundle(out: Path, *, root: Path, snapshot: str | Path, force: bool = False) -> list[str]:
    """Copia os arquivos do dataset para `out`. Devolve os destinos (posix), em ordem."""
    plan = bundle_plan(root, snapshot)
    _check_sources(plan, root, root / snapshot)
    _prepare_out(out, root, force)
    for dest, src in plan.items():
        target = out / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
    return list(plan)
