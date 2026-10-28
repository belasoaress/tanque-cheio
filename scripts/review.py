"""Revisão item a item dos conjuntos de dev/teste no terminal (M3) e auditoria por amostra (M4).

Uso (Windows/PowerShell, Linux ou macOS):

    uv run python scripts/review.py datasets/dev.jsonl
    uv run python scripts/review.py datasets/train.jsonl --audit 50 --seed 7

Modo revisão: mostra cada item ainda não revisado (pergunta, nível, order_matters,
uses_window, tags, SQL e as 5 primeiras linhas do resultado) e lê uma tecla:

    a  aprovar   -> grava `reviewed: true`
    e  editar    -> abre o item como JSON no editor ($VISUAL/$EDITOR; no Windows,
                    notepad), revalida (campos + execução da SQL) e mostra de novo
    d  descartar -> remove o item do arquivo e o registra, com o motivo, em
                    datasets/discarded.jsonl (na mesma pasta do dataset)
    s  pular     -> deixa para depois (continua pendente)
    q  sair

O arquivo é regravado de forma atômica (arquivo temporário + `os.replace`) após
cada decisão; ao reabrir, os itens já revisados são pulados.

Modo `--audit N`: sorteia N itens com `--seed` (padrão 0), pergunta só
aprovado (a) / reprovado (r) e acrescenta cada veredito em
results/audit-<nome do arquivo>.jsonl; retoma pulando os ids já auditados e,
no fim, imprime o % de reprovados. O dataset não é alterado.

Sem curses: só `input()`, para funcionar no PowerShell.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from tanque import config
from tanque.sql import execute

REQUIRED_FIELDS = (
    "id",
    "question",
    "sql",
    "difficulty",
    "uses_window",
    "order_matters",
    "split",
    "source",
    "reviewed",
    "tags",
)
DIFFICULTIES = ("facil", "medio", "dificil")
SOURCES = ("human_reviewed", "synthetic")
PREVIEW_ROWS = 5
DISCARDED_FILENAME = "discarded.jsonl"

Item = dict[str, Any]


# ---------------------------------------------------------------- arquivos


def load_jsonl(path: str | Path) -> list[Item]:
    items = []
    with Path(path).open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: JSON inválido: {e}") from e
    return items


def _dumps(item: Item) -> str:
    return json.dumps(item, ensure_ascii=False)


def write_jsonl_atomic(path: str | Path, items: Iterable[Item]) -> None:
    """Grava `items` em `path` sem nunca deixar um arquivo pela metade.

    Escreve num temporário na mesma pasta e troca com `os.replace` (atômico no
    mesmo sistema de arquivos, inclusive no Windows).
    """
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            for item in items:
                f.write(_dumps(item) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def append_jsonl(path: str | Path, record: Item) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(_dumps(record) + "\n")


# ---------------------------------------------------------------- lógica pura


def pending_indices(items: list[Item], skipped: Iterable[str] = ()) -> list[int]:
    """Índices dos itens ainda não revisados, na ordem do arquivo, sem os pulados."""
    skipped = set(skipped)
    return [
        i for i, it in enumerate(items) if not it.get("reviewed") and it.get("id") not in skipped
    ]


def approve(items: list[Item], item_id: str) -> list[Item]:
    """Cópia da lista com o item `item_id` marcado como revisado."""
    return [{**it, "reviewed": True} if it.get("id") == item_id else it for it in items]


def discard(items: list[Item], item_id: str) -> tuple[list[Item], Item]:
    """Remove `item_id`; devolve a lista nova e o item removido."""
    removed = [it for it in items if it.get("id") == item_id]
    if not removed:
        raise KeyError(item_id)
    return [it for it in items if it.get("id") != item_id], removed[0]


def replace_item(items: list[Item], item_id: str, new: Item) -> list[Item]:
    return [new if it.get("id") == item_id else it for it in items]


def schema_problems(item: Item, expected_split: str | None = None) -> list[str]:
    """Problemas de formato (seção 5.2), sem executar a SQL."""
    problems = [f"campo ausente: {k}" for k in REQUIRED_FIELDS if k not in item]
    if problems:
        return problems
    if not isinstance(item["question"], str) or not item["question"].strip():
        problems.append("question vazia")
    if not isinstance(item["sql"], str) or not item["sql"].strip():
        problems.append("sql vazia")
    if item["difficulty"] not in DIFFICULTIES:
        problems.append(f"difficulty inválido: {item['difficulty']!r}")
    if item["source"] not in SOURCES:
        problems.append(f"source inválido: {item['source']!r}")
    for k in ("uses_window", "order_matters", "reviewed"):
        if not isinstance(item[k], bool):
            problems.append(f"{k} precisa ser true/false")
    if not isinstance(item["tags"], list) or not all(isinstance(t, str) for t in item["tags"]):
        problems.append("tags precisa ser uma lista de strings")
    if expected_split is not None and item["split"] != expected_split:
        problems.append(f"split {item['split']!r} diferente de {expected_split!r}")
    return problems


def exec_problems(result: execute.ExecResult) -> list[str]:
    if result.error:
        return [result.error]
    if not result.rows:
        return ["a SQL não retornou linhas"]
    return []


def audit_sample(items: list[Item], n: int, seed: int) -> list[Item]:
    """N itens sorteados com `seed` (reprodutível); todos, se houver menos de N."""
    if n <= 0:
        raise ValueError("N precisa ser positivo")
    rng = random.Random(seed)
    if n >= len(items):
        return list(items)
    return rng.sample(items, n)


def audit_summary(records: Iterable[Item]) -> tuple[int, int, float]:
    """(total, reprovados, % reprovados) dos registros de auditoria."""
    records = list(records)
    total = len(records)
    rejected = sum(1 for r in records if r.get("verdict") == "reprovado")
    pct = 100.0 * rejected / total if total else 0.0
    return total, rejected, pct


def audit_path(dataset: Path, results_dir: Path) -> Path:
    return results_dir / f"audit-{dataset.stem}.jsonl"


def editor_command(env: dict[str, str] | None = None, platform: str | None = None) -> list[str]:
    """Comando do editor: $VISUAL, $EDITOR; senão notepad (Windows) ou nano/vi."""
    env = os.environ if env is None else env
    platform = sys.platform if platform is None else platform
    windows = platform.startswith("win")
    for var in ("VISUAL", "EDITOR"):
        value = env.get(var, "").strip()
        if value:
            return shlex.split(value, posix=not windows)
    if windows:
        return ["notepad"]
    return ["nano"] if shutil.which("nano") else ["vi"]


# ---------------------------------------------------------------- terminal


def _cell(value: Any) -> str:
    text = "NULL" if value is None else str(value)
    return text if len(text) <= 40 else text[:37] + "..."


def format_result(result: execute.ExecResult, limit: int = PREVIEW_ROWS) -> str:
    if result.error:
        return f"  ERRO: {result.error}"
    lines = ["  " + " | ".join(result.columns)]
    for row in result.rows[:limit]:
        lines.append("  " + " | ".join(_cell(v) for v in row))
    more = len(result.rows) - limit
    tail = f" (+{more} linhas)" if more > 0 else ""
    lines.append(f"  [{len(result.rows)} linha(s), {result.elapsed_ms:.0f} ms]{tail}")
    return "\n".join(lines)


def show_item(item: Item, result: execute.ExecResult, position: str) -> None:
    print("\n" + "=" * 78)
    print(f"{position}  {item.get('id')}  [{item.get('difficulty')}]  split={item.get('split')}")
    print(
        f"order_matters={item.get('order_matters')}  uses_window={item.get('uses_window')}  "
        f"tags={', '.join(item.get('tags') or [])}"
    )
    print(f"\nPergunta: {item.get('question')}\n")
    print("SQL:")
    print("  " + str(item.get("sql")).replace("\n", "\n  "))
    print("\nResultado:")
    print(format_result(result))


def ask(prompt: str, options: str, read: Callable[[str], str] = input) -> str:
    while True:
        answer = read(prompt).strip().lower()[:1]
        if answer and answer in options:
            return answer
        print(f"Opção inválida. Use uma de: {', '.join(options)}")


def edit_item(item: Item) -> Item | None:
    """Abre o item no editor; devolve o JSON editado ou None se cancelado/inválido."""
    fd, tmp = tempfile.mkstemp(prefix="tanque-review-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(item, f, ensure_ascii=False, indent=2)
            f.write("\n")
        subprocess.run([*editor_command(), tmp], check=False)
        text = Path(tmp).read_text(encoding="utf-8-sig")
    finally:
        Path(tmp).unlink(missing_ok=True)
    try:
        edited = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"JSON inválido, edição ignorada: {e}")
        return None
    if not isinstance(edited, dict):
        print("O conteúdo editado precisa ser um objeto JSON; edição ignorada.")
        return None
    return edited


def _run(sql: str, db: Path | None) -> execute.ExecResult:
    return execute.run(sql, db_path=db)


def review(dataset: Path, db: Path | None) -> int:
    items = load_jsonl(dataset)
    expected_split = items[0].get("split") if items else None
    discarded_log = dataset.parent / DISCARDED_FILENAME
    skipped: list[str] = []
    total = len(items)
    while True:
        pending = pending_indices(items, skipped)
        if not pending:
            break
        item = items[pending[0]]
        done = sum(1 for it in items if it.get("reviewed"))
        result = _run(item["sql"], db)
        show_item(item, result, f"[{done + 1}/{len(items)}]")
        problems = schema_problems(item, expected_split) + exec_problems(result)
        if problems:
            print("\nATENÇÃO: " + "; ".join(problems))
        choice = ask("\n[a]provar  [e]ditar  [d]escartar  [s]pular  [q]sair > ", "aedsq")
        if choice == "q":
            break
        if choice == "s":
            skipped.append(item["id"])
        elif choice == "a":
            if problems:
                print("Corrija os problemas (e) ou descarte (d) antes de aprovar.")
                continue
            items = approve(items, item["id"])
            write_jsonl_atomic(dataset, items)
        elif choice == "d":
            reason = input("Motivo do descarte (opcional): ").strip()
            items, removed = discard(items, item["id"])
            append_jsonl(
                discarded_log,
                {"from": dataset.name, "reason": reason or None, "item": removed},
            )
            write_jsonl_atomic(dataset, items)
        elif choice == "e":
            edited = edit_item(item)
            if edited is None:
                continue
            new_id = edited.get("id")
            if new_id != item["id"] and any(it.get("id") == new_id for it in items):
                print(f"id {new_id!r} já existe; edição ignorada.")
                continue
            edited["reviewed"] = False  # revalida e volta a pedir a decisão
            items = replace_item(items, item["id"], edited)
            write_jsonl_atomic(dataset, items)
    reviewed = sum(1 for it in items if it.get("reviewed"))
    print(
        f"\n{reviewed} de {len(items)} itens revisados "
        f"({total - len(items)} descartados nesta sessão; {len(skipped)} pulados)."
    )
    return 0


def audit(dataset: Path, n: int, seed: int, db: Path | None, results_dir: Path) -> int:
    items = load_jsonl(dataset)
    sample = audit_sample(items, n, seed)
    out = audit_path(dataset, results_dir)
    done = {r.get("id") for r in load_jsonl(out)} if out.is_file() else set()
    todo = [it for it in sample if it.get("id") not in done]
    print(f"Auditoria de {len(sample)} itens (seed {seed}); {len(todo)} pendentes -> {out}")
    for k, item in enumerate(todo, start=1):
        show_item(item, _run(item["sql"], db), f"[auditoria {k}/{len(todo)}]")
        choice = ask("\n[a]provado  [r]eprovado  [q]sair > ", "arq")
        if choice == "q":
            break
        verdict = "aprovado" if choice == "a" else "reprovado"
        append_jsonl(out, {"id": item.get("id"), "verdict": verdict, "seed": seed, "n": n})
    total, rejected, pct = audit_summary(load_jsonl(out) if out.is_file() else [])
    print(f"\nAuditados: {total}; reprovados: {rejected} ({pct:.1f}%).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Revisão dos datasets do tanque-cheio.")
    parser.add_argument("dataset", type=Path, help="Arquivo JSONL (seção 5.2).")
    parser.add_argument("--audit", type=int, metavar="N", help="Sorteia N itens para auditar.")
    parser.add_argument("--seed", type=int, default=0, help="Seed do sorteio (padrão 0).")
    parser.add_argument("--db", type=Path, help="Banco DuckDB (padrão: [paths].db_path).")
    args = parser.parse_args(argv)
    if not args.dataset.is_file():
        parser.error(f"arquivo não encontrado: {args.dataset}")
    db = args.db if args.db is not None else execute.default_db_path()
    if not Path(db).is_file():
        parser.error(f"banco não encontrado: {db} (rode `tanque data build` ou passe --db)")
    if args.audit is not None:
        if args.audit <= 0:
            parser.error("--audit precisa de N > 0")
        return audit(args.dataset, args.audit, args.seed, db, config.setting_path("results_dir"))
    return review(args.dataset, db)


if __name__ == "__main__":
    sys.exit(main())
