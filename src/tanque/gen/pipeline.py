"""Do bruto ao `datasets/train.jsonl`, e o relatório `results/gen-stats.md` (M4).

`build_train`:

1. lê os pares crus de `results/gen/raw/*.jsonl` (caminho Batch; nenhuma chamada
   à API aqui) ou recebe os pares prontos (`pairs=`, caminho por templates, D23);
2. valida por execução (`tanque.gen.validate`);
3. remove pares que colidem com dev/teste (`tanque.gen.leakage`), motivo `vazamento`;
4. deduplica internamente, motivo `duplicado`: no caminho Batch com o mesmo
   critério do vazamento (`dedupe_mode="fuzzy"`); no de templates só por pergunta
   ou SQL normalizada idêntica (`dedupe_mode="exact"`, D23);
5. grava `datasets/train.jsonl` (ids `train-00001`...), os descartes em
   `data/gen/discarded.jsonl` (ignorado pelo git) e o resumo em
   `results/gen/validate-summary.json` (sem horário, para ser reproduzível).

Nos descartes por vazamento só o id do item de avaliação é gravado, nunca o
texto dele.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from tanque import config
from tanque.budget import Ledger
from tanque.gen import batch as batch_mod
from tanque.gen import intents as intents_mod
from tanque.gen import leakage
from tanque.gen import validate as validate_mod
from tanque.infer import api

WORK_RELDIR = Path("data") / "gen"
SUMMARY_RELPATH = Path("results") / "gen" / "validate-summary.json"
STATS_RELPATH = Path("results") / "gen-stats.md"
TRAIN_FILENAME = "train.jsonl"
TOLERANCE_PP = 5


def train_path() -> Path:
    return config.setting_path("datasets_dir") / TRAIN_FILENAME


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class BuildResult:
    raw_pairs: int
    valid: int
    final: int
    reasons: dict[str, int]
    train_path: Path
    summary_path: Path


def build_train(
    *,
    db_path: str | Path | None = None,
    raw_dir: str | Path | None = None,
    reference_dir: str | Path | None = None,
    out_path: str | Path | None = None,
    work_dir: str | Path | None = None,
    summary_path: str | Path | None = None,
    timeout_s: float | None = None,
    threshold: float = leakage.THRESHOLD,
    pairs: list[dict[str, Any]] | None = None,
    dedupe_mode: str = "fuzzy",
    check_repeated: bool = True,
    extra_summary: dict[str, Any] | None = None,
) -> BuildResult:
    if dedupe_mode not in ("fuzzy", "exact"):
        raise ValueError(f"dedupe_mode inválido: {dedupe_mode}")
    reference, _ = leakage.load_reference(reference_dir)  # falha cedo se faltar dev/teste
    if pairs is None:
        pairs = batch_mod.raw_candidates(raw_dir)
    requests = len({p["custom_id"] for p in pairs if "custom_id" in p})
    vr = validate_mod.validate_pairs(
        pairs, db_path=db_path, timeout_s=timeout_s, check_repeated=check_repeated
    )

    no_leak, leaks = leakage.remove_leaks(vr.valid, reference, threshold)
    leaked = {c.index: c for c in leaks}
    discards = [d.to_dict() for d in vr.discarded]
    for i, item in enumerate(vr.valid):
        if i in leaked:
            c = leaked[i]
            discards.append(
                {
                    **item,
                    "reason": "vazamento",
                    "detail": f"{c.kind} ~ {c.other_id} ({c.score:.0f})",
                }
            )

    if dedupe_mode == "exact":
        kept, dups = leakage.dedupe_exact(no_leak)
    else:
        kept, dups = leakage.dedupe(no_leak, threshold)
    for c in dups:
        discards.append(
            {
                **no_leak[c.index],
                "reason": "duplicado",
                "detail": f"{c.kind} ~ posição {c.other_index} ({c.score:.0f})",
            }
        )

    items = validate_mod.assign_ids(kept)
    out = Path(out_path) if out_path is not None else train_path()
    write_jsonl(out, items)
    work = config.resolve_path(work_dir or WORK_RELDIR)
    write_jsonl(work / "discarded.jsonl", discards)

    reasons = dict(Counter(d["reason"] for d in discards).most_common())
    summary = {
        "requests_with_pairs": requests,
        "raw_pairs": len(pairs),
        "valid_after_execution": len(vr.valid),
        "final": len(items),
        "discarded_by_reason": reasons,
        "leakage_threshold": threshold,
        "dedupe_mode": dedupe_mode,
        "check_repeated_results": check_repeated,
        "by_intent": _by_intent(pairs, discards, items),
        **(extra_summary or {}),
    }
    spath = config.resolve_path(summary_path or SUMMARY_RELPATH)
    spath.parent.mkdir(parents=True, exist_ok=True)
    spath.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return BuildResult(len(pairs), len(vr.valid), len(items), reasons, out, spath)


def _intent_key(row: dict[str, Any]) -> str:
    return row.get("intent") or validate_mod.intent_of(row) or "(sem intenção)"


def _by_intent(
    pairs: list[dict[str, Any]], discards: list[dict[str, Any]], items: list[dict[str, Any]]
) -> dict[str, dict[str, int]]:
    """{intenção: {raw, final, <motivo>: n}} (só contagens, nada de texto)."""
    out: dict[str, Counter] = {}
    for p in pairs:
        out.setdefault(_intent_key(p), Counter())["raw"] += 1
    for d in discards:
        out.setdefault(_intent_key(d), Counter())[d["reason"]] += 1
    for t in items:
        out.setdefault(_intent_key(t), Counter())["final"] += 1
    return {
        k: {
            "raw": v["raw"],
            "final": v["final"],
            **{r: n for r, n in sorted(v.items()) if r not in ("raw", "final")},
        }
        for k, v in sorted(out.items())
    }


# --- relatório --------------------------------------------------------------


def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "—"


def render_stats(
    train: list[dict[str, Any]],
    summary: dict[str, Any] | None,
    ledger: Ledger,
    registry: api.BatchRegistry,
) -> str:
    total = len(train)
    by_level = Counter(t.get("difficulty") for t in train)
    by_intent = Counter(validate_mod.intent_of(t) for t in train)
    windows = sum(1 for t in train if t.get("uses_window"))
    hard = by_level.get("dificil", 0)

    lines = [
        "# Estatísticas do dataset de treino (M4; v2 no M15)",
        "",
        "Gerado por `tanque gen stats`. Não edite à mão.",
        "",
        "Treino v2 (M15, D34): as 40 intenções da v1 mais 17 novas, ligadas aos erros do "
        "ft-v1 no dev2 (`results/training-v2.md`); distribuição alvo 28/36/36 "
        "(a v1 tinha 35/40/25 e 8.217 pares).",
        "",
    ]
    if summary and summary.get("generator") == "templates":
        lines += [
            f"- **Comando:** `uv run tanque gen templates --n {summary.get('requested')} "
            f"--seed {summary.get('seed')}`",
        ]
    lines += [
        f"- **Pares no treino:** {total} (`datasets/train.jsonl`; meta do plano: ≥ 3.000)",
        f"- **Com função de janela:** {windows} ({_pct(windows, hard)} do nível difícil; "
        "teto de 20%)",
        "",
        "## Por nível",
        "",
        "| nível | pares | % | alvo | dentro de ±5 p.p. |",
        "|---|---:|---:|---:|---|",
    ]
    for level in intents_mod.LEVELS:
        n = by_level.get(level, 0)
        target = float(intents_mod.TARGET_DISTRIBUTION[level]) * 100
        share = 100 * n / total if total else 0.0
        ok = "sim" if total and abs(share - target) <= TOLERANCE_PP else "não"
        lines.append(f"| {level} | {n} | {_pct(n, total)} | {target:.0f}% | {ok} |")

    per_intent = (summary or {}).get("by_intent") or {}
    lines += [
        "",
        "## Por intenção",
        "",
        "Contagens por intenção: crus = pares que entraram no pipeline; os motivos são os do",
        "pipeline (validação por execução, vazamento contra dev/teste, duplicado interno).",
        "",
        "| intenção | nível | pares | crus | mantidos | vazamento | duplicado | outros |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for intent in intents_mod.INTENTS:
        st = per_intent.get(intent.id, {})
        raw = int(st.get("raw", 0))
        leak, dup = int(st.get("vazamento", 0)), int(st.get("duplicado", 0))
        other = sum(
            int(v) for k, v in st.items() if k not in ("raw", "final", "vazamento", "duplicado")
        )
        n = by_intent.get(intent.id, 0)
        lines.append(
            f"| `{intent.id}` | {intent.level} | {n} | {raw} | {_pct(n, raw)} | {leak} | {dup} "
            f"| {other} |"
        )
    unknown = sum(c for k, c in by_intent.items() if k not in intents_mod.INTENTS_BY_ID)
    if unknown:
        lines.append(f"| (sem intenção) | — | {unknown} |")

    lines += ["", "## Descartes", ""]
    if summary:
        raw = int(summary.get("raw_pairs", 0))
        lines += [
            f"- Requisições com pares: {summary.get('requests_with_pairs', 0)}",
            f"- Pares crus: {raw}",
            f"- Válidos após execução: {summary.get('valid_after_execution', 0)}",
            f"- No treino final: {summary.get('final', 0)} ({_pct(summary.get('final', 0), raw)} "
            "dos crus)",
            "",
            "| motivo | pares | % dos crus |",
            "|---|---:|---:|",
        ]
        for reason, n in summary.get("discarded_by_reason", {}).items():
            lines.append(f"| `{reason}` | {n} | {_pct(n, raw)} |")
        if summary.get("dedupe_mode"):
            crit = (
                "pergunta ou SQL normalizada idêntica (D23)"
                if summary["dedupe_mode"] == "exact"
                else f"token_set_ratio >= {summary.get('leakage_threshold')} ou SQL idêntica"
            )
            lines += ["", f"- Deduplicação interna: {crit}."]
        if summary.get("check_repeated_results") is False:
            lines.append(
                "- Regra `resultado_repetido` desligada: a SQL é montada pelo template a partir "
                "dos parâmetros sorteados (D23), então não pode ser genérica."
            )
        lines.append(
            f"- Vazamento contra dev/teste: token_set_ratio >= "
            f"{summary.get('leakage_threshold')} ou SQL normalizada idêntica."
        )
        gen = summary.get("generation_rejected_by_reason")
        if summary.get("generator") == "templates":
            lines += [
                "",
                "### Geração por templates (antes do pipeline)",
                "",
                f"- Seed: {summary.get('seed')}; pedidos: {summary.get('requested')}; "
                f"gerados: {summary.get('generated')}",
                "- Tentativas recusadas na geração (o candidato nem chega ao pipeline):",
                "",
                "| motivo | tentativas |",
                "|---|---:|",
            ]
            for reason, n in (gen or {}).items():
                lines.append(f"| `{reason}` | {n} |")
            if summary.get("exhausted_intents"):
                lines += [
                    "",
                    "- Intenções que esgotaram o espaço de parâmetros (a cota foi repartida "
                    "no mesmo nível): " + ", ".join(f"`{i}`" for i in summary["exhausted_intents"]),
                ]
    else:
        lines.append("(sem resumo: rode `tanque gen validate`)")

    m4 = [e for e in ledger.entries() if e.get("milestone") == batch_mod.MILESTONE]
    cost = sum((Decimal(str(e["cost_usd"])) for e in m4), Decimal(0))
    tokens = Counter()
    for e in m4:
        for k in ("input_tokens", "cached_tokens", "cache_write_tokens", "output_tokens"):
            tokens[k] += int(e.get(k, 0))
    batches = registry.submitted_of_kind(batch_mod.KIND)
    estimate = sum(Decimal(str(b.get("estimate_usd", 0))) for b in batches)
    lines += ["", "## Custo (ledger, marco M4)", ""]
    if summary and summary.get("generator") == "templates":
        lines += [
            "- Origem do treino: templates parametrizados com valores reais do banco (D23), "
            "sem chamadas à API.",
        ]
    lines += [
        f"- Lotes enviados: {len(batches)}; requisições: "
        f"{sum(int(b.get('n_requests', 0)) for b in batches)}",
        f"- Chamadas cobradas: {len(m4)}",
        f"- Tokens: entrada {tokens['input_tokens']}, leitura de cache "
        f"{tokens['cached_tokens']}, escrita de cache {tokens['cache_write_tokens']}, "
        f"saída {tokens['output_tokens']}",
        f"- **Gasto real: US$ {cost:.4f}** (estimado no envio: US$ {estimate:.4f})",
        f"- Por par no treino: US$ {cost / total:.6f}" if total else "- Por par no treino: —",
        "",
    ]
    return "\n".join(lines)


def write_stats(
    *,
    train_file: str | Path | None = None,
    summary_path: str | Path | None = None,
    out_path: str | Path | None = None,
    ledger: Ledger | None = None,
    registry: api.BatchRegistry | None = None,
) -> tuple[Path, str]:
    train = read_jsonl(Path(train_file) if train_file else train_path())
    spath = config.resolve_path(summary_path or SUMMARY_RELPATH)
    summary = json.loads(spath.read_text(encoding="utf-8")) if spath.is_file() else None
    text = render_stats(train, summary, ledger or Ledger(), registry or api.BatchRegistry())
    out = config.resolve_path(out_path or STATS_RELPATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    return out, text
