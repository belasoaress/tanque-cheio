"""Relatório final (M7) em Markdown: `results/report.md`, gerado só por `tanque eval report`.

Fontes, todas versionadas:

- `results/metrics.json` (gravado por `tanque eval score`): tabelas principais e
  comparação base → ajustado;
- `results/predictions/ft-*__test.jsonl` + `datasets/test.jsonl` + o banco: os
  exemplos de erro do modelo ajustado (re-executa as SQLs; ver `errors.py`);
- `results/error-annotations.json`: classificação manual opcional dos erros;
- `configs/settings.toml` `[gpu_cost]`: preço de referência de 1 h de GPU T4, para o
  custo estimado por 1.000 perguntas dos modelos locais;
- `snapshot/coletas-2026-09.json` e `results/spend.jsonl`: nota de método.

Sistemas: todos os de `metrics.json` com o split pedido, na ordem `base-*`, `ft-*`,
`claude-*`, outros (alfabética dentro de cada grupo). Um terceiro sistema
(`claude-*__test.jsonl`, D24) entra sozinho quando for pontuado.

**Custo por 1.000 perguntas.** Para sistemas de API, `cost_per_1k_usd` das
previsões (custo real, pelo `pricing.toml`). Para modelos locais medidos numa T4
(`hardware` em `[gpu_cost].hardware`), estimativa: latência média × 1.000 /
3,6 milhões de ms por hora × `gpu_hour_usd`, marcada com `*`. As previsões
continuam com `cost_usd = 0` (formato 5.3).

A saída é determinística (sem data/hora), para o CI conferir com `--check`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tanque import config
from tanque.eval import errors as errors_mod
from tanque.eval.score import load_metrics, read_jsonl
from tanque.sql import execute

REPORT_FILENAME = "report.md"
MISSING = "—"
ESTIMATE_MARK = "*"
UNMERGED_MARK = "†"
N_EXAMPLES = 10
MS_PER_HOUR = 3_600_000
SYSTEM_ORDER = ("base-", "ft-", "claude-")
LEVELS = (("facil", "fácil"), ("medio", "médio"), ("dificil", "difícil"))

COLUMNS = (
    "Sistema",
    "Split",
    "N",
    "EX",
    "EX fácil",
    "EX médio",
    "EX difícil",
    "SQL válida",
    "Bloqueada",
    "p50 (ms)",
    "p95 (ms)",
    "US$ / 1.000",
)


def default_report_path() -> Path:
    return config.setting_path("results_dir") / REPORT_FILENAME


def default_annotations_path() -> Path:
    return config.setting_path("results_dir") / errors_mod.ANNOTATIONS_FILENAME


# --- formatação -----------------------------------------------------------------


def _pct(value: Any) -> str:
    if value is None:
        return MISSING
    return f"{float(value) * 100:.1f}%".replace(".", ",")


def _num(value: Any, places: int) -> str:
    if value is None:
        return MISSING
    text = f"{float(value):,.{places}f}"
    # Formato brasileiro: ponto para milhar, vírgula para decimal.
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def _signed(value: float | None, places: int, unit: str) -> str:
    if value is None:
        return MISSING
    sign = "+" if value > 0 else ("−" if value < 0 else "±")
    return f"{sign}{_num(abs(value), places)}{unit}"


def _table(header: list[str] | tuple[str, ...], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *("| " + " | ".join(r) + " |" for r in rows),
    ]


# --- custo de GPU ---------------------------------------------------------------


@dataclass(frozen=True)
class GpuPrice:
    hour_usd: float
    gpu: str
    includes_vm: bool
    source_url: str
    checked_on: str
    hardware: tuple[str, ...]


def load_gpu_price(settings: dict[str, Any] | None = None) -> GpuPrice | None:
    """`[gpu_cost]` de settings.toml (None se a seção não existir)."""
    settings = settings if settings is not None else config.load_settings()
    raw = settings.get("gpu_cost")
    if not raw:
        return None
    return GpuPrice(
        hour_usd=float(raw["gpu_hour_usd"]),
        gpu=str(raw["gpu"]),
        includes_vm=bool(raw.get("includes_vm", False)),
        source_url=str(raw["source_url"]),
        checked_on=str(raw["checked_on"]),
        hardware=tuple(raw.get("hardware", ())),
    )


def gpu_cost_per_1k(mean_latency_ms: float, hour_usd: float) -> float:
    """US$ por 1.000 perguntas com uma GPU dedicada a `hour_usd` por hora."""
    return mean_latency_ms * 1000 / MS_PER_HOUR * hour_usd


def cost_per_1k(m: dict[str, Any], gpu: GpuPrice | None) -> tuple[float | None, bool]:
    """(US$ por 1.000, é estimativa de GPU?) de uma entrada de metrics.json."""
    api = m.get("cost_per_1k_usd")
    if api:
        return float(api), False
    hardware = m.get("hardware") or []
    mean = (m.get("latency_ms") or {}).get("mean")
    if gpu and hardware and mean is not None and set(hardware) <= set(gpu.hardware):
        return gpu_cost_per_1k(float(mean), gpu.hour_usd), True
    return None, False


def _cost_cell(m: dict[str, Any], gpu: GpuPrice | None) -> str:
    value, estimated = cost_per_1k(m, gpu)
    if value is None:
        return MISSING
    return _num(value, 4) + (ESTIMATE_MARK if estimated else "")


# --- tabelas --------------------------------------------------------------------


def _system_key(system: str) -> tuple[int, str]:
    for rank, prefix in enumerate(SYSTEM_ORDER):
        if system.startswith(prefix):
            return rank, system
    return len(SYSTEM_ORDER), system


def systems_with(metrics: dict[str, Any], split: str) -> list[str]:
    return sorted((s for s in metrics if split in metrics[s]), key=_system_key)


def _row(
    system: str,
    split: str,
    m: dict[str, Any],
    gpu: GpuPrice | None = None,
    unmerged: bool = False,
) -> list[str]:
    levels = m.get("ex_by_difficulty") or {}
    latency = m.get("latency_ms") or {}
    mark = UNMERGED_MARK if unmerged else ""
    return [
        f"`{system}`",
        split,
        str(m["n"]) if m.get("n") is not None else MISSING,
        _pct(m.get("ex")),
        *(_pct((levels.get(key) or {}).get("ex")) for key, _ in LEVELS),
        _pct(m.get("valid_sql_rate")),
        _pct(m.get("blocked_rate")),
        _num(latency.get("p50"), 0) + mark,
        _num(latency.get("p95"), 0) + mark,
        _cost_cell(m, gpu),
    ]


def main_table(
    metrics: dict[str, Any],
    split: str,
    gpu: GpuPrice | None = None,
    unmerged: frozenset[tuple[str, str]] = frozenset(),
) -> list[str]:
    rows = [
        _row(s, split, metrics[s][split], gpu, (s, split) in unmerged)
        for s in systems_with(metrics, split)
    ]
    return _table(COLUMNS, rows) if rows else []


def comparison_pairs(metrics: dict[str, Any], split: str) -> list[tuple[str, str]]:
    """(base-<m>, ft-<m>) com os dois sistemas pontuados no split."""
    pairs = []
    for system in systems_with(metrics, split):
        if system.startswith("ft-"):
            base = "base-" + system[len("ft-") :]
            if split in metrics.get(base, {}):
                pairs.append((base, system))
    return pairs


def comparison_table(
    base: dict[str, Any], ft: dict[str, Any], gpu: GpuPrice | None = None
) -> list[str]:
    def pp(key_path: tuple[str, ...]) -> tuple[Any, Any]:
        def get(m: dict[str, Any]) -> Any:
            for k in key_path:
                m = (m or {}).get(k)
            return m

        return get(base), get(ft)

    rows: list[list[str]] = []
    rate_rows = [
        ("EX", ("ex",)),
        *((f"EX {label}", ("ex_by_difficulty", key, "ex")) for key, label in LEVELS),
        ("SQL válida", ("valid_sql_rate",)),
        ("Bloqueada", ("blocked_rate",)),
    ]
    for label, path in rate_rows:
        b, f = pp(path)
        delta = None if b is None or f is None else (float(f) - float(b)) * 100
        rows.append([label, _pct(b), _pct(f), _signed(delta, 1, " p.p.")])
    for label, key in (("Latência p50 (ms)", "p50"), ("Latência p95 (ms)", "p95")):
        b, f = pp(("latency_ms", key))
        delta = None if b is None or f is None else float(f) - float(b)
        rows.append([label, _num(b, 0), _num(f, 0), _signed(delta, 0, "")])
    (cb, eb), (cf, ef) = cost_per_1k(base, gpu), cost_per_1k(ft, gpu)
    delta = None if cb is None or cf is None else cf - cb
    rows.append(
        [
            "US$ / 1.000",
            MISSING if cb is None else _num(cb, 4) + (ESTIMATE_MARK if eb else ""),
            MISSING if cf is None else _num(cf, 4) + (ESTIMATE_MARK if ef else ""),
            _signed(delta, 4, ""),
        ]
    )
    return _table(("Métrica", "Base", "Ajustado", "Δ"), rows)


# --- exemplos de erro -----------------------------------------------------------


@dataclass
class ErrorExample:
    id: str
    difficulty: str
    question: str
    gold_sql: str
    pred_sql: str
    type: str
    source: str
    note: str | None = None
    detail: str | None = None


@dataclass
class ErrorSection:
    system: str
    split: str
    n_items: int
    counts: list[tuple[str, int]]
    examples: list[ErrorExample]


def collect_errors(
    dataset_path: str | Path,
    predictions_path: str | Path,
    db_path: str | Path,
    annotations: dict[str, dict[str, Any]] | None = None,
    k: int = N_EXAMPLES,
) -> ErrorSection:
    """Pontua as previsões, classifica cada erro e escolhe `k` exemplos."""
    from tanque.eval import score as score_mod

    result = score_mod.score(dataset_path, predictions_path, db_path=db_path)
    items = {str(i["id"]): i for i in read_jsonl(dataset_path)}
    preds = {str(p["id"]): p for p in read_jsonl(predictions_path)}
    heuristic: dict[str, str] = {}
    details: dict[str, str | None] = {}
    for r in result.items:
        if r.correct:
            continue
        item, pred = items[r.id], preds.get(r.id)
        gold_res = execute.run(str(item["sql"]), db_path=db_path)
        pred_sql = str((pred or {}).get("sql") or "")
        gen_error = (pred or {}).get("error") if pred else "sem previsão"
        pred_res = execute.run(pred_sql, db_path=db_path) if pred_sql.strip() else None
        heuristic[r.id] = errors_mod.classify(
            str(item["sql"]),
            pred_sql,
            gold_res,
            pred_res,
            order_matters=bool(item.get("order_matters")),
            generation_error=gen_error,
        )
        details[r.id] = gen_error or (pred_res.error if pred_res else None)
    classified = errors_mod.apply_annotations(heuristic, annotations or {})
    chosen = errors_mod.select_examples(classified, k)
    examples = [
        ErrorExample(
            id=e.id,
            difficulty=str(items[e.id].get("difficulty")),
            question=str(items[e.id]["question"]),
            gold_sql=str(items[e.id]["sql"]),
            pred_sql=str((preds.get(e.id) or {}).get("sql") or ""),
            type=e.type,
            source=e.source,
            note=e.note,
            detail=details.get(e.id),
        )
        for e in chosen
    ]
    return ErrorSection(
        result.system, result.split, len(result.items), errors_mod.type_counts(classified), examples
    )


def _first_line(text: str, limit: int = 200) -> str:
    line = " ".join(text.split())
    return line if len(line) <= limit else line[: limit - 1] + "…"


def render_errors(section: ErrorSection) -> list[str]:
    n_errors = sum(n for _, n in section.counts)
    lines = [
        f"### `{section.system}` no {section.split}",
        "",
        f"{n_errors} erros em {section.n_items} perguntas. Tipos (heurística por código, "
        "com a classificação manual de `results/error-annotations.json` quando houver):",
        "",
        *_table(
            ("Tipo", "N", "% dos erros"),
            [
                [errors_mod.ERROR_TYPES[t], str(n), _pct(n / n_errors if n_errors else None)]
                for t, n in section.counts
            ],
        ),
        "",
        f"{len(section.examples)} exemplos (um tipo por vez, em rodízio):",
    ]
    for n, ex in enumerate(section.examples, start=1):
        lines += [
            "",
            f"**{n}. `{ex.id}`** · {ex.difficulty} · {errors_mod.ERROR_TYPES[ex.type]} "
            f"({ex.source})",
            "",
            f"> {_first_line(ex.question, 400)}",
        ]
        if ex.note:
            lines += ["", f"Nota: {ex.note}"]
        if ex.detail:
            detail = _first_line(ex.detail).replace("`", "'")
            lines += ["", f"Erro: `{detail}`"]
        lines += [
            "",
            "Referência:",
            "",
            "```sql",
            ex.gold_sql.strip(),
            "```",
            "",
            "Prevista:",
            "",
            "```sql",
            ex.pred_sql.strip() or "-- (vazia)",
            "```",
        ]
    return lines


# --- documento ------------------------------------------------------------------


@dataclass
class ReportContext:
    gpu: GpuPrice | None = None
    snapshot: dict[str, Any] | None = None
    snapshot_path: str | None = None
    spent_usd: float | None = None
    errors: list[ErrorSection] = field(default_factory=list)
    # (sistema, split) cujas previsões usaram o adapter sem fundir (run-info do M6).
    unmerged: frozenset[tuple[str, str]] = frozenset()


def _method_note(metrics: dict[str, Any], ctx: ReportContext) -> list[str]:
    test_systems = systems_with(metrics, "test")
    hardware = sorted({h for s in test_systems for h in metrics[s]["test"].get("hardware", [])})
    has_api = any(s.startswith("claude-") for s in test_systems)
    lines = ["## Nota de método", ""]
    if ctx.snapshot:
        period = ctx.snapshot.get("period", {})
        lines.append(
            f"- **Dados:** snapshot fixo `{ctx.snapshot_path}` (SHA-256 "
            f"`{ctx.snapshot.get('sha256')}`), coletas de {period.get('data_min')} a "
            f"{period.get('data_max')}; o CI reconstrói o banco a partir dele e confere "
            "contagem e MD5 de cada tabela (D25)."
        )
        lines.append(
            f"- **Data de referência:** `{period.get('data_referencia')}` (= `max(data_coleta)`, "
            'D14); "último mês" = mês do calendário que a contém (D19).'
        )
    lines += [
        "- **Prompt único** (`tanque.prompting.build_prompt`, seção 5.4) para todos os "
        "sistemas, pelo chat template de cada modelo; decodificação gulosa, "
        "`max_new_tokens = 256`, lote de 1 pergunta.",
        "- **Métrica:** EX = execution accuracy (seção 5.6): a SQL prevista é executada com "
        "o guardrail e comparada ao resultado da referência; bloqueio, erro ou falta de "
        "previsão contam como erro.",
        "- **Modelos locais:** `base-*` (sem treino) e `ft-*` (mesmo base + adapter LoRA do "
        "M6, fundido no base antes de gerar), na mesma sessão do Kaggle e na mesma GPU, "
        "fp16. Hardware das previsões do teste: "
        + (", ".join(f"`{h}`" for h in hardware) if hardware else MISSING)
        + ". A latência é medida por pergunta, do template à decodificação, numa T4.",
    ]
    if ctx.gpu:
        vm = (
            "inclui a VM"
            if ctx.gpu.includes_vm
            else "não inclui a VM (vCPU e memória), então é um piso"
        )
        lines.append(
            f"- **Custo por 1.000 perguntas ({ESTIMATE_MARK} = estimativa):** latência média × "
            f"1.000 ÷ 3.600.000 ms/h × US$ {_num(ctx.gpu.hour_usd, 2)}/h, o preço de "
            f"{ctx.gpu.gpu} em {ctx.gpu.source_url} (consultado em {ctx.gpu.checked_on}); "
            f"{vm}. No Kaggle a execução não custou nada; as previsões têm `cost_usd = 0`."
        )
    if has_api:
        lines.append(
            "- **LLM de fronteira:** custo real da API pelas previsões (`pricing.toml`). A "
            "latência dele inclui a rede e a do modelo pequeno é de uma T4: não são "
            "diretamente comparáveis."
        )
    else:
        lines.append(
            "- **Sem LLM de fronteira por enquanto (D24):** sem crédito de API, o relatório "
            "compara só os 2 sistemas locais. `tanque infer claude` continua no repositório; "
            "se `results/predictions/claude-*__test.jsonl` for adicionado e pontuado, ele "
            "entra nas tabelas sozinho."
        )
    if ctx.spent_usd is not None:
        lines.append(
            f"- **Gasto total do projeto em API:** US$ {_num(ctx.spent_usd, 2)} "
            "(`results/spend.jsonl`; `tanque budget status`)."
        )
    lines.append(
        "- **Dev × teste:** o dev (50) foi usado na escolha do modelo (M5) e do fine-tuning "
        "(M6); o teste (150) só aqui, uma vez por sistema."
    )
    return lines


def render(metrics: dict[str, Any], ctx: ReportContext | None = None) -> str:
    ctx = ctx or ReportContext()
    lines = [
        "# Relatório final",
        "",
        "> Gerado por `uv run tanque eval report` a partir de `results/metrics.json`, das "
        "previsões em `results/predictions/` e de `results/error-annotations.json`. "
        "Não edite à mão: o CI confere com `tanque eval report --check`.",
        "",
        "EX = execution accuracy (seção 5.6 do plano); percentuais sobre todos os itens do "
        f"split. `{ESTIMATE_MARK}` = custo estimado de GPU (ver a nota de método).",
        "",
        "## Tabela principal: teste (150 perguntas)",
        "",
    ]
    test = main_table(metrics, "test", ctx.gpu, ctx.unmerged)
    lines += test or [
        "_Previsões do teste ainda não pontuadas. Passos 👤 em `notebooks/README.md` (seção M7)._"
    ]
    lines += ["", "## Base → ajustado (teste)", ""]
    pairs = comparison_pairs(metrics, "test")
    if pairs:
        for base, ft in pairs:
            lines += [f"`{base}` → `{ft}`:", ""]
            lines += comparison_table(metrics[base]["test"], metrics[ft]["test"], ctx.gpu)
            lines.append("")
    else:
        lines += ["_Pendente: precisa de `base-*` e `ft-*` pontuados no teste._", ""]
    lines += ["## Erros do modelo ajustado (teste)", ""]
    if ctx.errors:
        for section in ctx.errors:
            lines += render_errors(section)
            lines.append("")
    else:
        lines += ["_Pendente: precisa de `results/predictions/ft-*__test.jsonl`._", ""]
    lines += [
        "## Referência: dev (50 perguntas)",
        "",
        "Usado na escolha do modelo e do fine-tuning; não é o resultado final.",
        "",
    ]
    lines += main_table(metrics, "dev", ctx.gpu, ctx.unmerged) or ["_Nenhuma métrica no dev._"]
    other = sorted({sp for s in metrics for sp in metrics[s]} - {"dev", "test"})
    for split in other:
        lines += [
            "",
            f"## Split `{split}`",
            "",
            *main_table(metrics, split, ctx.gpu, ctx.unmerged),
        ]
    if ctx.unmerged:
        lines += [
            "",
            f"{UNMERGED_MARK} Latência medida com o adapter LoRA sem fundir (`PeftModel`, "
            "run-info com `adapter_merged` ausente ou falso): não é comparável à do base. "
            "No teste o adapter é fundido antes de gerar.",
        ]
    lines += ["", *_method_note(metrics, ctx)]
    return "\n".join(lines) + "\n"


# --- montagem a partir dos arquivos ---------------------------------------------


class ReportInputError(RuntimeError):
    """Falta algo para gerar o relatório (ex.: banco para os exemplos de erro)."""


def unmerged_runs(predictions_dir: Path) -> frozenset[tuple[str, str]]:
    """(sistema, split) dos run-info com adapter e sem `adapter_merged: true`."""
    import json

    out = set()
    for path in sorted(Path(predictions_dir).glob("run-info__*__*.json")):
        info = json.loads(path.read_text(encoding="utf-8"))
        if info.get("adapter_id") and not info.get("adapter_merged") and info.get("system"):
            out.add((str(info["system"]), str(info.get("split"))))
    return frozenset(out)


def build_context(
    metrics: dict[str, Any],
    *,
    predictions_dir: Path,
    dataset_dir: Path,
    db_path: Path | None,
    annotations_path: Path | None,
    settings: dict[str, Any] | None = None,
    spent_usd: float | None = None,
) -> ReportContext:
    settings = settings if settings is not None else config.load_settings()
    snapshot_rel = settings.get("paths", {}).get("snapshot")
    snapshot = None
    if snapshot_rel:
        from tanque.data.build import read_snapshot_meta, snapshot_meta_path

        meta_path = snapshot_meta_path(config.resolve_path(snapshot_rel))
        snapshot = read_snapshot_meta(meta_path) if meta_path.is_file() else None
    ctx = ReportContext(
        gpu=load_gpu_price(settings),
        snapshot=snapshot,
        snapshot_path=snapshot_rel,
        spent_usd=spent_usd,
    )
    ctx.unmerged = unmerged_runs(Path(predictions_dir))
    ft_files = sorted(Path(predictions_dir).glob("ft-*__test.jsonl"))
    if ft_files:
        if db_path is None or not Path(db_path).is_file():
            raise ReportInputError(
                f"Banco não encontrado ({db_path}): os exemplos de erro re-executam as SQLs. "
                "Rode `tanque data build --from-parquet snapshot/coletas-2026-09.parquet`."
            )
        annotations = errors_mod.load_annotations(annotations_path)
        for path in ft_files:
            ctx.errors.append(
                collect_errors(Path(dataset_dir) / "test.jsonl", path, db_path, annotations)
            )
    return ctx


def render_from_files(
    metrics_path: str | Path,
    *,
    predictions_dir: str | Path | None = None,
    dataset_dir: str | Path | None = None,
    db_path: str | Path | None = None,
    annotations_path: str | Path | None = None,
    spent_usd: float | None = None,
) -> str:
    metrics_path = Path(metrics_path)
    metrics = load_metrics(metrics_path)
    ctx = build_context(
        metrics,
        predictions_dir=Path(predictions_dir or metrics_path.parent / "predictions"),
        dataset_dir=Path(dataset_dir or config.setting_path("datasets_dir")),
        db_path=Path(db_path) if db_path else None,
        annotations_path=Path(annotations_path) if annotations_path else None,
        spent_usd=spent_usd,
    )
    return render(metrics, ctx)


def write_report(metrics_path: str | Path, out_path: str | Path, **kwargs: Any) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        f.write(render_from_files(metrics_path, **kwargs))
    return out
