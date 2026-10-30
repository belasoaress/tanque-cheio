"""Gráfico EX × custo por 1.000 perguntas (M17, R22): `results/ex-vs-cost.png`.

Gerado só por `tanque eval chart` a partir de `results/metrics.json` (split `test`):

- um ponto por sistema com custo conhecido (o mesmo `report.cost_per_1k` da tabela
  principal: latência média × GPUs × preço da T4, D35); sistemas sem custo (ex.: API
  sem preço) ficam de fora;
- eixo x em escala log (US$ por 1.000 perguntas), eixo y = EX rígida (%), com o IC 95%
  (D29) como barra de erro vertical;
- cor pelo tipo de sistema (sem ajuste, ft-v1, ft-v2); círculo cheio = single-shot,
  triângulo vazado = `+retry` (D32); rótulo curto com o tamanho (`1,5B ft-v2`, `32B
  base`) nos pontos single-shot.

**Conferência do CI sem matplotlib.** O PNG leva no chunk `tEXt` `tanque-data` um
SHA-256 dos pontos desenhados (sistema, EX, IC, custo). `tanque eval chart --check`
recalcula esse hash a partir do `metrics.json` e compara com o do PNG, lendo o arquivo
com um parser de PNG de poucas linhas (`png_text_chunks`): não precisa de matplotlib
nem depende de fontes ou da versão da biblioteca, que mudam os bytes da imagem entre
Windows e Linux. Sem métricas do teste, não há gráfico (e um PNG velho é erro).

O matplotlib fica no grupo opcional `chart` (`uv sync --group chart`), só para gerar.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tanque import config
from tanque.eval import report
from tanque.eval.score import is_retry, single_shot, system_metrics, system_sort_key

CHART_FILENAME = "ex-vs-cost.png"
DIGEST_KEY = "tanque-data"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_SIZE = re.compile(r"-(\d+(?:\.\d+)?)b(?=-|$)")

# Paleta categórica de referência (skill de dataviz, tema claro): 3 primeiras posições,
# que passam nas checagens de daltonismo com todos os pares (gráfico de dispersão).
GROUPS = {
    "base": ("Sem ajuste (base)", "#2a78d6"),
    "ft": ("Ajustado v1 (ft-v1)", "#eb6834"),
    "ft2": ("Ajustado v2 (ft-v2)", "#1baf7a"),
    "other": ("Outros", "#52514e"),
}
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
TRAINING = {"base": "base", "ft": "ft-v1", "ft2": "ft-v2"}
X_TICKS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]


def default_chart_path() -> Path:
    return config.setting_path("results_dir") / CHART_FILENAME


@dataclass(frozen=True)
class Point:
    system: str
    label: str
    group: str
    retry: bool
    ex: float
    lo: float | None
    hi: float | None
    cost: float


def _group(system: str) -> str:
    prefix = single_shot(system).split("-", 1)[0]
    return prefix if prefix in TRAINING else "other"


def short_label(system: str) -> str:
    """`1,5B ft-v2` de `ft2-qwen2.5-coder-1.5b-instruct(+retry)`; outros: o próprio nome."""
    name = single_shot(system)
    group = _group(name)
    match = _SIZE.search(name)
    if group == "other" or not match:
        return name
    return f"{match.group(1).replace('.', ',')}B {TRAINING[group]}"


def points(
    metrics: dict[str, Any], split: str = "test", gpu: report.GpuPrice | None = None
) -> list[Point]:
    """Pontos do gráfico (ordem dos sistemas), só os que têm EX e custo."""
    systems = system_metrics(metrics)
    out = []
    for system in sorted(systems, key=system_sort_key):
        m = systems[system].get(split)
        if not m or m.get("ex") is None:
            continue
        cost, _ = report.cost_per_1k(m, gpu)
        if cost is None or cost <= 0:
            continue
        ci = m.get("ex_ci95") or [None, None]
        out.append(
            Point(
                system=system,
                label=short_label(system),
                group=_group(system),
                retry=is_retry(system),
                ex=float(m["ex"]),
                lo=None if ci[0] is None else float(ci[0]),
                hi=None if ci[1] is None else float(ci[1]),
                cost=round(float(cost), 6),
            )
        )
    return out


def data_digest(pts: list[Point]) -> str:
    """SHA-256 dos pontos (independe da ordem das chaves do metrics.json)."""
    rows = sorted((asdict(p) for p in pts), key=lambda r: r["system"])
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode("utf-8")).hexdigest()


# --- PNG ------------------------------------------------------------------------


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def write_png_with_text(path: str | Path, text: dict[str, str]) -> Path:
    """PNG mínimo (1 pixel) com chunks `tEXt`; só para testes da conferência."""
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)  # 1×1, 8 bits, tons de cinza
    chunks = [_chunk(b"IHDR", ihdr)]
    for key, value in text.items():
        chunks.append(_chunk(b"tEXt", key.encode("latin-1") + b"\0" + value.encode("latin-1")))
    chunks += [_chunk(b"IDAT", zlib.compress(b"\0\xff")), _chunk(b"IEND", b"")]
    path = Path(path)
    path.write_bytes(PNG_SIGNATURE + b"".join(chunks))
    return path


def png_text_chunks(path: str | Path) -> dict[str, str] | None:
    """Chunks `tEXt` de um PNG (`None` se o arquivo não for PNG)."""
    data = Path(path).read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        return None
    out: dict[str, str] = {}
    pos = len(PNG_SIGNATURE)
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        if kind == b"tEXt" and b"\0" in body:
            key, value = body.split(b"\0", 1)
            out[key.decode("latin-1")] = value.decode("latin-1")
        if kind == b"IEND":
            break
        pos += 12 + length
    return out


def check(pts: list[Point], png_path: str | Path) -> tuple[bool, str]:
    """(ok, mensagem): o PNG existe e foi gerado a partir destes pontos."""
    png_path = Path(png_path)
    if not pts:
        if png_path.exists():
            return False, (
                f"{png_path} existe, mas não há métricas do teste com custo para desenhar: "
                "apague o arquivo ou pontue o teste."
            )
        return True, "OK: sem métricas do teste; gráfico ainda não gerado."
    if not png_path.is_file():
        return False, f"{png_path} não existe: rode `uv run tanque eval chart` e commite."
    text = png_text_chunks(png_path)
    if text is None or text.get(DIGEST_KEY) != data_digest(pts):
        return False, (
            f"{png_path} está desatualizado em relação ao metrics.json: rode "
            "`uv sync --group chart` e `uv run tanque eval chart`, e commite."
        )
    return True, f"OK: {png_path} confere com o metrics.json ({len(pts)} pontos)."


# --- desenho --------------------------------------------------------------------


def _money(x: float, _pos: Any = None) -> str:
    """Rótulo do eixo x: `US$ 0,10`, `US$ 1`, `US$ 10`."""
    places = 2 if x < 1 else (0 if float(x).is_integer() else 1)
    return "US$ " + report._num(x, places)


def render(
    pts: list[Point],
    out_path: str | Path,
    *,
    gpu: report.GpuPrice | None = None,
    split: str = "test",
) -> Path:
    """Desenha o gráfico e grava o PNG (com o hash dos pontos no chunk `tEXt`)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    except ImportError as e:  # pragma: no cover - depende do ambiente
        raise RuntimeError(
            "matplotlib não instalado: rode `uv sync --group chart` (grupo opcional)."
        ) from e
    if not pts:
        raise ValueError(f"nenhum sistema com EX e custo no split {split!r}")

    plt.rcParams.update({"svg.hashsalt": "tanque", "font.size": 10})
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for p in pts:
        color = GROUPS[p.group][1]
        y = p.ex * 100
        yerr = None
        if p.lo is not None and p.hi is not None:
            yerr = [[max(0.0, y - p.lo * 100)], [max(0.0, p.hi * 100 - y)]]
        ax.errorbar(
            [p.cost],
            [y],
            yerr=yerr,
            fmt="^" if p.retry else "o",
            markersize=8,
            color=color,
            markerfacecolor=SURFACE if p.retry else color,
            markeredgecolor=color,
            markeredgewidth=1.6,
            ecolor=color,
            elinewidth=1.2,
            capsize=3,
            alpha=0.9,
            zorder=3,
        )
        if not p.retry:
            ax.annotate(
                p.label,
                (p.cost, y),
                xytext=(8, -3),
                textcoords="offset points",
                fontsize=9,
                color=TEXT,
                zorder=4,
            )
    ax.set_xscale("log")
    costs = [p.cost for p in pts]
    ax.set_xlim(min(costs) / 1.6, max(costs) * 2.2)
    ax.xaxis.set_major_locator(FixedLocator(X_TICKS))
    ax.xaxis.set_major_formatter(FuncFormatter(_money))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0f}%"))
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY)
    price = f"US$ {gpu.hour_usd:.2f}".replace(".", ",") + "/h por T4" if gpu else "preço da GPU"
    ax.set_xlabel(
        f"Custo estimado por 1.000 perguntas (escala log; latência média × GPUs × {price})",
        color=TEXT_SECONDARY,
    )
    ax.set_ylabel("EX no teste (execution accuracy)", color=TEXT_SECONDARY)
    ax.set_title(
        f"EX × custo no {'teste' if split == 'test' else split} (barras = IC 95%)",
        loc="left",
        color=TEXT,
        fontsize=12,
    )
    used = [g for g in GROUPS if any(p.group == g for p in pts)]
    handles = [
        Line2D([], [], marker="o", linestyle="", color=GROUPS[g][1], label=GROUPS[g][0])
        for g in used
    ]
    handles += [
        Line2D([], [], marker="o", linestyle="", color=TEXT_SECONDARY, label="Sem nova tentativa"),
        Line2D(
            [],
            [],
            marker="^",
            linestyle="",
            color=TEXT_SECONDARY,
            markerfacecolor=SURFACE,
            label="+retry (1 nova tentativa)",
        ),
    ]
    legend = ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(TEXT)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out,
        format="png",
        facecolor=SURFACE,
        metadata={"Software": None, DIGEST_KEY: data_digest(pts)},
    )
    plt.close(fig)
    return out
