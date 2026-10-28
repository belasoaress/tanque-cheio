"""Gera `docs/schema.md` a partir do banco construído (não editar o .md à mão)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from tanque.data.build import TABLES, schema_sql
from tanque.data.capitals import CAPITALS

TOP_BANDEIRAS = 15

COLUMN_DOCS: dict[tuple[str, str], str] = {
    ("municipios", "municipio_id"): "ID determinístico (ordem de `uf`, `nome`)",
    ("municipios", "nome"): "Nome como na ANP, em maiúsculas e sem acento (ex.: `SAO CARLOS`)",
    ("municipios", "uf"): "Sigla da UF (ex.: `SP`)",
    ("municipios", "regiao"): "Sigla da região: `N`, `NE`, `CO`, `SE`, `S`",
    ("municipios", "capital"): "`true` para as 27 capitais (lista fixa no código, D18)",
    ("postos", "posto_id"): "ID determinístico (ordem de `cnpj`)",
    ("postos", "cnpj"): "CNPJ da revenda, como na ANP (ex.: `00.003.188/0001-21`)",
    ("postos", "razao_social"): "Razão social da revenda (da coleta mais recente)",
    ("postos", "bandeira"): "Bandeira da coleta mais recente; `BRANCA` = sem bandeira",
    ("postos", "bairro"): "Bairro (da coleta mais recente)",
    ("postos", "cep"): "CEP (da coleta mais recente)",
    ("postos", "municipio_id"): "Município do posto (da coleta mais recente)",
    ("coletas", "coleta_id"): (
        "ID determinístico (ordem de `data_coleta`, `cnpj`, `produto`, valores, `unidade`)"
    ),
    ("coletas", "posto_id"): "Posto pesquisado",
    ("coletas", "produto"): "Combustível; valores fechados abaixo",
    ("coletas", "data_coleta"): "Data da coleta",
    ("coletas", "valor_venda"): "Preço de venda ao consumidor, R$ por unidade; sempre > 0",
    ("coletas", "valor_compra"): "Preço de compra pelo posto; quase sempre nulo",
    ("coletas", "unidade"): "`R$ / litro` ou `R$ / m³` (GNV)",
}


def _md_table(header: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return lines


def _fmt(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def render_schema_doc(db_path: Path, manifest: dict[str, Any], *, provisional: bool = False) -> str:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        columns = con.execute(
            "SELECT table_name, column_name, data_type, is_nullable "
            "FROM information_schema.columns WHERE table_schema = 'main' "
            "ORDER BY table_name, ordinal_position"
        ).fetchall()
        produtos = con.execute(
            "SELECT produto, unidade, count(*), min(valor_venda), max(valor_venda) "
            "FROM coletas GROUP BY 1, 2 ORDER BY 1, 2"
        ).fetchall()
        regioes = con.execute(
            "SELECT regiao, count(DISTINCT uf), count(*) FROM municipios GROUP BY 1 ORDER BY 1"
        ).fetchall()
        bandeiras = con.execute(
            "SELECT bandeira, count(*) AS n FROM postos GROUP BY 1 ORDER BY n DESC, bandeira "
            f"LIMIT {TOP_BANDEIRAS}"
        ).fetchall()
        n_bandeiras = con.execute("SELECT count(DISTINCT bandeira) FROM postos").fetchone()[0]
        capitais = con.execute(
            "SELECT nome, uf FROM municipios WHERE capital ORDER BY uf"
        ).fetchall()
    finally:
        con.close()

    period = manifest["period"]
    tables = manifest["tables"]
    rows = manifest["rows"]
    out: list[str] = ["# Schema do banco `data/tanque.duckdb`", ""]
    if provisional:
        out += [
            "> **PROVISÓRIO — gerado das fixtures; será regerado com o snapshot real.**",
            "",
        ]
    out += [
        "Gerado por `tanque data build` (src/tanque/data/schema_doc.py). Não edite à mão.",
        "",
        f"- **Data de referência:** `{period['data_referencia']}` (= `max(data_coleta)`). "
        'Datas relativas são resolvidas contra ela; "último mês" = mês do calendário que '
        "contém a data de referência (D19).",
        f"- **Período:** `{period['data_min']}` a `{period['data_max']}` "
        f"({period['months']} meses com coletas).",
        f"- **Arquivos brutos:** {len(manifest['raw_files'])} CSVs da ANP "
        "(hashes em `data/manifest.json`).",
        f"- **Linhas:** {_fmt(rows['read'])} lidas, {_fmt(rows['accepted'])} aceitas, "
        f"{_fmt(rows['rejected_total'])} rejeitadas.",
        f"- **UFs presentes:** {len(manifest['ufs'])} ({', '.join(manifest['ufs'])}).",
        "",
        "## Tabelas",
        "",
    ]
    for table in TABLES:
        out += [f"### `{table}` ({_fmt(tables[table])} linhas)", ""]
        out += _md_table(
            ["coluna", "tipo", "nulo", "descrição"],
            [
                [f"`{c}`", t, "sim" if n == "YES" else "não", COLUMN_DOCS.get((tb, c), "")]
                for tb, c, t, n in columns
                if tb == table
            ],
        )
        out.append("")
    out += ["## DDL (`src/tanque/data/schema.sql`)", "", "```sql", schema_sql().strip(), "```", ""]
    out += ["## Valores fechados", "", "### `coletas.produto`", ""]
    out += _md_table(
        ["produto", "unidade", "coletas", "min valor_venda", "max valor_venda"],
        [[f"`{p}`", f"`{u}`", _fmt(n), lo, hi] for p, u, n, lo, hi in produtos],
    )
    out += ["", "### `municipios.regiao`", ""]
    out += _md_table(
        ["regiao", "UFs", "municípios"], [[f"`{r}`", u, _fmt(n)] for r, u, n in regioes]
    )
    out += [
        "",
        f"### `postos.bandeira`: as {TOP_BANDEIRAS} mais frequentes "
        f"(de {n_bandeiras} distintas, por número de postos)",
        "",
    ]
    out += _md_table(["bandeira", "postos"], [[f"`{b}`", _fmt(n)] for b, n in bandeiras])
    out += [
        "",
        f"### Capitais (`municipios.capital = true`): {len(capitais)} de {len(CAPITALS)}",
        "",
        ", ".join(f"{nome}/{uf}" for nome, uf in capitais) or "(nenhuma)",
        "",
        "## Regras de construção",
        "",
        "- Só entram os 6 produtos automotivos acima; GLP e outros vão para o relatório de "
        "rejeição.",
        "- `valor_venda` nulo, inválido ou ≤ 0 não entra no banco; as linhas rejeitadas são "
        "contadas por motivo em `data/manifest.json` (`rows.rejected`).",
        "- Coletas repetidas exatamente (mesmo CNPJ, produto, data, valores e unidade) entram "
        "uma vez só (`linha_duplicada`).",
        "- Se o posto mudou de bandeira (ou de endereço) no período, vale a coleta mais "
        "recente. Bandeira vazia vira `BRANCA`.",
        "- Nome da rua, número e complemento não entram no banco.",
        "- Nomes de município sem acento e em maiúsculas, como nos arquivos da ANP.",
        "",
    ]
    if rows["rejected"]:
        out += ["### Rejeitadas por motivo", ""]
        out += _md_table(
            ["motivo", "linhas"], [[f"`{k}`", _fmt(v)] for k, v in rows["rejected"].items()]
        )
        out.append("")
    return "\n".join(out)


def write_schema_doc(
    db_path: Path, path: Path, manifest: dict[str, Any], *, provisional: bool = False
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(render_schema_doc(db_path, manifest, provisional=provisional))
    return path
