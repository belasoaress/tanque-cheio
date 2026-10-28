"""Prompt único usado por todos os sistemas e extração da SQL da resposta (plan.md, 5.4).

O prompt tem duas partes:

- `system_prompt()`: o prefixo estável (instrução, DDL, valores fechados, regra
  dos 70%, data de referência e regra do "último mês"). É idêntico para todas as
  perguntas, para poder ser marcado para prompt caching (M4).
- a mensagem do usuário, só com a pergunta, sempre por último.

Nada aqui depende do banco local (o CI não tem banco): a DDL vem de
`src/tanque/data/schema.sql` e os valores do snapshot vêm de `data/manifest.json`.
"""

from __future__ import annotations

import json
import re
from datetime import date
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

from tanque import config

INSTRUCTION = "Responda apenas com uma consulta SQL para DuckDB."

# Siglas de `municipios.regiao` (as mesmas de tanque.data.normalize.REGIONS),
# na ordem do comentário da DDL.
REGION_NAMES: dict[str, str] = {
    "N": "Norte",
    "NE": "Nordeste",
    "CO": "Centro-Oeste",
    "SE": "Sudeste",
    "S": "Sul",
}

# Trechos dos comentários da DDL que só fazem sentido dentro do repositório.
_DDL_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("valores fechados, ver docs/schema.md", "valores fechados, ver abaixo"),
    (" (D18)", ""),
)


def schema_ddl() -> str:
    """DDL de `schema.sql` sem as linhas só de comentário (o aviso de contrato).

    Os comentários de coluna, que ajudam o modelo, são mantidos.
    """
    text = resources.files("tanque.data").joinpath("schema.sql").read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("--")]
    ddl = "\n".join(lines).strip()
    for old, new in _DDL_REPLACEMENTS:
        ddl = ddl.replace(old, new)
    return ddl


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Conteúdo de `data/manifest.json` (commitado no M1)."""
    manifest_path = config.resolve_path(path) if path else config.setting_path("manifest")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def reference_values(manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Valores do snapshot que entram no prompt: produtos e data de referência."""
    manifest = manifest if manifest is not None else load_manifest()
    return {
        "produtos": sorted(manifest["produtos"]),
        "data_referencia": manifest["period"]["data_referencia"],
    }


def render_system_prompt(ddl: str, manifest: dict[str, Any]) -> str:
    """Monta o prefixo estável a partir da DDL e do manifest (função pura)."""
    values = reference_values(manifest)
    ref = date.fromisoformat(values["data_referencia"])
    month_start = ref.replace(day=1)
    produtos = ", ".join(f"'{p}'" for p in values["produtos"])
    regioes = ", ".join(f"'{sigla}' = {nome}" for sigla, nome in REGION_NAMES.items())
    return "\n".join(
        [
            INSTRUCTION,
            "",
            "Esquema do banco:",
            "```sql",
            ddl,
            "```",
            "",
            "Valores:",
            f"- coletas.produto: {produtos}.",
            f"- municipios.regiao: {regioes}.",
            "- municipios.nome: em maiúsculas e sem acento, ex.: 'SAO PAULO'.",
            "- municipios.capital = true para as 27 capitais.",
            "",
            "Regra dos 70%: o etanol compensa quando preço_etanol / preço_gasolina <= 0,7.",
            "",
            f"Data de referência: {ref.isoformat()}",
            '"último mês" = mês do calendário que contém a data de referência '
            f"(de {month_start.isoformat()} a {ref.isoformat()}).",
        ]
    )


@cache
def system_prompt() -> str:
    """Prefixo estável do prompt, igual para todas as perguntas (alvo do cache no M4)."""
    return render_system_prompt(schema_ddl(), load_manifest())


def build_prompt(question: str) -> list[dict]:
    """Mensagens no formato chat: system com o prefixo estável, user com a pergunta."""
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": f"Pergunta: {question.strip()}"},
    ]


# --- extração da SQL -----------------------------------------------------------

# Cerca de abertura (``` ou ~~~, com linguagem opcional), corpo, e a mesma cerca
# de fechamento (ou o fim do texto, se o modelo não fechou).
_FENCE_RE = re.compile(
    r"(?P<fence>`{3,}|~{3,})[ \t]*(?P<lang>[\w+-]*)[^\n]*\n(?P<body>.*?)(?:(?P=fence)|\Z)",
    re.DOTALL,
)
_SQL_LANGS = {"sql", "duckdb", "postgresql", "postgres"}
_SQL_START_RE = re.compile(
    r"\bSELECT\b|\bWITH\s+(?:RECURSIVE\s+)?\w+\s*(?:\([^)]*\)\s*)?AS\s*\(",
    re.IGNORECASE,
)
_TRAILING_RE = re.compile(r"[\s;`~]+\Z")


def _clean(sql: str) -> str:
    return _TRAILING_RE.sub("", sql).strip()


def extract_sql(raw_output: str) -> str:
    """SQL contida na saída do modelo, sem cercas de código nem texto em volta.

    - Com cercas (```sql, ``` ou ~~~): usa a primeira cerca que parece SQL
      (linguagem sql ou corpo começando por SELECT/WITH); senão, a primeira cerca.
    - Sem cerca: tudo a partir do primeiro SELECT ou WITH ... AS (.
    - Remove ponto e vírgula e espaços finais. Saída vazia -> "".

    Não corta o texto em `;` intermediários: várias instruções devem chegar ao
    guardrail, que as bloqueia e registra o motivo.
    """
    if not raw_output or not raw_output.strip():
        return ""
    blocks = list(_FENCE_RE.finditer(raw_output))
    if blocks:
        for block in blocks:
            body = block["body"]
            if block["lang"].lower() in _SQL_LANGS or _SQL_START_RE.match(body.lstrip()):
                return _clean(body)
        return _clean(blocks[0]["body"])
    match = _SQL_START_RE.search(raw_output)
    text = raw_output[match.start() :] if match else raw_output
    return _clean(text)
