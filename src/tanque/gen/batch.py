"""Geração de pares pergunta→SQL pela Message Batches API (M4).

Cada requisição pede `PAIRS_PER_REQUEST` (10) pares para uma intenção do
catálogo (`tanque.gen.intents`). O `system` tem um **prefixo estável**, igual em
todas as requisições:

1. instruções do gerador e convenções (preço médio, regra dos 70%, datas
   relativas, nomes de município, capitais, bandeiras);
2. o prompt de sistema da seção 5.4 (`prompting.system_prompt()`), o mesmo que
   o modelo treinado vai ver;
3. a lista de municípios do banco (nome/UF), para o gerador não inventar
   cidades que não estão no snapshot.

Esse prefixo passa de 512 tokens, o mínimo de cache do modelo `cheap` (ver
`tanque.infer.api`), e por isso o último bloco leva
`cache_control: {"type": "ephemeral"}`; a decisão é tomada pela contagem de
tokens medida, não suposta. A mensagem do usuário, que muda, traz a intenção e
um "foco" sorteado de forma determinística (produto, região, período e estilo),
para que requisições repetidas da mesma intenção não gerem as mesmas perguntas.

Arquivos:

- `results/api-batches.jsonl`: registro dos lotes (envio e coleta). A coleta
  só cobra o lote uma vez.
- `results/gen/raw/<batch_id>.jsonl`: saída bruta de cada requisição (texto,
  uso de tokens, custo). É pequena e fica no git: é a prova do que foi pago e
  permite refazer a validação sem pagar de novo.
- Os arquivos intermediários da validação ficam em `data/gen/` (ignorado); só
  `datasets/train.jsonl` vai para o git.

O modelo vem sempre da chave `cheap` de `configs/pricing.toml`.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from tanque import config
from tanque.budget import Ledger, ModelPrice, find_model
from tanque.gen import intents as intents_mod
from tanque.gen.intents import Intent
from tanque.infer import api

MILESTONE = "M4"
KIND = "gen"
MODEL_KEY = "cheap"
PAIRS_PER_REQUEST = 10
PILOT_REQUESTS = 20
DEFAULT_MAX_TOKENS = 8000
DEFAULT_EFFORT = "low"
# Saída esperada por requisição (10 pares + raciocínio em esforço baixo); só para
# a estimativa. A estimativa de pior caso usa `max_tokens`.
DEFAULT_EST_OUTPUT_TOKENS = 3000
RAW_RELDIR = Path("results") / "gen" / "raw"

GENERATOR_INSTRUCTIONS = f"""\
Você gera dados de treino para um modelo de text-to-SQL em português do Brasil.
Cada exemplo é um par: uma pergunta que uma pessoa faria sobre preços de
combustíveis da ANP e a consulta SQL (DuckDB) que a responde corretamente no
banco descrito abaixo.

Formato da resposta: SOMENTE um array JSON, sem texto antes ou depois, com
exatamente {PAIRS_PER_REQUEST} objetos assim:
[{{"question": "...", "sql": "SELECT ...", "order_matters": true}}, ...]
- "question": pergunta natural em português, com acentos, como um usuário escreveria.
- "sql": uma única consulta SELECT (ou WITH ... SELECT) para DuckDB, sem ponto e vírgula.
- "order_matters": true se a pergunta pede ordenação ou ranking; senão false.

Regras das consultas:
- Use só as tabelas coletas, postos e municipios. Junções:
  coletas c JOIN postos p ON p.posto_id = c.posto_id
  JOIN municipios m ON m.municipio_id = p.municipio_id
- Preço médio = AVG(c.valor_venda) sobre as coletas do recorte (não faça média de médias).
- Regra dos 70%: o etanol compensa quando AVG(valor_venda) do 'ETANOL' dividido por
  AVG(valor_venda) da 'GASOLINA' (comum, nunca a aditivada), no mesmo local e período,
  é <= 0.7.
- Produtos: use exatamente 'GASOLINA', 'GASOLINA ADITIVADA', 'ETANOL', 'DIESEL',
  'DIESEL S10', 'GNV'. "Gasolina" sozinha é a comum ('GASOLINA'); "diesel" sozinho é
  'DIESEL'; "álcool" é 'ETANOL'. GNV é vendido em R$/m³.
- Datas: os dados vão de 2024-10-01 a 2026-09-30. A data de referência é 2026-09-30.
  "Último mês" = setembro de 2026 (data_coleta >= DATE '2026-09-01' AND data_coleta <=
  DATE '2026-09-30'). "Este ano" = 2026 até a data de referência; "ano passado" = 2025;
  "últimos 3 meses" = julho a setembro de 2026; "últimos 12 meses" = 2025-10-01 a
  2026-09-30. Não use "mês passado" (é ambíguo); prefira meses com nome e ano.
  Não pergunte sobre datas fora do período.
- Municípios: m.nome está em MAIÚSCULAS e SEM ACENTO ('SAO PAULO', 'GOIANIA'); na
  pergunta escreva o nome normal ("São Paulo", "Goiânia"). Filtre sempre por nome E uf,
  porque há nomes repetidos em UFs diferentes. Use só municípios da lista abaixo.
  "Cidade de São Paulo" é o município; "estado de São Paulo" ou "SP" é a UF.
- Capitais: m.capital = true (não liste nomes de capitais à mão).
- Regiões: m.regiao com as siglas 'N', 'NE', 'CO', 'SE', 'S'.
- Bandeiras: p.bandeira, em maiúsculas: 'VIBRA', 'IPIRANGA', 'RAIZEN', 'ALE' e
  'BRANCA' (= posto sem bandeira, "bandeira branca"). Na pergunta, cite a bandeira
  pelo nome ("postos Ipiranga", "Raízen", "Vibra").
- valor_compra é sempre nulo neste snapshot: não use.
- Não use LIMIT sem ORDER BY. Use LIMIT só quando a pergunta pede "os N ...".
- A SQL precisa responder exatamente o que a pergunta pede: mesmo produto, local,
  período, ordenação e número de itens.

Diversidade (importante): as {PAIRS_PER_REQUEST} perguntas de uma resposta devem ser
diferentes entre si na redação, e não só nos parâmetros. Varie a estrutura da frase
(pergunta direta, pedido "me mostre...", "liste...", "qual foi...", "em que..."), o
registro (formal, coloquial), a ordem das informações e os parâmetros (produto,
local, período, N). Não repita a mesma frase trocando só a UF.
"""

_PRODUCT_FOCUS = ("GASOLINA", "ETANOL", "DIESEL S10", "DIESEL", "GASOLINA ADITIVADA", "GNV")
_REGION_FOCUS = (
    "Norte",
    "Nordeste",
    "Centro-Oeste",
    "Sudeste",
    "Sul",
    "o Brasil inteiro",
    "capitais",
)
_PERIOD_FOCUS = (
    "o último mês (setembro de 2026)",
    "o ano de 2025",
    "2026 até setembro",
    "meses específicos (com nome e ano)",
    "trimestres ou semestres",
    "os últimos 3, 6 ou 12 meses",
    "o período inteiro dos dados",
)
_STYLE_FOCUS = (
    "perguntas curtas e diretas",
    "perguntas mais longas, com contexto (ex.: 'estou planejando uma viagem...')",
    "pedidos no imperativo ('liste', 'mostre', 'compare')",
    "linguagem coloquial",
    "linguagem formal de relatório",
)


@dataclass(frozen=True)
class GenRequest:
    custom_id: str
    intent_id: str
    params: dict[str, Any]


@dataclass
class CollectResult:
    batch_id: str
    succeeded: int = 0
    errored: int = 0
    cost_usd: Decimal = Decimal(0)
    pairs: list[dict[str, Any]] = field(default_factory=list)
    raw_path: Path | None = None
    already_collected: bool = False


# --- prompt -----------------------------------------------------------------


def municipality_list_text(municipios: Iterable[tuple[str, str]]) -> str:
    """Lista "UF: NOME, NOME, ..." agrupada por UF, em ordem."""
    by_uf: dict[str, list[str]] = {}
    for nome, uf in municipios:
        by_uf.setdefault(uf, []).append(nome)
    lines = [f"{uf}: {', '.join(sorted(names))}" for uf, names in sorted(by_uf.items())]
    return "Municípios do banco (m.nome / m.uf):\n" + "\n".join(lines)


def load_municipios(db_path: str | Path) -> list[tuple[str, str]]:
    """(nome, uf) de todos os municípios do banco, aberto só para leitura."""
    import duckdb

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return [(r[0], r[1]) for r in con.execute("SELECT nome, uf FROM municipios").fetchall()]
    finally:
        con.close()


def default_system_prompt() -> str:
    from tanque import prompting

    return prompting.system_prompt()


def prefix_texts(municipios: Iterable[tuple[str, str]], target_prompt: str) -> list[str]:
    """Textos do prefixo estável, na ordem em que vão para `system`."""
    return [
        GENERATOR_INSTRUCTIONS,
        "Prompt de sistema que o modelo treinado recebe (descreve o banco):\n"
        "<prompt_do_modelo>\n" + target_prompt + "\n</prompt_do_modelo>",
        municipality_list_text(municipios),
    ]


def request_focus(intent: Intent, k: int) -> dict[str, str]:
    """Foco determinístico da requisição `k` dessa intenção."""
    rng = random.Random(f"{intent.id}:{k}")
    return {
        "produtos": ", ".join(rng.sample(_PRODUCT_FOCUS, 2)),
        "local": rng.choice(_REGION_FOCUS),
        "periodo": rng.choice(_PERIOD_FOCUS),
        "estilo": rng.choice(_STYLE_FOCUS),
    }


def user_message(intent: Intent, k: int) -> str:
    focus = request_focus(intent, k)
    return "\n".join(
        [
            intent.instructions(),
            "",
            "Foco desta leva (use na maioria das perguntas, sem forçar quando não fizer "
            "sentido para a intenção):",
            f"- produtos: {focus['produtos']}",
            f"- local: {focus['local']}",
            f"- período: {focus['periodo']}",
            f"- estilo: {focus['estilo']}",
            "",
            f"Gere {PAIRS_PER_REQUEST} pares desta intenção, no formato JSON pedido.",
        ]
    )


def custom_id(k: int, intent: Intent) -> str:
    return f"g{k:05d}-{intent.id}"[:64]


def intent_of_custom_id(cid: str) -> str:
    return cid.split("-", 1)[1] if "-" in cid else cid


def build_requests(
    n_requests: int,
    *,
    model_id: str,
    system: list[dict],
    offset: int = 0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    effort: str | None = DEFAULT_EFFORT,
) -> list[GenRequest]:
    """`n_requests` requisições na distribuição alvo, numeradas a partir de `offset`."""
    out = []
    for i, intent in enumerate(intents_mod.allocate(n_requests)):
        k = offset + i
        params: dict[str, Any] = {
            "model": model_id,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user_message(intent, k)}],
        }
        if effort:
            params["output_config"] = {"effort": effort}
        out.append(GenRequest(custom_id(k, intent), intent.id, params))
    return out


# --- parse ------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _as_pairs(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        for key in ("pairs", "pares", "examples", "exemplos", "items"):
            if isinstance(value.get(key), list):
                return _as_pairs(value[key])
        return [value] if "question" in value or "sql" in value else []
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    return []


def parse_pairs(text: str) -> list[dict[str, Any]]:
    """Pares `{question, sql, order_matters?}` de uma resposta do gerador.

    Tolerante: aceita cercas de código, texto em volta, um objeto com a lista
    dentro, e, se o array estiver truncado ou inválido, recupera os objetos
    completos um a um. Campos que não são texto viram texto vazio.
    """
    if not text or not text.strip():
        return []
    candidates = [m.group(1) for m in _FENCE_RE.finditer(text)] + [text]
    for chunk in candidates:
        start, end = chunk.find("["), chunk.rfind("]")
        if start != -1 and end > start:
            try:
                return _clean_pairs(_as_pairs(json.loads(chunk[start : end + 1])))
            except json.JSONDecodeError:
                pass
        try:
            return _clean_pairs(_as_pairs(json.loads(chunk)))
        except json.JSONDecodeError:
            pass
    # Recuperação: decodifica cada objeto `{...}` completo que encontrar.
    decoder = json.JSONDecoder()
    found: list[dict[str, Any]] = []
    pos = 0
    while True:
        pos = text.find("{", pos)
        if pos == -1:
            break
        try:
            value, end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            pos += 1
            continue
        found.extend(_as_pairs(value))
        pos = end
    return _clean_pairs(found)


def _clean_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for p in pairs:
        question = p.get("question")
        sql = p.get("sql")
        order = p.get("order_matters")
        out.append(
            {
                "question": question.strip() if isinstance(question, str) else "",
                "sql": sql.strip() if isinstance(sql, str) else "",
                "order_matters": order if isinstance(order, bool) else None,
            }
        )
    return out


# --- envio e coleta ---------------------------------------------------------


def cheap_price(ledger: Ledger) -> ModelPrice:
    return find_model(ledger.pricing, MODEL_KEY)


def prepare_system(
    client: Any,
    price: ModelPrice,
    texts: list[str],
) -> tuple[list[dict], int, bool]:
    """(blocos de system, tokens do prefixo, cache marcado?)."""
    probe = [{"role": "user", "content": "."}]
    plain = api.system_blocks(texts, cache=False)
    measured = api.count_tokens(client, price.id, plain, probe)
    tokens = measured if measured is not None else api.heuristic_tokens("".join(texts))
    cache = api.should_cache(tokens)
    return api.system_blocks(texts, cache=cache), tokens, cache


@dataclass
class Estimate:
    n_requests: int
    input_tokens_each: int
    prefix_tokens: int
    cache: bool
    expected_usd: Decimal
    worst_usd: Decimal


def estimate(
    price: ModelPrice,
    requests: list[GenRequest],
    prefix_tokens: int,
    cache: bool,
    *,
    est_output_tokens: int = DEFAULT_EST_OUTPUT_TOKENS,
) -> Estimate:
    """Estimativa esperada e de pior caso (saída = max_tokens) do lote."""
    if not requests:
        return Estimate(0, 0, prefix_tokens, cache, Decimal(0), Decimal(0))
    longest_user = max(api.heuristic_tokens(r.params["messages"][0]["content"]) for r in requests)
    input_each = prefix_tokens + longest_user
    cached = prefix_tokens if cache else 0
    n = len(requests)
    max_tokens = int(requests[0].params["max_tokens"])
    expected = api.estimate_cost(
        price, n, input_each, est_output_tokens, batch=True, cached_prefix_tokens=cached
    )
    worst = api.estimate_cost(
        price, n, input_each, max_tokens, batch=True, cached_prefix_tokens=cached
    )
    return Estimate(n, input_each, prefix_tokens, cache, expected, worst)


def next_offset(registry: api.BatchRegistry) -> int:
    """Primeiro número de requisição livre (as requisições são numeradas no projeto todo)."""
    return sum(int(e.get("n_requests", 0)) for e in registry.submitted_of_kind(KIND))


def submit(
    client: Any,
    requests: list[GenRequest],
    est: Estimate,
    *,
    ledger: Ledger,
    registry: api.BatchRegistry,
    offset: int,
    approved: bool = False,
    label: str = "submit",
) -> str:
    """Reserva no orçamento, checa a aprovação, envia o lote e registra. Devolve o id."""
    api.check_approval(est.expected_usd, approved)
    ledger.reserve(est.expected_usd)
    batch = client.messages.batches.create(
        requests=[{"custom_id": r.custom_id, "params": r.params} for r in requests]
    )
    batch_id = str(api.get(batch, "id"))
    registry.append(
        "submitted",
        batch_id,
        kind=KIND,
        label=label,
        milestone=MILESTONE,
        model=MODEL_KEY,
        n_requests=len(requests),
        offset=offset,
        prefix_tokens=est.prefix_tokens,
        cache=est.cache,
        estimate_usd=float(est.expected_usd),
        worst_case_usd=float(est.worst_usd),
        intents={r.custom_id: r.intent_id for r in requests},
    )
    return batch_id


def collect(
    client: Any,
    batch_id: str,
    *,
    ledger: Ledger,
    registry: api.BatchRegistry,
    raw_dir: str | Path | None = None,
) -> CollectResult:
    """Lê os resultados de um lote encerrado, cobra o uso real e grava a saída bruta.

    Se o lote já foi coletado, não cobra de novo: relê a saída bruta gravada.
    """
    raw_dir = config.resolve_path(raw_dir or RAW_RELDIR)
    raw_path = raw_dir / f"{batch_id}.jsonl"
    result = CollectResult(batch_id=batch_id, raw_path=raw_path)
    if registry.is_collected(batch_id):
        result.already_collected = True
        for row in read_raw(raw_path):
            result.succeeded += row["type"] == "succeeded"
            result.errored += row["type"] != "succeeded"
            result.cost_usd += Decimal(str(row.get("cost_usd") or 0))
            result.pairs.extend(row.get("pairs", []))
        return result

    price = cheap_price(ledger)
    rows = []
    for item in client.messages.batches.results(batch_id):
        cid = str(api.get(item, "custom_id"))
        res = api.get(item, "result")
        rtype = str(api.get(res, "type"))
        row: dict[str, Any] = {
            "custom_id": cid,
            "intent": intent_of_custom_id(cid),
            "type": rtype,
        }
        if rtype == "succeeded":
            message = api.get(res, "message")
            counts = api.usage_counts(api.get(message, "usage"))
            cost = ledger.charge(price.key, batch=True, milestone=MILESTONE, **counts)
            text = api.text_of(message)
            pairs = parse_pairs(text)
            row.update(
                {
                    "stop_reason": api.get(message, "stop_reason"),
                    "usage": counts,
                    "cost_usd": float(cost),
                    "text": text,
                    "pairs": pairs,
                }
            )
            result.succeeded += 1
            result.cost_usd += cost
            result.pairs.extend({**p, "intent": row["intent"], "custom_id": cid} for p in pairs)
        else:
            error = api.get(res, "error")
            row["error"] = str(api.get(error, "type", error)) if error is not None else None
            result.errored += 1
        rows.append(row)

    raw_dir.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in sorted(rows, key=lambda r: r["custom_id"]):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    registry.append(
        "collected",
        batch_id,
        kind=KIND,
        succeeded=result.succeeded,
        errored=result.errored,
        cost_usd=float(result.cost_usd),
    )
    return result


def read_raw(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def raw_candidates(raw_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Todos os pares crus de `results/gen/raw/*.jsonl`, com intenção e custom_id."""
    raw_dir = config.resolve_path(raw_dir or RAW_RELDIR)
    out = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        for row in read_raw(path):
            if row.get("type") != "succeeded":
                continue
            for p in row.get("pairs", []):
                out.append({**p, "intent": row["intent"], "custom_id": row["custom_id"]})
    return out


def sample_pairs(pairs: list[dict[str, Any]], n: int = 10, seed: int = 0) -> list[dict[str, Any]]:
    if len(pairs) <= n:
        return list(pairs)
    return random.Random(seed).sample(pairs, n)
