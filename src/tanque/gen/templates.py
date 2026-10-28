# ruff: noqa: E501
# (as redações das perguntas ficam numa linha só, para serem lidas como frases)
"""Treino sintético por templates parametrizados, sem API paga (M4, D23).

Para cada uma das 40 intenções de `tanque.gen.intents` há um gerador que:

1. sorteia parâmetros **reais** do banco (UF, região, município + UF que existe,
   capital, produto, bandeira com a grafia exata, bairro, ano, mês, intervalo de
   meses, N do top-N) com um `random.Random` de seed fixa;
2. preenche a SQL seguindo as convenções do gold (`docs/review-guide.md`, seção
   3, e D22): preço médio = `AVG(valor_venda)` sem `ROUND`; colunas = entidade
   (+ valor quando a pergunta pede valor ou ranking), só a entidade em "em
   quais"; município filtrado por `nome` + `uf`; mês = `date_trunc('month', ...)`;
   "último mês" = setembro de 2026; sem período = o snapshot todo; regra dos 70%
   com a `GASOLINA` comum;
3. escreve a pergunta escolhendo entre várias redações (formal, informal, curta,
   com "vc"/"qto"), com nomes de UF e região por extenso e cidades com acento;
   ~10% das perguntas recebem ruído leve (sem acento, minúsculas ou um erro de
   digitação de uma letra), com a tag `typo`.

Cada candidato é executado no banco (conexão somente leitura) antes de entrar
na saída. São recusados, com o motivo contado por intenção:

| motivo | regra |
|---|---|
| `vazio` | a consulta não devolve linhas |
| `nulo` | alguma célula do resultado é NULL |
| `zero` | a resposta é um único 0 (contagem ou diferença trivial) |
| `muitas_linhas` | mais de `MAX_RESULT_ROWS` linhas |
| `poucas_linhas` | ranking com menos linhas que o N pedido |
| `empate` | empate exato entre vizinhos do ranking ou no corte do `LIMIT` |
| `sinal` | "maiores altas"/"maiores quedas" ou "a mais" com o sinal errado |
| `repetido` | pergunta ou SQL normalizada já gerada |
| `sem_parametro` | o sorteio não achou parâmetro válido (ex.: cidade sem bairros) |

O empate é verificado como no M3: a consulta roda sem o `LIMIT` (a "sonda") e
os valores na posição k e k+1, e entre linhas vizinhas do top-k, precisam ser
diferentes.

A saída crua (`data/gen/templates_raw.jsonl`, ignorado pelo git) passa depois
pelo mesmo pipeline do caminho Batch: validação por execução, filtro contra
dev/teste e deduplicação interna (aqui por pergunta ou SQL normalizada idêntica,
D23). Nenhum dado de dev/teste é lido neste módulo.
"""

from __future__ import annotations

import datetime as dt
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb

from tanque import config
from tanque.gen import intents as intents_mod
from tanque.gen import leakage

RAW_RELPATH = Path("data") / "gen" / "templates_raw.jsonl"
NOISE_RATE = 0.10
MAX_RESULT_ROWS = 100
# Tentativas seguidas sem sucesso até considerar a intenção esgotada.
MAX_CONSECUTIVE_FAILURES = 400

SNAPSHOT_START = dt.date(2024, 10, 1)
SNAPSHOT_END = dt.date(2026, 10, 1)  # exclusivo: o snapshot vai até 2026-09-30

MONTHS_PT = (
    "janeiro",
    "fevereiro",
    "março",
    "abril",
    "maio",
    "junho",
    "julho",
    "agosto",
    "setembro",
    "outubro",
    "novembro",
    "dezembro",
)
ORDINAL_SEMESTER = {1: ("primeiro", "1º"), 2: ("segundo", "2º")}
ORDINAL_QUARTER = {
    1: ("primeiro", "1º"),
    2: ("segundo", "2º"),
    3: ("terceiro", "3º"),
    4: ("quarto", "4º"),
}

# Nome por extenso e artigo ("o" -> "no"/"do", "a" -> "na"/"da", "" -> "em"/"de").
# SP e RJ levam "estado" para não confundir com a capital.
UF_NAMES: dict[str, tuple[str, str]] = {
    "AC": ("Acre", "o"),
    "AL": ("Alagoas", ""),
    "AP": ("Amapá", "o"),
    "AM": ("Amazonas", "o"),
    "BA": ("Bahia", "a"),
    "CE": ("Ceará", "o"),
    "DF": ("Distrito Federal", "o"),
    "ES": ("Espírito Santo", "o"),
    "GO": ("Goiás", ""),
    "MA": ("Maranhão", "o"),
    "MT": ("Mato Grosso", ""),
    "MS": ("Mato Grosso do Sul", ""),
    "MG": ("Minas Gerais", ""),
    "PA": ("Pará", "o"),
    "PB": ("Paraíba", "a"),
    "PR": ("Paraná", "o"),
    "PE": ("Pernambuco", ""),
    "PI": ("Piauí", "o"),
    "RJ": ("estado do Rio de Janeiro", "o"),
    "RN": ("Rio Grande do Norte", "o"),
    "RS": ("Rio Grande do Sul", "o"),
    "RO": ("Rondônia", ""),
    "RR": ("Roraima", ""),
    "SC": ("Santa Catarina", ""),
    "SP": ("estado de São Paulo", "o"),
    "SE": ("Sergipe", ""),
    "TO": ("Tocantins", "o"),
}

REGION_TEXT: dict[str, tuple[tuple[str, str], ...]] = {
    "N": (("Norte", "o"), ("região Norte", "a")),
    "NE": (("Nordeste", "o"), ("região Nordeste", "a")),
    "CO": (("Centro-Oeste", "o"), ("região Centro-Oeste", "a")),
    "SE": (("Sudeste", "o"), ("região Sudeste", "a")),
    "S": (("Sul", "o"), ("região Sul", "a")),
}

# Capitais cujo nome é igual ao do estado.
_CITY_SPECIAL = {
    ("SAO PAULO", "SP"): (
        ("na cidade de São Paulo", "da cidade de São Paulo"),
        ("na capital paulista", "da capital paulista"),
    ),
    ("RIO DE JANEIRO", "RJ"): (
        ("na cidade do Rio de Janeiro", "da cidade do Rio de Janeiro"),
        ("na capital fluminense", "da capital fluminense"),
    ),
}

# Palavras dos nomes da ANP (maiúsculas, sem acento) com a grafia acentuada.
_ACCENTED_WORDS = {
    "ACAILANDIA": "Açailândia", "AGUAS": "Águas", "GOIAS": "Goiás", "MARANHAO": "Maranhão",
    "ANAPOLIS": "Anápolis", "GOIANIA": "Goiânia", "ARACATUBA": "Araçatuba",
    "ARAGUAINA": "Araguaína", "ARARANGUA": "Araranguá", "ARAUCARIA": "Araucária",
    "ARAXA": "Araxá", "AVARE": "Avaré", "BAGE": "Bagé", "PIRAI": "Piraí", "BELEM": "Belém",
    "GONCALVES": "Gonçalves", "BRAGANCA": "Bragança", "BRASILIA": "Brasília",
    "CACADOR": "Caçador", "CACAPAVA": "Caçapava", "CACERES": "Cáceres", "CAETITE": "Caetité",
    "CAICO": "Caicó", "CAMBE": "Cambé", "MOURAO": "Mourão", "CANINDE": "Canindé",
    "CARAPICUIBA": "Carapicuíba", "CATALAO": "Catalão", "CHAPECO": "Chapecó", "CODO": "Codó",
    "CONCEICAO": "Conceição", "CONCORDIA": "Concórdia", "CORNELIO": "Cornélio",
    "PROCOPIO": "Procópio", "CORUMBA": "Corumbá", "COSMOPOLIS": "Cosmópolis",
    "CRATEUS": "Crateús", "CRICIUMA": "Criciúma", "CUBATAO": "Cubatão", "CUIABA": "Cuiabá",
    "DIVINOPOLIS": "Divinópolis", "EUNAPOLIS": "Eunápolis", "FLORIANOPOLIS": "Florianópolis",
    "IGUACU": "Iguaçu", "BELTRAO": "Beltrão", "GARCA": "Garça", "GAVIAO": "Gavião",
    "GRAVATAI": "Gravataí", "GUAIBA": "Guaíba", "GUARATINGUETA": "Guaratinguetá",
    "GUARUJA": "Guarujá", "GUAXUPE": "Guaxupé", "HORTOLANDIA": "Hortolândia", "ICO": "Icó",
    "IJUI": "Ijuí", "ILHEUS": "Ilhéus", "IPIRA": "Ipirá", "IRECE": "Irecê",
    "ITABORAI": "Itaboraí", "ITAJAI": "Itajaí", "ITAJUBA": "Itajubá", "ITANHAEM": "Itanhaém",
    "ITAPOLIS": "Itápolis", "ITAUNA": "Itaúna", "JABOATAO": "Jaboatão", "JACAREI": "Jacareí",
    "JANAUBA": "Janaúba", "JANUARIA": "Januária", "JAU": "Jaú", "JEQUIE": "Jequié",
    "PARANA": "Paraná", "JOSE": "José", "BONIFACIO": "Bonifácio", "JUNDIAI": "Jundiaí",
    "LUZIANIA": "Luziânia", "MACAE": "Macaé", "MACAPA": "Macapá", "MACEIO": "Maceió",
    "MARABA": "Marabá", "MARICA": "Maricá", "MARILIA": "Marília", "MARINGA": "Maringá",
    "MATAO": "Matão", "MAUA": "Mauá", "GUACU": "Guaçu", "MOSSORO": "Mossoró",
    "MURIAE": "Muriaé", "NILOPOLIS": "Nilópolis", "NITEROI": "Niterói", "OLIMPIA": "Olímpia",
    "OSORIO": "Osório", "INDIOS": "Índios", "PARA": "Pará", "PARAGUACU": "Paraguaçu",
    "PARAISO": "Paraíso", "PARANAGUA": "Paranaguá", "PARANAVAI": "Paranavaí",
    "PARNAIBA": "Parnaíba", "PATROCINIO": "Patrocínio", "PAULINIA": "Paulínia",
    "PETROPOLIS": "Petrópolis", "POA": "Poá", "POCOES": "Poções", "POCOS": "Poços",
    "PORA": "Porã", "QUIXADA": "Quixadá", "RIBEIRAO": "Ribeirão",
    "RONDONOPOLIS": "Rondonópolis", "SABARA": "Sabará", "SANTAREM": "Santarém",
    "ANDRE": "André", "ANGELO": "Ângelo", "ANTONIO": "Antônio", "PADUA": "Pádua",
    "SAO": "São", "GONCALO": "Gonçalo", "JOAO": "João", "LUIS": "Luís",
    "SEBASTIAO": "Sebastião", "SERTAOZINHO": "Sertãozinho", "SUMARE": "Sumaré",
    "TABOAO": "Taboão", "TATUI": "Tatuí", "TAUBATE": "Taubaté", "TEOFILO": "Teófilo",
    "TIMOTEO": "Timóteo", "TRES": "Três", "CORACOES": "Corações", "TUBARAO": "Tubarão",
    "TUPA": "Tupã", "UBA": "Ubá", "UBERLANDIA": "Uberlândia", "UNAI": "Unaí",
    "UNIAO": "União", "VITORIA": "Vitória", "VALENCA": "Valença", "VALPARAISO": "Valparaíso",
    "VARZEA": "Várzea", "VIAMAO": "Viamão", "VICOSA": "Viçosa", "ANTAO": "Antão",
    "XANXERE": "Xanxerê", "PIAUI": "Piauí", "AMAPA": "Amapá", "CEARA": "Ceará",
}  # fmt: skip
_LOWER_WORDS = {"DE", "DA", "DO", "DOS", "DAS", "E", "DEL"}

# Bandeiras: grafia exata do banco -> nome na pergunta. "SP" fica de fora (confunde com a UF).
BAND_NAMES: dict[str, str] = {
    "VIBRA": "Vibra",
    "IPIRANGA": "Ipiranga",
    "RAIZEN": "Raízen",
    "ALE": "ALE",
    "SABBÁ": "Sabbá",
    "RODOIL": "Rodoil",
    "RAIZEN MIME": "Raízen Mime",
    "ATEM' S": "Atem's",
    "NEXTA": "Nexta",
    "EQUADOR": "Equador",
    "CHARRUA": "Charrua",
    "TAURUS": "Taurus",
    "DISLUB": "Dislub",
    "STANG": "Stang",
    "ATLÂNTICA": "Atlântica",
    "TEMAPE": "Temape",
    "MAXSUL": "Maxsul",
    "LARCO": "Larco",
}
BIG_BANDS = ("VIBRA", "IPIRANGA", "RAIZEN", "ALE")

PRODUCTS = ("GASOLINA", "ETANOL", "DIESEL S10", "DIESEL", "GASOLINA ADITIVADA", "GNV")
PRODUCT_WEIGHTS = (26, 22, 16, 14, 12, 10)
# (texto, feminino, peso)
PRODUCT_TEXT: dict[str, tuple[tuple[str, bool, int], ...]] = {
    "GASOLINA": (("gasolina", True, 8), ("gasolina comum", True, 2)),
    "GASOLINA ADITIVADA": (("gasolina aditivada", True, 1),),
    "ETANOL": (("etanol", False, 8), ("álcool", False, 2)),
    "DIESEL": (("diesel", False, 7), ("diesel comum", False, 3)),
    "DIESEL S10": (("diesel S10", False, 7), ("S10", False, 2)),
    "GNV": (("GNV", False, 8), ("gás natural veicular", False, 1)),
}
PRODUCT_TAG = {
    "GASOLINA": "gasolina",
    "GASOLINA ADITIVADA": "aditivada",
    "ETANOL": "etanol",
    "DIESEL": "diesel",
    "DIESEL S10": "diesel_s10",
    "GNV": "gnv",
}
PRODUCT_ALIAS = {
    "GASOLINA": "gasolina",
    "GASOLINA ADITIVADA": "gasolina_aditivada",
    "ETANOL": "etanol",
    "DIESEL": "diesel",
    "DIESEL S10": "diesel_s10",
    "GNV": "gnv",
}

J_P = "JOIN postos p ON p.posto_id = c.posto_id"
J_M = "JOIN municipios m ON m.municipio_id = p.municipio_id"
RATIO = (
    "AVG(CASE WHEN c.produto = 'ETANOL' THEN c.valor_venda END) / "
    "AVG(CASE WHEN c.produto = 'GASOLINA' THEN c.valor_venda END)"
)
E_G = "c.produto IN ('ETANOL', 'GASOLINA')"


class Skip(Exception):
    """O sorteio não deu um candidato utilizável; tente de novo."""


# --- textos ------------------------------------------------------------------


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    out = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return unicodedata.normalize("NFC", out)


def city_display(nome: str) -> str:
    """Nome da ANP ('SAO JOSE DOS CAMPOS') como uma pessoa escreveria ('São José dos Campos')."""

    def word(w: str, first: bool) -> str:
        if "-" in w:
            return "-".join(word(part, first and i == 0) for i, part in enumerate(w.split("-")))
        if not first and w in _LOWER_WORDS:
            return w.lower()
        return _ACCENTED_WORDS.get(w, w.capitalize())

    return " ".join(word(w, i == 0) for i, w in enumerate(nome.split()))


def say(text: str, cap: bool = True) -> str:
    """Limpa espaços e pontuação de uma pergunta montada por partes."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([?.,:;!)])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r",([?.])", r"\1", text)
    if cap and text:
        text = text[0].upper() + text[1:]
    return text


def money(value: float) -> str:
    return "R$ " + f"{value:.2f}".replace(".", ",")


def sql_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]+")


def typo(text: str, rng: random.Random) -> str:
    """Um erro de digitação de uma letra numa palavra comum (>= 5 letras, não sigla)."""
    words = [
        m
        for m in _WORD_RE.finditer(text)
        if len(m.group()) >= 5
        and not m.group().isupper()
        and not any(c.isdigit() for c in m.group())
    ]
    if not words:
        return text
    m = rng.choice(words)
    w = m.group()
    i = rng.randrange(1, len(w) - 1)
    op = rng.choice(("apaga", "troca", "dobra"))
    if op == "apaga":
        new = w[:i] + w[i + 1 :]
    elif op == "troca" and w[i] != w[i + 1]:
        new = w[:i] + w[i + 1] + w[i] + w[i + 2 :]
    else:
        new = w[: i + 1] + w[i] + w[i + 1 :]
    return text[: m.start()] + new + text[m.end() :]


def add_noise(text: str, rng: random.Random) -> str:
    kind = rng.choice(("sem_acento", "minusculas", "typo", "typo"))
    if kind == "sem_acento" and strip_accents(text) != text:
        return strip_accents(text)
    if kind == "minusculas":
        out = text.lower().rstrip("?")
        if out != text:
            return out
    return typo(text, rng)


# --- períodos ----------------------------------------------------------------


def _add_months(d: dt.date, n: int) -> dt.date:
    k = d.year * 12 + d.month - 1 + n
    return dt.date(k // 12, k % 12 + 1, 1)


def _month_text(d: dt.date) -> str:
    return f"{MONTHS_PT[d.month - 1]} de {d.year}"


@dataclass(frozen=True)
class Period:
    kind: str
    start: dt.date | None
    end: dt.date | None  # exclusivo
    texts: tuple[str, ...]  # "em 2025", "no último mês"...
    de_texts: tuple[str, ...] = ()  # "de 2025", "do primeiro semestre de 2025"...
    tag: str = ""

    def pred(self, col: str = "c.data_coleta") -> str:
        if self.start is None:
            return ""
        return f"{col} >= DATE '{self.start.isoformat()}' AND {col} < DATE '{self.end.isoformat()}'"

    def months(self) -> int:
        start = max(self.start or SNAPSHOT_START, SNAPSHOT_START)
        end = min(self.end or SNAPSHOT_END, SNAPSHOT_END)
        return (end.year - start.year) * 12 + end.month - start.month

    def text(self, rng: random.Random) -> str:
        return rng.choice(self.texts)

    def de(self, rng: random.Random) -> str:
        return rng.choice(self.de_texts) if self.de_texts else ""


def whole_period() -> Period:
    return Period(
        "none",
        None,
        None,
        ("", "", "", "", "em todo o período", "considerando todos os dados"),
        ("",),
        "",
    )


def last_month_period(ref_month: dt.date = dt.date(2026, 9, 1)) -> Period:
    return Period(
        "last_month",
        ref_month,
        _add_months(ref_month, 1),
        ("no último mês", "no último mês", "no mês mais recente", "no último mês dos dados"),
        (),
        "ultimo_mes",
    )


def year_period(year: int) -> Period:
    return Period(
        "year",
        dt.date(year, 1, 1),
        dt.date(year + 1, 1, 1),
        (f"em {year}", f"em {year}", f"no ano de {year}", f"ao longo de {year}"),
        (f"de {year}",),
        "ano",
    )


def month_period(year: int, month: int) -> Period:
    d = dt.date(year, month, 1)
    name = MONTHS_PT[month - 1]
    return Period(
        "month",
        d,
        _add_months(d, 1),
        (f"em {name} de {year}", f"em {name} de {year}", f"no mês de {name} de {year}",
         f"em {name}/{year}", f"em {month:02d}/{year}"),
        (),
        "mes",
    )  # fmt: skip


def month_range_period(year: int, first: int, last: int) -> Period:
    a, b = MONTHS_PT[first - 1], MONTHS_PT[last - 1]
    texts = (f"entre {a} e {b} de {year}", f"de {a} a {b} de {year}")
    return Period(
        "range", dt.date(year, first, 1), _add_months(dt.date(year, last, 1), 1), texts, texts,
        "intervalo",
    )  # fmt: skip


def semester_period(year: int, sem: int) -> Period:
    word, num = ORDINAL_SEMESTER[sem]
    start = dt.date(year, 1 if sem == 1 else 7, 1)
    return Period(
        "semester",
        start,
        _add_months(start, 6),
        (f"no {word} semestre de {year}", f"no {num} semestre de {year}"),
        (f"do {word} semestre de {year}", f"do {num} semestre de {year}"),
        "semestre",
    )


def quarter_period(year: int, q: int) -> Period:
    word, num = ORDINAL_QUARTER[q]
    start = dt.date(year, 3 * (q - 1) + 1, 1)
    return Period(
        "quarter",
        start,
        _add_months(start, 3),
        (f"no {word} trimestre de {year}", f"no {num} trimestre de {year}"),
        (f"do {word} trimestre de {year}", f"do {num} trimestre de {year}"),
        "trimestre",
    )


def last_n_months_period(n: int, ref_month: dt.date = dt.date(2026, 9, 1)) -> Period:
    word = {12: "doze", 6: "seis", 3: "três"}[n]
    return Period(
        f"last{n}",
        _add_months(ref_month, 1 - n),
        _add_months(ref_month, 1),
        (f"nos últimos {n} meses", f"nos últimos {n} meses", f"nos últimos {word} meses"),
        (f"dos últimos {n} meses",),
        f"ultimos_{n}_meses",
    )


# Pesos dos tipos de período.
PERIODS_ANY = {
    "none": 22, "last_month": 20, "year": 18, "month": 14, "range": 10, "semester": 6,
    "quarter": 4, "last12": 4, "last6": 2,
}  # fmt: skip
PERIODS_MULTI = {
    "none": 15, "year": 30, "range": 20, "semester": 12, "quarter": 8, "last12": 10, "last6": 5,
}  # fmt: skip


def _in_snapshot(start: dt.date, end: dt.date) -> bool:
    return SNAPSHOT_START <= start and end <= SNAPSHOT_END


def sample_period(rng: random.Random, weights: dict[str, int], min_months: int = 1) -> Period:
    for _ in range(50):
        kind = rng.choices(list(weights), weights=list(weights.values()))[0]
        if kind == "none":
            p = whole_period()
        elif kind == "last_month":
            p = last_month_period()
        elif kind == "year":
            p = year_period(rng.choice((2025, 2026)))
        elif kind == "month":
            d = _add_months(SNAPSHOT_START, rng.randrange(24))
            p = month_period(d.year, d.month)
        elif kind == "range":
            year = rng.choice((2025, 2025, 2026, 2026, 2024))
            lo, hi = (10, 12) if year == 2024 else (1, 12 if year == 2025 else 9)
            first = rng.randint(lo, hi - 1)
            last = rng.randint(first + 1, min(hi, first + 6))
            p = month_range_period(year, first, last)
        elif kind == "semester":
            year, sem = rng.choice(((2025, 1), (2025, 2), (2026, 1)))
            p = semester_period(year, sem)
        elif kind == "quarter":
            year, q = rng.choice(
                ((2024, 4), (2025, 1), (2025, 2), (2025, 3), (2025, 4), (2026, 1), (2026, 2),
                 (2026, 3))
            )  # fmt: skip
            p = quarter_period(year, q)
        elif kind == "last12":
            p = last_n_months_period(12)
        else:
            p = last_n_months_period(6)
        if p.start is not None and not _in_snapshot(p.start, min(p.end, SNAPSHOT_END)):
            continue
        if p.months() >= min_months:
            return p
    raise Skip("período")


# --- lugares e produtos ------------------------------------------------------


def _de_art(phrase: str) -> str:
    """'o etanol' -> 'do etanol'; 'a comum' -> 'da comum'."""
    art, _, rest = phrase.partition(" ")
    return {"o": "do", "a": "da"}.get(art, "de " + art) + " " + rest


def _em(article: str) -> str:
    return {"o": "no", "a": "na", "": "em"}[article]


def _de(article: str) -> str:
    return {"o": "do", "a": "da", "": "de"}[article]


@dataclass(frozen=True)
class Loc:
    kind: str  # brasil, regiao, uf, municipio, capitais
    preds: tuple[str, ...]
    em: str  # "em Goiânia", "no Paraná"
    de: str  # "de Goiânia", "do Paraná"
    tag: str
    uf: str | None = None
    nome: str | None = None
    regiao: str | None = None

    @property
    def needs_m(self) -> bool:
        return bool(self.preds)


@dataclass(frozen=True)
class Muni:
    nome: str
    uf: str
    regiao: str
    capital: bool
    postos: int


@dataclass(frozen=True)
class Prod:
    code: str
    text: str
    fem: bool

    @property
    def o(self) -> str:
        return ("a " if self.fem else "o ") + self.text

    @property
    def do(self) -> str:
        return ("da " if self.fem else "do ") + self.text

    @property
    def pelo(self) -> str:
        return ("pela " if self.fem else "pelo ") + self.text

    @property
    def art(self) -> str:
        return "a" if self.fem else "o"

    def adj(self, desc: bool) -> str:
        return ("cara" if self.fem else "caro") if desc else ("barata" if self.fem else "barato")

    @property
    def pred(self) -> str:
        return f"c.produto = {sql_str(self.code)}"

    @property
    def tag(self) -> str:
        return PRODUCT_TAG[self.code]


@dataclass
class Catalog:
    ufs: list[str]
    uf_region: dict[str, str]
    region_ufs: dict[str, list[str]]
    municipios: list[Muni]
    bairros: dict[tuple[str, str], int]  # (nome, uf) -> bairros distintos não nulos
    bandeiras: list[str]  # das BAND_NAMES, as que existem no banco

    def __post_init__(self) -> None:
        counts = Counter(m.nome for m in self.municipios)
        self.homonyms = {n for n, k in counts.items() if k > 1}
        self.by_uf: dict[str, list[Muni]] = defaultdict(list)
        for m in self.municipios:
            self.by_uf[m.uf].append(m)
        self.capitals = [m for m in self.municipios if m.capital]


def load_catalog(con: duckdb.DuckDBPyConnection) -> Catalog:
    rows = con.execute(
        "SELECT m.nome, m.uf, m.regiao, m.capital, COUNT(p.posto_id) "
        "FROM municipios m LEFT JOIN postos p ON p.municipio_id = m.municipio_id "
        "GROUP BY ALL ORDER BY m.uf, m.nome"
    ).fetchall()
    municipios = [Muni(n, u, r, bool(c), int(k)) for n, u, r, c, k in rows]
    uf_region = {m.uf: m.regiao for m in municipios}
    region_ufs: dict[str, list[str]] = defaultdict(list)
    for uf, reg in sorted(uf_region.items()):
        region_ufs[reg].append(uf)
    bairros = {
        (n, u): int(k)
        for n, u, k in con.execute(
            "SELECT m.nome, m.uf, COUNT(DISTINCT p.bairro) FROM postos p "
            "JOIN municipios m ON m.municipio_id = p.municipio_id "
            "WHERE p.bairro IS NOT NULL GROUP BY ALL"
        ).fetchall()
    }
    present = {b for (b,) in con.execute("SELECT DISTINCT bandeira FROM postos").fetchall()}
    bandeiras = [b for b in BAND_NAMES if b in present]
    return Catalog(sorted(uf_region), uf_region, dict(region_ufs), municipios, bairros, bandeiras)


# --- sondas e checagens ------------------------------------------------------


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float | Decimal):
        return float(value)
    return None


def _tie(a: Any, b: Any) -> bool:
    x, y = _num(a), _num(b)
    if x is None or y is None:
        return a == b
    return abs(x - y) <= 1e-9 * max(1.0, abs(x), abs(y))


def rank_problem(
    rows: Sequence[Sequence[Any]],
    n: int | None,
    value_index: int,
    group_index: int | None = None,
    sign: int = 0,
) -> str | None:
    """Problema de um ranking pela sonda (consulta sem LIMIT, já ordenada).

    `n=None`: a pergunta pede a lista inteira ordenada (todos os vizinhos contam).
    `sign=+1/-1`: os valores do top-n precisam ser positivos/negativos.
    """
    groups: dict[Any, list[Sequence[Any]]] = {}
    for row in rows:
        groups.setdefault(row[group_index] if group_index is not None else None, []).append(row)
    if not groups:
        return "vazio"
    for grp in groups.values():
        k = len(grp) if n is None else n
        if len(grp) < k:
            return "poucas_linhas"
        for i in range(min(k, len(grp) - 1)):
            if _tie(grp[i][value_index], grp[i + 1][value_index]):
                return "empate"
        if sign:
            for row in grp[:k]:
                v = _num(row[value_index])
                if v is None or v * sign <= 0:
                    return "sinal"
    return None


def result_problem(rows: Sequence[Sequence[Any]]) -> str | None:
    if not rows:
        return "vazio"
    if any(v is None for row in rows for v in row):
        return "nulo"
    if len(rows) == 1 and len(rows[0]) == 1 and _num(rows[0][0]) == 0:
        return "zero"
    if len(rows) > MAX_RESULT_ROWS:
        return "muitas_linhas"
    return None


@dataclass(frozen=True)
class Probe:
    sql: str
    n: int | None
    value_index: int
    group_index: int | None = None
    sign: int = 0


def positive_last_column(rows: Sequence[Sequence[Any]]) -> str | None:
    for row in rows:
        v = _num(row[-1])
        if v is None or v <= 0:
            return "sinal"
    return None


@dataclass
class Cand:
    question: str
    sql: str
    order_matters: bool
    tags: list[str] = field(default_factory=list)
    probe: Probe | None = None
    check: Callable[[Sequence[Sequence[Any]]], str | None] | None = None


# --- contexto de sorteio -----------------------------------------------------


class Ctx:
    def __init__(self, rng: random.Random, cat: Catalog, con: duckdb.DuckDBPyConnection):
        self.rng = rng
        self.cat = cat
        self.con = con

    # sorteios básicos
    def pick(self, seq: Sequence[Any]) -> Any:
        if not seq:
            raise Skip("sem_parametro")
        return self.rng.choice(seq)

    def chance(self, p: float) -> bool:
        return self.rng.random() < p

    def query(self, sql: str) -> list[tuple[Any, ...]]:
        return self.con.execute(sql).fetchall()

    def prod(self, codes: Sequence[str] = PRODUCTS) -> Prod:
        weights = [PRODUCT_WEIGHTS[PRODUCTS.index(c)] for c in codes]
        code = self.rng.choices(list(codes), weights=weights)[0]
        return self.prod_text(code)

    def prod_text(self, code: str) -> Prod:
        forms = PRODUCT_TEXT[code]
        text, fem, _ = self.rng.choices(forms, weights=[f[2] for f in forms])[0]
        return Prod(code, text, fem)

    def period(self, weights: dict[str, int] = PERIODS_ANY, min_months: int = 1) -> Period:
        return sample_period(self.rng, weights, min_months)

    def n(self, options: Sequence[int] = (3, 5, 5, 10), limit: int | None = None) -> int:
        opts = [k for k in options if limit is None or k < limit]
        if not opts:
            raise Skip("sem_parametro")
        return self.pick(opts)

    # lugares
    def brasil(self, allow_empty: bool = True) -> Loc:
        ems = ["no Brasil", "no país", "em todo o Brasil"] + ([""] if allow_empty else [])
        return Loc("brasil", (), self.pick(ems), self.pick(["do Brasil", "do país"]), "brasil")

    def uf(self, uf: str | None = None) -> Loc:
        uf = uf or self.pick(self.cat.ufs)
        name, art = UF_NAMES[uf]
        if self.chance(0.45):
            sart = "" if uf == "SP" else art
            em, de = f"{_em(sart)} {uf}", f"{_de(sart)} {uf}"
            if self.chance(0.15):
                em, de = f"no estado de {uf}", f"do estado de {uf}"
        else:
            em, de = f"{_em(art)} {name}", f"{_de(art)} {name}"
        reg = self.cat.uf_region[uf]
        return Loc("uf", (f"m.uf = '{uf}'",), em, de, "uf", uf=uf, regiao=reg)

    def regiao(self, reg: str | None = None) -> Loc:
        reg = reg or self.pick(sorted(self.cat.region_ufs))
        name, art = self.pick(REGION_TEXT[reg])
        return Loc(
            "regiao",
            (f"m.regiao = '{reg}'",),
            f"{_em(art)} {name}",
            f"{_de(art)} {name}",
            "regiao",
            regiao=reg,
        )

    def municipio(
        self, min_postos: int = 1, capital: bool | None = None, uf: str | None = None
    ) -> Loc:
        pool = [
            m
            for m in self.cat.municipios
            if m.postos >= min_postos
            and (capital is None or m.capital == capital)
            and (uf is None or m.uf == uf)
        ]
        if not pool:
            raise Skip("sem_parametro")
        m = self.pick(pool)
        return self.municipio_loc(m)

    def municipio_loc(self, m: Muni) -> Loc:
        d = city_display(m.nome)
        special = _CITY_SPECIAL.get((m.nome, m.uf))
        if special and self.chance(0.8):
            em, de = self.pick(special)
        else:
            forms = [
                (f"em {d} ({m.uf})", f"de {d} ({m.uf})"),
                (f"em {d}/{m.uf}", f"de {d}/{m.uf}"),
                (f"em {d}, {m.uf},", f"de {d}, {m.uf},"),
                (f"em {d} - {m.uf}", f"de {d} - {m.uf}"),
            ]
            if m.nome not in self.cat.homonyms and not special:
                forms += [(f"em {d}", f"de {d}")] * 4
            em, de = self.pick(forms)
        preds = (f"m.nome = {sql_str(m.nome)}", f"m.uf = '{m.uf}'")
        return Loc("municipio", preds, em, de, "municipio", uf=m.uf, nome=m.nome, regiao=m.regiao)

    def loc(self, weights: dict[str, int]) -> Loc:
        kind = self.rng.choices(list(weights), weights=list(weights.values()))[0]
        if kind == "brasil":
            return self.brasil()
        if kind == "uf":
            return self.uf()
        if kind == "regiao":
            return self.regiao()
        if kind == "municipio":
            return self.municipio(min_postos=3)
        if kind == "capital":
            return self.municipio(capital=True)
        if kind == "big_city":
            return self.municipio(min_postos=30)
        raise ValueError(kind)

    # bandeiras
    def bandeira(self, pool: Sequence[str] | None = None, branca: float = 0.2) -> str:
        if self.chance(branca):
            return "BRANCA"
        pool = [b for b in (pool or self.cat.bandeiras) if b in self.cat.bandeiras]
        if not pool:
            raise Skip("sem_parametro")
        weights = [3 if b in BIG_BANDS else 1 for b in pool]
        return self.rng.choices(pool, weights=weights)[0]

    def band_postos(self, b: str) -> str:
        """'postos Ipiranga', 'postos de bandeira branca'..."""
        if b == "BRANCA":
            return self.pick(["postos de bandeira branca", "postos sem bandeira"])
        name = BAND_NAMES[b]
        return self.pick([f"postos {name}", f"postos {name}", f"postos da bandeira {name}"])

    def band_name(self, b: str) -> str:
        return "bandeira branca" if b == "BRANCA" else BAND_NAMES[b]

    def branca_clause(self) -> tuple[str, str]:
        """(predicado, texto) explicitando se a bandeira branca conta como bandeira."""
        if self.chance(0.5):
            return "p.bandeira <> 'BRANCA'", self.pick(
                [
                    ", sem contar os postos de bandeira branca",
                    ", fora a bandeira branca",
                    " (sem contar postos sem bandeira)",
                ]
            )
        return "", self.pick(
            [
                ", contando a bandeira branca como uma bandeira",
                ", incluindo os postos de bandeira branca",
                " (bandeira branca conta como bandeira)",
            ]
        )


# --- montagem de SQL ---------------------------------------------------------


def frm(need_p: bool = False, need_m: bool = False) -> str:
    s = "FROM coletas c"
    if need_p or need_m:
        s += " " + J_P
    if need_m:
        s += " " + J_M
    return s


def where(*conds: str) -> str:
    parts = [c for c in conds if c]
    return (" WHERE " + " AND ".join(parts)) if parts else ""


def order(expr: str, desc: bool) -> str:
    return f" ORDER BY {expr}" + (" DESC" if desc else "")


def ranked(base: str, expr: str, desc: bool, n: int | None, value_index: int, sign: int = 0):
    """(sql, sonda) de um ranking: a sonda é a mesma consulta sem LIMIT."""
    probe_sql = base + order(expr, desc)
    sql = probe_sql + (f" LIMIT {n}" if n else "")
    return sql, Probe(probe_sql, n, value_index, sign=sign)


def _choose(t: Ctx, options: Sequence[str | tuple[str, bool]]) -> str:
    opt = t.pick(options)
    text, cap = (opt, True) if isinstance(opt, str) else opt
    return say(text, cap)


def _tags(*tags: str) -> list[str]:
    out: list[str] = []
    for tg in tags:
        if tg and tg not in out:
            out.append(tg)
    return out


# --- perguntas reaproveitadas --------------------------------------------------


def _media_questions(t: Ctx, P: Prod, where_txt: str, per: str) -> list[Any]:
    return [
        f"Qual o preço médio {P.do} {where_txt} {per}?",
        f"Qual foi o preço médio {P.do} {where_txt} {per}?",
        f"Quanto custou, em média, {P.o} {where_txt} {per}?",
        (f"preço médio {P.do} {where_txt} {per}", t.chance(0.5)),
        (f"qto custou {P.o} em média {where_txt} {per}?", False),
        (f"vc sabe o preço médio {P.do} {where_txt} {per}?", False),
        f"Me diga o valor médio de venda {P.do} {where_txt} {per}.",
        f"Em média, quanto se pagou {P.pelo} {where_txt} {per}?",
    ]


# =============================================================================
# Geradores: fácil
# =============================================================================


def g_media_produto_uf(t: Ctx) -> Cand:
    P, loc, per = t.prod(), t.uf(), t.period()
    sql = f"SELECT AVG(c.valor_venda) AS preco_medio {frm(need_m=True)}" + where(
        P.pred, *loc.preds, per.pred()
    )
    q = _choose(t, _media_questions(t, P, loc.em, per.text(t.rng)))
    return Cand(q, sql, False, _tags(P.tag, loc.tag, per.tag))


def g_media_produto_municipio(t: Ctx) -> Cand:
    P, loc, per = t.prod(), t.municipio(min_postos=2), t.period()
    sql = f"SELECT AVG(c.valor_venda) AS preco_medio {frm(need_m=True)}" + where(
        P.pred, *loc.preds, per.pred()
    )
    q = _choose(t, _media_questions(t, P, loc.em, per.text(t.rng)))
    return Cand(q, sql, False, _tags(P.tag, loc.tag, per.tag))


def g_media_produto_regiao(t: Ctx) -> Cand:
    P, loc, per = t.prod(), t.regiao(), t.period()
    sql = f"SELECT AVG(c.valor_venda) AS preco_medio {frm(need_m=True)}" + where(
        P.pred, *loc.preds, per.pred()
    )
    q = _choose(t, _media_questions(t, P, loc.em, per.text(t.rng)))
    return Cand(q, sql, False, _tags(P.tag, loc.tag, per.tag))


def g_media_nacional_periodo(t: Ctx) -> Cand:
    P = t.prod()
    weights = {k: v for k, v in PERIODS_ANY.items() if k != "none"}
    per = t.period(weights)
    loc = t.brasil(allow_empty=False)
    sql = f"SELECT AVG(c.valor_venda) AS preco_medio {frm()}" + where(P.pred, per.pred())
    opts = _media_questions(t, P, loc.em, per.text(t.rng)) + [
        f"Qual foi o preço médio nacional {P.do} {per.text(t.rng)}?",
        (f"média nacional {P.do} {per.text(t.rng)}", True),
    ]
    q = _choose(t, opts)
    return Cand(q, sql, False, _tags(P.tag, "brasil", per.tag))


def g_extremo_preco(t: Ctx) -> Cand:
    P = t.prod()
    loc = t.loc({"brasil": 2, "uf": 3, "municipio": 3, "regiao": 2})
    per = t.period()
    w, p = loc.em, per.text(t.rng)
    kind = t.pick(("min", "max", "both"))
    body = frm(need_m=loc.needs_m) + where(P.pred, *loc.preds, per.pred())
    if kind == "min":
        sql = f"SELECT MIN(c.valor_venda) AS menor_preco {body}"
        opts = [
            f"Qual foi o menor preço {P.do} registrado {w} {p}?",
            f"Qual o preço mais baixo {P.do} encontrado {w} {p}?",
            (f"menor preço {P.do} {w} {p}", True),
            (f"vc sabe qual foi o menor preço {P.do} coletado {w} {p}?", False),
            f"Qual o valor mínimo de venda {P.do} {w} {p}?",
        ]
    elif kind == "max":
        sql = f"SELECT MAX(c.valor_venda) AS maior_preco {body}"
        opts = [
            f"Qual foi o maior preço {P.do} registrado {w} {p}?",
            f"Qual o preço mais alto {P.do} encontrado {w} {p}?",
            (f"maior preço {P.do} {w} {p}", True),
            (f"qual foi o preço mais alto que pagaram {P.pelo} {w} {p}?", False),
            f"Qual o valor máximo de venda {P.do} {w} {p}?",
        ]
    else:
        sql = f"SELECT MIN(c.valor_venda) AS menor_preco, MAX(c.valor_venda) AS maior_preco {body}"
        opts = [
            f"Quais foram o menor e o maior preço {P.do} registrados {w} {p}?",
            (f"menor e maior preço {P.do} {w} {p}", True),
            f"Me passa o preço mínimo e o máximo {P.do} {w} {p}.",
            (f"qual o menor e o maior valor {P.do} {w} {p}?", False),
        ]
    return Cand(_choose(t, opts), sql, False, _tags(P.tag, loc.tag, per.tag, "min_max"))


def g_conta_postos_local(t: Ctx) -> Cand:
    loc = t.loc({"brasil": 1, "uf": 4, "regiao": 2, "municipio": 4})
    if loc.kind == "brasil":
        loc = t.brasil(allow_empty=False)
    if t.chance(0.6):
        join = " JOIN municipios m ON m.municipio_id = p.municipio_id" if loc.needs_m else ""
        sql = f"SELECT COUNT(*) AS postos FROM postos p{join}" + where(*loc.preds)
        opts = [
            f"Quantos postos {loc.em} aparecem na pesquisa da ANP?",
            f"Quantos postos de combustível foram pesquisados {loc.em}?",
            (f"número de postos {loc.em}", True),
            (f"qtos postos tem {loc.em} nos dados?", False),
            f"Quantas revendas {loc.em} estão no levantamento?",
        ]
        return Cand(_choose(t, opts), sql, False, _tags(loc.tag, "postos"))
    P, per = t.prod(), t.period()
    p = per.text(t.rng)
    sql = (
        "SELECT COUNT(DISTINCT c.posto_id) AS postos "
        + frm(need_m=loc.needs_m)
        + where(P.pred, *loc.preds, per.pred())
    )
    opts = [
        f"Quantos postos {loc.em} tiveram preço {P.do} coletado {p}?",
        f"Em quantos postos {loc.em} houve coleta de {P.text} {p}?",
        (f"qtos postos {loc.em} tiveram {P.text} pesquisado {p}?", False),
        f"Quantos postos diferentes tiveram coleta de {P.text} {loc.em} {p}?",
    ]
    return Cand(_choose(t, opts), sql, False, _tags(P.tag, loc.tag, per.tag, "postos"))


def g_conta_coletas(t: Ctx) -> Cand:
    loc = t.loc({"brasil": 3, "uf": 4, "regiao": 2, "municipio": 3})
    per = t.period()
    P = t.prod() if t.chance(0.75) else None
    dep = f"de {P.text}" if P else ""
    w, p = loc.em, per.text(t.rng)
    sql = (
        "SELECT COUNT(*) AS coletas "
        + frm(need_m=loc.needs_m)
        + where(P.pred if P else "", *loc.preds, per.pred())
    )
    opts = [
        f"Quantas coletas de preço {dep} foram feitas {w} {p}?",
        f"Quantos preços {dep} foram coletados {w} {p}?",
        (f"número de coletas {dep} {w} {p}", True),
        (f"qtas pesquisas de preço {dep} rolaram {w} {p}?", False),
        f"Quantos registros de preço {dep} existem {w} {p}?",
    ]
    return Cand(_choose(t, opts), sql, False, _tags(P.tag if P else "", loc.tag, per.tag))


def g_conta_postos_bandeira(t: Ctx) -> Cand:
    b = t.bandeira(branca=0.25)
    loc = t.loc({"brasil": 2, "uf": 4, "regiao": 2, "big_city": 3})
    if loc.kind == "brasil":
        loc = t.brasil(allow_empty=True)
    noun = t.band_postos(b)
    join = " JOIN municipios m ON m.municipio_id = p.municipio_id" if loc.needs_m else ""
    sql = f"SELECT COUNT(*) AS postos FROM postos p{join}" + where(
        f"p.bandeira = {sql_str(b)}", *loc.preds
    )
    opts = [
        f"Quantos {noun} existem {loc.em}?",
        f"Quantos {noun} aparecem na pesquisa {loc.em}?",
        (f"número de {noun} {loc.em}", True),
        (f"qtos {noun} tem {loc.em}?", False),
        f"Me diz quantos {noun} há {loc.em}.",
    ]
    return Cand(_choose(t, opts), sql, False, _tags(loc.tag, "bandeira"))


def g_media_bandeira(t: Ctx) -> Cand:
    P, b, per = t.prod(), t.bandeira(branca=0.25), t.period()
    loc = t.loc({"brasil": 3, "uf": 3, "regiao": 2, "big_city": 2})
    noun = "nos " + t.band_postos(b)
    w, p = loc.em, per.text(t.rng)
    sql = (
        "SELECT AVG(c.valor_venda) AS preco_medio "
        + frm(need_p=True, need_m=loc.needs_m)
        + (where(P.pred, f"p.bandeira = {sql_str(b)}", *loc.preds, per.pred()))
    )
    opts = [
        f"Qual o preço médio {P.do} {noun} {w} {p}?",
        f"Quanto custou em média {P.o} {noun} {w} {p}?",
        (f"preço médio {P.do} {noun} {w} {p}", True),
        (f"qto saiu {P.o} em média {noun} {w} {p}?", False),
        (f"vc sabe o preço médio {P.do} {noun} {w} {p}?", False),
    ]
    return Cand(_choose(t, opts), sql, False, _tags(P.tag, loc.tag, per.tag, "bandeira"))


def g_media_capitais(t: Ctx) -> Cand:
    P, per = t.prod(), t.period()
    kind = t.pick(("todas", "todas", "fora", "regiao", "uf"))
    if kind == "todas":
        preds = ("m.capital",)
        w = t.pick(["nas capitais", "nas capitais brasileiras", "nas capitais do país"])
    elif kind == "fora":
        preds = ("NOT m.capital",)
        w = t.pick(["fora das capitais", "nas cidades que não são capitais"])
    elif kind == "regiao":
        r = t.regiao()
        preds = ("m.capital", *r.preds)
        w = "nas capitais " + r.de
    else:
        u = t.uf()
        preds = ("m.capital", *u.preds)
        w = "na capital " + u.de
    sql = f"SELECT AVG(c.valor_venda) AS preco_medio {frm(need_m=True)}" + where(
        P.pred, *preds, per.pred()
    )
    q = _choose(t, _media_questions(t, P, w, per.text(t.rng)))
    return Cand(q, sql, False, _tags(P.tag, "capital", per.tag))


def g_conta_municipios(t: Ctx) -> Cand:
    loc = t.uf() if t.chance(0.7) else t.regiao()
    if t.chance(0.45):
        sql = "SELECT COUNT(*) AS municipios FROM municipios m" + where(*loc.preds)
        opts = [
            f"Quantos municípios {loc.de} aparecem nos dados?",
            f"Quantas cidades {loc.em} estão na pesquisa de preços?",
            (f"número de municípios pesquisados {loc.em}", True),
            (f"qtas cidades {loc.de} tem na base?", False),
            f"Em quantos municípios {loc.de} a ANP coletou preços?",
        ]
        return Cand(_choose(t, opts), sql, False, _tags(loc.tag, "municipio"))
    P, per = t.prod(), t.period()
    p = per.text(t.rng)
    sql = (
        "SELECT COUNT(DISTINCT m.municipio_id) AS municipios "
        + frm(need_m=True)
        + where(P.pred, *loc.preds, per.pred())
    )
    opts = [
        f"Em quantas cidades {loc.em} houve coleta de {P.text} {p}?",
        f"Quantos municípios {loc.de} tiveram preço {P.do} pesquisado {p}?",
        (f"número de cidades {loc.em} com coleta de {P.text} {p}", True),
        (f"qtas cidades {loc.em} tiveram {P.text} pesquisado {p}?", False),
    ]
    return Cand(_choose(t, opts), sql, False, _tags(P.tag, loc.tag, per.tag, "municipio"))


def g_media_ultimo_mes(t: Ctx) -> Cand:
    P = t.prod()
    loc = t.loc({"brasil": 3, "uf": 3, "municipio": 3, "regiao": 2})
    per = last_month_period()
    sql = f"SELECT AVG(c.valor_venda) AS preco_medio {frm(need_m=loc.needs_m)}" + where(
        P.pred, *loc.preds, per.pred()
    )
    q = _choose(t, _media_questions(t, P, loc.em, per.text(t.rng)))
    return Cand(q, sql, False, _tags(P.tag, loc.tag, "ultimo_mes"))


def g_amplitude_local(t: Ctx) -> Cand:
    P, per = t.prod(), t.period()
    loc = t.loc({"brasil": 2, "uf": 3, "municipio": 4, "regiao": 2})
    w, p = loc.em, per.text(t.rng)
    sql = (
        "SELECT MAX(c.valor_venda) - MIN(c.valor_venda) AS amplitude "
        + frm(need_m=loc.needs_m)
        + where(P.pred, *loc.preds, per.pred())
    )
    opts = [
        f"Qual a diferença entre o maior e o menor preço {P.do} {w} {p}?",
        f"Qual a diferença entre o preço mais alto e o mais baixo {P.do} registrados {w} {p}?",
        (f"amplitude de preço {P.do} {w} {p}", True),
        (f"qto de diferença tem entre o maior e o menor preço {P.do} {w} {p}?", False),
        f"De quanto foi a distância entre o preço máximo e o mínimo {P.do} {w} {p}?",
    ]
    return Cand(_choose(t, opts), sql, False, _tags(P.tag, loc.tag, per.tag, "amplitude"))


def g_datas_coleta(t: Ctx) -> Cand:
    P = t.prod()
    loc = t.municipio(min_postos=2) if t.chance(0.7) else t.uf()
    per = t.period({"none": 2, "year": 3, "month": 3, "range": 2, "semester": 1, "quarter": 1})
    w, p, dep = loc.em, per.text(t.rng), f"de {P.text}"
    body = frm(need_m=True) + where(P.pred, *loc.preds, per.pred())
    kind = t.pick(("dias", "dias", "ultima", "primeira"))
    if kind == "dias":
        sql = f"SELECT COUNT(DISTINCT c.data_coleta) AS dias {body}"
        opts = [
            f"Em quantos dias diferentes houve coleta {dep} {w} {p}?",
            f"Quantos dias tiveram coleta de preço {dep} {w} {p}?",
            (f"número de dias com coleta {dep} {w} {p}", True),
            (f"em qtos dias rolou coleta {dep} {w} {p}?", False),
        ]
    elif kind == "ultima":
        sql = f"SELECT MAX(c.data_coleta) AS ultima_coleta {body}"
        opts = [
            f"Qual foi a última data com coleta {dep} {w} {p}?",
            f"Quando foi feita a última coleta {dep} {w} {p}?",
            (f"data da última coleta {dep} {w} {p}", True),
            (f"qdo foi a última coleta {dep} {w} {p}?", False),
        ]
    else:
        sql = f"SELECT MIN(c.data_coleta) AS primeira_coleta {body}"
        opts = [
            f"Qual foi a primeira data com coleta {dep} {w} {p}?",
            f"Quando foi feita a primeira coleta {dep} {w} {p}?",
            (f"data da primeira coleta {dep} {w} {p}", True),
            (f"qdo foi a primeira coleta {dep} {w} {p}?", False),
        ]
    return Cand(_choose(t, opts), sql, False, _tags(P.tag, loc.tag, per.tag, "datas"))


# =============================================================================
# Geradores: médio
# =============================================================================


def g_ranking_uf_preco(t: Ctx) -> Cand:
    P, per = t.prod(), t.period()
    desc = t.chance(0.5)
    adj = P.adj(desc)
    if t.chance(0.7):
        scope_preds: tuple[str, ...] = ()
        sc, n_ufs = "", len(t.cat.ufs)
    else:
        r = t.regiao()
        scope_preds, sc, n_ufs = r.preds, r.de, len(t.cat.region_ufs[r.regiao])
    n = 1 if t.chance(0.2) else t.n((3, 5, 5, 10, 3, 2), limit=n_ufs)
    p = per.text(t.rng)
    base = (
        "SELECT m.uf, AVG(c.valor_venda) AS preco_medio "
        + frm(need_m=True)
        + where(P.pred, *scope_preds, per.pred())
        + " GROUP BY m.uf"
    )
    sql, probe = ranked(base, "preco_medio", desc, n, 1)
    alto = "mais alto" if desc else "mais baixo"
    if n == 1:
        opts = [
            f"Qual o estado {sc} com {P.o} mais {adj} {p}?",
            f"Em qual estado {sc} {P.o} foi mais {adj} {p}?",
            (f"qual UF {sc} teve {P.o} mais {adj} {p}?", False),
            (f"estado {sc} com o preço médio {P.do} {alto} {p}", True),
        ]
    else:
        opts = [
            f"Quais os {n} estados {sc} com {P.o} mais {adj} {p}?",
            (f"top {n} UFs {sc} com {P.o} mais {adj} {p}", t.chance(0.5)),
            f"Liste os {n} estados {sc} onde {P.o} foi mais {adj} em média {p}.",
            (f"quais {n} estados {sc} tiveram {P.o} mais {adj} {p}?", False),
            (f"me mostra os {n} estados {sc} com o preço médio {P.do} {alto} {p}", False),
        ]
    return Cand(_choose(t, opts), sql, True, _tags(P.tag, "ranking", "uf", per.tag), probe)


def g_ranking_municipios(t: Ctx) -> Cand:
    P, per = t.prod(), t.period()
    desc = t.chance(0.5)
    adj = P.adj(desc)
    kind = t.pick(("uf", "uf", "regiao", "brasil"))
    if kind == "uf":
        uf = t.pick([u for u in t.cat.ufs if len(t.cat.by_uf[u]) >= 4])
        loc = t.uf(uf)
        cols, group, sc, total = "m.nome", "m.nome", loc.de, len(t.cat.by_uf[uf])
        preds = loc.preds
    else:
        loc = t.regiao() if kind == "regiao" else t.brasil(allow_empty=False)
        cols, group, sc, preds = "m.nome, m.uf", "m.nome, m.uf", loc.de, loc.preds
        total = 1000
    n = 1 if t.chance(0.15) else t.n((3, 5, 5, 10), limit=total)
    vi = 2 if "," in cols else 1
    p = per.text(t.rng)
    base = (
        f"SELECT {cols}, AVG(c.valor_venda) AS preco_medio "
        + frm(need_m=True)
        + where(P.pred, *preds, per.pred())
        + f" GROUP BY {group}"
    )
    sql, probe = ranked(base, "preco_medio", desc, n, vi)
    if n == 1:
        opts = [
            f"Qual a cidade {sc} com {P.o} mais {adj} {p}?",
            f"Em qual município {sc} {P.o} foi mais {adj}, em média, {p}?",
            (f"qual cidade {sc} teve {P.o} mais {adj} {p}?", False),
        ]
    else:
        opts = [
            f"Quais as {n} cidades {sc} com {P.o} mais {adj} {p}?",
            (f"top {n} municípios {sc} com {P.o} mais {adj} {p}", t.chance(0.5)),
            f"Liste os {n} municípios {sc} onde {P.o} foi mais {adj} em média {p}.",
            (f"me fala as {n} cidades {sc} com {P.o} mais {adj} {p}", False),
        ]
    return Cand(_choose(t, opts), sql, True, _tags(P.tag, "ranking", "municipio", per.tag), probe)


def g_ranking_bandeiras(t: Ctx) -> Cand:
    P, per = t.prod(), t.period()
    desc = t.chance(0.5)
    adj = P.adj(desc)
    loc = t.loc({"brasil": 3, "regiao": 2, "uf": 3})
    if loc.kind == "brasil":
        loc = t.brasil(allow_empty=True)
    branca_pred, branca_txt = t.branca_clause()
    having, having_txt = "", ""
    if t.chance(0.3):
        k = t.pick((50, 100, 200))
        having = f" HAVING COUNT(*) >= {k}"
        having_txt = f", considerando só bandeiras com pelo menos {k} coletas"
    n = t.n((3, 5, 5))
    w, p = loc.em, per.text(t.rng)
    base = (
        "SELECT p.bandeira, AVG(c.valor_venda) AS preco_medio "
        + frm(need_p=True, need_m=loc.needs_m)
        + where(P.pred, branca_pred, *loc.preds, per.pred())
        + " GROUP BY p.bandeira"
        + having
    )
    sql, probe = ranked(base, "preco_medio", desc, n, 1)
    caro = "mais caro" if desc else "mais barato"
    opts = [
        f"Quais as {n} bandeiras com {P.o} mais {adj} {w} {p}{branca_txt}{having_txt}?",
        (f"top {n} bandeiras com {P.o} mais {adj} {w} {p}{branca_txt}{having_txt}", True),
        f"Liste as {n} bandeiras que cobraram {caro} {P.pelo} {w} {p}{branca_txt}{having_txt}.",
        (f"quais {n} bandeiras venderam {P.o} {caro} {w} {p}{branca_txt}{having_txt}?", False),
    ]
    return Cand(_choose(t, opts), sql, True, _tags(P.tag, "ranking", "bandeira", per.tag), probe)


def g_media_por_regiao(t: Ctx) -> Cand:
    P, per = t.prod(), t.period()
    p = per.text(t.rng)
    kind = t.pick(("cada", "cada", "ordenado", "ufs"))
    if kind == "ufs":
        r = t.regiao()
        sql = (
            "SELECT m.uf, AVG(c.valor_venda) AS preco_medio "
            + frm(need_m=True)
            + where(P.pred, *r.preds, per.pred())
            + " GROUP BY m.uf"
        )
        opts = [
            f"Qual o preço médio {P.do} em cada estado {r.de} {p}?",
            (f"preço médio {P.do} por UF {r.em} {p}", True),
            (f"qto custou {P.o} em média em cada estado {r.de} {p}?", False),
            f"Mostre o preço médio {P.do} nos estados {r.de} {p}.",
        ]
        return Cand(_choose(t, opts), sql, False, _tags(P.tag, "regiao", "uf", per.tag))
    base = (
        "SELECT m.regiao, AVG(c.valor_venda) AS preco_medio "
        + frm(need_m=True)
        + where(P.pred, per.pred())
        + " GROUP BY m.regiao"
    )
    if kind == "cada":
        opts = [
            f"Qual o preço médio {P.do} em cada região {p}?",
            (f"preço médio {P.do} por região {p}", True),
            f"Compare o preço médio {P.do} entre as regiões do país {p}.",
            (f"qto custou {P.o} em média em cada região {p}?", False),
        ]
        return Cand(_choose(t, opts), base, False, _tags(P.tag, "regiao", per.tag))
    desc = t.chance(0.5)
    sql, probe = ranked(base, "preco_medio", desc, None, 1)
    a, b = ("mais cara", "mais barata") if desc else ("mais barata", "mais cara")
    opts = [
        f"Ordene as regiões da {a} para a {b} pelo preço médio {P.do} {p}.",
        f"Quais regiões têm {P.o} mais {P.adj(desc)}? Ordene da {a} para a {b} {p}.",
        (f"regiões do país da {a} pra {b} no preço {P.do} {p}", False),
        f"Liste as regiões por preço médio {P.do} {p}, da {a} para a {b}.",
    ]
    return Cand(_choose(t, opts), sql, True, _tags(P.tag, "regiao", "ranking", per.tag), probe)


def g_serie_mensal(t: Ctx) -> Cand:
    P = t.prod()
    loc = t.loc({"brasil": 3, "uf": 4, "big_city": 3, "regiao": 2})
    per = t.period(PERIODS_MULTI, min_months=3)
    w, p = loc.em, per.text(t.rng)
    sql = (
        "SELECT date_trunc('month', c.data_coleta) AS mes, AVG(c.valor_venda) AS preco_medio "
        + frm(need_m=loc.needs_m)
        + where(P.pred, *loc.preds, per.pred())
        + " GROUP BY mes ORDER BY mes"
    )
    opts = [
        f"Como evoluiu o preço médio {P.do} mês a mês {w} {p}?",
        f"Qual o preço médio mensal {P.do} {w} {p}?",
        (f"preço médio {P.do} por mês {w} {p}", True),
        (f"me mostra a evolução mensal do preço {P.do} {w} {p}", False),
        (f"qto custou {P.o} em média em cada mês {w} {p}?", False),
    ]
    return Cand(_choose(t, opts), sql, False, _tags(P.tag, loc.tag, per.tag, "serie"))


def g_media_por_produto(t: Ctx) -> Cand:
    loc = t.loc({"uf": 3, "municipio": 3, "regiao": 2, "brasil": 1})
    per = t.period()
    w, p = loc.em, per.text(t.rng)
    base = (
        "SELECT c.produto, AVG(c.valor_venda) AS preco_medio "
        + frm(need_m=loc.needs_m)
        + where(*loc.preds, per.pred())
        + " GROUP BY c.produto"
    )
    if t.chance(0.65):
        opts = [
            f"Qual o preço médio de cada combustível {w} {p}?",
            (f"preço médio por produto {w} {p}", True),
            (f"qto custou em média cada combustível {w} {p}?", False),
            f"Mostre o preço médio de todos os combustíveis {w} {p}.",
        ]
        return Cand(_choose(t, opts), base, False, _tags(loc.tag, per.tag, "produto"))
    desc = t.chance(0.7)
    sql, probe = ranked(base, "preco_medio", desc, None, 1)
    a, b = ("mais caro", "mais barato") if desc else ("mais barato", "mais caro")
    opts = [
        f"Liste os combustíveis do {a} para o {b} {w} {p}, pelo preço médio.",
        f"Ordene os produtos pelo preço médio {w} {p}, do {a} para o {b}.",
        (f"combustíveis do {a} pro {b} {w} {p}", False),
    ]
    return Cand(_choose(t, opts), sql, True, _tags(loc.tag, per.tag, "produto", "ranking"), probe)


def g_ranking_capitais(t: Ctx) -> Cand:
    P, per = t.prod(), t.period()
    desc = t.chance(0.5)
    adj = P.adj(desc)
    preds: tuple[str, ...] = ("m.capital",)
    sc, total = "", 27
    if t.chance(0.35):
        r = t.regiao()
        preds, sc, total = ("m.capital", *r.preds), r.de, len(t.cat.region_ufs[r.regiao])
    p = per.text(t.rng)
    base = (
        "SELECT m.nome, m.uf, AVG(c.valor_venda) AS preco_medio "
        + frm(need_m=True)
        + where(P.pred, *preds, per.pred())
        + " GROUP BY m.nome, m.uf"
    )
    mode = t.pick(("n", "n", "n", "um", "todas"))
    if mode == "todas":
        sql, probe = ranked(base, "preco_medio", desc, None, 2)
        a, b = ("mais cara", "mais barata") if desc else ("mais barata", "mais cara")
        opts = [
            f"Liste as capitais {sc}, da {a} para a {b}, pelo preço médio {P.do} {p}.",
            f"Ordene as capitais {sc} pelo preço médio {P.do} {p}, da {a} para a {b}.",
            (f"capitais {sc} da {a} pra {b} no preço {P.do} {p}", False),
        ]
    elif mode == "um":
        sql, probe = ranked(base, "preco_medio", desc, 1, 2)
        opts = [
            f"Qual a capital {sc} com {P.o} mais {adj} {p}?",
            f"Em qual capital {sc} {P.o} foi mais {adj} {p}?",
            (f"qual capital {sc} teve {P.o} mais {adj} {p}?", False),
        ]
    else:
        n = t.n((3, 5, 5, 10, 2), limit=total)
        sql, probe = ranked(base, "preco_medio", desc, n, 2)
        opts = [
            f"Quais as {n} capitais {sc} com {P.o} mais {adj} {p}?",
            (f"top {n} capitais {sc} com {P.o} mais {adj} {p}", t.chance(0.5)),
            f"Liste as {n} capitais {sc} onde {P.o} foi mais {adj} em média {p}.",
            (f"quais {n} capitais {sc} tiveram {P.o} mais {adj} {p}?", False),
        ]
    return Cand(_choose(t, opts), sql, True, _tags(P.tag, "ranking", "capital", per.tag), probe)


def _join_names(names: Sequence[str], conj: str = "e") -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" {conj} " + names[-1]


def g_compara_bandeiras(t: Ctx) -> Cand:
    P, per = t.prod(), t.period()
    pool = ["BRANCA", *[b for b in BIG_BANDS if b in t.cat.bandeiras], "SABBÁ"]
    pool = [b for b in pool if b == "BRANCA" or b in t.cat.bandeiras]
    k = 3 if t.chance(0.3) else 2
    bands = t.rng.sample(pool, k)
    loc = t.loc({"brasil": 2, "regiao": 2, "uf": 3, "big_city": 2})
    w, p = loc.em, per.text(t.rng)
    names = [t.band_name(b) for b in bands]
    lst = ", ".join(sql_str(b) for b in bands)
    sql = (
        "SELECT p.bandeira, AVG(c.valor_venda) AS preco_medio "
        + frm(need_p=True, need_m=loc.needs_m)
        + where(P.pred, f"p.bandeira IN ({lst})", *loc.preds, per.pred())
        + " GROUP BY p.bandeira"
    )
    opts = [
        f"Compare o preço médio {P.do} entre {_join_names(names)} {w} {p}.",
        f"Qual o preço médio {P.do} em cada uma destas bandeiras {w} {p}: {_join_names(names)}?",
        (f"{' x '.join(names)}: preço médio {P.do} {w} {p}", True),
        (f"qto custou {P.o} em média na {_join_names(names, 'e na')} {w} {p}?", False),
    ]
    return Cand(
        _choose(t, opts),
        sql,
        False,
        _tags(P.tag, loc.tag, per.tag, "bandeira", "comparacao"),
        check=lambda rows, k=k: None if len(rows) == k else "poucas_linhas",
    )


def g_postos_por_bandeira(t: Ctx) -> Cand:
    loc = t.loc({"brasil": 2, "regiao": 2, "uf": 4, "big_city": 2})
    if loc.kind == "brasil":
        loc = t.brasil(allow_empty=False)
    branca_pred, branca_txt = t.branca_clause()
    join = " JOIN municipios m ON m.municipio_id = p.municipio_id" if loc.needs_m else ""
    base = (
        f"SELECT p.bandeira, COUNT(*) AS postos FROM postos p{join}"
        + where(*loc.preds, branca_pred)
        + " GROUP BY p.bandeira"
    )
    w = loc.em
    if t.chance(0.25):
        sql, probe = ranked(base, "postos", True, 1, 1)
        opts = [
            f"Qual a bandeira com mais postos {w}{branca_txt}?",
            (f"qual bandeira tem mais postos {w}{branca_txt}?", False),
            f"Que bandeira domina {w} em número de postos{branca_txt}?",
        ]
    else:
        n = t.n((3, 5, 5))
        sql, probe = ranked(base, "postos", True, n, 1)
        opts = [
            f"Quais as {n} bandeiras com mais postos {w}{branca_txt}?",
            (f"top {n} bandeiras em número de postos {w}{branca_txt}", True),
            f"Quais bandeiras dominam {w}? Liste as {n} com mais postos{branca_txt}.",
            (f"quais {n} bandeiras tem mais postos {w}{branca_txt}?", False),
        ]
    return Cand(_choose(t, opts), sql, True, _tags(loc.tag, "bandeira", "ranking"), probe)


def _pick_threshold(t: Ctx, values: list[float], step: float = 0.05) -> float:
    """Limiar redondo que deixa de 1 a ~40 itens de um lado, sem empate com o limiar."""
    vals = sorted(values)
    if len(vals) < 3:
        raise Skip("sem_parametro")
    for _ in range(20):
        i = t.rng.randrange(1, len(vals))
        x = (vals[i - 1] + vals[i]) / 2
        thr = round(round(x / step) * step, 2)
        above = sum(1 for v in vals if v > thr)
        if any(abs(v - thr) < 1e-6 for v in vals):
            continue
        if 1 <= above <= len(vals) - 1 and min(above, len(vals) - above) <= 40:
            return thr
    raise Skip("sem_parametro")


def g_municipios_acima_limite(t: Ctx) -> Cand:
    P, per = t.prod(), t.period()
    kind = t.pick(("uf", "uf", "regiao", "brasil"))
    if kind == "uf":
        uf = t.pick([u for u in t.cat.ufs if len(t.cat.by_uf[u]) >= 5])
        loc = t.uf(uf)
        cols, group = "m.nome", "m.nome"
    else:
        loc = t.regiao() if kind == "regiao" else t.brasil(allow_empty=False)
        cols, group = "m.nome, m.uf", "m.nome, m.uf"
    body = frm(need_m=True) + where(P.pred, *loc.preds, per.pred()) + f" GROUP BY {group}"
    values = [float(v) for (v,) in t.query(f"SELECT AVG(c.valor_venda) {body}")]
    thr = _pick_threshold(t, values)
    above = t.chance(0.6)
    op = ">" if above else "<"
    rel = "acima de" if above else "abaixo de"
    having = f" HAVING AVG(c.valor_venda) {op} {thr}"
    R, sc, p = money(thr), loc.de, per.text(t.rng)
    if t.chance(0.65):
        sql = f"SELECT {cols} {body}{having}"
        opts = [
            f"Em quais cidades {sc} {P.o} ficou {rel} {R} em média {p}?",
            f"Quais municípios {sc} tiveram preço médio {P.do} {rel} {R} {p}?",
            (f"cidades {sc} com {P.o} {rel} {R} na média {p}", True),
            (f"em quais cidades {sc} a média {P.do} ficou {rel} {R} {p}?", False),
        ]
        return Cand(_choose(t, opts), sql, False, _tags(P.tag, "municipio", per.tag, "limite"))
    sql = f"SELECT COUNT(*) AS municipios FROM (SELECT {cols} {body}{having}) t"
    opts = [
        f"Quantas cidades {sc} tiveram preço médio {P.do} {rel} {R} {p}?",
        f"Em quantos municípios {sc} {P.o} ficou {rel} {R} em média {p}?",
        (f"número de cidades {sc} com {P.o} {rel} {R} na média {p}", True),
        (f"qtas cidades {sc} tiveram {P.o} {rel} {R} em média {p}?", False),
    ]
    return Cand(_choose(t, opts), sql, False, _tags(P.tag, "municipio", per.tag, "limite"))


def g_postos_mais_baratos(t: Ctx) -> Cand:
    P = t.prod()
    city = t.municipio(min_postos=25)
    per = t.period({"none": 2, "last_month": 4, "month": 3, "year": 2, "range": 1})
    desc = t.chance(0.3)
    n = t.n((3, 5, 5))
    bairro = t.chance(0.25)
    cols = "p.razao_social, p.bairro" if bairro else "p.razao_social"
    group = "p.posto_id, p.razao_social, p.bairro" if bairro else "p.posto_id, p.razao_social"
    base = (
        f"SELECT {cols}, AVG(c.valor_venda) AS preco_medio "
        + frm(need_m=True)
        + where(P.pred, *city.preds, per.pred())
        + f" GROUP BY {group}"
    )
    sql, probe = ranked(base, "preco_medio", desc, n, 2 if bairro else 1)
    adjp = "caros" if desc else "baratos"
    menor = "maior" if desc else "menor"
    extra = t.pick([", com o bairro de cada um", " e em que bairro ficam"]) if bairro else ""
    p = per.text(t.rng)
    opts = [
        f"Quais os {n} postos com {P.o} mais {P.adj(desc)} {city.em} {p}{extra}?",
        (f"top {n} postos mais {adjp} para {P.text} {city.em} {p}{extra}", True),
        f"Liste os {n} postos {city.de} com o {menor} preço médio {P.do} {p}{extra}.",
        (f"quais os {n} postos mais {adjp} pra {P.text} {city.em} {p}{extra}?", False),
    ]
    return Cand(_choose(t, opts), sql, True, _tags(P.tag, "ranking", "posto", per.tag), probe)


def g_coletas_por_mes(t: Ctx) -> Cand:
    loc = t.loc({"brasil": 3, "uf": 4, "big_city": 2, "regiao": 2})
    per = t.period(PERIODS_MULTI, min_months=3)
    P = t.prod() if t.chance(0.6) else None
    dep = f"de {P.text}" if P else ""
    w, p = loc.em, per.text(t.rng)
    sql = (
        "SELECT date_trunc('month', c.data_coleta) AS mes, COUNT(*) AS coletas "
        + frm(need_m=loc.needs_m)
        + where(P.pred if P else "", *loc.preds, per.pred())
        + " GROUP BY mes ORDER BY mes"
    )
    opts = [
        f"Quantas coletas {dep} foram feitas por mês {w} {p}?",
        (f"número de coletas por mês {dep} {w} {p}", True),
        f"Mostre o volume mensal de coletas de preço {dep} {w} {p}.",
        (f"qtas pesquisas de preço {dep} teve em cada mês {w} {p}?", False),
    ]
    return Cand(_choose(t, opts), sql, False, _tags(P.tag if P else "", loc.tag, per.tag, "serie"))


def g_ranking_bairros(t: Ctx) -> Cand:
    P = t.prod(("GASOLINA", "ETANOL", "DIESEL S10", "GASOLINA ADITIVADA", "DIESEL"))
    pool = [
        m for m in t.cat.municipios if m.postos >= 40 and t.cat.bairros.get((m.nome, m.uf), 0) >= 8
    ]
    if not pool:
        raise Skip("sem_parametro")
    city = t.municipio_loc(t.pick(pool))
    per = t.period({"none": 4, "last_month": 2, "year": 3, "range": 1, "semester": 1})
    desc = t.chance(0.5)
    adj = P.adj(desc)
    n = 1 if t.chance(0.2) else t.n((3, 5, 5))
    base = (
        "SELECT p.bairro, AVG(c.valor_venda) AS preco_medio "
        + frm(need_m=True)
        + where(P.pred, *city.preds, "p.bairro IS NOT NULL", per.pred())
        + " GROUP BY p.bairro"
    )
    sql, probe = ranked(base, "preco_medio", desc, n, 1)
    p = per.text(t.rng)
    if n == 1:
        opts = [
            f"Qual o bairro {city.de} com {P.o} mais {adj} {p}?",
            f"Em que bairro {city.de} {P.o} foi mais {adj}, em média, {p}?",
            (f"qual bairro {city.de} tem {P.o} mais {adj} {p}?", False),
        ]
    else:
        opts = [
            f"Quais os {n} bairros {city.de} com {P.o} mais {adj} {p}?",
            (f"top {n} bairros {city.em} com {P.o} mais {adj} {p}", True),
            f"Liste os {n} bairros {city.de} onde {P.o} foi mais {adj} em média {p}.",
            (f"me fala os {n} bairros {city.de} com {P.o} mais {adj} {p}", False),
        ]
    return Cand(_choose(t, opts), sql, True, _tags(P.tag, "ranking", "bairro", per.tag), probe)


def g_amplitude_por_uf(t: Ctx) -> Cand:
    P, per = t.prod(), t.period()
    desc = t.chance(0.75)
    maior = "maior" if desc else "menor"
    p = per.text(t.rng)
    if t.chance(0.65):
        cols, ent, ents = "m.uf", "estado", "estados"
        preds: tuple[str, ...] = ()
        sc, total = "", len(t.cat.ufs)
    else:
        uf = t.pick([u for u in t.cat.ufs if len(t.cat.by_uf[u]) >= 4])
        loc = t.uf(uf)
        cols, ent, ents = "m.nome", "cidade", "cidades"
        preds, sc, total = loc.preds, loc.de, len(t.cat.by_uf[uf])
    base = (
        f"SELECT {cols}, MAX(c.valor_venda) - MIN(c.valor_venda) AS amplitude "
        + frm(need_m=True)
        + where(P.pred, *preds, per.pred())
        + f" GROUP BY {cols}"
    )
    if t.chance(0.2):
        sql, probe = ranked(base, "amplitude", desc, 1, 1)
        art = "o" if ent == "estado" else "a"
        opts = [
            f"Qual {art} {ent} {sc} com a {maior} diferença entre o maior e o menor preço {P.do} {p}?",
            f"Em qual {ent} {sc} foi {maior} a distância entre o preço mais alto e o mais baixo {P.do} {p}?",
        ]  # fmt: skip
    else:
        n = t.n((3, 5, 5), limit=total)
        sql, probe = ranked(base, "amplitude", desc, n, 1)
        opts = [
            f"Quais os {n} {ents} {sc} com a {maior} diferença entre o preço mais alto e o mais baixo {P.do} {p}?",
            (f"top {n} {ents} {sc} com {maior} amplitude de preço {P.do} {p}", True),
            f"Liste os {n} {ents} {sc} onde a diferença entre o maior e o menor preço {P.do} foi {maior} {p}.",
            (f"quais {n} {ents} {sc} tiveram a {maior} diferença entre o preço máximo e o mínimo {P.do} {p}?", False),
        ]  # fmt: skip
    if ent == "cidade":
        opts = [
            o.replace("Quais os", "Quais as").replace("Liste os", "Liste as")
            if isinstance(o, str)
            else o
            for o in opts
        ]
    return Cand(_choose(t, opts), sql, True, _tags(P.tag, "amplitude", "ranking", per.tag), probe)


def g_presenca_bandeira(t: Ctx) -> Cand:
    kind = t.pick(("rank_mun", "rank_mun", "rank_uf", "um_mun", "um_mun", "um_uf"))
    if kind in ("rank_mun", "rank_uf"):
        branca_pred, branca_txt = t.branca_clause()
        if kind == "rank_uf":
            loc = t.brasil(allow_empty=True)
            metric, alias, noun = "COUNT(DISTINCT m.uf)", "estados", "estados"
        else:
            loc = t.loc({"brasil": 3, "regiao": 2, "uf": 2})
            if loc.kind == "brasil":
                loc = t.brasil(allow_empty=True)
            metric, alias, noun = "COUNT(DISTINCT p.municipio_id)", "municipios", "cidades"
        need_m = loc.needs_m or kind == "rank_uf"
        join = " JOIN municipios m ON m.municipio_id = p.municipio_id" if need_m else ""
        base = (
            f"SELECT p.bandeira, {metric} AS {alias} FROM postos p{join}"
            + where(*loc.preds, branca_pred)
            + " GROUP BY p.bandeira"
        )
        n = t.n((3, 5, 5))
        sql, probe = ranked(base, alias, True, n, 1)
        w = loc.em
        opts = [
            f"Quais as {n} bandeiras presentes em mais {noun} {w}{branca_txt}?",
            (f"top {n} bandeiras por número de {noun} com posto {w}{branca_txt}", True),
            f"Quais {n} bandeiras estão em mais {noun} {w}{branca_txt}?",
            (f"quais {n} bandeiras aparecem em mais {noun} {w}{branca_txt}?", False),
        ]
        return Cand(_choose(t, opts), sql, True, _tags(loc.tag, "bandeira", "ranking"), probe)
    b = t.bandeira(branca=0.15)
    if kind == "um_uf":
        loc = t.regiao() if t.chance(0.3) else t.brasil(allow_empty=True)
        metric, alias = "COUNT(DISTINCT m.uf)", "estados"
        need_m = True
    else:
        loc = t.loc({"brasil": 3, "regiao": 2, "uf": 3})
        if loc.kind == "brasil":
            loc = t.brasil(allow_empty=True)
        metric, alias = "COUNT(DISTINCT p.municipio_id)", "municipios"
        need_m = loc.needs_m
    join = " JOIN municipios m ON m.municipio_id = p.municipio_id" if need_m else ""
    sql = f"SELECT {metric} AS {alias} FROM postos p{join}" + where(
        f"p.bandeira = {sql_str(b)}", *loc.preds
    )
    w = loc.em
    if b == "BRANCA":
        noun = "estados" if kind == "um_uf" else "cidades"
        opts = [
            f"Em quantos {noun} {w} há postos de bandeira branca?",
            f"Em quantos {noun} {w} existem postos sem bandeira?",
            (f"número de {noun} com posto de bandeira branca {w}", True),
        ]
    else:
        name = BAND_NAMES[b]
        if kind == "um_uf":
            opts = [
                f"Em quantos estados {w} a {name} tem postos?",
                f"A {name} está presente em quantos estados {w}?",
                (f"número de estados com posto {name} {w}", True),
                (f"a {name} tá em qtos estados {w}?", False),
            ]
        else:
            opts = [
                f"Em quantas cidades {w} a {name} tem postos?",
                f"Em quantos municípios {w} há postos {name}?",
                (f"número de cidades com posto {name} {w}", True),
                (f"a {name} tá presente em qtas cidades {w}?", False),
            ]
    return Cand(_choose(t, opts), sql, False, _tags(loc.tag, "bandeira", "contagem"))


_SIDE_PAIRS = (
    ("DIESEL", "DIESEL S10"),
    ("GASOLINA", "GASOLINA ADITIVADA"),
    ("ETANOL", "GASOLINA"),
    ("GASOLINA", "DIESEL S10"),
    ("GASOLINA", "DIESEL"),
)


def _avg_case(code: str) -> str:
    return f"AVG(CASE WHEN c.produto = {sql_str(code)} THEN c.valor_venda END)"


def g_lado_a_lado_produtos(t: Ctx) -> Cand:
    a, b = t.pick(_SIDE_PAIRS)
    A, B = t.prod_text(a), t.prod_text(b)
    per = t.period()
    p = per.text(t.rng)
    cols = f"{_avg_case(a)} AS {PRODUCT_ALIAS[a]}, {_avg_case(b)} AS {PRODUCT_ALIAS[b]}"
    prods = f"c.produto IN ({sql_str(a)}, {sql_str(b)})"
    kind = t.pick(("ufs", "ufs", "capitais", "um"))
    tags = _tags(A.tag, B.tag, per.tag, "comparacao", "produto")
    if kind == "ufs":
        r = t.regiao()
        sql = (
            f"SELECT m.uf, {cols} "
            + frm(need_m=True)
            + where(prods, *r.preds, per.pred())
            + " GROUP BY m.uf"
        )
        opts = [
            f"Compare o preço médio {A.do} e {B.do} em cada estado {r.de} {p}.",
            (f"preço médio {A.do} e {B.do} por UF {r.em} {p}, lado a lado", True),
            f"Qual o preço médio {A.do} e {B.do} em cada estado {r.de} {p}?",
            (f"qto custou {A.o} e {B.o} em média em cada estado {r.de} {p}?", False),
        ]
    elif kind == "capitais":
        preds: tuple[str, ...] = ("m.capital",)
        sc = ""
        if t.chance(0.5):
            r = t.regiao()
            preds, sc = ("m.capital", *r.preds), r.de
        sql = (
            f"SELECT m.nome, m.uf, {cols} "
            + frm(need_m=True)
            + where(prods, *preds, per.pred())
            + " GROUP BY m.nome, m.uf"
        )
        opts = [
            f"Compare o preço médio {A.do} e {B.do} em cada capital {sc} {p}.",
            f"Qual o preço médio {A.do} e {B.do} nas capitais {sc} {p}? Uma linha por capital.",
            (f"preço médio {A.do} e {B.do} por capital {sc} {p}", True),
            (f"qto custou {A.o} e {B.o} em média em cada capital {sc} {p}?", False),
        ]
    else:
        loc = t.loc({"brasil": 2, "uf": 3, "municipio": 3, "regiao": 2})
        sql = f"SELECT {cols} " + frm(need_m=loc.needs_m) + where(prods, *loc.preds, per.pred())
        w = loc.em
        opts = [
            f"Qual o preço médio {A.do} e {B.do} {w} {p}?",
            f"Compare o preço médio {A.do} com o {B.do} {w} {p}.",
            (f"preço médio {A.do} e {B.do} {w} {p}", True),
            (f"qto custou em média {A.o} e {B.o} {w} {p}?", False),
        ]
    return Cand(_choose(t, opts), sql, False, tags)


# =============================================================================
# Geradores: difícil
# =============================================================================


def _precos_eg(group: str, preds: Sequence[str], per: Period) -> str:
    """CTE `precos` com etanol e gasolina médios por grupo."""
    return (
        f"WITH precos AS (SELECT {group}, {_avg_case('ETANOL')} AS etanol, "
        f"{_avg_case('GASOLINA')} AS gasolina "
        + frm(need_m=True)
        + where(E_G, *preds, per.pred())
        + f" GROUP BY {group})"
    )


_COMPENSA = (
    "o etanol compensou",
    "o etanol compensou",
    "valeu a pena abastecer com etanol",
    "compensou usar álcool em vez de gasolina",
    "o etanol saiu a até 70% do preço da gasolina",
)
_NAO_COMPENSA = (
    "o etanol não compensou",
    "o etanol não compensou",
    "não valeu a pena abastecer com etanol",
    "o etanol passou de 70% do preço da gasolina",
)


def g_regra70_ufs(t: Ctx) -> Cand:
    per = t.period()
    p = per.text(t.rng)
    kind = t.pick(("lista", "lista", "nao", "razoes", "cidades", "um", "conta", "regioes"))
    tags = _tags("regra70", per.tag)
    scope_preds: tuple[str, ...] = ()
    sc = ""
    if kind in ("lista", "nao", "razoes", "conta") and t.chance(0.35):
        r = t.regiao()
        scope_preds, sc = r.preds, r.de
    if kind in ("lista", "nao"):
        cte = _precos_eg("m.uf", scope_preds, per)
        op = "<=" if kind == "lista" else ">"
        sql = f"{cte} SELECT uf FROM precos WHERE etanol / gasolina {op} 0.7 ORDER BY uf"
        verb = t.pick(_COMPENSA if kind == "lista" else _NAO_COMPENSA)
        opts = [
            f"Em quais estados {sc} {verb} {p}?",
            f"Em quais UFs {sc} {verb} {p}, pela regra dos 70%?",
            (f"estados {sc} onde {verb} {p}", True),
            (f"em quais estados {sc} {verb} {p}?", False),
        ]
        return Cand(_choose(t, opts), sql, False, tags + ["uf"])
    if kind == "razoes":
        cte = _precos_eg("m.uf", scope_preds, per)
        desc = t.chance(0.3)
        base = f"{cte} SELECT uf, etanol / gasolina AS razao FROM precos"
        if t.chance(0.5):
            sql, probe = ranked(base, "razao", desc, None, 1)
            a, b = ("maior", "menor") if desc else ("menor", "maior")
            opts = [
                f"Qual a razão entre o preço do etanol e o da gasolina em cada estado {sc} {p}, da {a} para a {b}?",
                f"Ordene os estados {sc} pela razão etanol/gasolina {p}, da {a} para a {b}.",
                (f"razão etanol/gasolina por UF {sc} {p}, da {a} pra {b}", False),
            ]  # fmt: skip
        else:
            n = t.n((3, 5, 5), limit=len(t.cat.ufs) if not sc else 4)
            sql, probe = ranked(base, "razao", desc, n, 1)
            a = "maior" if desc else "menor"
            opts = [
                f"Quais os {n} estados {sc} com a {a} razão entre o preço do etanol e o da gasolina {p}?",
                (f"top {n} UFs {sc} com {a} razão etanol/gasolina {p}", True),
                f"Liste os {n} estados {sc} onde o etanol esteve {'mais caro' if desc else 'mais barato'} em relação à gasolina {p}.",
            ]  # fmt: skip
        return Cand(_choose(t, opts), sql, True, tags + ["uf", "ranking"], probe)
    if kind == "cidades":
        uf = t.pick([u for u in t.cat.ufs if len(t.cat.by_uf[u]) >= 4])
        loc = t.uf(uf)
        cte = _precos_eg("m.nome", loc.preds, per)
        sql = f"{cte} SELECT nome FROM precos WHERE etanol / gasolina <= 0.7 ORDER BY nome"
        verb = t.pick(_COMPENSA)
        opts = [
            f"Em quais cidades {loc.de} {verb} {p}?",
            f"Quais municípios {loc.de} tiveram etanol compensando {p}, pela regra dos 70%?",
            (f"cidades {loc.de} onde {verb} {p}", True),
        ]
        return Cand(_choose(t, opts), sql, False, tags + ["municipio"])
    if kind == "conta":
        cte = _precos_eg("m.uf", scope_preds, per)
        sql = f"{cte} SELECT COUNT(*) AS estados FROM precos WHERE etanol / gasolina <= 0.7"
        verb = t.pick(_COMPENSA)
        opts = [
            f"Em quantos estados {sc} {verb} {p}?",
            f"Quantas UFs {sc} tiveram o etanol valendo a pena {p}, pela regra dos 70%?",
            (f"qtos estados {sc} tiveram o etanol compensando {p}?", False),
        ]
        return Cand(_choose(t, opts), sql, False, tags + ["uf", "contagem"])
    if kind == "regioes":
        cte = _precos_eg("m.regiao", (), per)
        sql = f"{cte} SELECT regiao FROM precos WHERE etanol / gasolina <= 0.7 ORDER BY regiao"
        verb = t.pick(_COMPENSA)
        opts = [
            f"Em quais regiões {verb} {p}?",
            (f"regiões onde {verb} {p}", True),
            (f"em que regiões do país {verb} {p}?", False),
        ]
        return Cand(_choose(t, opts), sql, False, tags + ["regiao"])
    loc = t.loc({"brasil": 2, "uf": 3, "municipio": 3, "regiao": 2})
    sql = f"SELECT {RATIO} AS razao " + frm(need_m=loc.needs_m) + where(E_G, *loc.preds, per.pred())
    w = loc.em
    opts = [
        f"Qual a razão entre os preços médios do etanol e da gasolina {w} {p}?",
        f"Quanto o etanol custou em relação à gasolina {w} {p}? Quero a razão etanol/gasolina.",
        (f"razão etanol/gasolina {w} {p}", True),
        (f"qual foi a relação entre o preço do etanol e o da gasolina {w} {p}?", False),
    ]
    return Cand(_choose(t, opts), sql, False, tags + [loc.tag])


def g_regra70_todos_meses(t: Ctx) -> Cand:
    per = t.period({"year": 5, "semester": 3, "last12": 2, "range": 3, "quarter": 1}, min_months=3)
    months = per.months()
    de = per.de(t.rng)
    scope = t.pick(("capitais", "capitais", "ufs", "cidades"))
    if scope == "capitais":
        group, sel, preds, ent = "m.nome, m.uf", "nome, uf", ("m.capital",), "capitais"
    elif scope == "ufs":
        group, sel, preds, ent = "m.uf", "uf", (), "estados"
    else:
        uf = t.pick([u for u in t.cat.ufs if len(t.cat.by_uf[u]) >= 4])
        loc = t.uf(uf)
        group, sel, preds, ent = "m.nome", "nome", loc.preds, f"cidades {loc.de}"
    cte = (
        f"WITH mensal AS (SELECT {group}, date_trunc('month', c.data_coleta) AS mes, "
        f"{RATIO} AS razao " + frm(need_m=True) + where(E_G, *preds, per.pred())
        + f" GROUP BY {group}, mes)"
    )  # fmt: skip
    gsel = sel
    mode = t.pick(("todos", "todos", "nenhum", "pelo_menos", "conta_meses"))
    tags = _tags("regra70", "mes", "cte", per.tag)
    if mode == "conta_meses":
        loc = t.loc({"brasil": 2, "uf": 3, "capital": 3})
        sql = (
            f"WITH mensal AS (SELECT date_trunc('month', c.data_coleta) AS mes, {RATIO} AS razao "
            + frm(need_m=loc.needs_m) + where(E_G, *loc.preds, per.pred())
            + " GROUP BY mes) SELECT COUNT(*) AS meses FROM mensal WHERE razao <= 0.7"
        )  # fmt: skip
        verb = t.pick(_COMPENSA)
        opts = [
            f"Em quantos meses {de} {verb} {loc.em}?",
            f"Quantos meses {de} tiveram o etanol compensando {loc.em}?",
            (f"em qtos meses {de} {verb} {loc.em}?", False),
        ]
        return Cand(_choose(t, opts), sql, False, tags + [loc.tag])
    if mode == "todos":
        having = f"HAVING COUNT(razao) = {months} AND MAX(razao) <= 0.7"
        verb = t.pick(_COMPENSA)
        opts = [
            f"Em quais {ent} {verb} em todos os meses {de}?",
            f"Quais {ent} tiveram o etanol compensando em todos os meses {de}?",
            (f"{ent} onde {verb} em todos os meses {de}", True),
            (f"em quais {ent} {verb} todo mês {de}?", False),
        ]
    elif mode == "nenhum":
        having = "HAVING MIN(razao) > 0.7"
        opts = [
            f"Em quais {ent} o etanol não compensou em nenhum mês {de}?",
            f"Quais {ent} não tiveram nenhum mês {de} com o etanol compensando?",
            (f"{ent} onde o etanol não compensou em nenhum mês {de}", True),
        ]
    else:
        k = t.rng.randint(2, max(2, months - 1))
        having = f"HAVING COUNT(CASE WHEN razao <= 0.7 THEN 1 END) >= {k}"
        verb = t.pick(_COMPENSA)
        opts = [
            f"Em quais {ent} {verb} em pelo menos {k} meses {de}?",
            f"Quais {ent} tiveram pelo menos {k} meses {de} com o etanol compensando?",
            (f"{ent} onde {verb} em {k} meses ou mais {de}", True),
        ]
    first = sel.split(",")[0]
    sql = f"{cte} SELECT {gsel} FROM mensal GROUP BY {gsel} {having} ORDER BY {first}"
    return Cand(_choose(t, opts), sql, False, tags)


def _period_pairs(t: Ctx) -> tuple[Period, Period, str]:
    kind = t.pick(("yoy", "yoy", "consec", "anos", "extremos", "semestres", "trimestres"))
    if kind == "yoy":
        m = t.rng.randint(1, 9)
        a, b = month_period(2025, m), month_period(2026, m)
        name = MONTHS_PT[m - 1]
        txt = t.pick([f"de {name} de 2025 para {name} de 2026",
                      f"comparando {name} de 2025 com {name} de 2026"])  # fmt: skip
    elif kind == "consec":
        d = _add_months(SNAPSHOT_START, t.rng.randrange(1, 24))
        prev = _add_months(d, -1)
        a, b = month_period(prev.year, prev.month), month_period(d.year, d.month)
        if prev.year == d.year:
            txt = f"de {MONTHS_PT[prev.month - 1]} para {_month_text(d)}"
        else:
            txt = f"de {_month_text(prev)} para {_month_text(d)}"
    elif kind == "anos":
        a, b = year_period(2025), year_period(2026)
        txt = t.pick(["de 2025 para 2026", "comparando 2025 com 2026"])
    elif kind == "extremos":
        a, b = month_period(2024, 10), month_period(2026, 9)
        txt = t.pick(["de outubro de 2024 para setembro de 2026",
                      "entre o primeiro e o último mês dos dados (out/2024 e set/2026)"])  # fmt: skip
    elif kind == "semestres":
        a, b = semester_period(2025, 1), semester_period(2026, 1)
        txt = "do primeiro semestre de 2025 para o primeiro semestre de 2026"
    else:
        q = t.rng.randint(1, 3)
        a, b = quarter_period(2025, q), quarter_period(2026, q)
        w = ORDINAL_QUARTER[q][0]
        txt = f"do {w} trimestre de 2025 para o {w} trimestre de 2026"
    return a, b, txt


def g_variacao_periodos(t: Ctx) -> Cand:
    P = t.prod()
    pa, pb, pair = _period_pairs(t)
    kind = t.pick(("pct", "pct", "reais", "ufs", "ufs", "capitais"))
    tags = _tags(P.tag, "variacao", "periodo", "cte")
    if kind in ("pct", "reais"):
        loc = t.loc({"brasil": 3, "uf": 3, "municipio": 2, "regiao": 2})
        body = lambda per: frm(need_m=loc.needs_m) + where(P.pred, *loc.preds, per.pred())  # noqa: E731
        cte = f"WITH a AS (SELECT AVG(c.valor_venda) AS p {body(pa)}), b AS (SELECT AVG(c.valor_venda) AS p {body(pb)})"
        w = loc.em
        if kind == "pct":
            sql = f"{cte} SELECT 100 * (b.p - a.p) / a.p AS variacao_pct FROM a CROSS JOIN b"
            opts = [
                f"Qual foi a variação percentual do preço médio {P.do} {w} {pair}?",
                f"Quanto variou, em %, o preço médio {P.do} {w} {pair}?",
                (f"variação % do preço médio {P.do} {w} {pair}", True),
                (f"qto o preço médio {P.do} subiu ou caiu em % {w} {pair}?", False),
            ]
        else:
            sql = f"{cte} SELECT b.p - a.p AS diferenca FROM a CROSS JOIN b"
            opts = [
                f"Quanto o preço médio {P.do} mudou, em reais, {w} {pair}?",
                f"Qual a diferença em R$ entre os preços médios {P.do} {w} {pair}?",
                (f"variação em reais do preço médio {P.do} {w} {pair}", True),
                (f"qto o preço {P.do} subiu ou caiu em R$ {w} {pair}, na média?", False),
            ]
        return Cand(_choose(t, opts), sql, False, tags + [loc.tag])
    if kind == "ufs":
        group, sel, on, ents, total = "m.uf", "a.uf", "a.uf = b.uf", "estados", 27
        preds: tuple[str, ...] = ()
    else:
        group, sel, on, ents, total = (
            "m.nome, m.uf", "a.nome, a.uf", "a.nome = b.nome AND a.uf = b.uf", "capitais", 27,
        )  # fmt: skip
        preds = ("m.capital",)

    def cte_part(per: Period) -> str:
        return (
            f"SELECT {group}, AVG(c.valor_venda) AS p "
            + frm(need_m=True)
            + where(P.pred, *preds, per.pred())
            + f" GROUP BY {group}"
        )

    cte = f"WITH a AS ({cte_part(pa)}), b AS ({cte_part(pb)})"
    alta = t.chance(0.6)
    n = t.n((3, 5, 5), limit=total)
    base = f"{cte} SELECT {sel}, 100 * (b.p - a.p) / a.p AS variacao_pct FROM a JOIN b ON {on}"
    vi = 2 if "," in sel else 1
    sql, probe = ranked(base, "variacao_pct", alta, n, vi, sign=1 if alta else -1)
    art = "as" if ents == "capitais" else "os"
    if alta:
        opts = [
            f"Quais {art} {n} {ents} com a maior alta percentual do preço médio {P.do} {pair}?",
            (f"top {n} {ents} com maior aumento % {P.do} {pair}", True),
            f"Liste {art} {n} {ents} onde o preço médio {P.do} mais subiu, em %, {pair}.",
            (f"quais {n} {ents} tiveram a maior alta {P.do} em % {pair}?", False),
        ]
    else:
        opts = [
            f"Quais {art} {n} {ents} com a maior queda percentual do preço médio {P.do} {pair}?",
            (f"top {n} {ents} com maior queda % {P.do} {pair}", True),
            f"Liste {art} {n} {ents} onde o preço médio {P.do} mais caiu, em %, {pair}.",
            (f"quais {n} {ents} tiveram a maior queda {P.do} em % {pair}?", False),
        ]
    return Cand(_choose(t, opts), sql, True, tags + ["ranking"], probe)


def g_razao_mensal(t: Ctx) -> Cand:
    loc = t.loc({"brasil": 3, "uf": 4, "big_city": 3, "regiao": 2})
    per = t.period(PERIODS_MULTI, min_months=3)
    w, p = loc.em, per.text(t.rng)
    series = (
        f"SELECT date_trunc('month', c.data_coleta) AS mes, {RATIO} AS razao "
        + frm(need_m=loc.needs_m) + where(E_G, *loc.preds, per.pred()) + " GROUP BY mes"
    )  # fmt: skip
    tags = _tags("regra70", "serie", loc.tag, per.tag)
    if t.chance(0.5):
        sql = series + " ORDER BY mes"
        opts = [
            f"Qual foi a razão entre o preço do etanol e o da gasolina mês a mês {w} {p}?",
            (f"razão etanol/gasolina por mês {w} {p}", True),
            f"Mostre, mês a mês, quanto o etanol custou em relação à gasolina {w} {p}.",
            (f"como ficou a relação etanol/gasolina em cada mês {w} {p}?", False),
        ]
        return Cand(_choose(t, opts), sql, False, tags)
    de = per.de(t.rng)
    if t.chance(0.65):
        op, verb = "<=", t.pick(_COMPENSA)
    else:
        op, verb = ">", t.pick(_NAO_COMPENSA)
    sql = f"WITH mensal AS ({series}) SELECT mes FROM mensal WHERE razao {op} 0.7 ORDER BY mes"
    opts = [
        f"Em quais meses {de} {verb} {w}?",
        (f"meses {de} em que {verb} {w}", True),
        (f"em que meses {de} {verb} {w}?", False),
        f"Liste os meses {de} em que {verb} {w}.",
    ]
    return Cand(_choose(t, opts), sql, False, tags + ["cte"])


def g_acima_media_nacional(t: Ctx) -> Cand:
    P, per = t.prod(), t.period()
    p = per.text(t.rng)
    above = t.chance(0.6)
    op = ">" if above else "<"
    rel = (
        t.pick(["acima da", "acima da", "mais alto que a"])
        if above
        else t.pick(["abaixo da", "abaixo da", "mais baixo que a"])
    )
    nacional = (
        "WITH nacional AS (SELECT AVG(valor_venda) AS media FROM coletas"
        + where(P.pred.replace("c.", ""), per.pred("data_coleta")) + ")"
    )  # fmt: skip
    media = t.pick(["média nacional", "média do Brasil", "média do país"])
    kind = t.pick(("ufs", "ufs", "ufs_valor", "capitais", "cidades_uf", "conta"))
    tags = _tags(P.tag, per.tag, "comparacao", "subconsulta", "cte")

    def body(preds: Sequence[str], group: str) -> str:
        return frm(need_m=True) + where(P.pred, *preds, per.pred()) + f" GROUP BY {group}"

    having = f" HAVING AVG(c.valor_venda) {op} (SELECT media FROM nacional)"
    if kind == "ufs":
        sql = f"{nacional} SELECT m.uf {body((), 'm.uf')}{having} ORDER BY m.uf"
        opts = [
            f"Em quais estados {P.o} ficou {rel} {media} {p}?",
            f"Quais UFs tiveram preço médio {P.do} {rel} {media} {p}?",
            (f"estados com {P.o} {rel} {media} {p}", True),
            (f"em quais estados o preço médio {P.do} ficou {rel} {media} {p}?", False),
        ]
        return Cand(_choose(t, opts), sql, False, tags + ["uf"])
    if kind == "ufs_valor":
        sql = (
            f"{nacional} SELECT m.uf, AVG(c.valor_venda) AS preco_medio "
            f"{body((), 'm.uf')}{having} ORDER BY m.uf"
        )
        opts = [
            f"Quais estados ficaram com {P.o} {rel} {media} {p}, e qual o preço médio de cada um?",
            f"Mostre os estados com preço médio {P.do} {rel} {media} {p}, com o preço de cada um.",
            (f"estados com {P.o} {rel} {media} {p} e seus preços médios", True),
        ]
        return Cand(_choose(t, opts), sql, False, tags + ["uf"])
    if kind == "capitais":
        sql = (
            f"{nacional} SELECT m.nome, m.uf {body(('m.capital',), 'm.nome, m.uf')}{having} "
            "ORDER BY m.nome"
        )
        opts = [
            f"Em quais capitais {P.o} ficou {rel} {media} {p}?",
            f"Quais capitais tiveram preço médio {P.do} {rel} {media} {p}?",
            (f"capitais com {P.o} {rel} {media} {p}", True),
        ]
        return Cand(_choose(t, opts), sql, False, tags + ["capital"])
    if kind == "conta":
        sql = (
            f"{nacional} SELECT COUNT(*) AS estados FROM (SELECT m.uf {body((), 'm.uf')}{having}) t"
        )
        opts = [
            f"Em quantos estados {P.o} ficou {rel} {media} {p}?",
            f"Quantas UFs tiveram preço médio {P.do} {rel} {media} {p}?",
            (f"qtos estados ficaram com {P.o} {rel} {media} {p}?", False),
        ]
        return Cand(_choose(t, opts), sql, False, tags + ["uf", "contagem"])
    uf = t.pick([u for u in t.cat.ufs if len(t.cat.by_uf[u]) >= 5])
    loc = t.uf(uf)
    estadual = (
        "WITH estadual AS (SELECT AVG(c.valor_venda) AS media " + frm(need_m=True)
        + where(P.pred, *loc.preds, per.pred()) + ")"
    )  # fmt: skip
    hav = f" HAVING AVG(c.valor_venda) {op} (SELECT media FROM estadual)"
    sql = f"{estadual} SELECT m.nome {body(loc.preds, 'm.nome')}{hav} ORDER BY m.nome"
    rel_e = ("acima" if above else "abaixo") + " da média do estado"
    opts = [
        f"Em quais cidades {loc.de} {P.o} ficou {rel_e} {p}?",
        f"Quais municípios {loc.de} tiveram preço médio {P.do} {rel_e} {p}?",
        (f"cidades {loc.de} com {P.o} {rel_e} {p}", True),
    ]
    return Cand(_choose(t, opts), sql, False, tags + ["municipio"])


def g_subiu_no_periodo(t: Ctx) -> Cand:
    P = t.prod()
    i = t.rng.randrange(0, 23)
    j = t.rng.randrange(i + 1, 24)
    d1, d2 = _add_months(SNAPSHOT_START, i), _add_months(SNAPSHOT_START, j)
    pa, pb = month_period(d1.year, d1.month), month_period(d2.year, d2.month)
    if d1.year == d2.year:
        pair = t.pick([f"de {MONTHS_PT[d1.month - 1]} para {_month_text(d2)}",
                       f"de {_month_text(d1)} para {_month_text(d2)}",
                       f"comparando {MONTHS_PT[d1.month - 1]} com {_month_text(d2)}"])  # fmt: skip
    else:
        pair = t.pick([f"de {_month_text(d1)} para {_month_text(d2)}",
                       f"comparando {_month_text(d1)} com {_month_text(d2)}"])  # fmt: skip
    scope = t.pick(("ufs", "ufs", "capitais", "cidades"))
    if scope == "ufs":
        group, sel, on, ents, preds = "m.uf", "inicio.uf", "inicio.uf = fim.uf", "estados", ()
    elif scope == "capitais":
        group, sel = "m.nome, m.uf", "inicio.nome, inicio.uf"
        on, ents, preds = (
            "inicio.nome = fim.nome AND inicio.uf = fim.uf",
            "capitais",
            ("m.capital",),
        )
    else:
        uf = t.pick([u for u in t.cat.ufs if len(t.cat.by_uf[u]) >= 4])
        loc = t.uf(uf)
        group, sel, on = "m.nome", "inicio.nome", "inicio.nome = fim.nome"
        ents, preds = f"cidades {loc.de}", loc.preds

    def part(per: Period) -> str:
        return (
            f"SELECT {group}, AVG(c.valor_venda) AS p "
            + frm(need_m=True)
            + where(P.pred, *preds, per.pred())
            + f" GROUP BY {group}"
        )

    cte = f"WITH inicio AS ({part(pa)}), fim AS ({part(pb)})"
    up = t.chance(0.6)
    tags = _tags(P.tag, "variacao", "cte")
    mode = t.pick(("lista", "lista", "pct", "conta"))
    if mode == "pct":
        rows = t.query(
            f"{cte} SELECT 100 * (fim.p - inicio.p) / inicio.p FROM inicio JOIN fim ON {on}"
        )
        vals = [float(v) * (1 if up else -1) for (v,) in rows if v is not None]
        vals = [v for v in vals if v > 0]
        thr = _pick_threshold(t, vals, step=0.5) if len(vals) >= 3 else None
        if thr is None or thr <= 0:
            raise Skip("sem_parametro")
        cond = (
            f"100 * (fim.p - inicio.p) / inicio.p > {thr}" if up
            else f"100 * (fim.p - inicio.p) / inicio.p < -{thr}"
        )  # fmt: skip
        verb = "subiu" if up else "caiu"
        pct = f"{thr:g}".replace(".", ",") + "%"
        sql = f"{cte} SELECT {sel} FROM inicio JOIN fim ON {on} WHERE {cond} ORDER BY {sel}"
        opts = [
            f"Em quais {ents} o preço médio {P.do} {verb} mais de {pct} {pair}?",
            (f"{ents} onde {P.o} {verb} mais de {pct} {pair}", True),
            (f"em quais {ents} o preço {P.do} {verb} mais de {pct} {pair}?", False),
        ]
        return Cand(_choose(t, opts), sql, False, tags)
    cond = "fim.p > inicio.p" if up else "fim.p < inicio.p"
    verb = "subiu" if up else "caiu"
    if mode == "conta":
        sql = f"{cte} SELECT COUNT(*) AS total FROM inicio JOIN fim ON {on} WHERE {cond}"
        fem = not ents.startswith("estados")
        em_q, q_, qt = ("quantas", "Quantas", "qtas") if fem else ("quantos", "Quantos", "qtos")
        what = "alta" if up else "queda"
        opts = [
            f"Em {em_q} {ents} o preço médio {P.do} {verb} {pair}?",
            (f"{qt} {ents} tiveram {what} no preço {P.do} {pair}?", False),
            f"{q_} {ents} tiveram {what} no preço médio {P.do} {pair}?",
        ]
        return Cand(_choose(t, opts), sql, False, tags + ["contagem"])
    sql = f"{cte} SELECT {sel} FROM inicio JOIN fim ON {on} WHERE {cond} ORDER BY {sel}"
    opts = [
        f"Em quais {ents} o preço médio {P.do} {verb} {pair}?",
        f"Quais {ents} tiveram {'alta' if up else 'queda'} no preço médio {P.do} {pair}?",
        (f"{ents} onde o preço médio {P.do} {verb} {pair}", True),
        (f"em quais {ents} {P.o} ficou mais {P.adj(up)} {pair}?", False),
    ]
    return Cand(_choose(t, opts), sql, False, tags)


def g_premio_bandeira(t: Ctx) -> Cand:
    P, per = t.prod(), t.period()
    loc = t.loc({"brasil": 3, "regiao": 2, "uf": 3, "big_city": 2})
    w, p = loc.em, per.text(t.rng)
    tags = _tags(P.tag, loc.tag, per.tag, "comparacao", "bandeira")
    bigs = [b for b in BIG_BANDS if b in t.cat.bandeiras]
    kind = t.pick(("lista", "uma", "uma", "pct"))
    if kind == "lista":
        bands = sorted(t.rng.sample(bigs, t.pick((2, 3))))
        lst = ", ".join(sql_str(b) for b in bands)
        names = _join_names([BAND_NAMES[b] for b in bands])
        das = _join_names([f"da {BAND_NAMES[b]}" for b in bands])
        sql = (
            "WITH por_bandeira AS (SELECT p.bandeira, AVG(c.valor_venda) AS preco "
            + frm(need_p=True, need_m=loc.needs_m) + where(P.pred, *loc.preds, per.pred())
            + " GROUP BY p.bandeira) SELECT b.bandeira, b.preco - br.preco AS diferenca "
            "FROM por_bandeira b CROSS JOIN (SELECT preco FROM por_bandeira "
            f"WHERE bandeira = 'BRANCA') br WHERE b.bandeira IN ({lst})"
        )  # fmt: skip
        opts = [
            f"Quanto {P.o} {das} custou a mais que {P.art} dos postos de bandeira branca {w} {p}, em média?",
            f"Qual a diferença de preço médio {P.do} entre {names} e os postos de bandeira branca {w} {p}?",
            (f"diferença em R$ entre o preço médio {P.do} de {names} e o da bandeira branca {w} {p}", True),
            (f"qto {das.replace('da ', 'a ')} cobraram a mais {P.pelo} que os postos sem bandeira {w} {p}?", False),
        ]  # fmt: skip
        nb = len(bands)

        def check(rows: Sequence[Sequence[Any]]) -> str | None:
            return positive_last_column(rows) if len(rows) == nb else "poucas_linhas"

        return Cand(_choose(t, opts), sql, False, tags + ["cte"], check=check)
    b = t.pick(bigs + ["SABBÁ"] if "SABBÁ" in t.cat.bandeiras else bigs)
    name = BAND_NAMES[b]
    pair = f"p.bandeira IN ({sql_str(b)}, 'BRANCA')"
    ab = f"AVG(CASE WHEN p.bandeira = {sql_str(b)} THEN c.valor_venda END)"
    ar = "AVG(CASE WHEN p.bandeira = 'BRANCA' THEN c.valor_venda END)"
    body = frm(need_p=True, need_m=loc.needs_m) + where(P.pred, pair, *loc.preds, per.pred())
    if kind == "uma":
        sql = f"SELECT {ab} - {ar} AS diferenca {body}"
        opts = [
            f"Quanto {P.o} da {name} custou a mais que {P.art} de bandeira branca {w} {p}, em média?",
            f"Qual a diferença entre o preço médio {P.do} na {name} e nos postos sem bandeira {w} {p}?",
            (f"qto a mais a {name} cobrou {P.pelo} que os postos de bandeira branca {w} {p}, em média?", False),
            (f"diferença de preço médio {P.do} entre {name} e bandeira branca {w} {p}", True),
        ]  # fmt: skip
    else:
        sql = f"SELECT 100 * ({ab} - {ar}) / {ar} AS diferenca_pct {body}"
        opts = [
            f"Quantos por cento {P.o} da {name} foi mais {P.adj(True)} que {P.art} de bandeira branca {w} {p}?",
            f"Em %, quanto {P.o} da {name} custou a mais que {P.art} dos postos sem bandeira {w} {p}?",
            (f"diferença percentual de preço {P.do} entre {name} e bandeira branca {w} {p}", True),
        ]  # fmt: skip
    return Cand(_choose(t, opts), sql, False, tags, check=positive_last_column)


_DIFF_PAIRS = (
    ("GASOLINA ADITIVADA", "GASOLINA", "a gasolina aditivada", "a comum"),
    ("DIESEL S10", "DIESEL", "o diesel S10", "o diesel comum"),
    ("GASOLINA", "ETANOL", "a gasolina", "o etanol"),
    ("GASOLINA ADITIVADA", "GASOLINA", "a aditivada", "a gasolina comum"),
)


def g_diferenca_produtos_uf(t: Ctx) -> Cand:
    a, b, ta, tb = t.pick(_DIFF_PAIRS)
    per = t.period()
    p = per.text(t.rng)
    scope = t.pick(("ufs", "ufs", "capitais", "regioes"))
    if scope == "ufs":
        group, sel, preds, ents, total = "m.uf", "uf", (), "estados", 27
    elif scope == "capitais":
        group, sel, preds, ents, total = "m.nome, m.uf", "nome, uf", ("m.capital",), "capitais", 27
    else:
        group, sel, preds, ents, total = "m.regiao", "regiao", (), "regiões", 5
    aa, ab = PRODUCT_ALIAS[a], PRODUCT_ALIAS[b]
    cte = (
        f"WITH precos AS (SELECT {group}, {_avg_case(a)} AS {aa}, {_avg_case(b)} AS {ab} "
        + frm(need_m=True) + where(f"c.produto IN ({sql_str(a)}, {sql_str(b)})", *preds, per.pred())
        + f" GROUP BY {group})"
    )  # fmt: skip
    tags = _tags(PRODUCT_TAG[a], PRODUCT_TAG[b], per.tag, "comparacao", "produto", "cte")
    fem = ents != "estados"
    art = "as" if fem else "os"
    mode = (
        t.pick(("reais", "reais", "pct", "limite"))
        if scope != "regioes"
        else t.pick(("reais", "pct"))
    )
    vi = 2 if "," in sel else 1
    if mode == "limite":
        diffs = [
            float(v) for (v,) in t.query(f"{cte} SELECT {aa} - {ab} FROM precos") if v is not None
        ]
        thr = _pick_threshold(t, diffs, step=0.05)
        if thr <= 0:
            raise Skip("sem_parametro")
        sql = (
            f"{cte} SELECT {sel} FROM precos WHERE {aa} - {ab} > {thr} ORDER BY {sel.split(',')[0]}"
        )
        R = money(thr)
        opts = [
            f"Em quais {ents} {ta} custou mais de {R} a mais que {tb} {p}?",
            f"Em quais {ents} a diferença entre {ta} e {tb} passou de {R} {p}?",
            (f"{ents} onde {ta} ficou mais de {R} acima {_de_art(tb)} {p}", True),
        ]
        return Cand(_choose(t, opts), sql, False, tags)
    desc = t.chance(0.75)
    maior = "maior" if desc else "menor"
    n = t.n((3, 5, 5), limit=total) if scope != "regioes" else None
    if mode == "pct":
        base = f"{cte} SELECT {sel}, 100 * ({aa} - {ab}) / {ab} AS diferenca_pct FROM precos"
        sql, probe = ranked(base, "diferenca_pct", desc, n, vi, sign=1)
        metric = "percentual"
    else:
        base = f"{cte} SELECT {sel}, {aa} - {ab} AS diferenca FROM precos"
        sql, probe = ranked(base, "diferenca", desc, n, vi, sign=1)
        metric = "em reais"
    if n is None:
        a1, b1 = ("maior", "menor") if desc else ("menor", "maior")
        opts = [
            f"Ordene as regiões pela diferença {metric} entre {ta} e {tb} {p}, da {a1} para a {b1}.",
            f"Qual a diferença {metric} entre {ta} e {tb} em cada região {p}? Da {a1} para a {b1}.",
        ]
    else:
        opts = [
            f"Quais {art} {n} {ents} com a {maior} diferença {metric} entre {ta} e {tb} {p}?",
            (f"top {n} {ents} com {maior} diferença {metric} entre {ta} e {tb} {p}", True),
            f"Liste {art} {n} {ents} onde a diferença {metric} entre {ta} e {tb} foi {maior} {p}.",
        ]
    return Cand(_choose(t, opts), sql, True, tags + ["ranking"], probe)


def g_janela_variacao_mensal(t: Ctx) -> Cand:
    P = t.prod(("GASOLINA", "ETANOL", "DIESEL S10", "DIESEL", "GASOLINA ADITIVADA"))
    loc = t.loc({"brasil": 3, "uf": 4, "capital": 2, "regiao": 2})
    per = t.period({"year": 5, "last12": 2, "last6": 1, "semester": 2, "range": 3}, min_months=3)
    if per.start <= SNAPSHOT_START:
        raise Skip("período")
    prev = _add_months(per.start, -1)
    w, p = loc.em, per.text(t.rng)
    cte = (
        "WITH mensal AS (SELECT date_trunc('month', c.data_coleta) AS mes, AVG(c.valor_venda) AS preco "
        + frm(need_m=loc.needs_m)
        + where(P.pred, *loc.preds, f"c.data_coleta >= DATE '{prev.isoformat()}'",
                f"c.data_coleta < DATE '{per.end.isoformat()}'")
        + " GROUP BY mes)"
    )  # fmt: skip
    lag = "LAG(preco) OVER (ORDER BY mes)"
    pct = t.chance(0.4)
    expr = f"100 * (preco - {lag}) / {lag}" if pct else f"preco - {lag}"
    alias = "variacao_pct" if pct else "variacao"
    var = f"{cte}, variacao AS (SELECT mes, {expr} AS {alias} FROM mensal)"
    flt = f" FROM variacao WHERE mes >= DATE '{per.start.isoformat()}'"
    tags = _tags(P.tag, loc.tag, per.tag, "janela", "serie", "variacao")
    unit = "percentual" if pct else "em reais"
    if t.chance(0.3):
        up = t.chance(0.6)
        base = f"{var} SELECT mes, {alias}{flt}"
        sql, probe = ranked(base, alias, up, 1, 1, sign=1 if up else -1)
        de = per.de(t.rng)
        what = "a maior alta" if up else "a maior queda"
        opts = [
            f"Em qual mês {de} {P.o} teve {what} {unit} em relação ao mês anterior {w}?",
            f"Qual mês {de} teve {what} {unit} do preço médio {P.do} sobre o mês anterior {w}?",
            (f"mês {de} com {what} {unit} {P.do} frente ao mês anterior {w}", True),
        ]
        return Cand(_choose(t, opts), sql, True, tags + ["ranking"], probe)
    sql = f"{var} SELECT mes, {alias}{flt} ORDER BY mes"
    opts = [
        f"Qual foi a variação {unit} do preço médio {P.do} em relação ao mês anterior, mês a mês, {w} {p}?",
        f"Mês a mês, quanto o preço médio {P.do} variou ({unit}) em relação ao mês anterior {w} {p}?",
        (f"variação mensal {unit} do preço médio {P.do} {w} {p}, comparando cada mês com o anterior", True),
        (f"mês a mês, qto variou o preço {P.do} ({unit}) {w} {p} em relação ao mês anterior?", False),
    ]  # fmt: skip
    return Cand(_choose(t, opts), sql, False, tags)


def g_janela_ranking_por_grupo(t: Ctx) -> Cand:
    P, per = t.prod(), t.period()
    desc = t.chance(0.5)
    adj = P.adj(desc)
    p = per.text(t.rng)
    kind = t.pick(("uf_regiao", "uf_regiao", "cidade_uf", "top2", "capital_regiao"))
    tags = _tags(P.tag, per.tag, "janela", "ranking")
    k = 2 if kind == "top2" else 1
    if kind in ("uf_regiao", "top2"):
        part, cols, sel, preds = "regiao", "m.regiao, m.uf", "regiao, uf", ()
    elif kind == "capital_regiao":
        part, cols, sel, preds = (
            "regiao",
            "m.regiao, m.nome, m.uf",
            "regiao, nome, uf",
            ("m.capital",),
        )
    else:
        r = t.regiao()
        part, cols, sel, preds = "uf", "m.uf, m.nome", "uf, nome", r.preds
    precos = (
        f"WITH precos AS (SELECT {cols}, AVG(c.valor_venda) AS preco " + frm(need_m=True)
        + where(P.pred, *preds, per.pred()) + f" GROUP BY {cols})"
    )  # fmt: skip
    dir_ = " DESC" if desc else ""
    fn = "ROW_NUMBER()" if k > 1 and t.chance(0.5) else "RANK()"
    sql = (
        f"{precos}, r AS (SELECT *, {fn} OVER (PARTITION BY {part} ORDER BY preco{dir_}) AS pos "
        f"FROM precos) SELECT {sel}, preco FROM r WHERE pos {'= 1' if k == 1 else '<= 2'} "
        f"ORDER BY {sel}"
    )
    n_cols = len(sel.split(","))
    probe = Probe(
        f"SELECT {sel}, preco FROM ({precos} SELECT * FROM precos) q ORDER BY {part}, preco{dir_}",
        k, n_cols, group_index=0,
    )  # fmt: skip
    if kind == "uf_regiao":
        opts = [
            f"Em cada região, qual o estado com {P.o} mais {adj} {p}?",
            f"Qual o estado com {P.o} mais {adj} de cada região {p}?",
            (f"me fala, região por região, qual UF teve {P.o} mais {adj} {p}", False),
            f"Para cada região do país, mostre o estado com o {'maior' if desc else 'menor'} preço médio {P.do} {p}.",
        ]  # fmt: skip
    elif kind == "top2":
        opts = [
            f"Quais os 2 estados com {P.o} mais {adj} em cada região {p}?",
            f"Em cada região, quais são os dois estados com {P.o} mais {adj} {p}?",
            (f"2 estados mais {'caros' if desc else 'baratos'} para {P.text} em cada região {p}", True),
        ]  # fmt: skip
    elif kind == "capital_regiao":
        opts = [
            f"Em cada região, qual a capital com {P.o} mais {adj} {p}?",
            f"Qual a capital com {P.o} mais {adj} de cada região {p}?",
            (f"capital com {P.o} mais {adj} em cada região {p}", True),
        ]
    else:
        opts = [
            f"Em cada estado {r.de}, qual a cidade com {P.o} mais {adj} {p}?",
            f"Qual o município com {P.o} mais {adj} em cada estado {r.de} {p}?",
            (f"cidade com {P.o} mais {adj} de cada UF {r.de} {p}", True),
        ]
    return Cand(_choose(t, opts), sql, False, tags + ["cte"], probe)


GENERATORS: dict[str, Callable[[Ctx], Cand]] = {
    "media_produto_uf": g_media_produto_uf,
    "media_produto_municipio": g_media_produto_municipio,
    "media_produto_regiao": g_media_produto_regiao,
    "media_nacional_periodo": g_media_nacional_periodo,
    "extremo_preco": g_extremo_preco,
    "conta_postos_local": g_conta_postos_local,
    "conta_coletas": g_conta_coletas,
    "conta_postos_bandeira": g_conta_postos_bandeira,
    "media_bandeira": g_media_bandeira,
    "media_capitais": g_media_capitais,
    "conta_municipios": g_conta_municipios,
    "media_ultimo_mes": g_media_ultimo_mes,
    "amplitude_local": g_amplitude_local,
    "datas_coleta": g_datas_coleta,
    "ranking_uf_preco": g_ranking_uf_preco,
    "ranking_municipios": g_ranking_municipios,
    "ranking_bandeiras": g_ranking_bandeiras,
    "media_por_regiao": g_media_por_regiao,
    "serie_mensal": g_serie_mensal,
    "media_por_produto": g_media_por_produto,
    "ranking_capitais": g_ranking_capitais,
    "compara_bandeiras": g_compara_bandeiras,
    "postos_por_bandeira": g_postos_por_bandeira,
    "municipios_acima_limite": g_municipios_acima_limite,
    "postos_mais_baratos": g_postos_mais_baratos,
    "coletas_por_mes": g_coletas_por_mes,
    "ranking_bairros": g_ranking_bairros,
    "amplitude_por_uf": g_amplitude_por_uf,
    "presenca_bandeira": g_presenca_bandeira,
    "lado_a_lado_produtos": g_lado_a_lado_produtos,
    "regra70_ufs": g_regra70_ufs,
    "regra70_todos_meses": g_regra70_todos_meses,
    "variacao_periodos": g_variacao_periodos,
    "razao_mensal": g_razao_mensal,
    "acima_media_nacional": g_acima_media_nacional,
    "subiu_no_periodo": g_subiu_no_periodo,
    "premio_bandeira": g_premio_bandeira,
    "diferenca_produtos_uf": g_diferenca_produtos_uf,
    "janela_variacao_mensal": g_janela_variacao_mensal,
    "janela_ranking_por_grupo": g_janela_ranking_por_grupo,
}


# =============================================================================
# Laço de geração
# =============================================================================


@dataclass
class GenerationResult:
    candidates: list[dict[str, Any]]
    quotas: dict[str, int]
    produced: dict[str, int]
    rejected: dict[str, dict[str, int]]
    exhausted: list[str]
    seed: int

    def summary(self) -> dict[str, Any]:
        total_rej: Counter[str] = Counter()
        for reasons in self.rejected.values():
            total_rej.update(reasons)
        return {
            "generator": "templates",
            "seed": self.seed,
            "requested": sum(self.quotas.values()),
            "generated": len(self.candidates),
            "generation_rejected_by_reason": dict(total_rej.most_common()),
            "generation_by_intent": {
                i: {
                    "quota": self.quotas.get(i, 0),
                    "generated": self.produced.get(i, 0),
                    "rejected": dict(sorted(self.rejected.get(i, {}).items())),
                }
                for i in GENERATORS
            },
            "exhausted_intents": self.exhausted,
        }


class _Producer:
    def __init__(self, con: duckdb.DuckDBPyConnection, cat: Catalog, seed: int) -> None:
        self.con = con
        self.cat = cat
        self.seed = seed
        self.seen_q: set[str] = set()
        self.seen_sql: set[str] = set()
        self.rejected: dict[str, Counter[str]] = defaultdict(Counter)
        self.out: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.ctx: dict[str, Ctx] = {}
        self.noise_rng: dict[str, random.Random] = {}

    def _ctx(self, intent_id: str) -> Ctx:
        if intent_id not in self.ctx:
            self.ctx[intent_id] = Ctx(random.Random(f"{self.seed}:{intent_id}"), self.cat, self.con)
            self.noise_rng[intent_id] = random.Random(f"{self.seed}:{intent_id}:ruido")
        return self.ctx[intent_id]

    def attempt(self, intent_id: str) -> bool:
        t = self._ctx(intent_id)
        rej = self.rejected[intent_id]
        try:
            cand = GENERATORS[intent_id](t)
        except Skip:
            rej["sem_parametro"] += 1
            return False
        nrng = self.noise_rng[intent_id]
        tags = list(cand.tags)
        question = cand.question
        if nrng.random() < NOISE_RATE:
            noisy = add_noise(question, nrng)
            if noisy != question:
                question = noisy
                tags.append("typo")
        nq, ns = leakage.normalize_question(question), leakage.normalize_sql(cand.sql)
        if nq in self.seen_q or ns in self.seen_sql:
            rej["repetido"] += 1
            return False
        rows = self.con.execute(cand.sql).fetchmany(MAX_RESULT_ROWS + 1)
        problem = result_problem(rows)
        if problem is None and cand.probe is not None:
            probe_rows = self.con.execute(cand.probe.sql).fetchall()
            problem = rank_problem(
                probe_rows, cand.probe.n, cand.probe.value_index, cand.probe.group_index,
                cand.probe.sign,
            )  # fmt: skip
        if problem is None and cand.check is not None:
            problem = cand.check(rows)
        if problem is not None:
            rej[problem] += 1
            return False
        self.seen_q.add(nq)
        self.seen_sql.add(ns)
        self.out[intent_id].append(
            {
                "question": question,
                "sql": cand.sql,
                "intent": intent_id,
                "order_matters": cand.order_matters,
                "tags": tags,
            }
        )
        return True

    def fill(self, intent_id: str, target: int) -> bool:
        """Gera até `target` itens da intenção; False se esgotou antes."""
        fails = 0
        while len(self.out[intent_id]) < target:
            if self.attempt(intent_id):
                fails = 0
            else:
                fails += 1
                if fails >= MAX_CONSECUTIVE_FAILURES:
                    return False
        return True


def generate(
    n: int,
    seed: int = 42,
    db_path: str | Path | None = None,
    quotas: dict[str, int] | None = None,
) -> GenerationResult:
    """Gera `n` candidatos na distribuição alvo (ou nas cotas dadas por intenção).

    Se uma intenção esgota o espaço de parâmetros, o que falta é repartido entre
    as outras intenções do mesmo nível, para manter a distribuição por nível.
    """
    path = Path(db_path) if db_path is not None else config.setting_path("db_path")
    con = duckdb.connect(str(path), read_only=True)
    try:
        cat = load_catalog(con)
        if quotas is None:
            quotas = dict(Counter(i.id for i in intents_mod.allocate(n)))
        prod = _Producer(con, cat, seed)
        exhausted: set[str] = set()
        for level in intents_mod.LEVELS:
            ids = [i.id for i in intents_mod.by_level(level) if quotas.get(i.id)]
            target = {i: quotas[i] for i in ids}
            for i in ids:
                if not prod.fill(i, target[i]):
                    exhausted.add(i)
            # Redistribui o que faltou entre as intenções ainda vivas do nível.
            while True:
                missing = sum(max(0, target[i] - len(prod.out[i])) for i in ids)
                alive = [i for i in ids if i not in exhausted]
                if not missing or not alive:
                    break
                for i in exhausted:
                    target[i] = len(prod.out[i])
                k = 0
                while missing:
                    target[alive[k % len(alive)]] += 1
                    missing -= 1
                    k += 1
                for i in alive:
                    if not prod.fill(i, target[i]):
                        exhausted.add(i)
    finally:
        con.close()

    candidates = [c for i in GENERATORS for c in prod.out[i]]
    random.Random(seed).shuffle(candidates)
    return GenerationResult(
        candidates=candidates,
        quotas=dict(quotas),
        produced={i: len(prod.out[i]) for i in GENERATORS},
        rejected={i: dict(prod.rejected[i]) for i in GENERATORS if prod.rejected[i]},
        exhausted=sorted(exhausted),
        seed=seed,
    )


def raw_path() -> Path:
    return config.resolve_path(RAW_RELPATH)


def write_raw(result: GenerationResult, path: str | Path | None = None) -> Path:
    out = Path(path) if path is not None else raw_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for c in result.candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return out
