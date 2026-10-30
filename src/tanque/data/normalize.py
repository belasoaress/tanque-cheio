"""Normalização dos CSVs da ANP para um Parquet único (`coletas.parquet`).

Cada CSV é lido como texto (todas as colunas `String`), com o encoding
detectado (UTF-8 com ou sem BOM; senão Latin-1). As colunas são casadas pelo
nome normalizado (sem acento, minúsculas, pontuação e espaços colapsados), então
"Estado - Sigla", "Estado-Sigla" e "ESTADO  SIGLA" são a mesma coluna.

Linhas que não podem entrar no banco não quebram a execução: vão para um
relatório de rejeição, contado por motivo (ver `REJECT_REASONS`).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

# Lista fechada de produtos automotivos (GLP fica de fora do escopo).
PRODUCTS: tuple[str, ...] = (
    "DIESEL",
    "DIESEL S10",
    "ETANOL",
    "GASOLINA",
    "GASOLINA ADITIVADA",
    "GNV",
)
REGIONS: tuple[str, ...] = ("CO", "N", "NE", "S", "SE")
DEFAULT_BANDEIRA = "BRANCA"

# Nome normalizado da coluna no CSV -> nome canônico.
COLUMN_ALIASES: dict[str, str] = {
    "regiao sigla": "regiao",
    "estado sigla": "uf",
    "municipio": "municipio",
    "revenda": "razao_social",
    "razao social": "razao_social",
    "cnpj da revenda": "cnpj",
    "nome da rua": "rua",
    "numero rua": "numero",
    "complemento": "complemento",
    "bairro": "bairro",
    "cep": "cep",
    "produto": "produto",
    "data da coleta": "data_coleta",
    "valor de venda": "valor_venda",
    "valor de compra": "valor_compra",
    "unidade de medida": "unidade",
    "bandeira": "bandeira",
}
REQUIRED_COLUMNS: tuple[str, ...] = (
    "regiao",
    "uf",
    "municipio",
    "cnpj",
    "produto",
    "data_coleta",
    "valor_venda",
    "unidade",
)
OPTIONAL_COLUMNS: tuple[str, ...] = ("razao_social", "bairro", "cep", "valor_compra", "bandeira")

# Motivos de rejeição, na ordem em que são testados (cada linha recebe o primeiro).
REJECT_REASONS: tuple[str, ...] = (
    "produto_fora_da_lista",
    "localizacao_invalida",
    "cnpj_vazio",
    "data_invalida",
    "valor_venda_vazio",
    "valor_venda_nao_positivo",
    "valor_venda_fora_da_faixa",
    "unidade_vazia",
    "linha_duplicada",
)
# DECIMAL(6,3) aceita até 999.999.
MAX_VALOR = 999.999

# Colunas que identificam uma coleta (uma linha de `coletas`); repetição exata é duplicata.
COLETA_KEY: tuple[str, ...] = (
    "cnpj",
    "produto",
    "data_coleta",
    "valor_venda",
    "valor_compra",
    "unidade",
)

# Schema do Parquet de saída, na ordem das colunas.
PARQUET_SCHEMA: dict[str, pl.DataType] = {
    "regiao": pl.String(),
    "uf": pl.String(),
    "municipio": pl.String(),
    "cnpj": pl.String(),
    "razao_social": pl.String(),
    "bandeira": pl.String(),
    "bairro": pl.String(),
    "cep": pl.String(),
    "produto": pl.String(),
    "data_coleta": pl.Date(),
    "valor_venda": pl.Decimal(6, 3),
    "valor_compra": pl.Decimal(6, 3),
    "unidade": pl.String(),
}


class NormalizeError(ValueError):
    """CSV que não pode ser interpretado (ex.: coluna obrigatória ausente)."""


@dataclass
class FileReport:
    """Contagens de um arquivo bruto."""

    name: str
    encoding: str
    rows_read: int
    rows_accepted: int = 0
    rejected: dict[str, int] = field(default_factory=dict)


@dataclass
class NormalizeResult:
    frame: pl.DataFrame
    files: list[FileReport]

    @property
    def rows_read(self) -> int:
        return sum(f.rows_read for f in self.files)

    @property
    def rejected(self) -> dict[str, int]:
        total: Counter[str] = Counter()
        for f in self.files:
            total.update(f.rejected)
        return dict(sorted(total.items()))


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def normalize_column_name(name: str) -> str:
    """'Estado - Sigla' -> 'estado sigla'; 'Valor  de Venda' -> 'valor de venda'."""
    name = strip_accents(name.replace("﻿", "")).lower()
    return re.sub(r"[^a-z0-9]+", " ", name).strip()


def map_columns(columns: list[str]) -> dict[str, str]:
    """Mapa coluna original -> nome canônico. Colunas desconhecidas são ignoradas."""
    mapping: dict[str, str] = {}
    for col in columns:
        canonical = COLUMN_ALIASES.get(normalize_column_name(col))
        if canonical is None:
            continue
        if canonical in mapping.values():
            raise NormalizeError(f"coluna {canonical!r} aparece mais de uma vez: {columns}")
        mapping[col] = canonical
    missing = [c for c in REQUIRED_COLUMNS if c not in mapping.values()]
    if missing:
        raise NormalizeError(f"colunas obrigatórias ausentes: {missing}; cabeçalho: {columns}")
    return mapping


def normalize_product(value: str | None) -> str | None:
    """Grafia canônica do produto, ou None se ele está fora da lista fechada."""
    if value is None:
        return None
    key = re.sub(r"[\s\-]+", " ", strip_accents(value).upper()).strip()
    key = key.replace("S 10", "S10")
    return key if key in PRODUCTS else None


def decode_bytes(data: bytes) -> tuple[str, str]:
    """Texto e encoding usado: UTF-8 (com ou sem BOM) ou, se falhar, Latin-1."""
    try:
        return data.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        return data.decode("latin-1"), "latin-1"


def read_raw_csv(path: Path) -> tuple[pl.DataFrame, str]:
    """Lê um CSV da ANP como texto, com colunas canônicas. Devolve (frame, encoding)."""
    text, encoding = decode_bytes(Path(path).read_bytes())
    frame = pl.read_csv(
        text.encode("utf-8"),
        separator=";",
        infer_schema=False,
    )
    mapping = map_columns(frame.columns)
    frame = frame.select([pl.col(src).alias(dst) for src, dst in mapping.items()])
    for col in OPTIONAL_COLUMNS:
        if col not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=pl.String).alias(col))
    return frame, encoding


def _clean_text(col: str) -> pl.Expr:
    """Tira espaços das pontas, colapsa espaços internos; vazio vira nulo."""
    expr = pl.col(col).str.strip_chars().str.replace_all(r"\s+", " ")
    return pl.when(expr == "").then(None).otherwise(expr).alias(col)


def _parse_decimal(col: str) -> pl.Expr:
    """'5,39' -> 5.39 (Float64); vazio ou inválido -> nulo."""
    return (
        pl.col(col).str.replace_all(r"\s", "").str.replace(",", ".").cast(pl.Float64, strict=False)
    )


def _upper_no_accents(col: str) -> pl.Expr:
    return pl.col(col).map_elements(
        lambda v: strip_accents(v).upper(), return_dtype=pl.String, skip_nulls=True
    )


def clean_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Aplica limpeza e tipos; acrescenta a coluna `motivo` (nula para linhas aceitas)."""
    text_cols = ["regiao", "uf", "municipio", "cnpj", "razao_social", "bairro", "cep"]
    text_cols += ["produto", "data_coleta", "valor_venda", "valor_compra", "unidade", "bandeira"]
    df = frame.with_columns([_clean_text(c) for c in text_cols])
    df = df.with_columns(
        _upper_no_accents("municipio").alias("municipio"),
        pl.col("uf").str.to_uppercase(),
        pl.col("regiao").str.to_uppercase(),
        pl.col("bandeira").str.to_uppercase().fill_null(DEFAULT_BANDEIRA),
        pl.col("produto")
        .map_elements(normalize_product, return_dtype=pl.String, skip_nulls=True)
        .alias("produto_norm"),
        pl.col("data_coleta").str.strptime(pl.Date, "%d/%m/%Y", strict=False).alias("data"),
        _parse_decimal("valor_venda").alias("venda"),
        _parse_decimal("valor_compra").alias("compra"),
    )
    loc_ok = (
        pl.col("municipio").is_not_null()
        & pl.col("uf").str.contains(r"^[A-Z]{2}$").fill_null(False)
        & pl.col("regiao").is_in(list(REGIONS)).fill_null(False)
    )
    motivo = (
        pl.when(pl.col("produto_norm").is_null())
        .then(pl.lit("produto_fora_da_lista"))
        .when(~loc_ok)
        .then(pl.lit("localizacao_invalida"))
        .when(pl.col("cnpj").is_null())
        .then(pl.lit("cnpj_vazio"))
        .when(pl.col("data").is_null())
        .then(pl.lit("data_invalida"))
        .when(pl.col("venda").is_null())
        .then(pl.lit("valor_venda_vazio"))
        .when(pl.col("venda") <= 0)
        .then(pl.lit("valor_venda_nao_positivo"))
        .when(pl.col("venda") > MAX_VALOR)
        .then(pl.lit("valor_venda_fora_da_faixa"))
        .when(pl.col("unidade").is_null())
        .then(pl.lit("unidade_vazia"))
        .otherwise(None)
        .alias("motivo")
    )
    # valor_compra inválido ou fora da faixa vira nulo (a coluna é opcional).
    compra = (
        pl.when((pl.col("compra") > 0) & (pl.col("compra") <= MAX_VALOR))
        .then(pl.col("compra").round(3))
        .otherwise(None)
    )
    return df.with_columns(motivo).select(
        "regiao",
        "uf",
        "municipio",
        "cnpj",
        "razao_social",
        "bandeira",
        "bairro",
        "cep",
        pl.col("produto_norm").alias("produto"),
        pl.col("data").alias("data_coleta"),
        pl.col("venda").round(3).cast(pl.Decimal(6, 3), strict=False).alias("valor_venda"),
        compra.cast(pl.Decimal(6, 3), strict=False).alias("valor_compra"),
        "unidade",
        "motivo",
    )


def normalize_files(paths: list[Path]) -> NormalizeResult:
    """Normaliza os CSVs dados (na ordem dada) e devolve as linhas aceitas e o relatório.

    Uma coleta repetida exatamente (mesmas colunas de `COLETA_KEY`), no mesmo arquivo
    ou em arquivos diferentes, entra uma vez só; as repetições contam como
    `linha_duplicada` no arquivo em que aparecem.
    """
    reports: list[FileReport] = []
    reasons_by_file: list[Counter[str]] = []
    accepted: list[pl.DataFrame] = []
    for i, path in enumerate(paths):
        path = Path(path)
        raw, encoding = read_raw_csv(path)
        df = clean_frame(raw)
        reports.append(FileReport(name=path.name, encoding=encoding, rows_read=df.height))
        reasons_by_file.append(Counter(df.get_column("motivo").drop_nulls().to_list()))
        ok = df.filter(pl.col("motivo").is_null()).drop("motivo")
        accepted.append(ok.with_columns(pl.lit(i, dtype=pl.UInt32).alias("_file")))

    if accepted:
        frame = pl.concat(accepted)
        dup = frame.select(~pl.struct(list(COLETA_KEY)).is_first_distinct()).to_series()
        dup_counts = frame.filter(dup).get_column("_file").value_counts()
        for file_idx, n in dup_counts.iter_rows():
            reasons_by_file[file_idx]["linha_duplicada"] += n
        frame = frame.filter(~dup).drop("_file")
    else:
        frame = pl.DataFrame(schema=PARQUET_SCHEMA)

    for report, reasons in zip(reports, reasons_by_file, strict=True):
        report.rejected = {r: reasons[r] for r in REJECT_REASONS if reasons.get(r)}
        report.rows_accepted = report.rows_read - sum(report.rejected.values())

    frame = frame.select(list(PARQUET_SCHEMA)).cast(PARQUET_SCHEMA)
    # Ordem total e estável: o Parquet não depende da ordem das linhas nos CSVs.
    frame = frame.sort(list(PARQUET_SCHEMA), nulls_last=False, maintain_order=True)
    return NormalizeResult(frame=frame, files=reports)


def write_parquet(frame: pl.DataFrame, path: Path) -> Path:
    """Grava o Parquet em arquivo temporário e renomeia no fim."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.write_parquet(tmp, compression="zstd", statistics=True)
    tmp.replace(path)
    return path
