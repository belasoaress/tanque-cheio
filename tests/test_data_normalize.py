"""Testes de src/tanque/data/normalize.py sobre as fixtures e casos pequenos."""

from datetime import date
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest

from tanque.data.normalize import (
    PRODUCTS,
    NormalizeError,
    decode_bytes,
    map_columns,
    normalize_column_name,
    normalize_files,
    normalize_product,
    read_raw_csv,
)

FIXTURES = Path(__file__).parent / "fixtures"
FIX_A = FIXTURES / "anp_sample_2025-01_gasolina-etanol.csv"  # UTF-8 com BOM, CRLF
FIX_B = FIXTURES / "anp_sample_2025-02_diesel-gnv.csv"  # Latin-1, cabeçalho com acentos
FIX_C = FIXTURES / "anp_sample_2025-03_gasolina-etanol.csv"  # UTF-8 sem BOM, colunas trocadas
ALL = [FIX_A, FIX_B, FIX_C]

HEADER = (
    "Regiao - Sigla;Estado - Sigla;Municipio;Revenda;CNPJ da Revenda;Nome da Rua;Numero Rua;"
    "Complemento;Bairro;Cep;Produto;Data da Coleta;Valor de Venda;Valor de Compra;"
    "Unidade de Medida;Bandeira"
)


def _csv(tmp_path: Path, rows: list[str], name: str = "x.csv", encoding="utf-8") -> Path:
    path = tmp_path / name
    path.write_bytes(("\n".join([HEADER, *rows]) + "\n").encode(encoding))
    return path


def _row(produto="GASOLINA", data="01/10/2024", venda="5,39", compra="", bandeira="IPIRANGA"):
    return (
        f"SE;SP;SAO CARLOS;POSTO X LTDA; 01.234.567/0001-89;RUA A;1;;CENTRO;13560-000;"
        f"{produto};{data};{venda};{compra};R$ / litro;{bandeira}"
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Estado - Sigla", "estado sigla"),
        ("Estado-Sigla", "estado sigla"),
        ("Valor  de Venda", "valor de venda"),
        ("﻿Região - Sigla", "regiao sigla"),
        ("MUNICÍPIO", "municipio"),
        ("Razão Social", "razao social"),
    ],
)
def test_normalize_column_name(raw, expected):
    assert normalize_column_name(raw) == expected


def test_map_columns_accepts_spelling_variants():
    cols = [
        "Região - Sigla",
        "Estado-Sigla",
        "Município",
        "Razão Social",
        "CNPJ da Revenda",
        "Produto",
        "Data  da Coleta",
        "Valor  de Venda",
        "Unidade de Medida",
        "Bandeira",
        "Coluna Nova Desconhecida",
    ]
    mapping = map_columns(cols)
    assert mapping["Razão Social"] == "razao_social"
    assert mapping["Valor  de Venda"] == "valor_venda"
    assert "Coluna Nova Desconhecida" not in mapping


def test_map_columns_missing_required_column_is_clear():
    with pytest.raises(NormalizeError, match="valor_venda"):
        map_columns(["Regiao - Sigla", "Estado - Sigla", "Municipio", "CNPJ da Revenda"])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("GASOLINA", "GASOLINA"),
        ("Gasolina Aditivada", "GASOLINA ADITIVADA"),
        ("  etanol ", "ETANOL"),
        ("Diesel S10", "DIESEL S10"),
        ("DIESEL S-10", "DIESEL S10"),
        ("gnv", "GNV"),
        ("GLP", None),
        ("OLEO DIESEL B S500", None),
        (None, None),
    ],
)
def test_normalize_product(raw, expected):
    assert normalize_product(raw) == expected


def test_closed_product_list_has_six_automotive_products():
    assert set(PRODUCTS) == {
        "GASOLINA",
        "GASOLINA ADITIVADA",
        "ETANOL",
        "DIESEL",
        "DIESEL S10",
        "GNV",
    }


def test_decode_bytes_utf8_bom_and_latin1():
    assert decode_bytes("﻿SÃO".encode()) == ("SÃO", "utf-8")
    assert decode_bytes("SÃO".encode("latin-1")) == ("SÃO", "latin-1")


def test_fixture_encodings_detected():
    assert read_raw_csv(FIX_A)[1] == "utf-8"
    assert read_raw_csv(FIX_B)[1] == "latin-1"
    assert read_raw_csv(FIX_C)[1] == "utf-8"
    # Mesmas colunas canônicas, apesar das grafias e da ordem diferentes.
    cols = [set(read_raw_csv(p)[0].columns) for p in ALL]
    assert cols[0] == cols[1] == cols[2]


def test_decimal_comma_and_dates(tmp_path):
    path = _csv(tmp_path, [_row(venda="5,39", compra="4,871", data="31/12/2024")])
    df = normalize_files([path]).frame
    assert df["valor_venda"].to_list() == [Decimal("5.390")]
    assert df["valor_compra"].to_list() == [Decimal("4.871")]
    assert df["data_coleta"].to_list() == [date(2024, 12, 31)]
    assert df.schema["valor_venda"] == pl.Decimal(6, 3)
    assert df["cnpj"].to_list() == ["01.234.567/0001-89"]  # espaço inicial removido


def test_rejections_counted_by_reason_instead_of_failing(tmp_path):
    rows = [
        _row(),
        _row(produto="GLP"),
        _row(venda=""),
        _row(venda="0,00"),
        _row(venda="-2,00"),
        _row(data="31/02/2025"),
        _row(data=""),
        _row(),  # duplicada exata da primeira
    ]
    result = normalize_files([_csv(tmp_path, rows)])
    assert result.frame.height == 1
    assert result.rejected == {
        "data_invalida": 2,
        "linha_duplicada": 1,
        "produto_fora_da_lista": 1,
        "valor_venda_nao_positivo": 2,
        "valor_venda_vazio": 1,
    }
    report = result.files[0]
    assert (report.rows_read, report.rows_accepted) == (8, 1)


def test_empty_bandeira_becomes_branca(tmp_path):
    df = normalize_files([_csv(tmp_path, [_row(bandeira="")])]).frame
    assert df["bandeira"].to_list() == ["BRANCA"]


def test_duplicates_across_files_counted_in_later_file(tmp_path):
    a = _csv(tmp_path, [_row()], name="a.csv")
    b = _csv(tmp_path, [_row(), _row(produto="ETANOL")], name="b.csv", encoding="latin-1")
    result = normalize_files([a, b])
    assert result.frame.height == 2
    assert result.files[0].rejected == {}
    assert result.files[1].rejected == {"linha_duplicada": 1}


def test_fixtures_end_to_end_counts():
    result = normalize_files(ALL)
    by_file = {f.name: f for f in result.files}
    assert by_file[FIX_A.name].rejected == {"valor_venda_vazio": 1, "valor_venda_nao_positivo": 1}
    assert by_file[FIX_B.name].rejected == {
        "produto_fora_da_lista": 4,
        "data_invalida": 1,
        "valor_venda_nao_positivo": 1,
    }
    assert by_file[FIX_C.name].rejected == {"data_invalida": 1, "linha_duplicada": 1}
    df = result.frame
    assert result.rows_read == 113
    assert df.height == 103
    assert set(df["produto"].unique()) == set(PRODUCTS)
    assert df.filter(pl.col("valor_venda") <= 0).height == 0
    # Latin-1 decodificado e acento removido do município: "SÃO PAULO" -> "SAO PAULO".
    assert "SÃO PAULO" not in df["municipio"].to_list()
    assert df.filter(pl.col("produto") == "GNV")["unidade"].unique().to_list() == ["R$ / m³"]
    # O complemento com ';' entre aspas não desalinha as colunas.
    assert df.filter(pl.col("cnpj") == "33.333.333/0001-33")["cep"].unique().to_list() == [
        "13025-000"
    ]


def test_output_independent_of_file_order():
    a = normalize_files(ALL).frame
    b = normalize_files(list(reversed(ALL))).frame
    assert a.equals(b)
