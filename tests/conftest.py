"""Fixtures compartilhadas.

As fixtures do M2 têm prefixo `m2_` para não colidir com as de outros marcos.
Nenhuma delas usa o banco real (data/tanque.duckdb): o banco de brinquedo é
criado em `tmp_path` com as três tabelas da seção 5.1 reduzidas ao mínimo.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

M2_TOY_DDL = """
CREATE TABLE municipios (
    municipio_id INTEGER PRIMARY KEY,
    nome VARCHAR NOT NULL,
    uf VARCHAR(2) NOT NULL,
    regiao VARCHAR(2) NOT NULL
);
CREATE TABLE postos (
    posto_id INTEGER PRIMARY KEY,
    cnpj VARCHAR NOT NULL UNIQUE,
    bandeira VARCHAR,
    municipio_id INTEGER NOT NULL REFERENCES municipios(municipio_id)
);
CREATE TABLE coletas (
    coleta_id BIGINT PRIMARY KEY,
    posto_id INTEGER NOT NULL REFERENCES postos(posto_id),
    produto VARCHAR NOT NULL,
    data_coleta DATE NOT NULL,
    valor_venda DECIMAL(6,3) NOT NULL
);
INSERT INTO municipios VALUES
    (1, 'SAO CARLOS', 'SP', 'SE'),
    (2, 'RECIFE', 'PE', 'NE'),
    (3, 'CURITIBA', 'PR', 'S');
INSERT INTO postos VALUES
    (10, '00.000.000/0001-10', 'IPIRANGA', 1),
    (11, '00.000.000/0001-11', 'BRANCA', 1),
    (20, '00.000.000/0001-20', 'SHELL', 2),
    (30, '00.000.000/0001-30', 'IPIRANGA', 3);
INSERT INTO coletas VALUES
    (1, 10, 'GASOLINA', DATE '2026-08-03', 6.190),
    (2, 10, 'ETANOL', DATE '2026-08-03', 4.290),
    (3, 11, 'GASOLINA', DATE '2026-08-10', 6.090),
    (4, 20, 'GASOLINA', DATE '2026-09-01', 6.490),
    (5, 20, 'ETANOL', DATE '2026-09-01', 4.990),
    (6, 30, 'DIESEL S10', DATE '2026-09-15', 6.010),
    (7, 30, 'GASOLINA', DATE '2026-09-15', 6.350);
"""

# Exemplos no formato da seção 5.2, com SQLs que rodam no banco de brinquedo.
M2_TOY_ITEMS = [
    {
        "id": "toy-0001",
        "question": "Qual o preço médio da gasolina em SP?",
        "sql": (
            "SELECT AVG(c.valor_venda) FROM coletas c JOIN postos p USING (posto_id) "
            "JOIN municipios m USING (municipio_id) "
            "WHERE c.produto = 'GASOLINA' AND m.uf = 'SP'"
        ),
        "difficulty": "facil",
        "uses_window": False,
        "order_matters": False,
    },
    {
        "id": "toy-0002",
        "question": "Quantas coletas há por produto?",
        "sql": "SELECT produto, COUNT(*) FROM coletas GROUP BY produto",
        "difficulty": "facil",
        "uses_window": False,
        "order_matters": False,
    },
    {
        "id": "toy-0003",
        "question": "Quais estados têm a gasolina mais cara, do maior para o menor?",
        "sql": (
            "SELECT m.uf, MAX(c.valor_venda) AS preco FROM coletas c "
            "JOIN postos p USING (posto_id) JOIN municipios m USING (municipio_id) "
            "WHERE c.produto = 'GASOLINA' GROUP BY m.uf ORDER BY preco DESC"
        ),
        "difficulty": "medio",
        "uses_window": False,
        "order_matters": True,
    },
    {
        "id": "toy-0004",
        "question": "Qual a razão etanol/gasolina por município?",
        "sql": (
            "WITH p AS (SELECT m.nome, c.produto, AVG(c.valor_venda) AS v FROM coletas c "
            "JOIN postos USING (posto_id) JOIN municipios m USING (municipio_id) "
            "GROUP BY ALL) "
            "SELECT e.nome, ROUND(e.v / g.v, 3) FROM p e JOIN p g ON e.nome = g.nome "
            "WHERE e.produto = 'ETANOL' AND g.produto = 'GASOLINA' ORDER BY 1"
        ),
        "difficulty": "dificil",
        "uses_window": False,
        "order_matters": True,
    },
]


def _m2_complete(item: dict, split: str) -> dict:
    return {
        **item,
        "split": split,
        "source": "human_reviewed",
        "reviewed": True,
        "tags": [],
    }


@pytest.fixture
def m2_toy_db(tmp_path: Path) -> Path:
    """Caminho de um DuckDB minúsculo com municipios, postos e coletas."""
    path = tmp_path / "toy.duckdb"
    con = duckdb.connect(str(path))
    try:
        con.execute(M2_TOY_DDL)
    finally:
        con.close()
    return path


@pytest.fixture
def m2_toy_items() -> list[dict]:
    """Itens de dataset (seção 5.2) válidos no banco de brinquedo."""
    return [_m2_complete(item, "dev") for item in M2_TOY_ITEMS]


def _m2_write_jsonl(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


@pytest.fixture
def m2_write_jsonl():
    """Função `(path, rows) -> path` que grava um JSONL em UTF-8."""
    return _m2_write_jsonl


@pytest.fixture
def m2_dataset(tmp_path: Path, m2_toy_items: list[dict]) -> Path:
    """Arquivo JSONL com os itens de brinquedo."""
    return _m2_write_jsonl(tmp_path / "dev.jsonl", m2_toy_items)
