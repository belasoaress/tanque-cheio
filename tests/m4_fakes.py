"""Peças de teste do M4: cliente falso da API e banco de brinquedo.

Nenhum teste chama a API real nem abre o banco real ou os datasets de dev/teste.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb

M4_TOY_DDL = """
CREATE TABLE municipios (
    municipio_id INTEGER PRIMARY KEY,
    nome VARCHAR NOT NULL,
    uf VARCHAR(2) NOT NULL,
    regiao VARCHAR(2) NOT NULL,
    capital BOOLEAN NOT NULL
);
CREATE TABLE postos (
    posto_id INTEGER PRIMARY KEY,
    cnpj VARCHAR NOT NULL UNIQUE,
    razao_social VARCHAR,
    bandeira VARCHAR,
    bairro VARCHAR,
    cep VARCHAR,
    municipio_id INTEGER NOT NULL REFERENCES municipios(municipio_id)
);
CREATE TABLE coletas (
    coleta_id BIGINT PRIMARY KEY,
    posto_id INTEGER NOT NULL REFERENCES postos(posto_id),
    produto VARCHAR NOT NULL,
    data_coleta DATE NOT NULL,
    valor_venda DECIMAL(6,3) NOT NULL,
    valor_compra DECIMAL(6,3),
    unidade VARCHAR NOT NULL
);
INSERT INTO municipios VALUES
    (1, 'SAO PAULO', 'SP', 'SE', true),
    (2, 'CAMPINAS', 'SP', 'SE', false),
    (3, 'RECIFE', 'PE', 'NE', true);
INSERT INTO postos VALUES
    (10, '00.000.000/0001-10', 'POSTO A', 'IPIRANGA', 'CENTRO', '01000-000', 1),
    (11, '00.000.000/0001-11', 'POSTO B', 'BRANCA', 'MOOCA', '03100-000', 1),
    (20, '00.000.000/0001-20', 'POSTO C', 'VIBRA', 'CAMBUI', '13000-000', 2),
    (30, '00.000.000/0001-30', 'POSTO D', 'IPIRANGA', 'BOA VIAGEM', '51000-000', 3);
INSERT INTO coletas VALUES
    (1, 10, 'GASOLINA', DATE '2026-08-03', 6.190, NULL, 'R$ / litro'),
    (2, 10, 'ETANOL', DATE '2026-08-03', 4.290, NULL, 'R$ / litro'),
    (3, 11, 'GASOLINA', DATE '2026-09-10', 6.090, NULL, 'R$ / litro'),
    (4, 20, 'GASOLINA', DATE '2026-09-01', 6.490, NULL, 'R$ / litro'),
    (5, 20, 'ETANOL', DATE '2026-09-01', 4.990, NULL, 'R$ / litro'),
    (6, 30, 'DIESEL S10', DATE '2026-09-15', 6.010, NULL, 'R$ / litro'),
    (7, 30, 'GASOLINA', DATE '2026-09-15', 6.350, NULL, 'R$ / litro');
"""


def make_toy_db(path: Path) -> Path:
    con = duckdb.connect(str(path))
    try:
        con.execute(M4_TOY_DDL)
    finally:
        con.close()
    return path


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


DEFAULT_USAGE = {
    "input_tokens": 100,
    "output_tokens": 50,
    "cache_read_input_tokens": 1000,
    "cache_creation_input_tokens": 0,
}

Responder = Callable[[dict[str, Any]], str]


def message(text: str, usage: dict | None = None, stop_reason: str = "end_turn") -> dict:
    return {
        "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": text}],
        "usage": dict(usage or DEFAULT_USAGE),
        "stop_reason": stop_reason,
    }


class FakeBatches:
    def __init__(self, owner: FakeMessages) -> None:
        self.owner = owner
        self.created: dict[str, list[dict]] = {}
        self.status = "ended"
        self.fail_ids: set[str] = set()
        self.drop_ids: set[str] = set()
        self.results_calls = 0

    def create(self, requests: list[dict]) -> dict:
        batch_id = f"msgbatch_fake{len(self.created) + 1}"
        self.created[batch_id] = list(requests)
        return {"id": batch_id, "processing_status": "in_progress"}

    def retrieve(self, batch_id: str) -> dict:
        return {
            "id": batch_id,
            "processing_status": self.status,
            "request_counts": {"processing": 0, "succeeded": len(self.created.get(batch_id, []))},
        }

    def results(self, batch_id: str):
        self.results_calls += 1
        for req in self.created[batch_id]:
            cid = req["custom_id"]
            if cid in self.drop_ids:
                continue
            if cid in self.fail_ids:
                yield {
                    "custom_id": cid,
                    "result": {"type": "errored", "error": {"type": "api_error"}},
                }
                continue
            yield {
                "custom_id": cid,
                "result": {"type": "succeeded", "message": self.owner.respond(req["params"])},
            }


class FakeMessages:
    def __init__(self, responder: Responder, usage: dict | None, prefix_tokens: int | None):
        self.responder = responder
        self.usage = usage
        self.prefix_tokens = prefix_tokens
        self.batches = FakeBatches(self)
        self.create_calls: list[dict] = []
        self.count_calls: list[dict] = []
        self.stop_reason = "end_turn"
        self.raise_for: set[str] = set()

    def respond(self, params: dict) -> dict:
        return message(self.responder(params), self.usage, self.stop_reason)

    def create(self, **params: Any) -> dict:
        self.create_calls.append(params)
        question = params["messages"][-1]["content"]
        if any(token in question for token in self.raise_for):
            raise RuntimeError("falha simulada da API")
        return self.respond(params)

    def count_tokens(self, **kwargs: Any) -> dict:
        self.count_calls.append(kwargs)
        if self.prefix_tokens is None:
            raise RuntimeError("count_tokens indisponível")
        return {"input_tokens": self.prefix_tokens}


class FakeClient:
    """Imita `anthropic.Anthropic()` no que o projeto usa."""

    def __init__(
        self,
        responder: Responder | None = None,
        *,
        usage: dict | None = None,
        prefix_tokens: int | None = 900,
    ) -> None:
        self.messages = FakeMessages(responder or (lambda _: "SELECT 1"), usage, prefix_tokens)
