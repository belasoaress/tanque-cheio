import random
import time

import pytest

from m4_fakes import write_jsonl
from tanque.gen import leakage


def item(i, question, sql="SELECT 1"):
    return {"id": f"x-{i}", "question": question, "sql": sql}


def test_normalize_question():
    assert leakage.normalize_question("  Qual o PREÇO médio, em São Paulo?! ") == (
        "qual o preco medio em sao paulo"
    )
    assert leakage.normalize_question("Diesel-S10/etanol") == "diesel s10 etanol"


def test_normalize_sql_ignores_case_spaces_and_semicolon():
    a = "select AVG(valor_venda)  from coletas\n where produto='ETANOL';"
    b = "SELECT avg(valor_venda) FROM coletas WHERE produto = 'ETANOL'"
    assert leakage.normalize_sql(a) == leakage.normalize_sql(b)
    assert leakage.normalize_sql(a) != leakage.normalize_sql(b.replace("ETANOL", "GNV"))


def test_normalize_sql_falls_back_on_unparseable():
    assert leakage.normalize_sql("SELECT ((( Foo") == "select(((foo"


def test_question_collision_at_threshold():
    ref = [item(0, "Qual o preço médio do etanol em Minas Gerais no último mês?", "SELECT 0")]
    train = [
        item(1, "qual o preco medio do etanol em minas gerais no ultimo mes"),  # igual
        item(2, "Me diga qual o preço médio do etanol em Minas Gerais no último mês, por favor"),
        item(3, "Quantos postos da Vibra existem no Rio Grande do Sul?"),
    ]
    found = leakage.find_collisions(train, ref)
    assert [(c.index, c.kind, c.other_id) for c in found] == [
        (0, "pergunta", "x-0"),
        (1, "pergunta", "x-0"),
    ]
    assert all(c.score >= 85 for c in found)


def test_sql_collision_even_with_different_question():
    ref = [item(0, "Pergunta de referência sobre bandeiras", "SELECT bandeira FROM postos")]
    train = [item(1, "Texto totalmente diferente aqui", "select BANDEIRA\nfrom postos;")]
    (c,) = leakage.find_collisions(train, ref)
    assert c.kind == "sql" and c.other_id == "x-0"


def test_remove_leaks_keeps_clean_items():
    ref = [item(0, "Qual o preço médio do GNV no Brasil?", "SELECT 2")]
    train = [
        item(1, "qual o preço médio do gnv no brasil"),
        item(2, "Ranking de bairros de Recife"),
    ]
    kept, leaks = leakage.remove_leaks(train, ref)
    assert [k["id"] for k in kept] == ["x-2"]
    assert len(leaks) == 1


def test_dedupe_keeps_first_of_each_group():
    items = [
        item(1, "Qual o preço médio do etanol em Goiás?", "SELECT 1"),
        item(2, "qual o preco medio do etanol em goias", "SELECT 2"),  # pergunta repetida
        item(3, "Liste os postos mais baratos de Natal", "select 1"),  # SQL repetida
        item(4, "Quantas coletas de GNV houve em 2025?", "SELECT 4"),
    ]
    kept, removed = leakage.dedupe(items)
    assert [k["id"] for k in kept] == ["x-1", "x-4"]
    assert [(c.index, c.kind, c.other_index) for c in removed] == [
        (1, "pergunta", 0),
        (2, "sql", 0),
    ]


def test_load_reference_requires_dev_and_test(tmp_path):
    write_jsonl(tmp_path / "dev.jsonl", [item(0, "a")])
    with pytest.raises(leakage.MissingEvalSetError):
        leakage.load_reference(tmp_path)
    write_jsonl(tmp_path / "test.jsonl", [item(1, "b")])
    items, files = leakage.load_reference(tmp_path)
    assert [i["id"] for i in items] == ["x-0", "x-1"]
    write_jsonl(tmp_path / "candidates_reserve.jsonl", [item(2, "c")])
    items, files = leakage.load_reference(tmp_path)
    assert [f.name for f in files] == ["dev.jsonl", "test.jsonl", "candidates_reserve.jsonl"]
    assert len(items) == 3


def _random_questions(n, seed):
    rng = random.Random(seed)
    words = (
        "preco medio gasolina etanol diesel gnv postos bandeira capital municipio estado regiao "
        "ultimo mes ano ranking mais caro barato variacao comparacao coletas media nordeste sul "
        "norte sudeste quantos quais onde quando compensa razao bairro cidade"
    ).split()
    return [" ".join(rng.sample(words, rng.randint(6, 12))) + f" {i}" for i in range(n)]


def test_performance_5k_against_200_and_internal_dedupe():
    train = [item(i, q, f"SELECT {i}") for i, q in enumerate(_random_questions(5000, 1))]
    ref = [item(i, q, f"SELECT -{i}") for i, q in enumerate(_random_questions(200, 2))]
    start = time.perf_counter()
    leakage.find_collisions(train, ref)
    leakage.dedupe(train[:1500])
    assert time.perf_counter() - start < 30
