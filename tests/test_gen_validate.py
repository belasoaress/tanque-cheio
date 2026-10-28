import pytest
from m4_fakes import make_toy_db

from tanque.gen import validate
from tanque.sql.execute import ExecResult

J = (
    "FROM coletas c JOIN postos p ON p.posto_id = c.posto_id "
    "JOIN municipios m ON m.municipio_id = p.municipio_id"
)


@pytest.fixture
def m4_db(tmp_path):
    return make_toy_db(tmp_path / "toy.duckdb")


def pair(question, sql, intent="media_produto_uf", **extra):
    return {"question": question, "sql": sql, "intent": intent, **extra}


def run_one(db, p):
    return validate.validate_pairs([p], db_path=db, timeout_s=5)


def test_valid_pair_becomes_train_example(m4_db):
    sql = f"SELECT   AVG(c.valor_venda)\n  {J}\n WHERE m.uf = 'SP' AND c.produto = 'GASOLINA';"
    res = run_one(m4_db, pair("Qual o preço médio da gasolina em SP?", sql, order_matters=False))
    assert res.discarded == []
    (item,) = res.valid
    assert item["sql"] == (f"SELECT AVG(c.valor_venda) {J} WHERE m.uf = 'SP' AND c.produto = 'GASOLINA'")
    assert item["difficulty"] == "facil"
    assert item["split"] == "train" and item["source"] == "synthetic"
    assert item["reviewed"] is False and item["uses_window"] is False
    assert item["order_matters"] is False
    assert item["tags"][0] == "intent:media_produto_uf"
    assert validate.intent_of(item) == "media_produto_uf"


def test_normalize_spaces_keeps_string_literals():
    sql = "SELECT  'R$  /  litro'  AS u,\n\t\"a  b\"  FROM t ;;"
    assert validate.normalize_spaces(sql) == "SELECT 'R$  /  litro' AS u, \"a  b\" FROM t"


@pytest.mark.parametrize(
    ("p", "reason"),
    [
        (pair("", "SELECT 1"), "campos_vazios"),
        (pair("Q?", "  "), "campos_vazios"),
        (pair("Q?", "SELECT 1", intent="nao_existe"), "intencao_desconhecida"),
        (pair("Q?", "SELECT FROM WHERE ((("), "parse"),
        (pair("Q?", "SELECT 1; SELECT 2"), "parse"),
        (pair("Q?", "SELECT * FROM read_csv('x.csv')"), "guardrail"),
        (pair("Q?", "DROP TABLE coletas"), "guardrail"),
        (pair("Q?", "SELECT coluna_que_nao_existe FROM coletas"), "erro_execucao"),
        (pair("Q?", "SELECT * FROM coletas WHERE produto = 'GLP'"), "resultado_vazio"),
        (pair("Q?", "SELECT AVG(valor_venda) FROM coletas WHERE produto = 'GNV'"), "resultado_nulo"),
    ],
)
def test_discard_reasons(m4_db, p, reason):
    res = run_one(m4_db, p)
    assert res.valid == []
    assert [d.reason for d in res.discarded] == [reason]
    assert res.reasons() == {reason: 1}


def test_timeout_reason_from_execution():
    def fake_run(sql, timeout_s=None, db_path=None):
        return ExecResult(error="timeout após 1s")

    res = validate.validate_pairs([pair("Q?", "SELECT 1")], run=fake_run)
    assert res.discarded[0].reason == "timeout"


def test_same_result_for_different_questions_of_same_intent_is_discarded(m4_db):
    generic = "SELECT AVG(valor_venda) FROM coletas WHERE produto = 'GASOLINA'"
    pairs = [
        pair("Preço médio da gasolina em SP?", generic),
        pair("Preço médio da gasolina em Pernambuco?", generic + " AND 1 = 1"),
        # mesma pergunta normalizada: não é sinal de SQL genérica (a dedupe trata)
        pair("preço médio da gasolina em SP", generic),
        # mesmo resultado, mas outra intenção: fica
        pair("Média nacional da gasolina?", generic, intent="media_nacional_periodo"),
    ]
    res = validate.validate_pairs(pairs, db_path=m4_db)
    assert [v["question"] for v in res.valid] == [
        "Preço médio da gasolina em SP?",
        "preço médio da gasolina em SP",
        "Média nacional da gasolina?",
    ]
    assert [d.reason for d in res.discarded] == ["resultado_repetido"]


def test_fingerprint_ignores_row_order_and_float_noise():
    a = ExecResult(columns=["x", "y"], rows=[("SP", 6.19), ("PE", 6.35)])
    b = ExecResult(columns=["u", "v"], rows=[("PE", 6.350000001), ("SP", 6.19)])
    c = ExecResult(columns=["x", "y"], rows=[("SP", 6.20), ("PE", 6.35)])
    assert validate.result_fingerprint(a) == validate.result_fingerprint(b)
    assert validate.result_fingerprint(a) != validate.result_fingerprint(c)


def test_window_detection_and_order_fallback(m4_db):
    sql = (
        "WITH x AS (SELECT data_coleta, AVG(valor_venda) AS p FROM coletas GROUP BY 1) "
        "SELECT data_coleta, p - LAG(p) OVER (ORDER BY data_coleta) FROM x ORDER BY data_coleta"
    )
    res = run_one(m4_db, pair("Variação diária?", sql, intent="janela_variacao_mensal"))
    (item,) = res.valid
    assert item["uses_window"] is True
    assert item["order_matters"] is True  # sem campo do gerador: vem do ORDER BY
    assert item["difficulty"] == "dificil"


def test_assign_ids_puts_id_first():
    items = validate.assign_ids([{"question": "a"}, {"question": "b"}])
    assert [i["id"] for i in items] == ["train-00001", "train-00002"]
    assert list(items[0])[0] == "id"
