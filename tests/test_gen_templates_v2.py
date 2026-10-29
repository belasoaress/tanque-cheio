"""Templates do treino v2 (M15, D34): padrões que o ft-v1 errou no dev2.

Como em `test_gen_templates.py`, nada aqui abre o banco real nem os conjuntos de
avaliação: tudo roda no banco sintético de lá.
"""

from __future__ import annotations

import re

import duckdb
import pytest

from tanque.gen import intents, templates
from test_gen_templates import make_synthetic_db

V2_INTENTS = [
    "postos_bairro",
    "filtro_contagem",
    "serie_trimestral",
    "postos_condicao_preco",
    "tanto_quanto",
    "todos_os",
    "participacao_pct",
    "regiao_vs_estado",
    "media_da_serie",
    "nunca_coleta",
    "variacao_grupos",
    "regra70_dois_periodos",
    "compara_produtos_grupo",
    "razao_por_grupo",
    "posicao_ranking",
    "acumulado",
    "pico_por_grupo",
]
V2_WINDOWS = {"posicao_ranking", "acumulado", "pico_por_grupo"}


@pytest.fixture(scope="module")
def synth_db(tmp_path_factory):
    return make_synthetic_db(tmp_path_factory.mktemp("synth_v2") / "synth.duckdb")


@pytest.fixture(scope="module")
def produced(synth_db):
    """Até 40 candidatos aceitos por intenção v2 (e por algumas da v1 que mudaram)."""
    con = duckdb.connect(str(synth_db), read_only=True)
    try:
        prod = templates._Producer(con, templates.load_catalog(con), seed=42)
        for intent_id in [*V2_INTENTS, "ranking_capitais", "lado_a_lado_produtos"]:
            prod.fill(intent_id, 40)
        yield {i: list(prod.out[i]) for i in prod.out}, con
    finally:
        con.close()


def _sqls(produced, intent_id):
    return [c["sql"] for c in produced[0][intent_id]]


# --- catálogo ------------------------------------------------------------------------


def test_v2_intents_are_in_the_catalog_with_generators():
    for intent_id in V2_INTENTS:
        assert intent_id in intents.INTENTS_BY_ID
        assert intent_id in templates.GENERATORS
    assert {i for i in V2_INTENTS if intents.INTENTS_BY_ID[i].uses_window} == V2_WINDOWS
    assert intents.INTENTS_BY_ID["postos_bairro"].level == "facil"
    medios = {"filtro_contagem", "serie_trimestral", "postos_condicao_preco"}
    assert {i for i in V2_INTENTS if intents.INTENTS_BY_ID[i].level == "medio"} == medios


def test_gentilicos_cover_every_uf_but_df():
    assert set(templates.UF_GENTILICOS) == set(templates.UF_NAMES) - {"DF"}
    assert templates.UF_GENTILICOS["RS"] == ("gaúchos", "gaúchas")
    assert templates.UF_GENTILICOS["SP"] == ("paulistas", "paulistas")


@pytest.mark.parametrize("intent_id", V2_INTENTS)
def test_every_v2_intent_produces_candidates(produced, intent_id):
    cands = produced[0][intent_id]
    assert len(cands) >= 10, intent_id
    con = produced[1]
    for c in cands:
        assert c["intent"] == intent_id
        assert "ROUND(" not in c["sql"].upper()
        if "m.nome = " in c["sql"]:
            assert re.search(r"m\.nome = '[^']+' AND m\.uf = '[A-Z]{2}'", c["sql"]), c["sql"]
        if c["order_matters"]:
            assert "ORDER BY" in c["sql"]
        rows = con.execute(c["sql"]).fetchall()
        assert rows and all(v is not None for r in rows for v in r)
        assert ("OVER (" in c["sql"]) == (intent_id in V2_WINDOWS)


# --- convenções de cada padrão ---------------------------------------------------------


def test_postos_bairro_filters_with_the_anp_spelling(produced):
    for c in produced[0]["postos_bairro"]:
        bairro = re.search(r"p\.bairro = '([^']+)'", c["sql"]).group(1)
        assert bairro == bairro.upper()
        assert bairro not in c["question"]  # a pergunta usa a grafia de uma pessoa
        assert templates.city_display(bairro).lower() in c["question"].lower()


def test_filtro_contagem_returns_entity_and_count(produced):
    for sql in _sqls(produced, "filtro_contagem"):
        assert re.search(r"HAVING COUNT\(\*\) [<>] \d+$", sql), sql
        assert re.match(r"SELECT (p\.bandeira|m\.uf|m\.nome|m\.nome, m\.uf), COUNT\(\*\) AS ", sql)
        if "m.uf = '" in sql and sql.startswith("SELECT m.nome"):  # cidades de uma UF
            assert sql.startswith("SELECT m.nome, COUNT")


def test_serie_trimestral_uses_quarter_and_year(produced):
    for sql in _sqls(produced, "serie_trimestral"):
        assert re.match(r"SELECT (quarter|year)\(c\.data_coleta\) AS (trimestre|ano), ", sql)
        assert "date_trunc('quarter'" not in sql


def test_postos_condicao_preco_counts_distinct_postos(produced):
    for sql in _sqls(produced, "postos_condicao_preco"):
        assert re.match(r"SELECT COUNT\(DISTINCT (c\.posto_id|m\.municipio_id)\) AS ", sql)
        assert re.search(r"c\.valor_venda [<>] \d+(\.\d+)?", sql)


def test_tanto_quanto_uses_intersect(produced):
    for sql in _sqls(produced, "tanto_quanto"):
        assert sql.count(" INTERSECT ") == 1


def test_todos_os_requires_every_member(produced):
    for sql in _sqls(produced, "todos_os"):
        assert re.search(
            r"HAVING (bool_and\(razao [<>]=? 0\.7\)|(MIN|MAX)\(preco\) [<>] )", sql
        ) or (re.search(r"HAVING COUNT\(\*\) = \d+ AND (MIN|MAX)\(preco\) [<>] ", sql)), sql


def test_participacao_pct_is_a_share_of_the_group(produced):
    for sql in _sqls(produced, "participacao_pct"):
        assert re.search(
            r"100\.0 \* COUNT\(CASE WHEN (p\.bandeira|c\.produto) = '[^']+' THEN 1 END\)"
            r" / COUNT\(\*\)",
            sql,
        )


def test_regiao_vs_estado_compares_levels(produced):
    for sql in _sqls(produced, "regiao_vs_estado"):
        assert (
            "AVG(CASE WHEN m.capital THEN" in sql
            and "AVG(CASE WHEN NOT m.capital THEN" in sql
            or "WITH regional AS" in sql
            or ("WITH e AS" in sql and "m.regiao = '" in sql)
        ), sql


def test_media_da_serie_compares_with_the_whole_period(produced):
    for sql in _sqls(produced, "media_da_serie"):
        assert sql.startswith(("WITH mensal AS", "WITH pp AS"))
        assert "(SELECT AVG(c.valor_venda) " in sql or "MAX(preco) - MIN(preco)" in sql


def test_nunca_coleta_uses_not_in(produced):
    for sql in _sqls(produced, "nunca_coleta"):
        assert re.search(r"(m\.municipio_id|p\.posto_id) NOT IN \(SELECT ", sql)


def test_variacao_grupos_joins_two_periods_by_group(produced):
    for sql in _sqls(produced, "variacao_grupos"):
        assert re.search(r"FROM a JOIN b ON a\.(regiao|bandeira|produto) = b\.\1", sql), sql


def test_regra70_dois_periodos_keeps_the_period_of_each_condition(produced):
    """'compensou em X, mas não em Y': a condição <= 0,7 vai para o período citado antes.

    Os candidatos crus do gerador (antes da execução), para cobrir as duas ordens.
    """
    import random

    con = produced[1]
    t = templates.Ctx(random.Random(7), templates.load_catalog(con), con)
    checked = 0
    for _ in range(300):
        try:
            c = templates.g_regra70_dois_periodos(t)
        except templates.Skip:
            continue
        q, sql = c.question, c.sql
        assert sql.count(" AS razao ") == 2
        assert re.search(r"DATE '2025-.*\), b AS \(.*DATE '2026-", sql)
        if ", mas não" not in q and " e não " not in q:
            continue
        first = re.search(r"20(25|26)", q).group(0)
        cte = "a" if first == "2025" else "b"  # a = recorte de 2025, b = de 2026
        other = "b" if cte == "a" else "a"
        assert f"{cte}.razao <= 0.7 AND {other}.razao > 0.7" in sql, (q, sql)
        checked += 1
    assert checked >= 20


def test_compara_produtos_grupo_is_not_the_70_rule(produced):
    for sql in _sqls(produced, "compara_produtos_grupo"):
        assert "0.7" not in sql
        assert re.search(r"HAVING AVG\(CASE .*END\) [<>] AVG\(CASE .*END\)", sql)


def test_razao_por_grupo_lists_the_ratio_without_filtering(produced):
    for sql in _sqls(produced, "razao_por_grupo"):
        assert templates.RATIO in sql and "0.7" not in sql
        assert "ORDER BY" not in sql


def test_posicao_ranking_returns_only_the_position(produced):
    for sql in _sqls(produced, "posicao_ranking"):
        assert "RANK() OVER (ORDER BY " in sql
        assert re.search(r"SELECT posicao FROM r WHERE ", sql)


def test_acumulado_is_a_running_sum(produced):
    for sql in _sqls(produced, "acumulado"):
        assert "SUM(coletas) OVER (ORDER BY mes) AS acumulado" in sql


def test_pico_por_grupo_returns_group_and_item_without_the_value(produced):
    for sql in _sqls(produced, "pico_por_grupo"):
        assert "ROW_NUMBER() OVER (PARTITION BY " in sql
        assert re.search(r"SELECT \w+, (mes|bandeira) FROM r WHERE pos = 1", sql), sql


# --- reforços em intenções da v1 ------------------------------------------------------------


def test_capitals_of_a_region_return_only_the_name(produced):
    for sql in _sqls(produced, "ranking_capitais"):
        if "m.regiao = '" in sql:
            assert sql.startswith("SELECT m.nome, AVG(") and "GROUP BY m.nome " in sql
        else:
            assert sql.startswith("SELECT m.nome, m.uf, AVG(")


def test_two_products_in_one_place_are_two_rows(produced):
    singles = [s for s in _sqls(produced, "lado_a_lado_produtos") if "GROUP BY" not in s]
    assert not singles  # "o preço médio de A e de B" num só lugar = GROUP BY c.produto
    assert any(
        s.startswith("SELECT c.produto, AVG(") for s in _sqls(produced, "lado_a_lado_produtos")
    )
