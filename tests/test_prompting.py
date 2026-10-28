"""Testes de src/tanque/prompting.py (plan.md, seção 5.4)."""

import json

import pytest

from tanque import config, prompting
from tanque.data.normalize import PRODUCTS, REGIONS

QUESTION = "Qual o preço médio da gasolina em SP?"


def _manifest() -> dict:
    return json.loads(config.setting_path("manifest").read_text(encoding="utf-8"))


# --- build_prompt -------------------------------------------------------------


def test_build_prompt_is_chat_with_system_then_user():
    messages = prompting.build_prompt(QUESTION)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert all(isinstance(m["content"], str) and m["content"] for m in messages)


def test_question_comes_last_in_user_message():
    messages = prompting.build_prompt(QUESTION)
    assert messages[-1]["content"].rstrip().endswith(QUESTION)
    assert QUESTION not in messages[0]["content"]


def test_system_starts_with_short_instruction():
    system = prompting.system_prompt()
    assert system.startswith("Responda apenas com uma consulta SQL para DuckDB.")


def test_system_has_ddl_of_three_tables_with_column_comments():
    system = prompting.system_prompt()
    for table in ("municipios", "postos", "coletas"):
        assert f"CREATE TABLE {table} (" in system
    assert "capital       BOOLEAN NOT NULL" in system
    assert "'BRANCA' para sem bandeira" in system
    assert "'R$ / litro' ou 'R$ / m³'" in system
    # Comentário de contrato da 1ª linha do schema.sql não interessa ao modelo.
    assert "Não mudar sem aprovação" not in system
    assert "docs/schema.md" not in system


def test_system_lists_closed_values_of_produto():
    system = prompting.system_prompt()
    assert len(PRODUCTS) == 6
    for produto in PRODUCTS:
        assert f"'{produto}'" in system


def test_system_lists_regioes_with_meaning():
    system = prompting.system_prompt()
    assert set(prompting.REGION_NAMES) == set(REGIONS)
    for sigla, nome in prompting.REGION_NAMES.items():
        assert f"'{sigla}' = {nome}" in system
    assert prompting.REGION_NAMES["N"] == "Norte"
    assert prompting.REGION_NAMES["CO"] == "Centro-Oeste"


def test_produtos_come_from_manifest_and_match_closed_list():
    assert sorted(_manifest()["produtos"]) == sorted(PRODUCTS)
    assert prompting.reference_values()["produtos"] == sorted(PRODUCTS)


def test_system_has_rule_of_70_percent_and_capital():
    system = prompting.system_prompt()
    assert "0,7" in system
    assert "etanol" in system.lower() and "gasolina" in system.lower()
    assert "capital = true" in system


def test_system_has_reference_date_and_last_month_rule():
    system = prompting.system_prompt()
    ref = _manifest()["period"]["data_referencia"]
    assert ref == "2026-09-30"
    assert f"Data de referência: {ref}" in system
    assert "último mês" in system
    assert "mês do calendário que contém a data de referência" in system


def test_system_has_answer_conventions_d22():
    # D22: a convenção de colunas e de período do gold vai no prompt.
    system = prompting.system_prompt()
    assert "AVG(valor_venda)" in system
    assert "nome e uf" in system
    assert "só as colunas que identificam a resposta" in system
    assert "Sem período na pergunta = todo o período do banco" in system
    assert "date_trunc('month', data_coleta)" in system
    assert "'BRANCA'" in system


def test_system_mentions_municipio_names_upper_without_accents():
    system = prompting.system_prompt()
    assert "'SAO PAULO'" in system


def test_no_bandeiras_in_prompt():
    # Contrato 5.4: só produto e regiao (+ regra dos 70%).
    system = prompting.system_prompt()
    for bandeira in ("VIBRA", "IPIRANGA", "RAIZEN"):
        assert bandeira not in system


# --- determinismo e prefixo estável -------------------------------------------


def test_build_prompt_is_deterministic():
    assert prompting.build_prompt(QUESTION) == prompting.build_prompt(QUESTION)


def test_stable_prefix_does_not_depend_on_question():
    a = prompting.build_prompt(QUESTION)
    b = prompting.build_prompt("Em quais capitais o etanol compensou em 2026?")
    assert a[0] == b[0]
    assert a[0]["content"] == prompting.system_prompt()
    assert a[1] != b[1]


def test_user_message_is_just_the_question():
    messages = prompting.build_prompt(f"  {QUESTION}\n")
    assert messages[1]["content"] == f"Pergunta: {QUESTION}"


def test_render_system_prompt_is_pure():
    manifest = {"produtos": {"B": 1, "A": 2}, "period": {"data_referencia": "2030-01-31"}}
    ddl = "CREATE TABLE t (x INTEGER);"
    out = prompting.render_system_prompt(ddl, manifest)
    assert "Data de referência: 2030-01-31" in out
    assert "'A', 'B'" in out
    assert ddl in out


def test_render_system_prompt_fails_without_reference_date():
    with pytest.raises(KeyError):
        prompting.render_system_prompt("CREATE TABLE t (x INTEGER);", {"produtos": {"A": 1}})


# --- extract_sql ---------------------------------------------------------------


def test_extract_sql_fence_with_language():
    raw = "```sql\nSELECT 1;\n```"
    assert prompting.extract_sql(raw) == "SELECT 1"


def test_extract_sql_fence_without_language():
    raw = "```\nSELECT uf FROM municipios\n```"
    assert prompting.extract_sql(raw) == "SELECT uf FROM municipios"


def test_extract_sql_tilde_fence():
    raw = "~~~sql\nSELECT 2\n~~~"
    assert prompting.extract_sql(raw) == "SELECT 2"


def test_extract_sql_text_before_and_after_fence():
    raw = "Aqui está a consulta:\n\n```sql\nSELECT 1\nFROM coletas;\n```\n\nEla calcula X."
    assert prompting.extract_sql(raw) == "SELECT 1\nFROM coletas"


def test_extract_sql_several_fences_takes_first_with_sql():
    raw = (
        "Exemplo de saída:\n```text\n| uf |\n```\n"
        "Consulta:\n```sql\nSELECT uf FROM municipios;\n```\n"
        "Alternativa:\n```sql\nSELECT 2\n```"
    )
    assert prompting.extract_sql(raw) == "SELECT uf FROM municipios"


def test_extract_sql_unclosed_fence():
    raw = "```sql\nSELECT 1"
    assert prompting.extract_sql(raw) == "SELECT 1"


def test_extract_sql_no_fence_with_prose_before():
    raw = "A consulta é: SELECT AVG(valor_venda) FROM coletas WHERE produto = 'GASOLINA';"
    assert prompting.extract_sql(raw) == (
        "SELECT AVG(valor_venda) FROM coletas WHERE produto = 'GASOLINA'"
    )


def test_extract_sql_no_fence_with_cte():
    raw = "Resposta:\nWITH m AS (SELECT 1 AS x) SELECT x FROM m"
    assert prompting.extract_sql(raw) == "WITH m AS (SELECT 1 AS x) SELECT x FROM m"


def test_extract_sql_plain_sql_unchanged():
    assert prompting.extract_sql("SELECT 1") == "SELECT 1"


def test_extract_sql_lowercase():
    raw = "claro! select uf from municipios where capital;"
    assert prompting.extract_sql(raw) == "select uf from municipios where capital"


def test_extract_sql_lowercase_in_fence():
    raw = "```SQL\nwith a as (select 1) select * from a;;\n```"
    assert prompting.extract_sql(raw) == "with a as (select 1) select * from a"


def test_extract_sql_does_not_match_inside_words():
    # "selecionei" não é SELECT; sem SQL reconhecível, devolve o texto limpo.
    raw = "Eu selecionei os dados."
    assert prompting.extract_sql(raw) == "Eu selecionei os dados."


@pytest.mark.parametrize("raw", ["", "   ", "\n\t \n", "```sql\n```", "```\n\n```"])
def test_extract_sql_empty(raw):
    assert prompting.extract_sql(raw) == ""


def test_extract_sql_none_is_empty():
    assert prompting.extract_sql(None) == ""  # type: ignore[arg-type]
