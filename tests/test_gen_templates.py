"""Testes do gerador de treino por templates (M4, D23).

Nenhum teste abre o banco real nem os conjuntos de dev/teste: o banco é
sintético, criado em `tmp_path` com dados suficientes para todas as intenções.
"""

from __future__ import annotations

import datetime as dt
import re

import duckdb
import pytest

from m4_fakes import M4_TOY_DDL, read_jsonl, write_jsonl
from tanque.gen import intents, leakage, pipeline, templates, validate

# --- banco sintético ------------------------------------------------------------

_UFS = [
    # uf, região, capital, outras cidades
    ("SP", "SE", "SAO PAULO", ["CAMPINAS", "SAO JOSE DOS CAMPOS", "RIBEIRAO PRETO"]),
    ("MG", "SE", "BELO HORIZONTE", ["UBERLANDIA", "JUIZ DE FORA"]),
    ("RJ", "SE", "RIO DE JANEIRO", ["NITEROI", "VALENCA"]),
    ("ES", "SE", "VITORIA", ["VILA VELHA"]),
    ("BA", "NE", "SALVADOR", ["FEIRA DE SANTANA", "VALENCA"]),
    ("PE", "NE", "RECIFE", ["CARUARU", "PETROLINA"]),
    ("CE", "NE", "FORTALEZA", ["SOBRAL", "JUAZEIRO DO NORTE"]),
    ("PR", "S", "CURITIBA", ["LONDRINA", "MARINGA"]),
    ("RS", "S", "PORTO ALEGRE", ["CAXIAS DO SUL", "PELOTAS"]),
    ("SC", "S", "FLORIANOPOLIS", ["JOINVILLE", "BLUMENAU"]),
    ("GO", "CO", "GOIANIA", ["ANAPOLIS", "RIO VERDE"]),
    ("MT", "CO", "CUIABA", ["SINOP", "RONDONOPOLIS"]),
    ("DF", "CO", "BRASILIA", []),
    ("PA", "N", "BELEM", ["SANTAREM", "MARABA"]),
    ("AM", "N", "MANAUS", ["PARINTINS", "MANACAPURU"]),
    ("TO", "N", "PALMAS", ["ARAGUAINA", "GURUPI"]),
]
_BANDEIRAS = ["BRANCA", "VIBRA", "IPIRANGA", "RAIZEN", "BRANCA", "ALE", "SABBÁ", "IPIRANGA"]
_BAIRROS = [
    "CENTRO",
    "JARDIM AMERICA",
    "VILA NOVA",
    "BOA VISTA",
    "SAO JOSE",
    "INDUSTRIAL",
    "AEROPORTO",
    "SANTA CRUZ",
    "BELA VISTA",
    "PARQUE DAS NACOES",
]
_BASE = {
    "GASOLINA": 6.0,
    "GASOLINA ADITIVADA": 6.3,
    "ETANOL": 4.1,
    "DIESEL": 6.1,
    "DIESEL S10": 6.25,
    "GNV": 4.6,
}


def make_synthetic_db(path):
    con = duckdb.connect(str(path))
    try:
        ddl = M4_TOY_DDL.split("INSERT INTO")[0]
        con.execute(ddl)
        mun_rows, posto_rows = [], []
        mid = pid = 0
        for ui, (uf, reg, cap, others) in enumerate(_UFS):
            for ci, nome in enumerate([cap, *others]):
                mid += 1
                mun_rows.append((mid, nome, uf, reg, ci == 0))
                n_postos = 42 if ci == 0 else 5
                for k in range(n_postos):
                    pid += 1
                    band = _BANDEIRAS[(pid + ui) % len(_BANDEIRAS)]
                    bairro = _BAIRROS[(k + ci) % len(_BAIRROS)] if k % 7 != 6 else None
                    posto_rows.append(
                        (pid, f"00.{pid:03d}/0001", f"AUTO POSTO {pid}", band, bairro, "", mid)
                    )
        con.executemany("INSERT INTO municipios VALUES (?, ?, ?, ?, ?)", mun_rows)
        con.executemany("INSERT INTO postos VALUES (?, ?, ?, ?, ?, ?, ?)", posto_rows)
        con.execute("CREATE TEMP TABLE base (produto VARCHAR, preco DOUBLE)")
        con.executemany("INSERT INTO base VALUES (?, ?)", list(_BASE.items()))
        con.execute(
            """
            INSERT INTO coletas
            SELECT row_number() OVER () AS coleta_id, p.posto_id, b.produto,
                   (DATE '2024-10-01' + INTERVAL (mi) MONTH + INTERVAL (d) DAY)::DATE,
                   CAST(b.preco + (m.municipio_id % 13) * 0.07 + mi * 0.011
                        + ((p.posto_id * 37 + mi * 11 + d * 5
                            + length(b.produto) * 3) % 97) / 331.0
                        + CASE WHEN b.produto = 'ETANOL' THEN (m.municipio_id % 5) * 0.12
                               ELSE 0 END
                        AS DECIMAL(6,3)),
                   NULL,
                   CASE WHEN b.produto = 'GNV' THEN 'R$ / m³' ELSE 'R$ / litro' END
            FROM postos p
            JOIN municipios m ON m.municipio_id = p.municipio_id
            CROSS JOIN base b
            CROSS JOIN range(24) AS t(mi)
            CROSS JOIN (VALUES (4), (17)) AS dd(d)
            WHERE NOT (b.produto = 'GNV' AND p.posto_id % 3 <> 0)
            """
        )
    finally:
        con.close()
    return path


@pytest.fixture(scope="module")
def synth_db(tmp_path_factory):
    return make_synthetic_db(tmp_path_factory.mktemp("synth") / "synth.duckdb")


# --- catálogo, textos e períodos -------------------------------------------------


def test_every_intent_has_a_template():
    assert set(templates.GENERATORS) == set(intents.INTENTS_BY_ID)


@pytest.mark.parametrize(
    ("nome", "esperado"),
    [
        ("GOIANIA", "Goiânia"),
        ("SAO JOSE DOS CAMPOS", "São José dos Campos"),
        ("JI-PARANA", "Ji-Paraná"),
        ("SAO JOAO DEL REI", "São João del Rei"),
        ("RIBEIRAO PRETO", "Ribeirão Preto"),
        ("CAMPINAS", "Campinas"),
    ],
)
def test_city_display_names_have_accents(nome, esperado):
    assert templates.city_display(nome) == esperado


def test_uf_and_region_names_cover_all():
    assert len(templates.UF_NAMES) == 27
    assert set(templates.REGION_TEXT) == {"N", "NE", "CO", "SE", "S"}
    assert templates.UF_NAMES["SP"][0] == "estado de São Paulo"


def test_periods_follow_the_review_guide():
    last = templates.last_month_period()
    assert (last.start, last.end) == (dt.date(2026, 9, 1), dt.date(2026, 10, 1))
    assert last.pred() == (
        "c.data_coleta >= DATE '2026-09-01' AND c.data_coleta < DATE '2026-10-01'"
    )
    rng_ = templates.month_range_period(2026, 3, 6)
    assert (rng_.start, rng_.end) == (dt.date(2026, 3, 1), dt.date(2026, 7, 1))
    assert any("entre março e junho de 2026" in t for t in rng_.texts)
    last12 = templates.last_n_months_period(12)
    assert (last12.start, last12.end) == (dt.date(2025, 10, 1), dt.date(2026, 10, 1))
    whole = templates.whole_period()
    assert whole.pred() == "" and whole.months() == 24
    year = templates.year_period(2026)
    assert year.months() == 9


def test_noise_is_light_and_deterministic():
    q = "Qual o preço médio da gasolina em Goiânia no último mês?"
    assert templates.strip_accents(q) == "Qual o preco medio da gasolina em Goiania no ultimo mes?"
    import random

    a = templates.typo(q, random.Random(3))
    b = templates.typo(q, random.Random(3))
    assert a == b and a != q
    # um erro de uma letra: comprimentos diferem em no máximo 1
    assert abs(len(a) - len(q)) <= 1
    # números e siglas não são alterados
    q2 = "Top 5 UFs com GNV mais caro em 2025"
    for seed in range(30):
        out = templates.typo(q2, random.Random(seed))
        assert "2025" in out and "GNV" in out and "UFs" in out


def test_rank_check_detects_ties_and_short_rankings():
    rows = [("A", 6.1), ("B", 6.0), ("C", 5.9), ("D", 5.9)]
    assert templates.rank_problem(rows, n=3, value_index=1) == "empate"
    assert templates.rank_problem(rows, n=2, value_index=1) is None
    assert templates.rank_problem(rows, n=5, value_index=1) == "poucas_linhas"
    rows2 = [("A", 6.1), ("B", 6.1), ("C", 5.0)]
    assert templates.rank_problem(rows2, n=2, value_index=1) == "empate"
    # por grupo: o empate vale dentro de cada grupo
    grouped = [("N", "PA", 5.0), ("N", "AM", 5.5), ("S", "PR", 5.0), ("S", "RS", 5.0)]
    assert templates.rank_problem(grouped, n=1, value_index=2, group_index=0) == "empate"
    assert templates.rank_problem(grouped[:3], n=1, value_index=2, group_index=0) is None
    # sinal exigido (ex.: "maiores quedas" precisam ser negativas)
    assert templates.rank_problem([("A", -1.0), ("B", 0.5)], 2, 1, sign=-1) == "sinal"


def test_degenerate_results():
    assert templates.result_problem([]) == "vazio"
    assert templates.result_problem([(None,)]) == "nulo"
    assert templates.result_problem([("SP", None)]) == "nulo"
    assert templates.result_problem([(0,)]) == "zero"
    assert templates.result_problem([(6.1,)]) is None


# --- geração no banco sintético --------------------------------------------------


@pytest.fixture(scope="module")
def generated(synth_db):
    return templates.generate(400, seed=42, db_path=synth_db)


def test_generate_is_deterministic(synth_db, generated):
    again = templates.generate(400, seed=42, db_path=synth_db)
    assert again.candidates == generated.candidates
    other = templates.generate(400, seed=7, db_path=synth_db)
    assert other.candidates != generated.candidates


def test_generate_covers_every_intent(generated):
    seen = {c["intent"] for c in generated.candidates}
    assert seen == set(intents.INTENTS_BY_ID)
    assert len(generated.candidates) >= 300


def test_candidates_follow_conventions(generated):
    for c in generated.candidates:
        sql, q = c["sql"], c["question"]
        assert set(c) >= {"question", "sql", "intent", "order_matters", "tags"}
        assert "ROUND(" not in sql.upper()
        assert q and q == q.strip()
        # município sempre por nome + uf
        if "m.nome = " in sql:
            assert re.search(r"m\.nome = '[^']+' AND m\.uf = '[A-Z]{2}'", sql), sql
        # mês sempre por date_trunc
        assert "strftime" not in sql and "EXTRACT(MONTH" not in sql.upper()
        # regra dos 70% sempre com a gasolina comum
        if "0.7" in sql:
            assert "'GASOLINA'" in sql and "ADITIVADA" not in sql
        # order_matters só com ORDER BY
        if c["order_matters"]:
            assert "ORDER BY" in sql


def test_noise_rate_close_to_ten_percent(generated):
    noisy = sum(1 for c in generated.candidates if "typo" in c["tags"])
    assert 0.05 <= noisy / len(generated.candidates) <= 0.16


def test_no_duplicate_sql_or_question_in_candidates(generated):
    sqls = [leakage.normalize_sql(c["sql"]) for c in generated.candidates]
    qs = [leakage.normalize_question(c["question"]) for c in generated.candidates]
    assert len(set(sqls)) == len(sqls)
    assert len(set(qs)) == len(qs)


def test_candidates_pass_execution_validation(synth_db, generated):
    # uma amostra basta: cada SQL já foi executada na geração
    vr = validate.validate_pairs(generated.candidates[::3], db_path=synth_db)
    reasons = vr.reasons()
    # por construção, nenhuma SQL quebra nem volta vazia/nula
    for bad in ("parse", "guardrail", "erro_execucao", "timeout", "resultado_vazio"):
        assert bad not in reasons, (bad, reasons)
    assert "resultado_nulo" not in reasons
    # as tags extras do gerador sobrevivem à validação
    some = vr.valid[0]
    assert some["tags"][0].startswith("intent:")


def test_level_distribution_of_candidates(generated):
    from collections import Counter

    levels = Counter(intents.INTENTS_BY_ID[c["intent"]].level for c in generated.candidates)
    total = sum(levels.values())
    for level, target in intents.TARGET_DISTRIBUTION.items():
        assert abs(levels[level] / total - float(target)) <= 0.05, levels


def test_validate_can_skip_repeated_result_rule(synth_db):
    pairs = [
        {
            "question": "Quantos postos há?",
            "sql": "SELECT COUNT(*) FROM postos",
            "intent": "conta_postos_local",
        },
        {
            "question": "Número de postos?",
            "sql": "SELECT COUNT(*) + 0 FROM postos",
            "intent": "conta_postos_local",
        },
    ]
    strict = validate.validate_pairs(pairs, db_path=synth_db)
    assert strict.reasons() == {"resultado_repetido": 1}
    loose = validate.validate_pairs(pairs, db_path=synth_db, check_repeated=False)
    assert len(loose.valid) == 2


# --- deduplicação interna exata (D23) ---------------------------------------------


def test_dedupe_exact_keeps_parameter_variations():
    items = [
        {"question": "Preço médio da gasolina em SP em 2025?", "sql": "SELECT 1"},
        {"question": "Preço médio da gasolina em MG em 2025?", "sql": "SELECT 2"},
        {"question": "preco medio da gasolina em sp em 2025", "sql": "SELECT 3"},
        {"question": "Outra redação qualquer", "sql": "select   1"},
    ]
    kept, removed = leakage.dedupe_exact(items)
    assert [k["sql"] for k in kept] == ["SELECT 1", "SELECT 2"]
    assert [(r.index, r.kind) for r in removed] == [(2, "pergunta"), (3, "sql")]
    # o fuzzy do plano trataria as duas primeiras como duplicatas
    kept_fuzzy, _ = leakage.dedupe(items[:2])
    assert len(kept_fuzzy) == 1


# --- pipeline e CLI ----------------------------------------------------------------


def _reference(tmp_path, question):
    ref = tmp_path / "datasets"
    write_jsonl(ref / "dev.jsonl", [{"id": "dev-0001", "question": "xyz abc", "sql": "SELECT 9"}])
    write_jsonl(ref / "test.jsonl", [{"id": "test-0001", "question": question, "sql": "SELECT 8"}])
    return ref


def test_build_train_from_template_pairs(tmp_path, synth_db, generated):
    pairs = generated.candidates[:120]
    ref = _reference(tmp_path, pairs[0]["question"])
    result = pipeline.build_train(
        db_path=synth_db,
        pairs=pairs,
        dedupe_mode="exact",
        check_repeated=False,
        reference_dir=ref,
        out_path=tmp_path / "train.jsonl",
        work_dir=tmp_path / "work",
        summary_path=tmp_path / "summary.json",
        extra_summary={"generator": "templates"},
    )
    train = read_jsonl(tmp_path / "train.jsonl")
    assert result.final == len(train) > 0
    assert result.reasons.get("vazamento", 0) >= 1
    assert train[0]["id"] == "train-00001"
    for t in train:
        assert t["split"] == "train" and t["source"] == "synthetic" and t["reviewed"] is False
        assert t["difficulty"] in intents.LEVELS
        assert isinstance(t["order_matters"], bool) and isinstance(t["uses_window"], bool)
    import json

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["dedupe_mode"] == "exact"
    assert summary["generator"] == "templates"
    by_intent = summary["by_intent"]
    assert sum(v["raw"] for v in by_intent.values()) == len(pairs)
    assert sum(v["final"] for v in by_intent.values()) == len(train)


def test_cli_templates_command(tmp_path, synth_db, monkeypatch):
    from typer.testing import CliRunner

    from tanque.cli import app

    ref = _reference(tmp_path, "pergunta que não existe no treino")
    monkeypatch.setattr(pipeline, "train_path", lambda: tmp_path / "train.jsonl")
    monkeypatch.setattr(pipeline, "SUMMARY_RELPATH", tmp_path / "summary.json")
    monkeypatch.setattr(pipeline, "STATS_RELPATH", tmp_path / "gen-stats.md")
    monkeypatch.setattr(pipeline, "WORK_RELDIR", tmp_path / "work")
    monkeypatch.setattr(templates, "RAW_RELPATH", tmp_path / "work" / "templates_raw.jsonl")
    monkeypatch.setattr(leakage, "datasets_dir", lambda: ref)
    result = CliRunner().invoke(
        app, ["gen", "templates", "--n", "150", "--seed", "42", "--db", str(synth_db)]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "work" / "templates_raw.jsonl").is_file()
    train = read_jsonl(tmp_path / "train.jsonl")
    assert len(train) > 50
    stats = (tmp_path / "gen-stats.md").read_text(encoding="utf-8")
    assert "templates" in stats
    assert "Por intenção" in stats
