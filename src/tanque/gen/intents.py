"""Catálogo de intenções para gerar o dataset de treino sintético (M4).

Cada intenção descreve um *tipo* de pergunta que o sistema precisa responder,
com o nível de dificuldade (seção 5.2), se usa função de janela, uma forma
típica de SQL e dicas de variação para o gerador. As intenções foram derivadas
do plano (seções 1 e 5) e do schema real (`docs/schema.md`), **nunca** dos
conjuntos de dev/teste, que este marco não lê.

Distribuição alvo do treino (M4): 35% fácil, 40% médio, 25% difícil. Dentro
do nível difícil, intenções com janela têm peso menor para que fiquem abaixo de
20% do nível (D6).

`allocate(n_requests)` reparte as requisições do lote entre as intenções de
forma determinística: primeiro entre os níveis (maior resto), depois entre as
intenções de cada nível pelos pesos (maior resto, desempate por ordem fixa
embaralhada com semente). Assim, o lote piloto de 20 requisições já cobre 20
intenções diferentes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from fractions import Fraction

LEVELS = ("facil", "medio", "dificil")
TARGET_DISTRIBUTION: dict[str, Fraction] = {
    "facil": Fraction(35, 100),
    "medio": Fraction(40, 100),
    "dificil": Fraction(25, 100),
}
MAX_WINDOW_SHARE_OF_HARD = Fraction(20, 100)
_ALLOCATION_SEED = 20261028


@dataclass(frozen=True)
class Intent:
    id: str
    level: str
    uses_window: bool
    title: str
    description: str
    sql_shape: str
    variations: tuple[str, ...]
    tags: tuple[str, ...] = field(default_factory=tuple)
    weight: Fraction = Fraction(1)

    def instructions(self) -> str:
        """Texto da intenção para a mensagem do gerador."""
        lines = [
            f"Intenção: {self.title} (id `{self.id}`)",
            f"Nível: {self.level}" + (" — usa função de janela" if self.uses_window else ""),
            "",
            f"O que a pergunta pede: {self.description}",
            "",
            "Forma típica da SQL (adapte filtros, produto, local e período a cada pergunta):",
            "```sql",
            self.sql_shape.strip(),
            "```",
            "",
            "Varie entre as perguntas:",
            *[f"- {v}" for v in self.variations],
        ]
        return "\n".join(lines)


# Trecho de JOIN padrão usado nos exemplos.
_J = (
    "FROM coletas c\n"
    "JOIN postos p ON p.posto_id = c.posto_id\n"
    "JOIN municipios m ON m.municipio_id = p.municipio_id"
)

_PARAMS = "produto, UF/município/região e período (mês, ano, último mês, intervalo de datas)"

INTENTS: tuple[Intent, ...] = (
    # ------------------------------------------------------------------ fácil
    Intent(
        id="media_produto_uf",
        level="facil",
        uses_window=False,
        title="Preço médio de um produto numa UF",
        description="preço médio de venda de um combustível em um estado, "
        "com ou sem recorte de período.",
        sql_shape=f"SELECT AVG(c.valor_venda) AS preco_medio\n{_J}\n"
        "WHERE c.produto = 'ETANOL' AND m.uf = 'MG'\n"
        "  AND c.data_coleta >= DATE '2025-01-01' AND c.data_coleta < DATE '2026-01-01'",
        variations=(
            "estados diferentes, citados pela sigla ou pelo nome ('Minas Gerais')",
            "todos os produtos, inclusive GNV (R$/m³) e gasolina aditivada",
            "sem período, com ano, com mês específico ou com 'último mês'",
        ),
        tags=("media", "uf"),
    ),
    Intent(
        id="media_produto_municipio",
        level="facil",
        uses_window=False,
        title="Preço médio de um produto num município",
        description="preço médio de um combustível em uma cidade (filtre por nome e UF).",
        sql_shape=f"SELECT AVG(c.valor_venda)\n{_J}\n"
        "WHERE c.produto = 'GASOLINA' AND m.nome = 'CAMPINAS' AND m.uf = 'SP'",
        variations=(
            "cidades de várias regiões, escritas com acento na pergunta",
            "perguntas diretas e indiretas ('quanto custa, em média, ...')",
            "com e sem recorte de período",
        ),
        tags=("media", "municipio"),
    ),
    Intent(
        id="media_produto_regiao",
        level="facil",
        uses_window=False,
        title="Preço médio de um produto numa região",
        description="preço médio de um combustível em uma das cinco regiões do país.",
        sql_shape=f"SELECT AVG(c.valor_venda)\n{_J}\n"
        "WHERE c.produto = 'DIESEL S10' AND m.regiao = 'NE'",
        variations=(
            "as cinco regiões, pelo nome ('Nordeste', 'Centro-Oeste')",
            "com ano, mês ou trimestre",
        ),
        tags=("media", "regiao"),
    ),
    Intent(
        id="media_nacional_periodo",
        level="facil",
        uses_window=False,
        title="Preço médio nacional num período",
        description="preço médio de um combustível no Brasil inteiro em um período.",
        sql_shape="SELECT AVG(valor_venda)\nFROM coletas\n"
        "WHERE produto = 'GNV' AND data_coleta >= DATE '2026-03-01' "
        "AND data_coleta < DATE '2026-04-01'",
        variations=(
            "meses específicos, anos, 'último mês', semestres",
            "'no país', 'no Brasil', 'nacionalmente'",
        ),
        tags=("media", "brasil", "periodo"),
    ),
    Intent(
        id="extremo_preco",
        level="facil",
        uses_window=False,
        title="Maior ou menor preço registrado",
        description="o maior ou o menor valor de venda registrado de um combustível "
        "em um local e/ou período.",
        sql_shape=f"SELECT MIN(c.valor_venda)\n{_J}\nWHERE c.produto = 'ETANOL' AND m.uf = 'GO'",
        variations=(
            "máximo e mínimo, ou os dois na mesma pergunta (duas colunas)",
            "Brasil, UF, cidade ou região",
            "com e sem período",
        ),
        tags=("min_max",),
    ),
    Intent(
        id="conta_postos_local",
        level="facil",
        uses_window=False,
        title="Quantos postos há num local",
        description="número de postos (tabela postos) em um município, UF ou região.",
        sql_shape="SELECT COUNT(*)\nFROM postos p\n"
        "JOIN municipios m ON m.municipio_id = p.municipio_id\n"
        "WHERE m.nome = 'SALVADOR' AND m.uf = 'BA'",
        variations=(
            "'quantos postos', 'quantas revendas', 'número de postos pesquisados'",
            "cidade, estado ou região",
        ),
        tags=("contagem", "postos"),
    ),
    Intent(
        id="conta_coletas",
        level="facil",
        uses_window=False,
        title="Quantas coletas de preço",
        description="número de coletas (registros de preço) de um produto, num local e/ou período.",
        sql_shape="SELECT COUNT(*)\nFROM coletas\n"
        "WHERE produto = 'DIESEL' AND data_coleta >= DATE '2025-07-01' "
        "AND data_coleta < DATE '2025-08-01'",
        variations=(
            "'quantas coletas', 'quantos preços foram coletados', 'quantas pesquisas'",
            "com filtro de UF (exige os JOINs) ou só de produto e data",
        ),
        tags=("contagem", "coletas"),
    ),
    Intent(
        id="conta_postos_bandeira",
        level="facil",
        uses_window=False,
        title="Quantos postos de uma bandeira",
        description="número de postos de uma bandeira (ou sem bandeira) em um local.",
        sql_shape="SELECT COUNT(*)\nFROM postos p\n"
        "JOIN municipios m ON m.municipio_id = p.municipio_id\n"
        "WHERE p.bandeira = 'IPIRANGA' AND m.uf = 'RS'",
        variations=(
            "bandeiras frequentes (VIBRA, IPIRANGA, RAIZEN, ALE) e 'bandeira branca' (= 'BRANCA')",
            "Brasil, região, UF ou cidade",
        ),
        tags=("contagem", "bandeira"),
    ),
    Intent(
        id="media_bandeira",
        level="facil",
        uses_window=False,
        title="Preço médio numa bandeira",
        description="preço médio de um combustível nos postos de uma bandeira.",
        sql_shape="SELECT AVG(c.valor_venda)\nFROM coletas c\n"
        "JOIN postos p ON p.posto_id = c.posto_id\n"
        "WHERE c.produto = 'GASOLINA' AND p.bandeira = 'VIBRA'",
        variations=(
            "várias bandeiras, inclusive postos de bandeira branca",
            "com período ou só no último mês",
        ),
        tags=("media", "bandeira"),
    ),
    Intent(
        id="media_capitais",
        level="facil",
        uses_window=False,
        title="Preço médio em capitais",
        description="preço médio de um combustível numa capital específica ou no conjunto "
        "das capitais (municipios.capital = true).",
        sql_shape=f"SELECT AVG(c.valor_venda)\n{_J}\nWHERE c.produto = 'ETANOL' AND m.capital",
        variations=(
            "'nas capitais', 'nas capitais brasileiras', 'fora das capitais' (NOT m.capital)",
            "uma capital pelo nome ('em Fortaleza')",
        ),
        tags=("media", "capital"),
    ),
    Intent(
        id="conta_municipios",
        level="facil",
        uses_window=False,
        title="Quantos municípios na pesquisa",
        description="número de municípios cobertos pela pesquisa em uma UF ou região.",
        sql_shape="SELECT COUNT(*)\nFROM municipios\nWHERE uf = 'PR'",
        variations=(
            "'quantas cidades', 'quantos municípios aparecem nos dados'",
            "UF ou região",
        ),
        tags=("contagem", "municipio"),
    ),
    Intent(
        id="media_ultimo_mes",
        level="facil",
        uses_window=False,
        title="Preço médio no último mês",
        description="preço médio de um combustível no último mês (mês da data de referência).",
        sql_shape="SELECT AVG(valor_venda)\nFROM coletas\nWHERE produto = 'DIESEL S10'\n"
        "  AND data_coleta >= DATE '2026-09-01' AND data_coleta <= DATE '2026-09-30'",
        variations=(
            "'no último mês', 'no mês mais recente', 'atualmente', 'hoje em dia'",
            "Brasil, UF ou cidade",
        ),
        tags=("media", "ultimo_mes"),
    ),
    Intent(
        id="amplitude_local",
        level="facil",
        uses_window=False,
        title="Diferença entre o maior e o menor preço",
        description="amplitude (MAX - MIN) do valor de venda de um combustível num local.",
        sql_shape="SELECT MAX(c.valor_venda) - MIN(c.valor_venda)\n"
        f"{_J}\nWHERE c.produto = 'GASOLINA' AND m.nome = 'RECIFE' AND m.uf = 'PE'",
        variations=(
            "'diferença entre o posto mais caro e o mais barato', 'amplitude', 'variação de preço'",
            "com período",
        ),
        tags=("min_max", "amplitude"),
    ),
    Intent(
        id="datas_coleta",
        level="facil",
        uses_window=False,
        title="Datas das coletas",
        description="a primeira ou a última data de coleta, ou em quantos dias distintos houve "
        "coleta, de um produto num local.",
        sql_shape=f"SELECT MAX(c.data_coleta)\n{_J}\n"
        "WHERE c.produto = 'GNV' AND m.nome = 'NATAL' AND m.uf = 'RN'",
        variations=(
            "primeira data, última data, número de dias com coleta (COUNT(DISTINCT data_coleta))",
            "por produto, cidade ou UF",
        ),
        tags=("datas",),
    ),
    # ------------------------------------------------------------------ médio
    Intent(
        id="ranking_uf_preco",
        level="medio",
        uses_window=False,
        title="Ranking de UFs pelo preço",
        description="os N estados com o combustível mais caro ou mais barato em média, "
        "num período.",
        sql_shape=f"SELECT m.uf, AVG(c.valor_venda) AS preco_medio\n{_J}\n"
        "WHERE c.produto = 'ETANOL' AND c.data_coleta >= DATE '2026-09-01'\n"
        "GROUP BY m.uf\nORDER BY preco_medio ASC\nLIMIT 5",
        variations=(
            "top 3, 5, 10; mais caro e mais barato",
            "período por ano, mês, 'último mês', semestre",
        ),
        tags=("ranking", "uf"),
    ),
    Intent(
        id="ranking_municipios",
        level="medio",
        uses_window=False,
        title="Ranking de municípios",
        description="os N municípios com o combustível mais caro ou mais barato em média, "
        "dentro de uma UF ou região.",
        sql_shape=f"SELECT m.nome, AVG(c.valor_venda) AS preco_medio\n{_J}\n"
        "WHERE c.produto = 'GASOLINA' AND m.regiao = 'S'\n"
        "GROUP BY m.nome, m.uf\nORDER BY preco_medio DESC\nLIMIT 10",
        variations=(
            "UF ou região; mais caro e mais barato; N variado",
            "pode devolver nome e UF",
        ),
        tags=("ranking", "municipio"),
    ),
    Intent(
        id="ranking_bandeiras",
        level="medio",
        uses_window=False,
        title="Bandeiras mais caras ou mais baratas",
        description="bandeiras ordenadas pelo preço médio de um combustível em uma região/UF "
        "e período.",
        sql_shape=f"SELECT p.bandeira, AVG(c.valor_venda) AS preco_medio\n{_J}\n"
        "WHERE c.produto = 'DIESEL' AND m.regiao = 'NE'\n"
        "  AND c.data_coleta >= DATE '2025-01-01' AND c.data_coleta < DATE '2026-01-01'\n"
        "GROUP BY p.bandeira\nORDER BY preco_medio DESC\nLIMIT 5",
        variations=(
            "com mínimo de coletas (HAVING COUNT(*) >= 100) em parte das perguntas",
            "região, UF ou Brasil",
        ),
        tags=("ranking", "bandeira"),
    ),
    Intent(
        id="media_por_regiao",
        level="medio",
        uses_window=False,
        title="Preço médio por região",
        description="o preço médio de um combustível em cada região, ordenado.",
        sql_shape=f"SELECT m.regiao, AVG(c.valor_venda) AS preco_medio\n{_J}\n"
        "WHERE c.produto = 'GNV' AND c.data_coleta >= DATE '2026-01-01'\n"
        "GROUP BY m.regiao\nORDER BY preco_medio DESC",
        variations=(
            "'por região', 'em cada região', 'compare as regiões'",
            "por UF dentro de uma região também vale",
        ),
        tags=("agrupamento", "regiao"),
    ),
    Intent(
        id="serie_mensal",
        level="medio",
        uses_window=False,
        title="Série mensal do preço médio",
        description="preço médio mês a mês de um combustível em um local e intervalo.",
        sql_shape="SELECT date_trunc('month', c.data_coleta) AS mes, "
        f"AVG(c.valor_venda) AS preco_medio\n{_J}\n"
        "WHERE c.produto = 'GASOLINA' AND m.uf = 'SP' AND c.data_coleta >= DATE '2025-01-01'\n"
        "GROUP BY mes\nORDER BY mes",
        variations=(
            "'mês a mês', 'evolução mensal', 'por mês'",
            "Brasil, UF ou cidade; ano ou últimos 6/12 meses",
        ),
        tags=("serie", "mes"),
    ),
    Intent(
        id="media_por_produto",
        level="medio",
        uses_window=False,
        title="Preço médio de cada produto num local",
        description="o preço médio de cada combustível em uma cidade ou UF, num período.",
        sql_shape=f"SELECT c.produto, AVG(c.valor_venda) AS preco_medio\n{_J}\n"
        "WHERE m.nome = 'BELO HORIZONTE' AND m.uf = 'MG'\n"
        "  AND c.data_coleta >= DATE '2026-09-01'\nGROUP BY c.produto\nORDER BY preco_medio DESC",
        variations=(
            "'cada combustível', 'todos os produtos'",
            "ordenado ou não",
        ),
        tags=("agrupamento", "produto"),
    ),
    Intent(
        id="ranking_capitais",
        level="medio",
        uses_window=False,
        title="Capitais ordenadas pelo preço",
        description="as capitais ordenadas pelo preço médio de um combustível num período.",
        sql_shape=f"SELECT m.nome, m.uf, AVG(c.valor_venda) AS preco_medio\n{_J}\n"
        "WHERE c.produto = 'ETANOL' AND m.capital AND c.data_coleta >= DATE '2026-09-01'\n"
        "GROUP BY m.nome, m.uf\nORDER BY preco_medio\nLIMIT 5",
        variations=(
            "top N mais caras/baratas, ou todas as capitais",
            "capitais de uma região",
        ),
        tags=("ranking", "capital"),
    ),
    Intent(
        id="compara_bandeiras",
        level="medio",
        uses_window=False,
        title="Comparação entre bandeiras escolhidas",
        description="preço médio de um combustível em duas ou três bandeiras citadas "
        "(ex.: Ipiranga x bandeira branca), num local.",
        sql_shape=f"SELECT p.bandeira, AVG(c.valor_venda) AS preco_medio\n{_J}\n"
        "WHERE c.produto = 'GASOLINA' AND m.uf = 'RJ' AND p.bandeira IN ('IPIRANGA', 'BRANCA')\n"
        "GROUP BY p.bandeira",
        variations=(
            "pares e trios de bandeiras; 'postos sem bandeira'",
            "UF, região ou cidade; com período",
        ),
        tags=("comparacao", "bandeira"),
    ),
    Intent(
        id="postos_por_bandeira",
        level="medio",
        uses_window=False,
        title="Número de postos por bandeira",
        description="quantos postos cada bandeira tem num local, em ordem decrescente.",
        sql_shape="SELECT p.bandeira, COUNT(*) AS postos\nFROM postos p\n"
        "JOIN municipios m ON m.municipio_id = p.municipio_id\nWHERE m.uf = 'BA'\n"
        "GROUP BY p.bandeira\nORDER BY postos DESC\nLIMIT 5",
        variations=(
            "top N bandeiras; UF, região ou cidade",
            "'quais bandeiras dominam', 'maiores redes'",
        ),
        tags=("contagem", "bandeira", "ranking"),
    ),
    Intent(
        id="municipios_acima_limite",
        level="medio",
        uses_window=False,
        title="Municípios acima ou abaixo de um preço",
        description="municípios cujo preço médio de um combustível ficou acima (ou abaixo) "
        "de um valor em reais, num período (HAVING).",
        sql_shape=f"SELECT m.nome, m.uf, AVG(c.valor_venda) AS preco_medio\n{_J}\n"
        "WHERE c.produto = 'GASOLINA' AND c.data_coleta >= DATE '2026-09-01'\n"
        "GROUP BY m.nome, m.uf\nHAVING AVG(c.valor_venda) > 7.0\nORDER BY preco_medio DESC",
        variations=(
            "limiares realistas para cada produto (veja as faixas de preço)",
            "acima e abaixo; Brasil ou UF/região; também a contagem desses municípios",
        ),
        tags=("filtro_agregado", "municipio"),
    ),
    Intent(
        id="postos_mais_baratos",
        level="medio",
        uses_window=False,
        title="Postos mais baratos de uma cidade",
        description="os postos (razão social e bairro) com menor preço médio de um combustível "
        "numa cidade, num período.",
        sql_shape="SELECT p.razao_social, p.bairro, AVG(c.valor_venda) AS preco_medio\n"
        f"{_J}\nWHERE c.produto = 'ETANOL' AND m.nome = 'GOIANIA' AND m.uf = 'GO'\n"
        "  AND c.data_coleta >= DATE '2026-09-01'\n"
        "GROUP BY p.posto_id, p.razao_social, p.bairro\nORDER BY preco_medio\nLIMIT 5",
        variations=(
            "mais baratos e mais caros; N variado",
            "último mês ou ano",
        ),
        tags=("ranking", "posto"),
    ),
    Intent(
        id="coletas_por_mes",
        level="medio",
        uses_window=False,
        title="Coletas por mês",
        description="número de coletas por mês num local ou de um produto, em ordem de mês.",
        sql_shape="SELECT date_trunc('month', c.data_coleta) AS mes, COUNT(*) AS coletas\n"
        f"{_J}\nWHERE m.uf = 'PE' AND c.data_coleta >= DATE '2026-01-01'\n"
        "GROUP BY mes\nORDER BY mes",
        variations=(
            "por produto ou por local",
            "'quantas pesquisas por mês', 'volume de coletas mensal'",
        ),
        tags=("contagem", "serie"),
    ),
    Intent(
        id="ranking_bairros",
        level="medio",
        uses_window=False,
        title="Bairros mais baratos",
        description="bairros de uma cidade ordenados pelo preço médio de um combustível.",
        sql_shape=f"SELECT p.bairro, AVG(c.valor_venda) AS preco_medio\n{_J}\n"
        "WHERE c.produto = 'GASOLINA' AND m.nome = 'CURITIBA' AND m.uf = 'PR'\n"
        "GROUP BY p.bairro\nORDER BY preco_medio\nLIMIT 5",
        variations=(
            "cidades grandes (capitais e cidades com muitos postos)",
            "mais caros e mais baratos",
        ),
        tags=("ranking", "bairro"),
    ),
    Intent(
        id="amplitude_por_uf",
        level="medio",
        uses_window=False,
        title="Amplitude de preço por UF",
        description="diferença entre o maior e o menor valor de venda de um combustível "
        "em cada UF, ordenada.",
        sql_shape="SELECT m.uf, MAX(c.valor_venda) - MIN(c.valor_venda) AS amplitude\n"
        f"{_J}\nWHERE c.produto = 'DIESEL S10' AND c.data_coleta >= DATE '2026-01-01'\n"
        "GROUP BY m.uf\nORDER BY amplitude DESC\nLIMIT 5",
        variations=(
            "por UF ou por município de uma UF",
            "'maior dispersão', 'maior diferença entre postos'",
        ),
        tags=("amplitude", "ranking"),
    ),
    Intent(
        id="presenca_bandeira",
        level="medio",
        uses_window=False,
        title="Presença das bandeiras",
        description="em quantos municípios (ou UFs) cada bandeira tem postos.",
        sql_shape="SELECT p.bandeira, COUNT(DISTINCT p.municipio_id) AS municipios\n"
        "FROM postos p\nGROUP BY p.bandeira\nORDER BY municipios DESC\nLIMIT 5",
        variations=(
            "municípios ou UFs distintos; Brasil ou região",
            "uma bandeira específica ('em quantas cidades a Vibra está')",
        ),
        tags=("contagem", "bandeira"),
    ),
    Intent(
        id="lado_a_lado_produtos",
        level="medio",
        uses_window=False,
        title="Dois produtos lado a lado",
        description="preço médio de dois combustíveis em colunas separadas, por UF ou cidade "
        "(agregação condicional).",
        sql_shape="SELECT m.uf,\n"
        "  AVG(CASE WHEN c.produto = 'DIESEL' THEN c.valor_venda END) AS diesel,\n"
        "  AVG(CASE WHEN c.produto = 'DIESEL S10' THEN c.valor_venda END) AS diesel_s10\n"
        f"{_J}\nWHERE m.regiao = 'N' AND c.data_coleta >= DATE '2026-09-01'\n"
        "GROUP BY m.uf\nORDER BY m.uf",
        variations=(
            "DIESEL x DIESEL S10, GASOLINA x GASOLINA ADITIVADA, ETANOL x GASOLINA",
            "por UF ou por capital",
        ),
        tags=("comparacao", "produto"),
    ),
    # ---------------------------------------------------------------- difícil
    Intent(
        id="regra70_ufs",
        level="dificil",
        uses_window=False,
        title="Regra dos 70% por UF",
        description="em quais estados o etanol compensa (razão etanol/gasolina comum <= 0,7) "
        "num período, ou a razão de cada estado.",
        sql_shape="WITH precos AS (\n"
        "  SELECT m.uf,\n"
        "    AVG(CASE WHEN c.produto = 'ETANOL' THEN c.valor_venda END) AS etanol,\n"
        "    AVG(CASE WHEN c.produto = 'GASOLINA' THEN c.valor_venda END) AS gasolina\n"
        f"  {_J.replace(chr(10), chr(10) + '  ')}\n"
        "  WHERE c.data_coleta >= DATE '2026-09-01'\n"
        "  GROUP BY m.uf\n)\n"
        "SELECT uf, etanol / gasolina AS razao\nFROM precos\n"
        "WHERE etanol / gasolina <= 0.7\nORDER BY razao",
        variations=(
            "'vale a pena abastecer com etanol', 'compensa o álcool', 'regra dos 70%'",
            "UFs, regiões ou municípios de uma UF; último mês, ano, mês específico",
            "lista das UFs onde compensa, ou razão de todas ordenada",
        ),
        tags=("regra70", "uf"),
    ),
    Intent(
        id="regra70_todos_meses",
        level="dificil",
        uses_window=False,
        title="Regra dos 70% em todos os meses",
        description="locais (capitais, UFs) onde o etanol compensou em todos os meses de um "
        "período (CTE por mês + HAVING).",
        sql_shape="WITH mensal AS (\n"
        "  SELECT m.nome, date_trunc('month', c.data_coleta) AS mes,\n"
        "    AVG(CASE WHEN c.produto = 'ETANOL' THEN c.valor_venda END)\n"
        "      / AVG(CASE WHEN c.produto = 'GASOLINA' THEN c.valor_venda END) AS razao\n"
        f"  {_J.replace(chr(10), chr(10) + '  ')}\n"
        "  WHERE m.capital AND c.data_coleta >= DATE '2026-01-01'\n"
        "  GROUP BY m.nome, mes\n)\n"
        "SELECT nome\nFROM mensal\nGROUP BY nome\nHAVING COUNT(*) = 9 AND MAX(razao) <= 0.7\n"
        "ORDER BY nome",
        variations=(
            "'todos os meses', 'em nenhum mês', 'em pelo menos N meses'",
            "capitais ou UFs; ano de 2025, 2026 até setembro, últimos 6 meses",
        ),
        tags=("regra70", "mes", "cte"),
    ),
    Intent(
        id="variacao_periodos",
        level="dificil",
        uses_window=False,
        title="Variação entre dois períodos",
        description="variação percentual do preço médio de um combustível entre dois períodos "
        "(ex.: set/2025 x set/2026), por UF ou nacional.",
        sql_shape="WITH a AS (\n"
        f"  SELECT m.uf, AVG(c.valor_venda) AS p\n  {_J.replace(chr(10), chr(10) + '  ')}\n"
        "  WHERE c.produto = 'GASOLINA' AND c.data_coleta >= DATE '2025-09-01'"
        " AND c.data_coleta < DATE '2025-10-01'\n  GROUP BY m.uf\n), b AS (\n"
        f"  SELECT m.uf, AVG(c.valor_venda) AS p\n  {_J.replace(chr(10), chr(10) + '  ')}\n"
        "  WHERE c.produto = 'GASOLINA' AND c.data_coleta >= DATE '2026-09-01'\n"
        "  GROUP BY m.uf\n)\n"
        "SELECT a.uf, 100 * (b.p - a.p) / a.p AS variacao_pct\n"
        "FROM a JOIN b ON a.uf = b.uf\nORDER BY variacao_pct DESC\nLIMIT 5",
        variations=(
            "anos inteiros (2025 x 2026 até setembro), meses, trimestres",
            "Brasil, UF, capitais; maiores altas e quedas",
        ),
        tags=("variacao", "periodo", "cte"),
    ),
    Intent(
        id="razao_mensal",
        level="dificil",
        uses_window=False,
        title="Razão etanol/gasolina mês a mês",
        description="a razão entre o preço do etanol e o da gasolina comum em cada mês, "
        "num local; ou os meses em que o etanol compensou.",
        sql_shape="SELECT date_trunc('month', c.data_coleta) AS mes,\n"
        "  AVG(CASE WHEN c.produto = 'ETANOL' THEN c.valor_venda END)\n"
        "    / AVG(CASE WHEN c.produto = 'GASOLINA' THEN c.valor_venda END) AS razao\n"
        f"{_J}\nWHERE m.uf = 'SP' AND c.data_coleta >= DATE '2025-01-01'\n"
        "GROUP BY mes\nORDER BY mes",
        variations=(
            "UF, cidade, capital, Brasil",
            "série completa ou só os meses com razão <= 0,7",
        ),
        tags=("regra70", "serie"),
    ),
    Intent(
        id="acima_media_nacional",
        level="dificil",
        uses_window=False,
        title="Acima da média nacional",
        description="UFs ou municípios cujo preço médio de um combustível ficou acima "
        "(ou abaixo) da média nacional no mesmo período (subconsulta/CTE).",
        sql_shape="WITH nacional AS (\n"
        "  SELECT AVG(valor_venda) AS media FROM coletas\n"
        "  WHERE produto = 'DIESEL S10' AND data_coleta >= DATE '2026-09-01'\n)\n"
        "SELECT m.uf, AVG(c.valor_venda) AS preco_medio\n"
        f"{_J}\nWHERE c.produto = 'DIESEL S10' AND c.data_coleta >= DATE '2026-09-01'\n"
        "GROUP BY m.uf\nHAVING AVG(c.valor_venda) > (SELECT media FROM nacional)\n"
        "ORDER BY preco_medio DESC",
        variations=(
            "acima e abaixo; média nacional ou da região",
            "UFs, capitais, municípios de uma região",
        ),
        tags=("comparacao", "subconsulta", "cte"),
    ),
    Intent(
        id="subiu_no_periodo",
        level="dificil",
        uses_window=False,
        title="Onde o preço subiu ou caiu",
        description="municípios ou UFs onde o preço médio de um combustível subiu (ou caiu) "
        "entre o primeiro e o último mês de um intervalo.",
        sql_shape="WITH inicio AS (\n"
        f"  SELECT m.uf, AVG(c.valor_venda) AS p\n  {_J.replace(chr(10), chr(10) + '  ')}\n"
        "  WHERE c.produto = 'ETANOL' AND c.data_coleta >= DATE '2026-01-01'"
        " AND c.data_coleta < DATE '2026-02-01'\n  GROUP BY m.uf\n), fim AS (\n"
        f"  SELECT m.uf, AVG(c.valor_venda) AS p\n  {_J.replace(chr(10), chr(10) + '  ')}\n"
        "  WHERE c.produto = 'ETANOL' AND c.data_coleta >= DATE '2026-09-01'\n"
        "  GROUP BY m.uf\n)\n"
        "SELECT inicio.uf\nFROM inicio JOIN fim ON inicio.uf = fim.uf\n"
        "WHERE fim.p < inicio.p\nORDER BY inicio.uf",
        variations=(
            "subiu, caiu, subiu mais de X%",
            "UFs, capitais ou municípios de uma UF; também a contagem",
        ),
        tags=("variacao", "cte"),
    ),
    Intent(
        id="premio_bandeira",
        level="dificil",
        uses_window=False,
        title="Prêmio de preço das bandeiras",
        description="diferença (em R$ ou %) entre o preço médio de cada bandeira e o dos postos "
        "de bandeira branca, num local e período.",
        sql_shape="WITH por_bandeira AS (\n"
        "  SELECT p.bandeira, AVG(c.valor_venda) AS preco\n"
        f"  {_J.replace(chr(10), chr(10) + '  ')}\n"
        "  WHERE c.produto = 'GASOLINA' AND m.regiao = 'SE'\n"
        "    AND c.data_coleta >= DATE '2026-09-01'\n"
        "  GROUP BY p.bandeira\n)\n"
        "SELECT b.bandeira, 100 * (b.preco - br.preco) / br.preco AS premio_pct\n"
        "FROM por_bandeira b CROSS JOIN (SELECT preco FROM por_bandeira "
        "WHERE bandeira = 'BRANCA') br\n"
        "WHERE b.bandeira IN ('VIBRA', 'IPIRANGA', 'RAIZEN')\nORDER BY premio_pct DESC",
        variations=(
            "uma bandeira contra a branca, ou as maiores bandeiras",
            "em reais ou percentual; região, UF ou Brasil",
        ),
        tags=("comparacao", "bandeira", "cte"),
    ),
    Intent(
        id="diferenca_produtos_uf",
        level="dificil",
        uses_window=False,
        title="Diferença entre produtos por UF",
        description="diferença (R$ ou %) entre dois produtos (ex.: aditivada x comum, "
        "S10 x comum) em cada UF, ordenada; ou onde a diferença é maior que X.",
        sql_shape="WITH p AS (\n"
        "  SELECT m.uf,\n"
        "    AVG(CASE WHEN c.produto = 'GASOLINA ADITIVADA' THEN c.valor_venda END) AS adit,\n"
        "    AVG(CASE WHEN c.produto = 'GASOLINA' THEN c.valor_venda END) AS comum\n"
        f"  {_J.replace(chr(10), chr(10) + '  ')}\n"
        "  WHERE c.data_coleta >= DATE '2026-01-01'\n  GROUP BY m.uf\n)\n"
        "SELECT uf, adit - comum AS diferenca\nFROM p\nORDER BY diferenca DESC\nLIMIT 5",
        variations=(
            "pares de produtos diferentes; reais ou percentual",
            "UFs, capitais ou regiões",
        ),
        tags=("comparacao", "produto", "cte"),
    ),
    Intent(
        id="janela_variacao_mensal",
        level="dificil",
        uses_window=True,
        title="Variação mês a mês com LAG",
        description="variação do preço médio de um combustível em relação ao mês anterior, "
        "mês a mês, num local (função de janela LAG).",
        sql_shape="WITH mensal AS (\n"
        "  SELECT date_trunc('month', c.data_coleta) AS mes, AVG(c.valor_venda) AS preco\n"
        f"  {_J.replace(chr(10), chr(10) + '  ')}\n"
        "  WHERE c.produto = 'GASOLINA' AND m.uf = 'RJ'\n  GROUP BY mes\n)\n"
        "SELECT mes, preco - LAG(preco) OVER (ORDER BY mes) AS variacao\n"
        "FROM mensal\nORDER BY mes",
        variations=(
            "variação absoluta ou percentual; mês com maior alta",
            "Brasil, UF, capital",
        ),
        tags=("janela", "serie", "variacao"),
        weight=Fraction(3, 4),
    ),
    Intent(
        id="janela_ranking_por_grupo",
        level="dificil",
        uses_window=True,
        title="Ranking dentro de cada grupo",
        description="o estado (ou município) mais caro/barato de cada região, ou a posição de "
        "cada UF no ranking da sua região (RANK/ROW_NUMBER OVER PARTITION BY).",
        sql_shape="WITH precos AS (\n"
        "  SELECT m.regiao, m.uf, AVG(c.valor_venda) AS preco\n"
        f"  {_J.replace(chr(10), chr(10) + '  ')}\n"
        "  WHERE c.produto = 'ETANOL' AND c.data_coleta >= DATE '2026-09-01'\n"
        "  GROUP BY m.regiao, m.uf\n), r AS (\n"
        "  SELECT *, RANK() OVER (PARTITION BY regiao ORDER BY preco) AS pos FROM precos\n)\n"
        "SELECT regiao, uf, preco\nFROM r\nWHERE pos = 1\nORDER BY regiao",
        variations=(
            "mais caro ou mais barato de cada região; top 2 por região",
            "UF por região, município por UF",
        ),
        tags=("janela", "ranking"),
        weight=Fraction(3, 4),
    ),
)

INTENTS_BY_ID: dict[str, Intent] = {i.id: i for i in INTENTS}


def by_level(level: str) -> list[Intent]:
    return [i for i in INTENTS if i.level == level]


def _largest_remainder(
    total: int, weights: dict[str, Fraction], order: list[str]
) -> dict[str, int]:
    """Reparte `total` proporcionalmente a `weights`; desempate pela ordem de `order`."""
    weight_sum = sum(weights.values())
    exact = {k: Fraction(total) * w / weight_sum for k, w in weights.items()}
    counts = {k: int(v) for k, v in exact.items()}
    missing = total - sum(counts.values())
    rank = {k: i for i, k in enumerate(order)}
    by_remainder = sorted(weights, key=lambda k: (-(exact[k] - counts[k]), rank[k]))
    for k in by_remainder[:missing]:
        counts[k] += 1
    return counts


def allocate(n_requests: int) -> list[Intent]:
    """Lista de `n_requests` intenções na distribuição alvo, numa ordem estável.

    A ordem intercala os níveis e as intenções, para que qualquer prefixo da
    lista (e o lote piloto) fique perto da distribuição alvo.
    """
    if n_requests < 0:
        raise ValueError("n_requests não pode ser negativo")
    per_level = _largest_remainder(n_requests, dict(TARGET_DISTRIBUTION), list(LEVELS))
    rng = random.Random(_ALLOCATION_SEED)
    queues: dict[str, list[Intent]] = {}
    for level in LEVELS:
        intents = by_level(level)
        order = [i.id for i in intents]
        rng.shuffle(order)
        counts = _largest_remainder(per_level[level], {i.id: i.weight for i in intents}, order)
        # Rodízio: uma de cada intenção por vez, na ordem embaralhada.
        queue: list[Intent] = []
        remaining = dict(counts)
        while any(remaining.values()):
            for intent_id in order:
                if remaining[intent_id]:
                    queue.append(INTENTS_BY_ID[intent_id])
                    remaining[intent_id] -= 1
        queues[level] = queue

    # Intercala os níveis proporcionalmente (o nível mais "atrasado" vai primeiro).
    out: list[Intent] = []
    taken = dict.fromkeys(LEVELS, 0)
    for step in range(1, n_requests + 1):
        level = max(
            (lv for lv in LEVELS if taken[lv] < len(queues[lv])),
            key=lambda lv: (
                Fraction(step) * TARGET_DISTRIBUTION[lv] - taken[lv],
                -LEVELS.index(lv),
            ),
        )
        out.append(queues[level][taken[level]])
        taken[level] += 1
    return out
