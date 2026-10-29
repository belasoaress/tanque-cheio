"""Catálogo de intenções para gerar o dataset de treino sintético (M4).

Cada intenção descreve um *tipo* de pergunta que o sistema precisa responder,
com o nível de dificuldade (seção 5.2), se usa função de janela, uma forma
típica de SQL e dicas de variação para o gerador. As intenções foram derivadas
do plano (seções 1 e 5) e do schema real (`docs/schema.md`), **nunca** dos
conjuntos de dev/teste, que este marco não lê.

Distribuição alvo do treino (M4): 35% fácil, 40% médio, 25% difícil. Dentro
do nível difícil, intenções com janela têm peso menor para que fiquem abaixo de
20% do nível (D6).

Treino v2 (M15, D34): 17 intenções novas, cada uma ligada a um padrão que o ft-v1
errou no **dev2** (`results/training-v2.md`; o teste não é lido). A distribuição
alvo passa a 28% fácil, 36% médio e 36% difícil: o fácil (97% no dev2) mantém o
volume da v1, e o difícil (50%) ganha o que é novo.

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
# v1 (M4): 35/40/25. v2 (M15): 28/36/36, com `tanque gen templates --n 13000`, o fácil
# fica com ~250 candidatos por intenção, como na v1.
TARGET_DISTRIBUTION: dict[str, Fraction] = {
    "facil": Fraction(28, 100),
    "medio": Fraction(36, 100),
    "dificil": Fraction(36, 100),
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
    # v2 (M15): bairro escrito como uma pessoa escreve, filtrado com a grafia da ANP.
    Intent(
        id="postos_bairro",
        level="facil",
        uses_window=False,
        title="Postos ou preço num bairro",
        description="quantos postos há num bairro de uma cidade, ou o preço médio de um "
        "combustível nele; o bairro é filtrado com a grafia do banco (maiúsculas, sem acento).",
        sql_shape="SELECT COUNT(*) AS postos\nFROM postos p\n"
        "JOIN municipios m ON m.municipio_id = p.municipio_id\n"
        "WHERE p.bairro = 'BOA VIAGEM' AND m.nome = 'RECIFE' AND m.uf = 'PE'",
        variations=(
            "bairro com acento e em minúsculas na pergunta ('Boa Viagem')",
            "contagem de postos ou preço médio de um produto, com e sem período",
        ),
        tags=("bairro", "contagem"),
        weight=Fraction(1, 2),
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
        "(agregação condicional). Num lugar só (v2), uma linha por produto.",
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
    # v2 (M15): padrões médios que o ft-v1 errou no dev2.
    Intent(
        id="filtro_contagem",
        level="medio",
        uses_window=False,
        title="Grupos com mais (ou menos) de N postos ou coletas",
        description="bandeiras, estados ou municípios com mais (ou menos) de N postos, ou de N "
        "coletas de um produto; devolve a entidade e a contagem (HAVING COUNT).",
        sql_shape="SELECT m.nome, COUNT(*) AS postos\nFROM postos p\n"
        "JOIN municipios m ON m.municipio_id = p.municipio_id\nWHERE m.uf = 'PR'\n"
        "GROUP BY m.nome\nHAVING COUNT(*) > 50",
        variations=(
            "bandeiras, estados ou municípios; mais de / menos de N",
            "estados pelo gentílico ('municípios gaúchos', 'cidades paulistas')",
        ),
        tags=("contagem", "filtro_agregado"),
    ),
    Intent(
        id="serie_trimestral",
        level="medio",
        uses_window=False,
        title="Preço médio por trimestre ou por ano",
        description="preço médio (ou número de coletas) por trimestre de um ano, ou por ano; "
        "trimestre = quarter(...) e ano = year(...), inteiros.",
        sql_shape="SELECT quarter(c.data_coleta) AS trimestre, AVG(c.valor_venda) AS preco_medio\n"
        "FROM coletas c\nWHERE c.produto = 'GASOLINA' AND c.data_coleta >= DATE '2026-01-01'"
        " AND c.data_coleta < DATE '2027-01-01'\nGROUP BY trimestre\nORDER BY trimestre",
        variations=(
            "'por trimestre', 'em cada trimestre', 'por ano', 'ano a ano'",
            "Brasil, UF, região ou cidade",
        ),
        tags=("serie", "trimestre"),
    ),
    Intent(
        id="postos_condicao_preco",
        level="medio",
        uses_window=False,
        title="Postos com preço acima ou abaixo de um valor",
        description="quantos postos (ou cidades) tiveram coleta de um combustível acima (ou "
        "abaixo) de um valor em reais; postos contados com COUNT(DISTINCT c.posto_id).",
        sql_shape=f"SELECT COUNT(DISTINCT c.posto_id) AS postos\n{_J}\n"
        "WHERE c.produto = 'ETANOL' AND m.uf = 'MG' AND c.valor_venda > 4.5\n"
        "  AND c.data_coleta >= DATE '2026-09-01' AND c.data_coleta < DATE '2026-10-01'",
        variations=(
            "acima e abaixo; postos ou cidades",
            "UF, região, cidade ou Brasil; com período",
        ),
        tags=("contagem", "posto", "limite"),
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
    # v2 (M15, D34): padrões difíceis que o ft-v1 errou no dev2.
    Intent(
        id="tanto_quanto",
        level="dificil",
        uses_window=False,
        title="Tanto em X quanto em Y",
        description="postos, cidades ou bandeiras presentes nos dois recortes: coleta de um "
        "produto em dois períodos, dois produtos no mesmo período, ou duas UFs (INTERSECT).",
        sql_shape="SELECT COUNT(*) AS postos FROM (\n"
        f"  SELECT c.posto_id {_J.replace(chr(10), ' ')}\n"
        "  WHERE c.produto = 'ETANOL' AND m.nome = 'RECIFE' AND m.uf = 'PE'\n"
        "    AND c.data_coleta >= DATE '2025-09-01' AND c.data_coleta < DATE '2025-10-01'\n"
        "  INTERSECT\n"
        f"  SELECT c.posto_id {_J.replace(chr(10), ' ')}\n"
        "  WHERE c.produto = 'ETANOL' AND m.nome = 'RECIFE' AND m.uf = 'PE'\n"
        "    AND c.data_coleta >= DATE '2026-09-01' AND c.data_coleta < DATE '2026-10-01'\n"
        ") t",
        variations=(
            "'tanto em ... quanto em ...', 'nos dois períodos', 'e também'",
            "postos de uma cidade, cidades de uma UF, bandeiras em duas UFs",
        ),
        tags=("intersecao", "cte"),
    ),
    Intent(
        id="todos_os",
        level="dificil",
        uses_window=False,
        title="Todos os municípios / todos os meses",
        description="grupos em que todos os membros cumprem a condição: estados em que todos "
        "os municípios tiveram o etanol compensando, ou locais com preço acima de R$ X em "
        "todos os meses de um período.",
        sql_shape="WITH r AS (\n  SELECT m.uf, m.nome,\n"
        "    AVG(CASE WHEN c.produto = 'ETANOL' THEN c.valor_venda END)\n"
        "      / AVG(CASE WHEN c.produto = 'GASOLINA' THEN c.valor_venda END) AS razao\n"
        f"  {_J.replace(chr(10), chr(10) + '  ')}\n"
        "  WHERE c.produto IN ('ETANOL', 'GASOLINA') AND m.regiao = 'NE'\n"
        "    AND c.data_coleta >= DATE '2025-01-01' AND c.data_coleta < DATE '2026-01-01'\n"
        "  GROUP BY m.uf, m.nome\n)\n"
        "SELECT uf FROM r\nGROUP BY uf\nHAVING bool_and(razao <= 0.7)\nORDER BY uf",
        variations=(
            "'todos os municípios', 'todas as cidades', 'em todos os meses', 'nenhum'",
            "regra dos 70% ou limiar em reais; estados, capitais ou cidades de uma UF",
        ),
        tags=("todos", "cte"),
    ),
    Intent(
        id="participacao_pct",
        level="dificil",
        uses_window=False,
        title="Participação percentual",
        description="percentual de postos de uma bandeira (ou de coletas de um produto) no "
        "total de um grupo: por região/UF, num local, ou o ranking dos maiores percentuais.",
        sql_shape="SELECT m.regiao,\n"
        "  100.0 * COUNT(CASE WHEN p.bandeira = 'VIBRA' THEN 1 END) / COUNT(*) AS pct\n"
        "FROM postos p\nJOIN municipios m ON m.municipio_id = p.municipio_id\n"
        "GROUP BY m.regiao",
        variations=(
            "'percentual', 'participação', 'que fração', 'que % dos'",
            "bandeiras (inclusive branca) ou produtos; região, UF ou cidade",
        ),
        tags=("percentual", "contagem"),
    ),
    Intent(
        id="regiao_vs_estado",
        level="dificil",
        uses_window=False,
        title="Capital x interior, estado x região",
        description="comparações entre níveis geográficos: a capital contra o resto do "
        "estado, ou cada estado contra a média da sua região.",
        sql_shape=f"SELECT m.uf\n{_J}\n"
        "WHERE c.produto = 'ETANOL' AND c.data_coleta >= DATE '2025-01-01'"
        " AND c.data_coleta < DATE '2026-01-01'\nGROUP BY m.uf\n"
        "HAVING AVG(CASE WHEN m.capital THEN c.valor_venda END)\n"
        "     > AVG(CASE WHEN NOT m.capital THEN c.valor_venda END)\nORDER BY m.uf",
        variations=(
            "'capital x interior', 'resto do estado', 'média da região'",
            "lista de estados, diferença em reais por estado ou de um estado",
        ),
        tags=("comparacao", "regiao", "capital"),
    ),
    Intent(
        id="media_da_serie",
        level="dificil",
        uses_window=False,
        title="Contra a média do período ou da cidade",
        description="meses acima da média do ano, amplitude entre o mês mais caro e o mais "
        "barato, ou postos abaixo da média da cidade (CTE + subconsulta).",
        sql_shape="WITH mensal AS (\n"
        "  SELECT date_trunc('month', c.data_coleta) AS mes, AVG(c.valor_venda) AS preco\n"
        "  FROM coletas c\n  WHERE c.produto = 'ETANOL' AND c.data_coleta >= DATE '2026-01-01'"
        " AND c.data_coleta < DATE '2027-01-01'\n  GROUP BY mes\n)\n"
        "SELECT COUNT(*) AS meses FROM mensal\nWHERE preco > (SELECT AVG(c.valor_venda) "
        "FROM coletas c WHERE c.produto = 'ETANOL'\n"
        "  AND c.data_coleta >= DATE '2026-01-01' AND c.data_coleta < DATE '2027-01-01')",
        variations=(
            "quantos/quais meses acima ou abaixo da média do ano",
            "amplitude das médias mensais; postos abaixo/acima da média da cidade",
        ),
        tags=("subconsulta", "serie", "cte"),
    ),
    Intent(
        id="nunca_coleta",
        level="dificil",
        uses_window=False,
        title="Nunca tiveram coleta",
        description="municípios ou postos sem nenhuma coleta de um produto (no período), "
        "por NOT IN; lista ou contagem.",
        sql_shape="SELECT m.nome\nFROM municipios m\n"
        "WHERE m.uf = 'BA' AND m.municipio_id NOT IN (\n"
        "  SELECT p.municipio_id FROM coletas c\n"
        "  JOIN postos p ON p.posto_id = c.posto_id\n  WHERE c.produto = 'GNV'\n)",
        variations=(
            "'nunca', 'nenhuma coleta', 'sem coleta de'",
            "municípios de uma UF ou postos de uma cidade; lista ou contagem",
        ),
        tags=("subconsulta", "negacao"),
    ),
    Intent(
        id="variacao_grupos",
        level="dificil",
        uses_window=False,
        title="Variação entre períodos por região, bandeira ou produto",
        description="variação (R$ ou %) do preço médio entre dois períodos em cada região, "
        "em bandeiras citadas ou em cada produto; ou o grupo com a maior alta/queda.",
        sql_shape="WITH a AS (\n"
        f"  SELECT m.regiao, AVG(c.valor_venda) AS p\n  {_J.replace(chr(10), chr(10) + '  ')}\n"
        "  WHERE c.produto = 'GASOLINA' AND c.data_coleta >= DATE '2026-01-01'"
        " AND c.data_coleta < DATE '2026-02-01'\n  GROUP BY m.regiao\n), b AS (\n"
        f"  SELECT m.regiao, AVG(c.valor_venda) AS p\n  {_J.replace(chr(10), chr(10) + '  ')}\n"
        "  WHERE c.produto = 'GASOLINA' AND c.data_coleta >= DATE '2026-04-01'"
        " AND c.data_coleta < DATE '2026-05-01'\n  GROUP BY m.regiao\n)\n"
        "SELECT a.regiao, b.p - a.p AS variacao\nFROM a JOIN b ON a.regiao = b.regiao",
        variations=(
            "regiões, bandeiras citadas (IN) ou produtos",
            "'comparando a média de junho com a de março', anos, meses, semestres",
        ),
        tags=("variacao", "periodo", "cte"),
    ),
    Intent(
        id="regra70_dois_periodos",
        level="dificil",
        uses_window=False,
        title="Regra dos 70% em dois períodos",
        description="locais em que o etanol compensou num período mas não no outro, nos dois "
        "ou em nenhum (CTE por período + JOIN).",
        sql_shape="WITH a AS (\n  SELECT m.uf,\n"
        "    AVG(CASE WHEN c.produto = 'ETANOL' THEN c.valor_venda END)\n"
        "      / AVG(CASE WHEN c.produto = 'GASOLINA' THEN c.valor_venda END) AS razao\n"
        f"  {_J.replace(chr(10), chr(10) + '  ')}\n"
        "  WHERE c.data_coleta >= DATE '2025-01-01' AND c.data_coleta < DATE '2025-07-01'\n"
        "  GROUP BY m.uf\n), b AS (\n  SELECT m.uf,\n"
        "    AVG(CASE WHEN c.produto = 'ETANOL' THEN c.valor_venda END)\n"
        "      / AVG(CASE WHEN c.produto = 'GASOLINA' THEN c.valor_venda END) AS razao\n"
        f"  {_J.replace(chr(10), chr(10) + '  ')}\n"
        "  WHERE c.data_coleta >= DATE '2026-01-01' AND c.data_coleta < DATE '2026-07-01'\n"
        "  GROUP BY m.uf\n)\n"
        "SELECT b.uf FROM a JOIN b ON a.uf = b.uf\n"
        "WHERE b.razao <= 0.7 AND a.razao > 0.7\nORDER BY b.uf",
        variations=(
            "'em X, mas não em Y', 'deixou de compensar', 'nos dois períodos'",
            "estados, capitais, regiões ou cidades de uma UF",
        ),
        tags=("regra70", "periodo", "cte"),
    ),
    Intent(
        id="compara_produtos_grupo",
        level="dificil",
        uses_window=False,
        title="Onde um produto foi mais barato que outro",
        description="locais em que o preço médio de um combustível ficou abaixo (ou acima) do "
        "de outro, sem a regra dos 70% (HAVING com duas médias condicionais).",
        sql_shape=f"SELECT m.nome, m.uf\n{_J}\n"
        "WHERE c.produto IN ('DIESEL', 'GASOLINA') AND m.capital\n"
        "  AND c.data_coleta >= DATE '2025-01-01' AND c.data_coleta < DATE '2026-01-01'\n"
        "GROUP BY m.nome, m.uf\n"
        "HAVING AVG(CASE WHEN c.produto = 'DIESEL' THEN c.valor_venda END)\n"
        "     < AVG(CASE WHEN c.produto = 'GASOLINA' THEN c.valor_venda END)\nORDER BY m.nome",
        variations=(
            "GNV x etanol, diesel x gasolina, S10 x gasolina",
            "capitais, estados, cidades de uma UF, regiões",
        ),
        tags=("comparacao", "produto"),
    ),
    Intent(
        id="razao_por_grupo",
        level="dificil",
        uses_window=False,
        title="Razão etanol/gasolina por grupo",
        description="a razão etanol/gasolina comum em cada bandeira citada, capital, região "
        "ou cidade de uma UF, sem ordenar nem filtrar pela regra.",
        sql_shape="SELECT p.bandeira,\n"
        "  AVG(CASE WHEN c.produto = 'ETANOL' THEN c.valor_venda END)\n"
        "    / AVG(CASE WHEN c.produto = 'GASOLINA' THEN c.valor_venda END) AS razao\n"
        f"{_J}\nWHERE c.produto IN ('ETANOL', 'GASOLINA') AND m.uf = 'SC'\n"
        "  AND p.bandeira IN ('VIBRA', 'IPIRANGA', 'RAIZEN', 'BRANCA')\nGROUP BY p.bandeira",
        variations=(
            "bandeiras citadas (IN), capitais de uma região, regiões, cidades de uma UF",
            "último mês, ano, mês específico",
        ),
        tags=("regra70", "agrupamento"),
    ),
    Intent(
        id="posicao_ranking",
        level="dificil",
        uses_window=True,
        title="Posição num ranking",
        description="a posição de um estado, capital, cidade ou bandeira no ranking por preço "
        "médio (ou número de postos): RANK() OVER e filtro na entidade.",
        sql_shape="WITH u AS (\n  SELECT m.uf, AVG(c.valor_venda) AS preco\n"
        f"  {_J.replace(chr(10), chr(10) + '  ')}\n"
        "  WHERE c.produto = 'ETANOL' AND c.data_coleta >= DATE '2025-01-01'"
        " AND c.data_coleta < DATE '2026-01-01'\n"
        "  GROUP BY m.uf\n), r AS (\n"
        "  SELECT uf, RANK() OVER (ORDER BY preco) AS posicao FROM u\n)\n"
        "SELECT posicao FROM r WHERE uf = 'MG'",
        variations=(
            "'posição', 'em que lugar ficou', 'colocação'",
            "do mais caro para o mais barato e vice-versa; UF, capital, cidade, bandeira",
        ),
        tags=("janela", "ranking"),
        weight=Fraction(3, 4),
    ),
    Intent(
        id="acumulado",
        level="dificil",
        uses_window=True,
        title="Acumulado mês a mês",
        description="soma acumulada, mês a mês, do número de coletas num local e período "
        "(SUM() OVER (ORDER BY mes)).",
        sql_shape="WITH mensal AS (\n"
        "  SELECT date_trunc('month', c.data_coleta) AS mes, COUNT(*) AS coletas\n"
        "  FROM coletas c\n  WHERE c.produto = 'DIESEL S10' AND c.data_coleta >= DATE '2026-01-01'"
        " AND c.data_coleta < DATE '2027-01-01'\n  GROUP BY mes\n)\n"
        "SELECT mes, SUM(coletas) OVER (ORDER BY mes) AS acumulado\nFROM mensal\nORDER BY mes",
        variations=(
            "'acumulado', 'total acumulado', 'soma corrida'",
            "por produto ou todas as coletas; Brasil, UF, região ou cidade",
        ),
        tags=("janela", "serie", "acumulado"),
        weight=Fraction(3, 4),
    ),
    Intent(
        id="pico_por_grupo",
        level="dificil",
        uses_window=True,
        title="O mês (ou a bandeira) de pico em cada grupo",
        description="para cada grupo, o mês com o preço médio mais alto/baixo, ou a bandeira "
        "com mais postos; devolve o grupo e o item escolhido (ROW_NUMBER por grupo).",
        sql_shape="WITH mensal AS (\n"
        "  SELECT m.nome, date_trunc('month', c.data_coleta) AS mes, AVG(c.valor_venda) AS preco\n"
        f"  {_J.replace(chr(10), chr(10) + '  ')}\n"
        "  WHERE c.produto = 'ETANOL' AND m.capital AND m.regiao = 'NE'\n"
        "    AND c.data_coleta >= DATE '2026-01-01' AND c.data_coleta < DATE '2027-01-01'\n"
        "  GROUP BY m.nome, mes\n), r AS (\n"
        "  SELECT nome, mes, ROW_NUMBER() OVER (PARTITION BY nome ORDER BY preco DESC) AS pos\n"
        "  FROM mensal\n)\nSELECT nome, mes FROM r WHERE pos = 1 ORDER BY nome",
        variations=(
            "'para cada capital', 'em cada estado', 'por região', 'cada combustível'",
            "mês mais caro ou mais barato; bandeira com mais postos por região",
        ),
        tags=("janela", "ranking", "serie"),
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
