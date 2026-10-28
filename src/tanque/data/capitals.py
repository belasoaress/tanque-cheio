"""As 27 capitais (26 estados + DF), no formato da ANP: maiúsculas, sem acento (D18)."""

CAPITALS: frozenset[tuple[str, str]] = frozenset(
    {
        ("RIO BRANCO", "AC"),
        ("MACEIO", "AL"),
        ("MACAPA", "AP"),
        ("MANAUS", "AM"),
        ("SALVADOR", "BA"),
        ("FORTALEZA", "CE"),
        ("BRASILIA", "DF"),
        ("VITORIA", "ES"),
        ("GOIANIA", "GO"),
        ("SAO LUIS", "MA"),
        ("CUIABA", "MT"),
        ("CAMPO GRANDE", "MS"),
        ("BELO HORIZONTE", "MG"),
        ("BELEM", "PA"),
        ("JOAO PESSOA", "PB"),
        ("CURITIBA", "PR"),
        ("RECIFE", "PE"),
        ("TERESINA", "PI"),
        ("RIO DE JANEIRO", "RJ"),
        ("NATAL", "RN"),
        ("PORTO ALEGRE", "RS"),
        ("PORTO VELHO", "RO"),
        ("BOA VISTA", "RR"),
        ("FLORIANOPOLIS", "SC"),
        ("SAO PAULO", "SP"),
        ("ARACAJU", "SE"),
        ("PALMAS", "TO"),
    }
)

assert len(CAPITALS) == 27
assert len({uf for _, uf in CAPITALS}) == 27
