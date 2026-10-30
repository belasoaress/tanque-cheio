-- Contrato do banco. Não mudar sem aprovação.
CREATE TABLE municipios (
    municipio_id  INTEGER PRIMARY KEY,
    nome          VARCHAR NOT NULL,      -- como na ANP, em maiúsculas, ex.: 'SAO CARLOS'
    uf            VARCHAR(2) NOT NULL,   -- 'SP'
    regiao        VARCHAR(2) NOT NULL,   -- 'N','NE','CO','SE','S'
    capital       BOOLEAN NOT NULL       -- true para as 27 capitais (D18)
);

CREATE TABLE postos (
    posto_id      INTEGER PRIMARY KEY,
    cnpj          VARCHAR NOT NULL UNIQUE,
    razao_social  VARCHAR,
    bandeira      VARCHAR,               -- normalizada; 'BRANCA' para sem bandeira
    bairro        VARCHAR,
    cep           VARCHAR,
    municipio_id  INTEGER NOT NULL REFERENCES municipios(municipio_id)
);

CREATE TABLE coletas (
    coleta_id     BIGINT PRIMARY KEY,
    posto_id      INTEGER NOT NULL REFERENCES postos(posto_id),
    produto       VARCHAR NOT NULL,      -- valores fechados, ver docs/schema.md
    data_coleta   DATE NOT NULL,
    valor_venda   DECIMAL(6,3) NOT NULL, -- R$ por unidade
    valor_compra  DECIMAL(6,3),          -- frequentemente nulo
    unidade       VARCHAR NOT NULL       -- 'R$ / litro' ou 'R$ / m³'
);
