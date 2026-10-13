-- Banco: rh (recursos humanos)
-- Regras de negócio:
--   funcionário ativo = data_desligamento IS NULL
--   headcount = número de funcionários ativos
--   métricas de salário (média, folha, ranking) consideram apenas funcionários ativos
--   folha salarial = SUM(salario) dos funcionários ativos
--   cada funcionário tem no máximo uma avaliação por ano
--   datas no formato 'AAAA-MM-DD'

CREATE TABLE departamentos (
    id INTEGER PRIMARY KEY,
    nome TEXT NOT NULL
);

CREATE TABLE cargos (
    id INTEGER PRIMARY KEY,
    titulo TEXT NOT NULL,
    nivel TEXT NOT NULL           -- 'Júnior', 'Pleno', 'Sênior', 'Gestor'
);

CREATE TABLE funcionarios (
    id INTEGER PRIMARY KEY,
    nome TEXT NOT NULL,
    departamento_id INTEGER NOT NULL REFERENCES departamentos(id),
    cargo_id INTEGER NOT NULL REFERENCES cargos(id),
    salario REAL NOT NULL,        -- salário mensal (R$)
    data_admissao TEXT NOT NULL,
    data_desligamento TEXT,       -- NULL se o funcionário está ativo
    cidade TEXT NOT NULL,
    modelo_trabalho TEXT NOT NULL -- 'presencial', 'híbrido', 'remoto'
);

CREATE TABLE avaliacoes (
    id INTEGER PRIMARY KEY,
    funcionario_id INTEGER NOT NULL REFERENCES funcionarios(id),
    ano INTEGER NOT NULL,
    nota REAL NOT NULL            -- de 1.0 a 5.0
);
