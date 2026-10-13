-- Banco: financeiro (contas a pagar e a receber)
-- Regras de negócio:
--   lancamentos.tipo = 'receita' (entrada) ou 'despesa' (saída); valor é sempre positivo
--   saldo = SUM(receitas) - SUM(despesas)
--   em aberto = status IN ('pendente', 'atrasado'); valor pago = status 'pago'
--   receitas não têm fornecedor (fornecedor_id NULL)
--   datas no formato 'AAAA-MM-DD'

CREATE TABLE centros_custo (
    id INTEGER PRIMARY KEY,
    nome TEXT NOT NULL
);

CREATE TABLE fornecedores (
    id INTEGER PRIMARY KEY,
    nome TEXT NOT NULL,
    categoria TEXT NOT NULL,      -- 'Tecnologia', 'Logística', 'Marketing', 'Serviços', 'Infraestrutura'
    estado TEXT NOT NULL          -- sigla: 'SP', 'RJ', ...
);

CREATE TABLE contas_bancarias (
    id INTEGER PRIMARY KEY,
    nome TEXT NOT NULL,
    banco TEXT NOT NULL,
    tipo TEXT NOT NULL            -- 'corrente', 'investimento'
);

CREATE TABLE lancamentos (
    id INTEGER PRIMARY KEY,
    data TEXT NOT NULL,
    tipo TEXT NOT NULL,           -- 'receita' ou 'despesa'
    valor REAL NOT NULL,          -- valor em R$ (sempre positivo)
    status TEXT NOT NULL,         -- 'pago', 'pendente', 'atrasado'
    descricao TEXT NOT NULL,
    centro_custo_id INTEGER NOT NULL REFERENCES centros_custo(id),
    fornecedor_id INTEGER REFERENCES fornecedores(id),
    conta_id INTEGER NOT NULL REFERENCES contas_bancarias(id)
);
