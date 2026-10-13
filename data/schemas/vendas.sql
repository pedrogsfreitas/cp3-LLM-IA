-- Banco: vendas (e-commerce)
-- Regras de negócio:
--   venda/compra = pedido com status <> 'cancelado'
--   faturamento = SUM(itens_pedido.quantidade * itens_pedido.preco_unitario * (1 - itens_pedido.desconto)) dos pedidos não cancelados
--   ticket médio = faturamento / número de pedidos não cancelados
--   margem bruta = faturamento - SUM(itens_pedido.quantidade * produtos.custo)
--   datas no formato 'AAAA-MM-DD'

CREATE TABLE categorias (
    id INTEGER PRIMARY KEY,
    nome TEXT NOT NULL
);

CREATE TABLE produtos (
    id INTEGER PRIMARY KEY,
    nome TEXT NOT NULL,
    categoria_id INTEGER NOT NULL REFERENCES categorias(id),
    preco REAL NOT NULL,          -- preço de venda atual (R$)
    custo REAL NOT NULL,          -- custo unitário (R$)
    ativo INTEGER NOT NULL        -- 1 = ativo, 0 = descontinuado
);

CREATE TABLE clientes (
    id INTEGER PRIMARY KEY,
    nome TEXT NOT NULL,
    cidade TEXT NOT NULL,
    estado TEXT NOT NULL,         -- sigla: 'SP', 'RJ', 'MG', ...
    segmento TEXT NOT NULL,       -- 'Varejo', 'PME', 'Corporativo'
    data_cadastro TEXT NOT NULL
);

CREATE TABLE pedidos (
    id INTEGER PRIMARY KEY,
    cliente_id INTEGER NOT NULL REFERENCES clientes(id),
    data_pedido TEXT NOT NULL,
    status TEXT NOT NULL,         -- 'entregue', 'em_transito', 'pendente', 'cancelado'
    canal TEXT NOT NULL           -- 'site', 'app', 'loja'
);

CREATE TABLE itens_pedido (
    id INTEGER PRIMARY KEY,
    pedido_id INTEGER NOT NULL REFERENCES pedidos(id),
    produto_id INTEGER NOT NULL REFERENCES produtos(id),
    quantidade INTEGER NOT NULL,
    preco_unitario REAL NOT NULL, -- preço praticado no pedido (R$)
    desconto REAL NOT NULL        -- fração de 0 a 1 (0.1 = 10%)
);
