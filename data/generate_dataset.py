"""
Gera o dataset pergunta (PT-BR, linguagem de negócio) -> SQL (SQLite).

Cada template tem 4 formulações da mesma pergunta. As 3 primeiras vão para
treino/validação e a ÚLTIMA fica exclusiva do teste. Assim o teste mede se o
modelo generaliza para perguntas escritas de um jeito que ele nunca viu.

Todo SQL gerado é executado no banco correspondente; o script falha se algum
SQL der erro ou retornar vazio.

    python data/generate_dataset.py
"""
import json
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
from sql_utils import run_query  # noqa: E402

OUT_DIR = ROOT / "dataset"
SEED = 42
AMOSTRAS_TREINO = 10  # variações de slots por formulação de treino
AMOSTRAS_TESTE = 3    # variações de slots na formulação de teste
FRACAO_VAL = 0.1

MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho",
         "agosto", "setembro", "outubro", "novembro", "dezembro"]

FAT = "SUM(i.quantidade * i.preco_unitario * (1 - i.desconto))"

# --------------------------------------------------------------------------
# Templates: (id, slots, [formulações...], sql)
# --------------------------------------------------------------------------
VENDAS = [
    ("faturamento_ano", ["ano"], [
        "Qual foi o faturamento total em {ano}?",
        "Quanto a empresa faturou no ano de {ano}?",
        "Me mostre a receita de vendas de {ano}.",
        "Preciso do valor total vendido em {ano}, sem contar pedidos cancelados.",
    ], f"SELECT ROUND({FAT}, 2) AS faturamento FROM itens_pedido i JOIN pedidos p ON p.id = i.pedido_id "
       "WHERE p.status <> 'cancelado' AND strftime('%Y', p.data_pedido) = '{ano}';"),

    ("faturamento_mensal", ["ano"], [
        "Qual o faturamento mês a mês em {ano}?",
        "Mostre o faturamento por mês no ano de {ano}.",
        "Como foi a evolução mensal das vendas em {ano}?",
        "Quero ver quanto vendemos em cada mês de {ano}.",
    ], f"SELECT strftime('%m', p.data_pedido) AS mes, ROUND({FAT}, 2) AS faturamento FROM itens_pedido i "
       "JOIN pedidos p ON p.id = i.pedido_id WHERE p.status <> 'cancelado' AND strftime('%Y', p.data_pedido) = '{ano}' "
       "GROUP BY mes ORDER BY mes;"),

    ("top_produtos_qtd", ["ano", "n"], [
        "Quais são os {n} produtos mais vendidos em {ano}?",
        "Liste os {n} produtos com maior quantidade vendida em {ano}.",
        "Top {n} produtos por volume de vendas em {ano}.",
        "Quais {n} itens tiveram mais unidades vendidas em {ano}?",
    ], "SELECT pr.nome, SUM(i.quantidade) AS quantidade FROM itens_pedido i JOIN pedidos p ON p.id = i.pedido_id "
       "JOIN produtos pr ON pr.id = i.produto_id WHERE p.status <> 'cancelado' AND strftime('%Y', p.data_pedido) = '{ano}' "
       "GROUP BY pr.id, pr.nome ORDER BY quantidade DESC, pr.nome LIMIT {n};"),

    ("top_clientes_valor", ["n"], [
        "Quem são os {n} clientes que mais gastaram?",
        "Liste os {n} maiores clientes por valor comprado.",
        "Top {n} clientes por faturamento.",
        "Quais {n} clientes trouxeram mais receita para a empresa?",
    ], f"SELECT c.nome, ROUND({FAT}, 2) AS total FROM clientes c JOIN pedidos p ON p.cliente_id = c.id "
       "JOIN itens_pedido i ON i.pedido_id = p.id WHERE p.status <> 'cancelado' GROUP BY c.id, c.nome "
       "ORDER BY total DESC LIMIT {n};"),

    ("pedidos_por_status", [], [
        "Quantos pedidos existem em cada status?",
        "Mostre a quantidade de pedidos por status.",
        "Qual a distribuição dos pedidos por status?",
        "Conte os pedidos agrupando pelo status.",
    ], "SELECT status, COUNT(*) AS total_pedidos FROM pedidos GROUP BY status ORDER BY total_pedidos DESC;"),

    ("ticket_medio_canal", [], [
        "Qual o ticket médio por canal de venda?",
        "Calcule o ticket médio de cada canal.",
        "Mostre o valor médio por pedido em cada canal.",
        "Qual é o valor médio dos pedidos em cada canal de venda?",
    ], f"SELECT p.canal, ROUND({FAT} / COUNT(DISTINCT p.id), 2) AS ticket_medio FROM pedidos p "
       "JOIN itens_pedido i ON i.pedido_id = p.id WHERE p.status <> 'cancelado' GROUP BY p.canal "
       "ORDER BY ticket_medio DESC;"),

    ("clientes_estado", ["estado"], [
        "Quantos clientes temos no estado {estado}?",
        "Qual o número de clientes de {estado}?",
        "Conte os clientes cadastrados em {estado}.",
        "Quantos clientes são do estado de {estado}?",
    ], "SELECT COUNT(*) AS total_clientes FROM clientes WHERE estado = '{estado}';"),

    ("faturamento_categoria", [], [
        "Qual o faturamento por categoria de produto?",
        "Mostre quanto cada categoria vendeu.",
        "Faturamento agrupado por categoria.",
        "Quanto cada categoria de produto gerou de receita?",
    ], f"SELECT ca.nome AS categoria, ROUND({FAT}, 2) AS faturamento FROM itens_pedido i "
       "JOIN pedidos p ON p.id = i.pedido_id JOIN produtos pr ON pr.id = i.produto_id "
       "JOIN categorias ca ON ca.id = pr.categoria_id WHERE p.status <> 'cancelado' "
       "GROUP BY ca.id, ca.nome ORDER BY faturamento DESC;"),

    ("cancelados_ano", ["ano"], [
        "Quantos pedidos foram cancelados em {ano}?",
        "Qual o total de cancelamentos em {ano}?",
        "Número de pedidos cancelados no ano de {ano}.",
        "Em {ano}, quantos pedidos acabaram cancelados?",
    ], "SELECT COUNT(*) AS cancelados FROM pedidos WHERE status = 'cancelado' AND strftime('%Y', data_pedido) = '{ano}';"),

    ("clientes_sem_pedido", [], [
        "Quantos clientes nunca fizeram um pedido?",
        "Conte os clientes que não têm nenhum pedido.",
        "Qual a quantidade de clientes sem pedidos?",
        "Quantos clientes cadastrados ainda não fizeram nenhum pedido?",
    ], "SELECT COUNT(*) AS clientes_sem_pedido FROM clientes c WHERE NOT EXISTS "
       "(SELECT 1 FROM pedidos p WHERE p.cliente_id = c.id);"),

    ("produtos_categoria_preco", ["categoria_valor"], [
        "Quais produtos da categoria {categoria} têm preço acima de R$ {valor}?",
        "Liste os produtos de {categoria} com preço acima de {valor} reais.",
        "Mostre os produtos da categoria {categoria} com preço maior que {valor}.",
        "Quais itens de {categoria} têm preço superior a R$ {valor}?",
    ], "SELECT pr.nome, pr.preco FROM produtos pr JOIN categorias ca ON ca.id = pr.categoria_id "
       "WHERE ca.nome = '{categoria}' AND pr.preco > {valor} ORDER BY pr.preco DESC;"),

    ("margem_produto", ["n"], [
        "Quais os {n} produtos com maior margem bruta?",
        "Top {n} produtos por lucro bruto.",
        "Liste os {n} produtos que deram mais margem.",
        "Quais {n} produtos geraram mais lucro bruto em reais?",
    ], "SELECT pr.nome, ROUND(SUM(i.quantidade * i.preco_unitario * (1 - i.desconto) - i.quantidade * pr.custo), 2) "
       "AS margem_bruta FROM itens_pedido i JOIN pedidos p ON p.id = i.pedido_id JOIN produtos pr ON pr.id = i.produto_id "
       "WHERE p.status <> 'cancelado' GROUP BY pr.id, pr.nome ORDER BY margem_bruta DESC LIMIT {n};"),

    ("faturamento_estado_top", ["n"], [
        "Quais os {n} estados com maior faturamento?",
        "Top {n} estados por receita de vendas.",
        "Liste os {n} estados que mais compraram em valor.",
        "Em quais {n} estados vendemos mais, em reais?",
    ], f"SELECT c.estado, ROUND({FAT}, 2) AS faturamento FROM clientes c JOIN pedidos p ON p.cliente_id = c.id "
       "JOIN itens_pedido i ON i.pedido_id = p.id WHERE p.status <> 'cancelado' GROUP BY c.estado "
       "ORDER BY faturamento DESC LIMIT {n};"),

    ("pedidos_canal_mes", ["mes_ano"], [
        "Quantos pedidos tivemos por canal em {mes_nome} de {ano}?",
        "Mostre o número de pedidos de cada canal no mês {mm}/{ano}.",
        "Qual a quantidade de pedidos por canal em {mes_nome}/{ano}?",
        "Como os pedidos de {mes_nome} de {ano} se dividiram entre os canais?",
    ], "SELECT canal, COUNT(*) AS total_pedidos FROM pedidos WHERE strftime('%Y-%m', data_pedido) = '{ano}-{mm}' "
       "GROUP BY canal ORDER BY total_pedidos DESC;"),

    ("clientes_segmento_ano", ["segmento", "ano_cadastro"], [
        "Quantos clientes do segmento {segmento} foram cadastrados em {ano}?",
        "Número de novos clientes {segmento} em {ano}.",
        "Conte os clientes {segmento} com cadastro em {ano}.",
        "Em {ano}, quantos clientes do segmento {segmento} entraram na base?",
    ], "SELECT COUNT(*) AS total_clientes FROM clientes WHERE segmento = '{segmento}' "
       "AND strftime('%Y', data_cadastro) = '{ano}';"),

    ("preco_medio_categoria", [], [
        "Qual o preço médio dos produtos de cada categoria?",
        "Mostre o preço médio por categoria.",
        "Preço médio de venda agrupado por categoria.",
        "Em média, qual é o preço dos produtos em cada categoria?",
    ], "SELECT ca.nome AS categoria, ROUND(AVG(pr.preco), 2) AS preco_medio FROM produtos pr "
       "JOIN categorias ca ON ca.id = pr.categoria_id GROUP BY ca.id, ca.nome ORDER BY preco_medio DESC;"),

    ("ticket_medio_ano", ["ano"], [
        "Qual foi o ticket médio em {ano}?",
        "Calcule o ticket médio dos pedidos de {ano}.",
        "Quanto foi o valor médio por pedido em {ano}?",
        "Em {ano}, qual o gasto médio por pedido?",
    ], f"SELECT ROUND({FAT} / COUNT(DISTINCT p.id), 2) AS ticket_medio FROM pedidos p "
       "JOIN itens_pedido i ON i.pedido_id = p.id WHERE p.status <> 'cancelado' "
       "AND strftime('%Y', p.data_pedido) = '{ano}';"),

    ("faturamento_canal_ano", ["canal", "ano"], [
        "Quanto o canal {canal} faturou em {ano}?",
        "Qual o faturamento do canal {canal} em {ano}?",
        "Receita de vendas pelo canal {canal} no ano de {ano}.",
        "Em {ano}, qual foi a receita gerada pelo canal {canal}?",
    ], f"SELECT ROUND({FAT}, 2) AS faturamento FROM itens_pedido i JOIN pedidos p ON p.id = i.pedido_id "
       "WHERE p.status <> 'cancelado' AND p.canal = '{canal}' AND strftime('%Y', p.data_pedido) = '{ano}';"),

    ("top_categorias_qtd", ["n_cat"], [
        "Quais as {n} categorias com mais unidades vendidas?",
        "Top {n} categorias por quantidade vendida.",
        "Liste as {n} categorias que mais venderam em volume.",
        "Quais {n} categorias venderam mais itens?",
    ], "SELECT ca.nome AS categoria, SUM(i.quantidade) AS quantidade FROM itens_pedido i "
       "JOIN pedidos p ON p.id = i.pedido_id JOIN produtos pr ON pr.id = i.produto_id "
       "JOIN categorias ca ON ca.id = pr.categoria_id WHERE p.status <> 'cancelado' "
       "GROUP BY ca.id, ca.nome ORDER BY quantidade DESC, ca.nome LIMIT {n};"),

    ("clientes_compraram_ano", ["ano"], [
        "Quantos clientes diferentes compraram em {ano}?",
        "Número de clientes únicos com compra em {ano}.",
        "Quantos clientes fizeram pelo menos uma compra em {ano}?",
        "Qual a quantidade de compradores distintos em {ano}?",
    ], "SELECT COUNT(DISTINCT cliente_id) AS clientes FROM pedidos WHERE status <> 'cancelado' "
       "AND strftime('%Y', data_pedido) = '{ano}';"),

    ("produtos_inativos", [], [
        "Quais produtos estão descontinuados?",
        "Liste os produtos inativos.",
        "Mostre os produtos que não estão mais ativos.",
        "Quais itens do catálogo foram descontinuados?",
    ], "SELECT nome FROM produtos WHERE ativo = 0 ORDER BY nome;"),
]

RH = [
    ("headcount_depto", [], [
        "Qual o headcount por departamento?",
        "Quantos funcionários ativos tem cada departamento?",
        "Mostre o número de colaboradores ativos por área.",
        "Quantas pessoas trabalham hoje em cada departamento?",
    ], "SELECT d.nome AS departamento, COUNT(*) AS headcount FROM funcionarios f "
       "JOIN departamentos d ON d.id = f.departamento_id WHERE f.data_desligamento IS NULL "
       "GROUP BY d.id, d.nome ORDER BY headcount DESC;"),

    ("salario_medio_nivel", [], [
        "Qual o salário médio por nível de cargo?",
        "Mostre a média salarial de cada nível.",
        "Salário médio agrupado por nível (Júnior, Pleno, Sênior, Gestor).",
        "Quanto ganha em média um funcionário de cada nível?",
    ], "SELECT c.nivel, ROUND(AVG(f.salario), 2) AS salario_medio FROM funcionarios f "
       "JOIN cargos c ON c.id = f.cargo_id WHERE f.data_desligamento IS NULL GROUP BY c.nivel "
       "ORDER BY salario_medio DESC;"),

    ("admitidos_ano", ["ano_adm"], [
        "Quantos funcionários foram admitidos em {ano}?",
        "Número de contratações em {ano}.",
        "Quantas pessoas entraram na empresa em {ano}?",
        "Em {ano}, quantas admissões tivemos?",
    ], "SELECT COUNT(*) AS admitidos FROM funcionarios WHERE strftime('%Y', data_admissao) = '{ano}';"),

    ("desligados_ano", ["ano_desl"], [
        "Quantos funcionários foram desligados em {ano}?",
        "Número de desligamentos em {ano}.",
        "Quantas pessoas saíram da empresa em {ano}?",
        "Em {ano}, quantos colaboradores deixaram a empresa?",
    ], "SELECT COUNT(*) AS desligados FROM funcionarios WHERE strftime('%Y', data_desligamento) = '{ano}';"),

    ("top_salarios", ["n"], [
        "Quais são os {n} maiores salários da empresa?",
        "Liste os {n} funcionários ativos com maior salário.",
        "Top {n} salários entre os colaboradores ativos.",
        "Quem são os {n} funcionários mais bem pagos?",
    ], "SELECT nome, salario FROM funcionarios WHERE data_desligamento IS NULL ORDER BY salario DESC LIMIT {n};"),

    ("nota_media_depto_ano", ["ano_aval"], [
        "Qual a nota média de avaliação por departamento em {ano}?",
        "Média das avaliações de desempenho de cada departamento em {ano}.",
        "Mostre a nota média por área no ciclo de {ano}.",
        "Como cada departamento foi avaliado, em média, em {ano}?",
    ], "SELECT d.nome AS departamento, ROUND(AVG(a.nota), 2) AS nota_media FROM avaliacoes a "
       "JOIN funcionarios f ON f.id = a.funcionario_id JOIN departamentos d ON d.id = f.departamento_id "
       "WHERE a.ano = {ano} GROUP BY d.id, d.nome ORDER BY nota_media DESC;"),

    ("funcionarios_depto", ["departamento"], [
        "Quais funcionários ativos trabalham no departamento de {departamento}?",
        "Liste os colaboradores ativos de {departamento}.",
        "Mostre o nome dos funcionários ativos da área de {departamento}.",
        "Quem faz parte hoje do time de {departamento}?",
    ], "SELECT f.nome FROM funcionarios f JOIN departamentos d ON d.id = f.departamento_id "
       "WHERE d.nome = '{departamento}' AND f.data_desligamento IS NULL ORDER BY f.nome;"),

    ("modelo_trabalho", ["modelo"], [
        "Quantos funcionários ativos trabalham no modelo {modelo}?",
        "Número de colaboradores ativos em regime {modelo}.",
        "Quantas pessoas ativas estão no modelo de trabalho {modelo}?",
        "Qual o total de funcionários ativos em formato {modelo}?",
    ], "SELECT COUNT(*) AS total FROM funcionarios WHERE modelo_trabalho = '{modelo}' AND data_desligamento IS NULL;"),

    ("folha_depto", [], [
        "Qual a folha salarial de cada departamento?",
        "Mostre o custo de folha por departamento.",
        "Soma dos salários dos ativos por departamento.",
        "Quanto cada área gasta por mês com salários?",
    ], "SELECT d.nome AS departamento, ROUND(SUM(f.salario), 2) AS folha_salarial FROM funcionarios f "
       "JOIN departamentos d ON d.id = f.departamento_id WHERE f.data_desligamento IS NULL "
       "GROUP BY d.id, d.nome ORDER BY folha_salarial DESC;"),

    ("nota_acima", ["nota", "ano_aval"], [
        "Quantos funcionários tiveram nota acima de {nota} em {ano}?",
        "Número de colaboradores com avaliação maior que {nota} em {ano}.",
        "Conte as avaliações acima de {nota} no ano de {ano}.",
        "Em {ano}, quantas pessoas foram avaliadas com nota superior a {nota}?",
    ], "SELECT COUNT(*) AS total FROM avaliacoes WHERE ano = {ano} AND nota > {nota};"),

    ("depto_maior_salario_medio", [], [
        "Qual departamento tem o maior salário médio?",
        "Qual área paga melhor, em média?",
        "Departamento com a maior média salarial.",
        "Em qual departamento o salário médio é mais alto?",
    ], "SELECT d.nome AS departamento, ROUND(AVG(f.salario), 2) AS salario_medio FROM funcionarios f "
       "JOIN departamentos d ON d.id = f.departamento_id WHERE f.data_desligamento IS NULL "
       "GROUP BY d.id, d.nome ORDER BY salario_medio DESC LIMIT 1;"),

    ("headcount_cidade", [], [
        "Quantos funcionários ativos temos em cada cidade?",
        "Headcount por cidade.",
        "Distribuição dos colaboradores ativos por cidade.",
        "Em quais cidades estão nossos funcionários ativos e quantos em cada uma?",
    ], "SELECT cidade, COUNT(*) AS headcount FROM funcionarios WHERE data_desligamento IS NULL "
       "GROUP BY cidade ORDER BY headcount DESC;"),

    ("cargos_depto", ["departamento"], [
        "Quantos funcionários ativos há em cada cargo no departamento de {departamento}?",
        "Distribuição dos ativos por cargo dentro de {departamento}.",
        "Mostre a quantidade de pessoas ativas por cargo na área de {departamento}.",
        "Como o time de {departamento} está dividido entre os cargos?",
    ], "SELECT c.titulo AS cargo, COUNT(*) AS total FROM funcionarios f JOIN cargos c ON c.id = f.cargo_id "
       "JOIN departamentos d ON d.id = f.departamento_id WHERE d.nome = '{departamento}' "
       "AND f.data_desligamento IS NULL GROUP BY c.id, c.titulo ORDER BY total DESC;"),

    ("salario_medio_depto", ["departamento"], [
        "Qual o salário médio do departamento de {departamento}?",
        "Média salarial da área de {departamento}.",
        "Quanto ganham em média os funcionários de {departamento}?",
        "Qual é a média de salário em {departamento}?",
    ], "SELECT ROUND(AVG(f.salario), 2) AS salario_medio FROM funcionarios f "
       "JOIN departamentos d ON d.id = f.departamento_id WHERE d.nome = '{departamento}' "
       "AND f.data_desligamento IS NULL;"),

    ("sem_avaliacao", ["ano_aval"], [
        "Quantos funcionários ativos não foram avaliados em {ano}?",
        "Número de colaboradores ativos sem avaliação no ciclo de {ano}.",
        "Conte os funcionários ativos que não têm nota em {ano}.",
        "Em {ano}, quantas pessoas ativas ficaram sem avaliação de desempenho?",
    ], "SELECT COUNT(*) AS sem_avaliacao FROM funcionarios f WHERE f.data_desligamento IS NULL "
       "AND NOT EXISTS (SELECT 1 FROM avaliacoes a WHERE a.funcionario_id = f.id AND a.ano = {ano});"),

    ("faixa_salarial_nivel", [], [
        "Qual o maior e o menor salário de cada nível?",
        "Mostre o salário mínimo e máximo por nível de cargo.",
        "Faixa salarial (mínimo e máximo) por nível.",
        "Para cada nível, qual o piso e o teto salarial praticado?",
    ], "SELECT c.nivel, MIN(f.salario) AS menor_salario, MAX(f.salario) AS maior_salario FROM funcionarios f "
       "JOIN cargos c ON c.id = f.cargo_id WHERE f.data_desligamento IS NULL GROUP BY c.nivel;"),
]

SALDO = "SUM(CASE WHEN tipo = 'receita' THEN valor ELSE -valor END)"

FINANCEIRO = [
    ("receitas_ano", ["ano"], [
        "Qual o total de receitas em {ano}?",
        "Quanto entrou de receita em {ano}?",
        "Some todas as receitas do ano de {ano}.",
        "Qual foi a receita total registrada em {ano}?",
    ], "SELECT ROUND(SUM(valor), 2) AS total_receitas FROM lancamentos WHERE tipo = 'receita' "
       "AND strftime('%Y', data) = '{ano}';"),

    ("despesas_cc_ano", ["ano"], [
        "Qual o total de despesas por centro de custo em {ano}?",
        "Mostre as despesas de {ano} agrupadas por centro de custo.",
        "Quanto cada centro de custo gastou em {ano}?",
        "Em {ano}, quais foram os gastos de cada centro de custo?",
    ], "SELECT cc.nome AS centro_custo, ROUND(SUM(l.valor), 2) AS total_despesas FROM lancamentos l "
       "JOIN centros_custo cc ON cc.id = l.centro_custo_id WHERE l.tipo = 'despesa' "
       "AND strftime('%Y', l.data) = '{ano}' GROUP BY cc.id, cc.nome ORDER BY total_despesas DESC;"),

    ("saldo_mensal", ["ano"], [
        "Qual foi o saldo de cada mês em {ano}?",
        "Mostre receitas menos despesas mês a mês em {ano}.",
        "Resultado mensal (receitas - despesas) de {ano}.",
        "Como ficou o saldo mensal da empresa ao longo de {ano}?",
    ], f"SELECT strftime('%m', data) AS mes, ROUND({SALDO}, 2) AS saldo FROM lancamentos "
       "WHERE strftime('%Y', data) = '{ano}' GROUP BY mes ORDER BY mes;"),

    ("top_fornecedores_pago", ["n"], [
        "Quais os {n} fornecedores que mais receberam pagamentos?",
        "Top {n} fornecedores por valor já pago.",
        "Liste os {n} fornecedores com maior total pago.",
        "Para quais {n} fornecedores a empresa mais pagou até agora?",
    ], "SELECT f.nome, ROUND(SUM(l.valor), 2) AS total_pago FROM lancamentos l "
       "JOIN fornecedores f ON f.id = l.fornecedor_id WHERE l.tipo = 'despesa' AND l.status = 'pago' "
       "GROUP BY f.id, f.nome ORDER BY total_pago DESC LIMIT {n};"),

    ("atrasados", [], [
        "Quantos lançamentos estão atrasados e qual o valor total?",
        "Mostre a quantidade e o valor dos lançamentos em atraso.",
        "Total de lançamentos com status atrasado (quantidade e soma).",
        "Quanto temos em atraso, em número de lançamentos e em reais?",
    ], "SELECT COUNT(*) AS quantidade, ROUND(SUM(valor), 2) AS valor_total FROM lancamentos WHERE status = 'atrasado';"),

    ("despesas_cc_mes", ["centro", "mes_ano"], [
        "Quanto o centro de custo {centro} gastou em {mes_nome} de {ano}?",
        "Total de despesas de {centro} no mês {mm}/{ano}.",
        "Despesas do centro de custo {centro} em {mes_nome}/{ano}.",
        "Qual foi o gasto do centro de custo {centro} em {mes_nome} de {ano}?",
    ], "SELECT ROUND(SUM(l.valor), 2) AS total_despesas FROM lancamentos l "
       "JOIN centros_custo cc ON cc.id = l.centro_custo_id WHERE cc.nome = '{centro}' AND l.tipo = 'despesa' "
       "AND strftime('%Y-%m', l.data) = '{ano}-{mm}';"),

    ("contas_a_pagar", [], [
        "Qual o total de contas a pagar em aberto?",
        "Quanto temos de despesas ainda não pagas?",
        "Soma das despesas pendentes ou atrasadas.",
        "Qual o valor total que ainda falta pagar aos fornecedores?",
    ], "SELECT ROUND(SUM(valor), 2) AS a_pagar FROM lancamentos WHERE tipo = 'despesa' "
       "AND status IN ('pendente', 'atrasado');"),

    ("saldo_conta", [], [
        "Qual o saldo de cada conta bancária?",
        "Mostre o saldo por conta.",
        "Receitas menos despesas por conta bancária.",
        "Quanto há de saldo em cada uma das contas da empresa?",
    ], "SELECT cb.nome AS conta, ROUND(SUM(CASE WHEN l.tipo = 'receita' THEN l.valor ELSE -l.valor END), 2) AS saldo "
       "FROM lancamentos l JOIN contas_bancarias cb ON cb.id = l.conta_id GROUP BY cb.id, cb.nome ORDER BY saldo DESC;"),

    ("fornecedores_estado", ["estado_forn"], [
        "Quais fornecedores são do estado {estado}?",
        "Liste os fornecedores de {estado}.",
        "Mostre os fornecedores localizados em {estado}.",
        "Que fornecedores temos no estado de {estado}?",
    ], "SELECT nome FROM fornecedores WHERE estado = '{estado}' ORDER BY nome;"),

    ("despesa_categoria_forn", ["ano"], [
        "Qual o total de despesas por categoria de fornecedor em {ano}?",
        "Mostre os gastos de {ano} agrupados pela categoria do fornecedor.",
        "Quanto gastamos com cada categoria de fornecedor em {ano}?",
        "Em {ano}, como as despesas se dividiram entre as categorias de fornecedores?",
    ], "SELECT f.categoria, ROUND(SUM(l.valor), 2) AS total_despesas FROM lancamentos l "
       "JOIN fornecedores f ON f.id = l.fornecedor_id WHERE l.tipo = 'despesa' AND strftime('%Y', l.data) = '{ano}' "
       "GROUP BY f.categoria ORDER BY total_despesas DESC;"),

    ("maior_despesa_ano", ["ano"], [
        "Qual foi a maior despesa de {ano}?",
        "Mostre o lançamento de despesa de maior valor em {ano}.",
        "Qual a despesa mais cara registrada em {ano}?",
        "Em {ano}, qual foi o maior gasto individual?",
    ], "SELECT descricao, valor, data FROM lancamentos WHERE tipo = 'despesa' AND strftime('%Y', data) = '{ano}' "
       "ORDER BY valor DESC LIMIT 1;"),

    ("lancamentos_status", [], [
        "Quantos lançamentos há em cada status?",
        "Distribuição dos lançamentos por status.",
        "Conte os lançamentos agrupados por status.",
        "Quantos lançamentos estão pagos, pendentes e atrasados?",
    ], "SELECT status, COUNT(*) AS total FROM lancamentos GROUP BY status ORDER BY total DESC;"),

    ("receitas_trimestre", ["ano"], [
        "Qual o total de receitas por trimestre em {ano}?",
        "Mostre as receitas de {ano} agrupadas por trimestre.",
        "Receita trimestral de {ano}.",
        "Quanto entrou de receita em cada trimestre de {ano}?",
    ], "SELECT ((CAST(strftime('%m', data) AS INTEGER) - 1) / 3) + 1 AS trimestre, ROUND(SUM(valor), 2) AS total_receitas "
       "FROM lancamentos WHERE tipo = 'receita' AND strftime('%Y', data) = '{ano}' GROUP BY trimestre ORDER BY trimestre;"),

    ("fornecedores_sem_lancamento", [], [
        "Quais fornecedores não têm nenhum lançamento?",
        "Liste os fornecedores sem despesas registradas.",
        "Mostre os fornecedores que nunca foram usados.",
        "Existem fornecedores cadastrados sem nenhum lançamento? Quais?",
    ], "SELECT f.nome FROM fornecedores f WHERE NOT EXISTS "
       "(SELECT 1 FROM lancamentos l WHERE l.fornecedor_id = f.id) ORDER BY f.nome;"),

    ("despesas_mes", ["mes_ano"], [
        "Qual o total de despesas em {mes_nome} de {ano}?",
        "Quanto gastamos no mês {mm}/{ano}?",
        "Some as despesas de {mes_nome}/{ano}.",
        "Qual foi o gasto total da empresa em {mes_nome} de {ano}?",
    ], "SELECT ROUND(SUM(valor), 2) AS total_despesas FROM lancamentos WHERE tipo = 'despesa' "
       "AND strftime('%Y-%m', data) = '{ano}-{mm}';"),
]

TEMPLATES = {"vendas": VENDAS, "rh": RH, "financeiro": FINANCEIRO}


# --------------------------------------------------------------------------
# Domínios dos slots (lidos dos próprios bancos)
# --------------------------------------------------------------------------
def dominios(db):
    conn = sqlite3.connect(ROOT / "db" / f"{db}.sqlite")
    q = lambda sql: [r[0] for r in conn.execute(sql)]  # noqa: E731
    d = {"n": [3, 5, 10]}
    if db == "vendas":
        d["ano"] = [2024, 2025, 2026]
        d["ano_cadastro"] = [2022, 2023, 2024, 2025, 2026]
        d["estado"] = q("SELECT DISTINCT estado FROM clientes ORDER BY estado")
        d["canal"] = ["site", "app", "loja"]
        d["segmento"] = ["Varejo", "PME", "Corporativo"]
        d["n_cat"] = [3, 5]
        precos = defaultdict(list)
        for cat, preco in conn.execute(
                "SELECT ca.nome, pr.preco FROM produtos pr JOIN categorias ca ON ca.id = pr.categoria_id"):
            precos[cat].append(preco)
        d["categoria_valor"] = [(cat, v) for cat, ps in precos.items()
                                for v in (50, 100, 200, 300, 500, 1000, 2000, 3000) if v < max(ps)]
    elif db == "rh":
        d["ano_adm"] = list(range(2018, 2027))
        d["ano_desl"] = list(range(2020, 2027))
        d["ano_aval"] = [2023, 2024, 2025]
        d["departamento"] = q("SELECT nome FROM departamentos ORDER BY id")
        d["modelo"] = ["presencial", "híbrido", "remoto"]
        d["nota"] = [3.0, 3.5, 4.0, 4.5]
    else:
        d["ano"] = [2024, 2025, 2026]
        d["centro"] = q("SELECT nome FROM centros_custo ORDER BY id")
        d["estado_forn"] = q("SELECT DISTINCT estado FROM fornecedores ORDER BY estado")
    d["mes_ano"] = [(a, m) for a in (2024, 2025, 2026) for m in range(1, 13) if (a, m) <= (2026, 9)]
    conn.close()
    return d


def sortear_slots(rng, slots, dom):
    valores = {}
    for s in slots:
        if s == "mes_ano":
            ano, mes = rng.choice(dom["mes_ano"])
            valores.update(ano=ano, mm=f"{mes:02d}", mes_nome=MESES[mes - 1])
        elif s == "categoria_valor":
            cat, v = rng.choice(dom["categoria_valor"])
            valores.update(categoria=cat, valor=v)
        elif s in ("ano_adm", "ano_desl", "ano_aval", "ano_cadastro"):
            valores["ano"] = rng.choice(dom[s])
        elif s == "n_cat":
            valores["n"] = rng.choice(dom[s])
        elif s == "estado_forn":
            valores["estado"] = rng.choice(dom[s])
        else:
            valores[s] = rng.choice(dom[s])
    return valores


def gerar_exemplos(rng, db, template, dom, formulacao_idx, quantidade):
    tid, slots, formulacoes, sql_tpl = template
    exemplos, vistos = [], set()
    tentativas = quantidade if slots else 1
    for _ in range(tentativas * 5):
        if len(exemplos) >= tentativas:
            break
        v = sortear_slots(rng, slots, dom)
        pergunta = formulacoes[formulacao_idx].format(**v)
        if pergunta in vistos:
            continue
        vistos.add(pergunta)
        exemplos.append({"db": db, "template": f"{db}.{tid}", "formulacao": formulacao_idx,
                         "question": pergunta, "sql": sql_tpl.format(**v)})
    return exemplos


def main():
    rng = random.Random(SEED)
    treino, teste = [], []
    for db, templates in TEMPLATES.items():
        dom = dominios(db)
        for t in templates:
            assert len(t[2]) == 4, f"{t[0]} precisa de 4 formulações"
            for idx in range(3):
                treino += gerar_exemplos(rng, db, t, dom, idx, AMOSTRAS_TREINO)
            teste += gerar_exemplos(rng, db, t, dom, 3, AMOSTRAS_TESTE)

    # valida: todo SQL precisa executar e retornar algo
    problemas = []
    for ex in treino + teste:
        try:
            _, linhas = run_query(ex["db"], ex["sql"])
            if not linhas or all(v is None for v in linhas[0]):
                problemas.append((ex["template"], ex["question"], "resultado vazio"))
        except Exception as e:  # noqa: BLE001
            problemas.append((ex["template"], ex["question"], str(e)))
    if problemas:
        for p in problemas:
            print("[ERRO]", *p)
        sys.exit(f"{len(problemas)} exemplos inválidos.")

    rng.shuffle(treino)
    n_val = int(len(treino) * FRACAO_VAL)
    splits = {"train": treino[n_val:], "val": treino[:n_val], "test": teste}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for nome, exemplos in splits.items():
        with open(OUT_DIR / f"{nome}.jsonl", "w", encoding="utf-8") as f:
            for i, ex in enumerate(exemplos):
                f.write(json.dumps({"id": f"{nome}-{i:04d}", **ex}, ensure_ascii=False) + "\n")

    n_templates = sum(len(t) for t in TEMPLATES.values())
    print(f"[ok] {n_templates} templates, todos os SQL validados nos bancos.")
    for nome, exemplos in splits.items():
        por_db = Counter(e["db"] for e in exemplos)
        print(f"     {nome:5s}: {len(exemplos):4d} exemplos  {dict(por_db)}")


if __name__ == "__main__":
    main()
