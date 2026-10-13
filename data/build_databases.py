"""
Cria os bancos SQLite de negócio (vendas, rh, financeiro) com dados sintéticos.

Os dados são gerados com semente fixa, então todo integrante do grupo obtém
exatamente os mesmos bancos ao rodar:

    python data/build_databases.py
"""
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCHEMAS = ROOT / "schemas"
DB_DIR = ROOT / "db"

DATA_INICIO = date(2024, 1, 1)
DATA_FIM = date(2026, 9, 30)

PRIMEIROS_NOMES = [
    "Ana", "Bruno", "Carla", "Daniel", "Eduarda", "Felipe", "Gabriela", "Henrique",
    "Isabela", "João", "Larissa", "Lucas", "Mariana", "Mateus", "Natália", "Otávio",
    "Paula", "Rafael", "Sofia", "Thiago", "Vanessa", "Vinícius", "Beatriz", "Caio",
    "Débora", "Fernanda", "Gustavo", "Helena", "Igor", "Juliana", "Leonardo", "Letícia",
    "Marcelo", "Nicole", "Pedro", "Renata", "Rodrigo", "Tatiana", "Victor", "Yasmin",
]
SOBRENOMES = [
    "Silva", "Santos", "Oliveira", "Souza", "Rodrigues", "Ferreira", "Alves", "Pereira",
    "Lima", "Gomes", "Costa", "Ribeiro", "Martins", "Carvalho", "Almeida", "Lopes",
    "Soares", "Fernandes", "Vieira", "Barbosa", "Rocha", "Dias", "Nascimento", "Andrade",
    "Moreira", "Nunes", "Marques", "Machado", "Mendes", "Freitas", "Cardoso", "Teixeira",
]


def nome_pessoa(rng):
    return f"{rng.choice(PRIMEIROS_NOMES)} {rng.choice(SOBRENOMES)} {rng.choice(SOBRENOMES)}"


def data_aleatoria(rng, inicio, fim):
    dias = (fim - inicio).days
    return inicio + timedelta(days=rng.randint(0, max(dias, 0)))


def criar_banco(nome):
    DB_DIR.mkdir(parents=True, exist_ok=True)
    caminho = DB_DIR / f"{nome}.sqlite"
    if caminho.exists():
        caminho.unlink()
    conn = sqlite3.connect(caminho)
    conn.executescript((SCHEMAS / f"{nome}.sql").read_text(encoding="utf-8"))
    return conn, caminho


# ---------------------------------------------------------------- vendas
CATEGORIAS = {
    "Eletrônicos": (["Smartphone", "Fone Bluetooth", "Smart TV 50\"", "Caixa de Som", "Smartwatch", "Tablet", "Câmera de Segurança"], 120, 4500),
    "Informática": (["Notebook", "Monitor 27\"", "Teclado Mecânico", "Mouse Sem Fio", "SSD 1TB", "Impressora", "Webcam Full HD"], 80, 6000),
    "Casa e Cozinha": (["Air Fryer", "Liquidificador", "Cafeteira", "Jogo de Panelas", "Aspirador Robô", "Micro-ondas", "Mixer"], 60, 1800),
    "Esporte e Lazer": (["Bicicleta Aro 29", "Esteira", "Kit Halteres", "Tênis de Corrida", "Barraca de Camping", "Bola de Futebol", "Tapete de Yoga"], 40, 3500),
    "Livros": (["Romance Best-seller", "Livro de Finanças", "Livro de Programação", "HQ Colecionável", "Livro Infantil", "Biografia", "Box de Ficção"], 25, 220),
    "Beleza": (["Perfume Importado", "Kit Skincare", "Secador de Cabelo", "Chapinha", "Barbeador Elétrico", "Hidratante", "Protetor Solar"], 30, 650),
    "Brinquedos": (["Kit de Blocos", "Boneca", "Carrinho de Controle", "Quebra-cabeça 1000 peças", "Jogo de Tabuleiro", "Pelúcia", "Drone Infantil"], 35, 800),
    "Moda": (["Jaqueta Jeans", "Camiseta Básica", "Calça Chino", "Vestido Midi", "Mochila", "Óculos de Sol", "Relógio Analógico"], 45, 900),
}

CIDADES = [
    ("São Paulo", "SP", 18), ("Campinas", "SP", 6), ("Santos", "SP", 3),
    ("Rio de Janeiro", "RJ", 10), ("Niterói", "RJ", 3),
    ("Belo Horizonte", "MG", 7), ("Uberlândia", "MG", 3),
    ("Curitiba", "PR", 5), ("Londrina", "PR", 2),
    ("Porto Alegre", "RS", 5), ("Florianópolis", "SC", 3), ("Joinville", "SC", 2),
    ("Salvador", "BA", 4), ("Recife", "PE", 4), ("Fortaleza", "CE", 3),
    ("Brasília", "DF", 4), ("Goiânia", "GO", 3), ("Manaus", "AM", 2),
    ("Belém", "PA", 2), ("Vitória", "ES", 2),
]


def build_vendas(rng):
    conn, caminho = criar_banco("vendas")
    cur = conn.cursor()

    produto_id = 0
    produtos = []
    for cat_id, (cat_nome, (itens, pmin, pmax)) in enumerate(CATEGORIAS.items(), start=1):
        cur.execute("INSERT INTO categorias VALUES (?, ?)", (cat_id, cat_nome))
        for item in itens:
            produto_id += 1
            preco = round(rng.uniform(pmin, pmax), 2)
            custo = round(preco * rng.uniform(0.45, 0.75), 2)
            ativo = 0 if rng.random() < 0.12 else 1
            cur.execute("INSERT INTO produtos VALUES (?, ?, ?, ?, ?, ?)",
                        (produto_id, item, cat_id, preco, custo, ativo))
            produtos.append((produto_id, preco))

    cidades = [(c, uf) for c, uf, peso in CIDADES for _ in range(peso)]
    clientes = []
    for cid in range(1, 401):
        cidade, uf = rng.choice(cidades)
        segmento = rng.choices(["Varejo", "PME", "Corporativo"], [0.6, 0.3, 0.1])[0]
        cadastro = data_aleatoria(rng, date(2022, 1, 1), DATA_FIM)
        cur.execute("INSERT INTO clientes VALUES (?, ?, ?, ?, ?, ?)",
                    (cid, nome_pessoa(rng), cidade, uf, segmento, cadastro.isoformat()))
        clientes.append((cid, cadastro))

    # ~10% dos clientes nunca compram
    compradores = [c for c in clientes if rng.random() > 0.10]
    item_id = 0
    for pid in range(1, 4001):
        cid, cadastro = rng.choice(compradores)
        inicio = max(cadastro, DATA_INICIO)
        data_pedido = data_aleatoria(rng, inicio, DATA_FIM)
        recente = (DATA_FIM - data_pedido).days < 20
        if recente:
            status = rng.choices(["entregue", "em_transito", "pendente", "cancelado"], [0.3, 0.35, 0.25, 0.1])[0]
        else:
            status = rng.choices(["entregue", "cancelado"], [0.88, 0.12])[0]
        canal = rng.choices(["site", "app", "loja"], [0.5, 0.3, 0.2])[0]
        cur.execute("INSERT INTO pedidos VALUES (?, ?, ?, ?, ?)",
                    (pid, cid, data_pedido.isoformat(), status, canal))
        for prod_id, preco in rng.sample(produtos, rng.choices([1, 2, 3, 4], [0.45, 0.3, 0.17, 0.08])[0]):
            item_id += 1
            qtd = rng.choices([1, 2, 3, 4, 5], [0.55, 0.25, 0.1, 0.06, 0.04])[0]
            preco_unit = round(preco * rng.uniform(0.95, 1.05), 2)
            desconto = rng.choice([0, 0, 0, 0, 0.05, 0.1, 0.15])
            cur.execute("INSERT INTO itens_pedido VALUES (?, ?, ?, ?, ?, ?)",
                        (item_id, pid, prod_id, qtd, preco_unit, desconto))
    conn.commit()
    conn.close()
    return caminho


# -------------------------------------------------------------------- rh
DEPARTAMENTOS = ["Tecnologia", "Comercial", "Marketing", "Financeiro",
                 "Recursos Humanos", "Operações", "Jurídico", "Atendimento"]
CARGOS = [
    ("Assistente", "Júnior"), ("Analista Júnior", "Júnior"), ("Desenvolvedor Júnior", "Júnior"),
    ("Analista Pleno", "Pleno"), ("Desenvolvedor Pleno", "Pleno"),
    ("Analista Sênior", "Sênior"), ("Desenvolvedor Sênior", "Sênior"), ("Especialista", "Sênior"),
    ("Coordenador", "Gestor"), ("Gerente", "Gestor"),
]
FAIXAS = {"Júnior": (2800, 5500), "Pleno": (5500, 9500), "Sênior": (9000, 16500), "Gestor": (14000, 28000)}
CIDADES_RH = [("São Paulo", 5), ("Rio de Janeiro", 2), ("Belo Horizonte", 2),
              ("Curitiba", 1), ("Porto Alegre", 1), ("Recife", 1)]


def build_rh(rng):
    conn, caminho = criar_banco("rh")
    cur = conn.cursor()
    for i, nome in enumerate(DEPARTAMENTOS, start=1):
        cur.execute("INSERT INTO departamentos VALUES (?, ?)", (i, nome))
    for i, (titulo, nivel) in enumerate(CARGOS, start=1):
        cur.execute("INSERT INTO cargos VALUES (?, ?, ?)", (i, titulo, nivel))

    dev_cargos = [i for i, (t, _) in enumerate(CARGOS, start=1) if t.startswith("Desenvolvedor")]
    outros_cargos = [i for i, (t, _) in enumerate(CARGOS, start=1) if not t.startswith("Desenvolvedor")]
    cidades = [c for c, peso in CIDADES_RH for _ in range(peso)]
    pesos_depto = [0.24, 0.18, 0.1, 0.1, 0.06, 0.16, 0.05, 0.11]

    funcionarios = []
    for fid in range(1, 321):
        depto = rng.choices(range(1, 9), pesos_depto)[0]
        if depto == 1 and rng.random() < 0.7:
            cargo = rng.choice(dev_cargos)
        else:
            cargo = rng.choices(outros_cargos, [3, 4, 4, 3, 2, 2, 1])[0]
        nivel = CARGOS[cargo - 1][1]
        salario = round(rng.uniform(*FAIXAS[nivel]), 2)
        admissao = data_aleatoria(rng, date(2016, 1, 1), DATA_FIM)
        desligamento = None
        if rng.random() < 0.22:
            d = data_aleatoria(rng, max(admissao + timedelta(days=90), date(2020, 1, 1)), DATA_FIM)
            if d > admissao:
                desligamento = d
        modelo = rng.choices(["presencial", "híbrido", "remoto"], [0.3, 0.5, 0.2])[0]
        cur.execute("INSERT INTO funcionarios VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (fid, nome_pessoa(rng), depto, cargo, salario, admissao.isoformat(),
                     desligamento.isoformat() if desligamento else None, rng.choice(cidades), modelo))
        funcionarios.append((fid, admissao, desligamento))

    aid = 0
    for ano in (2023, 2024, 2025):
        for fid, admissao, desligamento in funcionarios:
            elegivel = admissao <= date(ano, 9, 30) and (desligamento is None or desligamento > date(ano, 12, 31))
            if elegivel and rng.random() < 0.93:
                aid += 1
                nota = round(min(5.0, max(1.0, rng.gauss(3.6, 0.65))), 1)
                cur.execute("INSERT INTO avaliacoes VALUES (?, ?, ?, ?)", (aid, fid, ano, nota))
    conn.commit()
    conn.close()
    return caminho


# ------------------------------------------------------------ financeiro
CENTROS = ["Administrativo", "Comercial", "Marketing", "Tecnologia", "Operações", "Logística"]
CAT_FORNECEDOR = {
    "Tecnologia": ["Licença de software", "Serviço de nuvem", "Suporte de TI", "Compra de equipamentos"],
    "Logística": ["Frete", "Armazenagem", "Transporte de cargas"],
    "Marketing": ["Campanha digital", "Evento corporativo", "Produção de conteúdo"],
    "Serviços": ["Consultoria", "Auditoria", "Serviço de limpeza", "Assessoria jurídica"],
    "Infraestrutura": ["Aluguel", "Manutenção predial", "Energia elétrica", "Internet e telefonia"],
}
CENTRO_POR_CATEGORIA = {
    "Tecnologia": ["Tecnologia", "Administrativo"],
    "Logística": ["Logística", "Operações"],
    "Marketing": ["Marketing", "Comercial"],
    "Serviços": ["Administrativo", "Comercial", "Operações"],
    "Infraestrutura": ["Administrativo", "Operações", "Logística"],
}
PREFIXOS = ["Alfa", "Nova", "Prime", "Vetor", "Atlas", "Delta", "Sigma", "Orion", "Lumen", "Horizonte",
            "Ponto", "Rede", "Fortis", "Ágil", "Vértice", "Nexo", "Aurora", "Pilar", "Rota", "Zênite"]
SUFIXOS = {"Tecnologia": "Tech", "Logística": "Log", "Marketing": "Mídia",
           "Serviços": "Consultoria", "Infraestrutura": "Facilities"}
ESTADOS_FORN = ["SP", "SP", "SP", "RJ", "MG", "PR", "RS", "SC", "BA", "PE", "DF"]
RECEITAS = ["Venda de serviços", "Contrato de assinatura", "Recebimento de cliente",
            "Projeto de implantação", "Licenciamento"]


def build_financeiro(rng):
    conn, caminho = criar_banco("financeiro")
    cur = conn.cursor()
    for i, nome in enumerate(CENTROS, start=1):
        cur.execute("INSERT INTO centros_custo VALUES (?, ?)", (i, nome))
    contas = [(1, "Itaú Corrente", "Itaú", "corrente"), (2, "Bradesco Corrente", "Bradesco", "corrente"),
              (3, "Banco do Brasil Corrente", "Banco do Brasil", "corrente"),
              (4, "BTG Investimentos", "BTG", "investimento")]
    cur.executemany("INSERT INTO contas_bancarias VALUES (?, ?, ?, ?)", contas)

    fornecedores = []
    nomes_usados = set()
    for fid in range(1, 41):
        categoria = rng.choice(list(CAT_FORNECEDOR))
        while True:
            nome = f"{rng.choice(PREFIXOS)} {SUFIXOS[categoria]} Ltda"
            if nome not in nomes_usados:
                nomes_usados.add(nome)
                break
        cur.execute("INSERT INTO fornecedores VALUES (?, ?, ?, ?)",
                    (fid, nome, categoria, rng.choice(ESTADOS_FORN)))
        fornecedores.append((fid, categoria))
    # 5 fornecedores cadastrados que nunca tiveram lançamento
    ativos = fornecedores[:-5]
    centro_id = {nome: i for i, nome in enumerate(CENTROS, start=1)}

    lid = 0
    ano, mes = DATA_INICIO.year, DATA_INICIO.month
    while (ano, mes) <= (DATA_FIM.year, DATA_FIM.month):
        inicio_mes = date(ano, mes, 1)
        fim_mes = (date(ano + mes // 12, mes % 12 + 1, 1) - timedelta(days=1))
        meses_ate_fim = (DATA_FIM.year - ano) * 12 + DATA_FIM.month - mes

        def status():
            if meses_ate_fim >= 2:
                return rng.choices(["pago", "atrasado"], [0.93, 0.07])[0]
            return rng.choices(["pago", "pendente", "atrasado"], [0.5, 0.4, 0.1])[0]

        for _ in range(rng.randint(18, 28)):
            lid += 1
            conta = rng.choices([1, 2, 3, 4], [0.45, 0.3, 0.15, 0.1])[0]
            cur.execute("INSERT INTO lancamentos VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (lid, data_aleatoria(rng, inicio_mes, fim_mes).isoformat(), "receita",
                         round(rng.uniform(5000, 80000), 2), status(), rng.choice(RECEITAS),
                         rng.choice([2, 2, 2, 4, 5]), None, conta))
        for centro in CENTROS:
            for _ in range(rng.randint(5, 10)):
                candidatos = [f for f in ativos if centro in CENTRO_POR_CATEGORIA[f[1]]] or ativos
                fid, categoria = rng.choice(candidatos)
                lid += 1
                cur.execute("INSERT INTO lancamentos VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (lid, data_aleatoria(rng, inicio_mes, fim_mes).isoformat(), "despesa",
                             round(rng.uniform(400, 35000), 2), status(),
                             rng.choice(CAT_FORNECEDOR[categoria]), centro_id[centro], fid,
                             rng.choices([1, 2, 3], [0.5, 0.35, 0.15])[0]))
        mes += 1
        if mes == 13:
            ano, mes = ano + 1, 1
    conn.commit()
    conn.close()
    return caminho


def main():
    for nome, builder in [("vendas", build_vendas), ("rh", build_rh), ("financeiro", build_financeiro)]:
        caminho = builder(random.Random(f"cp3-{nome}"))
        conn = sqlite3.connect(caminho)
        tabelas = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        resumo = ", ".join(f"{t}={conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]}" for t in tabelas)
        conn.close()
        print(f"[ok] {caminho.name}: {resumo}")


if __name__ == "__main__":
    main()
