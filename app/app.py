"""
Frontend: o usuário escolhe o banco de negócio, faz a pergunta em português,
ajusta os parâmetros de geração e recebe o SQL gerado + o resultado da consulta.

    python app/app.py        ->  abre em http://127.0.0.1:7860
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from model_utils import DEFAULT_BASE_MODEL, generate_sql, load_base_model, load_tokenizer, require_gpu  # noqa: E402
from sql_utils import BANCOS, run_query, schema_text  # noqa: E402

BASE = "Modelo base (sem fine-tuning)"
EXEMPLOS = {
    "vendas": ["Qual foi o faturamento total em 2025?",
               "Quais são os 5 produtos mais vendidos em 2026?",
               "Qual o ticket médio por canal de venda?",
               "Quantos clientes nunca fizeram um pedido?"],
    "rh": ["Qual o headcount por departamento?",
           "Qual o salário médio por nível de cargo?",
           "Quantos funcionários foram desligados em 2025?",
           "Em 2024, quantas pessoas ativas ficaram sem avaliação de desempenho?"],
    "financeiro": ["Qual o total de despesas por centro de custo em 2025?",
                   "Qual o valor total que ainda falta pagar aos fornecedores?",
                   "Como ficou o saldo mensal da empresa ao longo de 2025?",
                   "Quais os 5 fornecedores que mais receberam pagamentos?"],
}

# ---------------------------------------------------------------- modelo
require_gpu()
print("Carregando modelo base na GPU...")
TOK = load_tokenizer(DEFAULT_BASE_MODEL)
MODELO = load_base_model(DEFAULT_BASE_MODEL)
PREFERIDO = "exp5_3epocas"  # melhor relação acerto x latência nos testes (ver README)
ADAPTADORES = {p.parent.name: p for p in sorted((ROOT / "outputs").glob("*/adapter")) if (p / "adapter_config.json").exists()}
if PREFERIDO in ADAPTADORES:  # coloca o modelo preferido como padrão no dropdown
    ADAPTADORES = {PREFERIDO: ADAPTADORES[PREFERIDO], **{k: v for k, v in ADAPTADORES.items() if k != PREFERIDO}}
if ADAPTADORES:
    from peft import PeftModel
    nomes = list(ADAPTADORES)
    MODELO = PeftModel.from_pretrained(MODELO, str(ADAPTADORES[nomes[0]]), adapter_name=nomes[0])
    for nome in nomes[1:]:
        MODELO.load_adapter(str(ADAPTADORES[nome]), adapter_name=nome)
    print(f"Adaptadores LoRA carregados: {', '.join(nomes)}")
else:
    print("Nenhum adaptador em outputs/*/adapter — só o modelo base estará disponível.")
MODELO.eval()
OPCOES_MODELO = [f"Fine-tuned: {n}" for n in ADAPTADORES] + [BASE]


def gerar(modelo_escolhido, db, pergunta, temperature, top_p, max_new_tokens):
    if modelo_escolhido == BASE or not ADAPTADORES:
        ctx = MODELO.disable_adapter() if ADAPTADORES else _nulo()
    else:
        MODELO.set_adapter(modelo_escolhido.removeprefix("Fine-tuned: "))
        ctx = _nulo()
    with ctx:
        return generate_sql(MODELO, TOK, db, pergunta, temperature, top_p, int(max_new_tokens))


class _nulo:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def executar(db, sql):
    try:
        colunas, linhas = run_query(db, sql, max_rows=200)
        return pd.DataFrame(linhas, columns=colunas), f"{len(linhas)} linha(s)"
    except Exception as e:  # noqa: BLE001
        return pd.DataFrame(), f"⚠️ Erro ao executar: {e}"


def info(g, status):
    vram = torch.cuda.memory_allocated() / 1024**3
    return (f"**{status}** · {g['latency_s']:.2f}s · {g['new_tokens']} tokens · "
            f"{g['tokens_per_s']:.0f} tokens/s · VRAM em uso {vram:.2f} GB ({torch.cuda.get_device_name(0)})")


def responder(modelo_escolhido, db, pergunta, temperature, top_p, max_new_tokens):
    if not pergunta.strip():
        return "", pd.DataFrame(), "Digite uma pergunta."
    g = gerar(modelo_escolhido, db, pergunta, temperature, top_p, max_new_tokens)
    df, status = executar(db, g["sql"])
    return g["sql"], df, info(g, status)


def comparar(modelo_escolhido, db, pergunta, temperature, top_p, max_new_tokens):
    a = responder(BASE, db, pergunta, temperature, top_p, max_new_tokens)
    b = responder(modelo_escolhido, db, pergunta, temperature, top_p, max_new_tokens)
    return (*a, *b)


def trocar_banco(db):
    return schema_text(db), gr.update(choices=EXEMPLOS[db], value=EXEMPLOS[db][0]), EXEMPLOS[db][0]


with gr.Blocks(title="Text-to-SQL de Negócios") as demo:
    gr.Markdown("# Text-to-SQL de Negócios\nPergunte em português sobre os dados da empresa. "
                "O modelo lê o esquema do banco escolhido e gera a consulta SQL, que é executada em modo somente leitura.")
    with gr.Row():
        with gr.Column(scale=1):
            db = gr.Dropdown(BANCOS, value="vendas", label="Banco de dados")
            modelo = gr.Dropdown(OPCOES_MODELO, value=OPCOES_MODELO[0], label="Modelo")
            with gr.Accordion("Parâmetros de geração", open=True):
                temperature = gr.Slider(0.0, 1.5, value=0.0, step=0.05, label="temperature (0 = determinístico)")
                top_p = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="top_p")
                max_new_tokens = gr.Slider(32, 512, value=256, step=16, label="max_new_tokens")
            with gr.Accordion("Esquema do banco", open=False):
                esquema = gr.Code(schema_text("vendas"), language="sql", interactive=False)
        with gr.Column(scale=2):
            exemplos = gr.Dropdown(EXEMPLOS["vendas"], value=EXEMPLOS["vendas"][0],
                                   label="Exemplos de perguntas", allow_custom_value=True)
            pergunta = gr.Textbox(EXEMPLOS["vendas"][0], label="Sua pergunta", lines=2)
            with gr.Tab("Perguntar"):
                botao = gr.Button("Gerar SQL", variant="primary")
                sql_out = gr.Code(language="sql", label="SQL gerado")
                info_out = gr.Markdown()
                tabela = gr.Dataframe(label="Resultado")
            with gr.Tab("Comparar base × fine-tuned"):
                botao_cmp = gr.Button("Comparar", variant="primary")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Modelo base")
                        sql_a = gr.Code(language="sql")
                        info_a = gr.Markdown()
                        tab_a = gr.Dataframe()
                    with gr.Column():
                        gr.Markdown("### Modelo selecionado")
                        sql_b = gr.Code(language="sql")
                        info_b = gr.Markdown()
                        tab_b = gr.Dataframe()

    entradas = [modelo, db, pergunta, temperature, top_p, max_new_tokens]
    db.change(trocar_banco, db, [esquema, exemplos, pergunta])
    exemplos.change(lambda q: q, exemplos, pergunta)
    botao.click(responder, entradas, [sql_out, tabela, info_out])
    pergunta.submit(responder, entradas, [sql_out, tabela, info_out])
    botao_cmp.click(comparar, entradas, [sql_a, tab_a, info_a, sql_b, tab_b, info_b])

if __name__ == "__main__":
    demo.launch()
