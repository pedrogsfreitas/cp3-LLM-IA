"""Formato do prompt (igual no treino, na avaliação e no app)."""
import re

from sql_utils import schema_text

SYSTEM_PROMPT = (
    "Você é um especialista em SQL que traduz perguntas de negócio em consultas SQLite. "
    "Use somente as tabelas e colunas do esquema fornecido e siga as regras de negócio "
    "descritas nos comentários do esquema. Responda apenas com a consulta SQL, sem explicações."
)


def build_messages(db: str, question: str, sql: str | None = None) -> list[dict]:
    """Monta as mensagens de chat. O banco escolhido entra como parâmetro:
    o esquema dele (DDL + regras de negócio) vai junto com a pergunta."""
    user = f"Banco de dados: {db}\n\nEsquema:\n{schema_text(db)}\n\nPergunta: {question}"
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    if sql is not None:
        msgs.append({"role": "assistant", "content": sql})
    return msgs


def extract_sql(texto: str) -> str:
    """Extrai a consulta da resposta do modelo (remove ```sql, explicações, etc.)."""
    t = texto.strip()
    bloco = re.search(r"```(?:sql)?\s*(.*?)```", t, re.DOTALL | re.IGNORECASE)
    if bloco:
        t = bloco.group(1).strip()
    inicio = re.search(r"\b(with|select)\b", t, re.IGNORECASE)
    if inicio:
        t = t[inicio.start():]
    if ";" in t:
        t = t[: t.index(";") + 1]
    return t.strip()
