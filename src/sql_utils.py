"""Execução segura de SQL nos bancos de negócio e comparação de resultados."""
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "data" / "db"
SCHEMA_DIR = ROOT / "data" / "schemas"
BANCOS = ["vendas", "rh", "financeiro"]

_PROIBIDOS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex)\b",
    re.IGNORECASE,
)


def schema_text(db: str) -> str:
    """Esquema (DDL + regras de negócio comentadas) usado no prompt."""
    return (SCHEMA_DIR / f"{db}.sql").read_text(encoding="utf-8").strip()


def db_path(db: str) -> Path:
    return DB_DIR / f"{db}.sqlite"


def is_safe_select(sql: str) -> bool:
    """Aceita apenas uma única consulta de leitura (SELECT/WITH)."""
    s = sql.strip().rstrip(";").strip()
    if not s or ";" in s:
        return False
    if not re.match(r"^(select|with)\b", s, re.IGNORECASE):
        return False
    return not _PROIBIDOS.search(s)


def run_query(db: str, sql: str, timeout_s: float = 5.0, max_rows: int = 5000):
    """Executa SQL em modo somente leitura. Retorna (colunas, linhas).

    Lança ValueError se a consulta não for um SELECT e sqlite3.Error em erro de execução.
    Consultas que passam do timeout são interrompidas (evita joins cartesianos travando).
    """
    if not is_safe_select(sql):
        raise ValueError("Apenas uma consulta SELECT é permitida.")
    uri = db_path(db).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    inicio = time.monotonic()
    conn.set_progress_handler(lambda: 1 if time.monotonic() - inicio > timeout_s else 0, 10_000)
    try:
        cur = conn.execute(sql.strip().rstrip(";"))
        colunas = [d[0] for d in cur.description] if cur.description else []
        linhas = cur.fetchmany(max_rows)
        return colunas, linhas
    finally:
        conn.close()


def _norm_valor(v):
    if isinstance(v, float):
        return round(v, 2)
    if isinstance(v, int):
        return round(float(v), 2)
    if isinstance(v, str):
        try:
            return round(float(v), 2)
        except ValueError:
            return v.strip()
    return v


def _norm_linha(linha):
    # ordem das colunas é ignorada: compara o conjunto de valores da linha
    return tuple(sorted((_norm_valor(v) for v in linha), key=lambda x: (str(type(x)), str(x))))


def results_match(linhas_gold, linhas_pred) -> bool:
    """Execution accuracy: mesmo multiconjunto de linhas (floats com 2 casas,
    ordem de linhas e de colunas ignorada, nomes/aliases de colunas ignorados)."""
    return Counter(map(_norm_linha, linhas_gold)) == Counter(map(_norm_linha, linhas_pred))


def normalize_sql(sql: str) -> str:
    """Normalização para exact match: minúsculas, espaços colapsados, sem ';'."""
    s = sql.strip().rstrip(";").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*([(),=<>])\s*", r"\1", s)
    return s
