"""
Avalia o modelo (base ou base + adaptador LoRA) no conjunto de teste.

    # modelo base, sem treino
    python src/evaluate.py --name base
    # modelo treinado
    python src/evaluate.py --name exp1_baseline --adapter outputs/exp1_baseline/adapter
    # testando parâmetros de geração
    python src/evaluate.py --name exp1_temp07 --adapter outputs/exp1_baseline/adapter --temperature 0.7 --top_p 0.9

Métricas:
  - execution accuracy: o SQL gerado retorna o mesmo resultado que o SQL de referência
  - SQL válido: o SQL gerado executa sem erro
  - exact match: texto igual ao de referência (após normalização)
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from model_utils import DEFAULT_BASE_MODEL, generate_sql, load_for_inference
from sql_utils import normalize_sql, results_match, run_query

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="nome do experimento (usado nos arquivos de saída)")
    ap.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--adapter", default=None, help="pasta do adaptador LoRA (omitir = modelo base)")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="avaliar só os N primeiros (teste rápido)")
    args = ap.parse_args()

    exemplos = [json.loads(l) for l in open(ROOT / "data" / "dataset" / f"{args.split}.jsonl", encoding="utf-8")]
    if args.limit:
        exemplos = exemplos[: args.limit]

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    model, tok = load_for_inference(args.base_model, args.adapter, args.load_in_4bit)
    print(f"== Avaliando '{args.name}' em {len(exemplos)} exemplos ({torch.cuda.get_device_name(0)})")

    resultados = []
    for i, ex in enumerate(exemplos, 1):
        g = generate_sql(model, tok, ex["db"], ex["question"], args.temperature, args.top_p, args.max_new_tokens)
        _, linhas_gold = run_query(ex["db"], ex["sql"])
        valido, acerto, erro = False, False, None
        try:
            _, linhas_pred = run_query(ex["db"], g["sql"])
            valido = True
            acerto = results_match(linhas_gold, linhas_pred)
        except Exception as e:  # noqa: BLE001
            erro = str(e)[:200]
        resultados.append({
            "id": ex["id"], "db": ex["db"], "template": ex["template"], "question": ex["question"],
            "gold_sql": ex["sql"], "pred_sql": g["sql"], "raw_output": g["raw"],
            "valid_sql": valido, "exec_match": acerto,
            "exact_match": normalize_sql(g["sql"]) == normalize_sql(ex["sql"]),
            "error": erro, "latency_s": round(g["latency_s"], 3), "tokens_per_s": round(g["tokens_per_s"], 1),
        })
        marca = "✓" if acerto else ("✗" if valido else "!")
        print(f"[{i:3d}/{len(exemplos)}] {marca} {ex['db']:10s} {ex['question'][:70]}")

    def taxa(lst, chave):
        return round(100 * sum(r[chave] for r in lst) / len(lst), 1) if lst else 0.0

    por_db = defaultdict(list)
    for r in resultados:
        por_db[r["db"]].append(r)
    resumo = {
        "name": args.name,
        "adapter": args.adapter,
        "base_model": args.base_model,
        "load_in_4bit": args.load_in_4bit,
        "generation": {"temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_new_tokens},
        "split": args.split,
        "n": len(resultados),
        "execution_accuracy": taxa(resultados, "exec_match"),
        "valid_sql": taxa(resultados, "valid_sql"),
        "exact_match": taxa(resultados, "exact_match"),
        "avg_latency_s": round(statistics.mean(r["latency_s"] for r in resultados), 3),
        "avg_tokens_per_s": round(statistics.mean(r["tokens_per_s"] for r in resultados), 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "gpu": torch.cuda.get_device_name(0),
        "by_db": {db: {"n": len(lst), "execution_accuracy": taxa(lst, "exec_match"),
                       "valid_sql": taxa(lst, "valid_sql")} for db, lst in por_db.items()},
    }

    saida = ROOT / "results"
    saida.mkdir(exist_ok=True)
    (saida / f"eval_{args.name}.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(saida / f"preds_{args.name}.jsonl", "w", encoding="utf-8") as f:
        for r in resultados:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n== {args.name}: execution accuracy {resumo['execution_accuracy']}% | "
          f"SQL válido {resumo['valid_sql']}% | exact match {resumo['exact_match']}% | "
          f"latência média {resumo['avg_latency_s']}s")
    for db, m in resumo["by_db"].items():
        print(f"   {db:10s} exec acc {m['execution_accuracy']}% (n={m['n']})")


if __name__ == "__main__":
    main()
