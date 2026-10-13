"""
Junta os resultados de treino e avaliação em results/resumo.md (tabela para o README).

    python src/compare_results.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def carregar(prefixo):
    return {p.stem[len(prefixo):]: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(RES.glob(f"{prefixo}*.json"))}


def main():
    treinos = carregar("train_")
    avals = carregar("eval_")
    if not avals:
        raise SystemExit("Nenhum results/eval_*.json encontrado. Rode src/evaluate.py primeiro.")

    linhas = [
        "## Treinos (parâmetros de fine-tuning)\n",
        "| Experimento | LoRA r | Learning rate | Épocas | 4-bit | Loss treino | Loss val | Tempo (min) | Pico VRAM (GB) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for nome, t in treinos.items():
        c = t["config"]
        linhas.append(
            f"| {nome} | {c['lora_r']} | {c['learning_rate']} | {c['num_epochs']} | "
            f"{'sim' if c['load_in_4bit'] else 'não'} | {t['train_loss']:.4f} | {t['eval_loss']:.4f} | "
            f"{t['train_time_s'] / 60:.1f} | {t['peak_vram_gb']} |")

    linhas += [
        "\n## Avaliação no conjunto de teste (formulações inéditas)\n",
        "| Modelo | Temperature | top_p | Execution acc. (%) | SQL válido (%) | Exact match (%) | vendas | rh | financeiro | Latência (s) | Tokens/s |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for nome, a in sorted(avals.items(), key=lambda kv: -kv[1]["execution_accuracy"]):
        g = a["generation"]
        db = a["by_db"]
        linhas.append(
            f"| {nome} | {g['temperature']} | {g['top_p']} | **{a['execution_accuracy']}** | {a['valid_sql']} | "
            f"{a['exact_match']} | {db.get('vendas', {}).get('execution_accuracy', '-')} | "
            f"{db.get('rh', {}).get('execution_accuracy', '-')} | "
            f"{db.get('financeiro', {}).get('execution_accuracy', '-')} | {a['avg_latency_s']} | {a['avg_tokens_per_s']} |")

    texto = "\n".join(linhas) + "\n"
    (RES / "resumo.md").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"[ok] tabela salva em {RES / 'resumo.md'}")


if __name__ == "__main__":
    main()
