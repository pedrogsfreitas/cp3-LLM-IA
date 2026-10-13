"""
Fine-tuning LoRA/QLoRA do modelo para text-to-SQL de negócio.

    python src/train.py --config configs/exp1_baseline.yaml

Salva o adaptador em outputs/<nome>/adapter e as métricas de treino em
results/train_<nome>.json.
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

from model_utils import DEFAULT_BASE_MODEL, load_base_model, load_tokenizer, require_gpu
from prompt import build_messages

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "dataset"

DEFAULTS = {
    "base_model": DEFAULT_BASE_MODEL,
    "load_in_4bit": False,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "learning_rate": 2e-4,
    "num_epochs": 2,
    "batch_size": 4,
    "grad_accum": 4,
    "max_length": 1024,
    "warmup_ratio": 0.05,
    "seed": 42,
}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(linha) for linha in f]


def tokenize_example(ex, tok, max_length):
    """Tokeniza prompt + resposta e mascara o prompt nos labels (-100):
    a loss é calculada só sobre o SQL que o modelo deve gerar."""
    prompt = tok.apply_chat_template(build_messages(ex["db"], ex["question"]),
                                     tokenize=False, add_generation_prompt=True)
    completo = tok.apply_chat_template(build_messages(ex["db"], ex["question"], ex["sql"]), tokenize=False)
    if not completo.startswith(prompt):
        raise ValueError("Template de chat inesperado: o prompt não é prefixo da conversa completa.")
    ids_prompt = tok(prompt, add_special_tokens=False)["input_ids"]
    ids_resposta = tok(completo[len(prompt):], add_special_tokens=False)["input_ids"]
    input_ids = (ids_prompt + ids_resposta)[:max_length]
    labels = ([-100] * len(ids_prompt) + ids_resposta)[:max_length]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = lambda seq, v: seq + [v] * (n - len(seq))  # noqa: E731
        return {
            "input_ids": torch.tensor([pad(b["input_ids"], self.pad_id) for b in batch]),
            "attention_mask": torch.tensor([pad(b["attention_mask"], 0) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
        }


class GpuStats(TrainerCallback):
    """Registra uso de VRAM durante o treino (evidência de que rodou na GPU)."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and torch.cuda.is_available():
            logs["vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = {**DEFAULTS, **(yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {})}
    nome = cfg.get("name", cfg_path.stem)
    out_dir = ROOT / "outputs" / nome
    require_gpu()
    set_seed(cfg["seed"])
    torch.cuda.reset_peak_memory_stats()
    print(f"== Experimento {nome} na GPU {torch.cuda.get_device_name(0)}")
    print(json.dumps(cfg, indent=2, ensure_ascii=False))

    tok = load_tokenizer(cfg["base_model"])
    tok.padding_side = "right"
    treino = read_jsonl(DATA / "train.jsonl")
    val = read_jsonl(DATA / "val.jsonl")
    ds_treino = Dataset.from_list([tokenize_example(e, tok, cfg["max_length"]) for e in treino])
    ds_val = Dataset.from_list([tokenize_example(e, tok, cfg["max_length"]) for e in val])
    maior = max(len(x) for x in ds_treino["input_ids"])
    print(f"Treino: {len(ds_treino)} exemplos | Validação: {len(ds_val)} | maior sequência: {maior} tokens")

    model = load_base_model(cfg["base_model"], cfg["load_in_4bit"])
    model.config.use_cache = False
    if cfg["load_in_4bit"]:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"], task_type="CAUSAL_LM"))
    # adaptadores em fp32 + autocast fp16 (evita erro de "unscale FP16 gradients")
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    passos = math.ceil(len(ds_treino) / (cfg["batch_size"] * cfg["grad_accum"])) * cfg["num_epochs"]
    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=float(cfg["learning_rate"]),
        num_train_epochs=cfg["num_epochs"],
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(passos * cfg["warmup_ratio"])),
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        seed=cfg["seed"],
        dataloader_pin_memory=True,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds_treino, eval_dataset=ds_val,
                      data_collator=Collator(tok.pad_token_id), callbacks=[GpuStats()])

    inicio = time.time()
    resultado = trainer.train()
    duracao = time.time() - inicio
    final_eval = trainer.evaluate()

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    metricas = {
        "name": nome,
        "config": cfg,
        "gpu": torch.cuda.get_device_name(0),
        "train_loss": resultado.training_loss,
        "eval_loss": final_eval.get("eval_loss"),
        "train_time_s": round(duracao, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "steps": resultado.global_step,
        "log_history": trainer.state.log_history,
        "adapter_dir": str(adapter_dir.relative_to(ROOT)),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"train_{nome}.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[ok] {nome}: loss treino={metricas['train_loss']:.4f} | loss val={metricas['eval_loss']:.4f} | "
          f"{duracao / 60:.1f} min | pico VRAM {metricas['peak_vram_gb']} GB")
    print(f"Adaptador salvo em {adapter_dir}")


if __name__ == "__main__":
    main()
