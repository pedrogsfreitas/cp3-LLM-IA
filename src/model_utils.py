"""Carregamento do modelo base (fp16 ou 4-bit) e dos adaptadores LoRA."""
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from prompt import build_messages, extract_sql

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def require_gpu():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA não disponível. Este projeto deve rodar na GPU (rode check_gpu.py).")
    return torch.device("cuda:0")


def load_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(model_id: str, load_in_4bit: bool = False):
    """RTX 2060 (Turing, sm_75): usa fp16 (sem bf16) e atenção SDPA (sem FlashAttention 2)."""
    kwargs = dict(dtype=torch.float16, device_map={"": 0}, attn_implementation="sdpa")
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


def load_for_inference(model_id: str, adapter_dir: str | None = None, load_in_4bit: bool = False):
    require_gpu()
    tok = load_tokenizer(adapter_dir or model_id)
    model = load_base_model(model_id, load_in_4bit)
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tok


@torch.inference_mode()
def generate_sql(model, tok, db: str, question: str, temperature: float = 0.0,
                 top_p: float = 1.0, max_new_tokens: int = 256):
    """Gera o SQL. Retorna dict com sql, resposta bruta, latência e tokens/s."""
    texto = tok.apply_chat_template(build_messages(db, question), tokenize=False, add_generation_prompt=True)
    entrada = tok(texto, return_tensors="pt", add_special_tokens=False).to(model.device)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    torch.cuda.synchronize()
    inicio = time.perf_counter()
    saida = model.generate(**entrada, **gen_kwargs)
    torch.cuda.synchronize()
    latencia = time.perf_counter() - inicio
    novos = saida[0, entrada["input_ids"].shape[1]:]
    resposta = tok.decode(novos, skip_special_tokens=True)
    return {
        "sql": extract_sql(resposta),
        "raw": resposta,
        "latency_s": latencia,
        "new_tokens": int(novos.shape[0]),
        "tokens_per_s": novos.shape[0] / latencia if latencia > 0 else 0.0,
    }
