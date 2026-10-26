"""PyTorch/PEFT model helpers for the isolated TRL environment."""

import gc
import hashlib
import os
from pathlib import Path
import time

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from ml_training.common import ROOT, read_json


def original_path():
    record = read_json(ROOT / "configs/models.json")["language_models"]["student"]
    cache = Path(os.environ.get("ML_DEMO_MODEL_CACHE", str(Path.home() / ".cache/huggingface/hub")))
    return cache / ("models--" + record["model_id"].replace("/", "--")) / "snapshots" / record["revision"]


def load(source, device="cpu", adapter=None, trainable=False, seed=41):
    set_seed(seed)
    model, details = AutoModelForCausalLM.from_pretrained(source, local_files_only=True, dtype=torch.float32,
                                                        attn_implementation="eager", output_loading_info=True)
    if details["missing_keys"] or details["unexpected_keys"] or details["mismatched_keys"]:
        raise AssertionError(f"Checkpoint conversion did not load exactly: {details}")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    elif trainable:
        layer_count = model.config.num_hidden_layers
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0., target_modules=["q_proj", "v_proj"],
                                               layers_to_transform=list(range(layer_count-4, layer_count)), task_type="CAUSAL_LM"))
    model.to(device)
    return model, tokenizer


def adapter_hash(model):
    value = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            value.update(name.encode())
            value.update(parameter.detach().float().cpu().numpy().tobytes())
    return value.hexdigest()


def prompt(tokenizer, row):
    return tokenizer.apply_chat_template([{"role": "system", "content": row["system"]}, {"role": "user", "content": row["user"]}],
                                          tokenize=False, add_generation_prompt=True)


def generate(model, tokenizer, rows, n=1, temperature=0., seed=41):
    set_seed(seed)
    model.eval()
    outputs = []
    for offset in range(0, len(rows), 4):
        part = rows[offset:offset+4]
        inputs = tokenizer([prompt(tokenizer, row) for row in part], padding=True, return_tensors="pt").to(model.device)
        started = time.perf_counter()
        kwargs = {"do_sample": temperature > 0, "num_return_sequences": n, "max_new_tokens": 160,
                  "pad_token_id": tokenizer.pad_token_id}
        if temperature:
            kwargs.update(temperature=temperature, top_p=1., top_k=0)
        with torch.inference_mode():
            sequences = model.generate(**inputs, **kwargs)[:, inputs["input_ids"].shape[1]:].cpu()
        elapsed = time.perf_counter() - started
        for i in range(len(part)):
            group = []
            for sequence in sequences[i*n:(i+1)*n]:
                tokens = sequence.tolist()
                if tokenizer.eos_token_id in tokens:
                    tokens = tokens[:tokens.index(tokenizer.eos_token_id)+1]
                group.append({"text": tokenizer.decode(tokens, skip_special_tokens=True), "generation_tokens": len(tokens),
                              "batch_elapsed_seconds": elapsed, "batch_size": len(part)*n, "temperature": temperature,
                              "max_tokens": 160, "prompt_tokens": int(inputs["attention_mask"][i].sum())})
            outputs.append(group)
    return outputs


def release():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
