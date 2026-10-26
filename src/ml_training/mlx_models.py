"""Local MLX causal model loading, LoRA updates, and generation evidence."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import time
import gc

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load, stream_generate
from mlx_lm.generate import batch_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.trainer import default_loss
from mlx_lm.tuner.utils import linear_to_lora_layers
import numpy as np

from ml_training.common import ROOT, digest, environment, read_json, write_json

LORA_CONFIG = {"rank": 8, "scale": 16., "dropout": 0., "keys": ["self_attn.q_proj", "self_attn.v_proj"]}
LORA_LAYERS = 4


def record(kind="student"):
    return read_json(ROOT / "configs" / "models.json")["language_models"][kind]


def path(kind="student"):
    config = record(kind)
    cache = Path(os.environ.get("ML_DEMO_MODEL_CACHE", str(Path.home() / ".cache/huggingface/hub")))
    result = cache / ("models--" + config["model_id"].replace("/", "--")) / "snapshots" / config["revision"]
    if not (result / "config.json").is_file():
        raise FileNotFoundError(f"Missing pinned language model: {result}")
    return result


def load_model(kind="student", adapter=None):
    return load(str(path(kind)), adapter_path=str(adapter) if adapter else None)


def prompt(tokenizer, system, user):
    return tokenizer.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                          tokenize=False, add_generation_prompt=True)


def infer(model, tokenizer, system, user, max_tokens=192, temperature=0.):
    model.eval()
    text = prompt(tokenizer, system, user)
    start = time.perf_counter()
    pieces, last = [], None
    for response in stream_generate(model, tokenizer, prompt=text, max_tokens=max_tokens,
                                    sampler=make_sampler(temp=temperature)):
        pieces.append(response.text)
        last = response
    return {"text": "".join(pieces), "elapsed_seconds": time.perf_counter() - start,
            "prompt_tokens": last.prompt_tokens if last else 0,
            "generation_tokens": last.generation_tokens if last else 0,
            "peak_memory_gb": last.peak_memory if last else 0,
            "temperature": temperature, "max_tokens": max_tokens}


def make_lora(model):
    model.freeze()
    linear_to_lora_layers(model, LORA_LAYERS, LORA_CONFIG)
    mx.eval(model.parameters())


def only_adapter_trainable(model):
    model.freeze()
    for _, module in model.named_modules():
        if hasattr(module, "lora_a"):
            module.unfreeze(keys=["lora_a", "lora_b"])


def release():
    gc.collect()
    mx.clear_cache()


def infer_rows(model, tokenizer, rows, max_tokens=96, batch_size=8, progress=None):
    model.eval()
    outputs = []
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        prompts = [tokenizer.encode(prompt(tokenizer, row["system"], row["user"])) for row in batch]
        start = time.perf_counter()
        result = batch_generate(model, tokenizer, prompts, max_tokens=max_tokens,
                                sampler=make_sampler(temp=0.), return_token_ids=True,
                                completion_batch_size=batch_size, prefill_batch_size=min(4, batch_size))
        elapsed = time.perf_counter() - start
        outputs.extend({"text": text, "prompt_tokens": len(p), "generation_tokens": len(tokens),
                        "batch_elapsed_seconds": elapsed, "batch_size": len(batch), "temperature": 0.,
                        "max_tokens": max_tokens}
                       for text, p, tokens in zip(result.texts, prompts, result.token_ids, strict=True))
        if progress:
            progress(offset + len(batch), len(rows))
    return outputs


def adapter_hash(model):
    result = hashlib.sha256()
    for name, value in tree_flatten(model.trainable_parameters()):
        result.update(name.encode())
        result.update(str(value.dtype).encode())
        result.update(np.asarray(value.astype(mx.float32)).tobytes())
    return result.hexdigest()


def save_adapter(model, destination):
    destination.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(destination / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    write_json(destination / "adapter_config.json", {"fine_tune_type": "lora", "num_layers": LORA_LAYERS,
                                                    "lora_parameters": LORA_CONFIG})


def training_batch(tokenizer, examples):
    tokens, lengths = [], []
    for example in examples:
        prefix = tokenizer.encode(prompt(tokenizer, example["system"], example["user"]))
        answer = tokenizer.encode(example["answer"], add_special_tokens=False) + [tokenizer.eos_token_id]
        sequence = prefix + answer
        if len(sequence) > 512:
            raise ValueError("Training example exceeds fixed 512-token budget")
        tokens.append(sequence)
        lengths.append((len(prefix), len(sequence) - 1))
    max_length = max(len(row) for row in tokens)
    return mx.array([row + [tokenizer.pad_token_id or tokenizer.eos_token_id] * (max_length - len(row)) for row in tokens]), mx.array(lengths)


def update_step(model, optimizer, batch):
    model.train()
    loss_and_grad = nn.value_and_grad(model, default_loss)
    (loss, tokens), gradients = loss_and_grad(model, *batch)
    optimizer.update(model, gradients)
    mx.eval(model.parameters(), optimizer.state, loss)
    value = float(loss)
    if not np.isfinite(value):
        raise ValueError("Nonfinite language-model training loss")
    return value, int(tokens)


def probe():
    mx.random.seed(41)
    output = ROOT / "artifacts" / "probes" / "mlx-student"
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    model, tokenizer = load_model()
    baseline = infer(model, tokenizer, "Return only compact JSON.", "Return an object with ok set to true.", max_tokens=32)
    make_lora(model)
    before = adapter_hash(model)
    batch = training_batch(tokenizer, [{"system": "Return only compact JSON.",
                                        "user": "Return an object with ok set to true.", "answer": '{"ok":true}'}])
    optimizer = optim.Adam(learning_rate=1e-4)
    loss, tokens = update_step(model, optimizer, batch)
    after = adapter_hash(model)
    if before == after:
        raise AssertionError("MLX LoRA update did not change weights")
    adapter = output / "adapter"
    save_adapter(model, adapter)
    expected = infer(model, tokenizer, "Return only compact JSON.", "Return an object with ok set to true.", max_tokens=32)
    restored, restored_tokenizer = load_model(adapter=adapter)
    # load_adapters reconstructs modules; freeze base weights before hashing only adapter leaves.
    restored.freeze()
    for name, module in restored.named_modules():
        if hasattr(module, "lora_a"):
            module.unfreeze(keys=["lora_a", "lora_b"])
    restored_hash = adapter_hash(restored)
    actual = infer(restored, restored_tokenizer, "Return only compact JSON.", "Return an object with ok set to true.", max_tokens=32)
    if restored_hash != after or expected["text"] != actual["text"]:
        raise AssertionError("MLX adapter reload differs")
    write_json(output / "probe.json", {"purpose": "compatibility probe, not task evaluation", "environment": environment(),
                                        "model": record(), "input_weights_sha256": digest(path() / "model.safetensors"),
                                        "baseline": baseline, "after_update": expected, "reloaded": actual,
                                        "before_adapter_sha256": before, "after_adapter_sha256": after,
                                        "reloaded_adapter_sha256": restored_hash, "optimizer_steps": 1,
                                        "loss": loss, "completion_tokens_in_loss": tokens,
                                        "elapsed_seconds": time.perf_counter() - start})
    return output / "probe.json"
