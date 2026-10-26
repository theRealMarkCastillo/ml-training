"""Shared SFT adapter training with validation selection and real update evidence."""

import random
import time

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from ml_training.common import write_json
from ml_training.mlx_models import (LORA_CONFIG, LORA_LAYERS, adapter_hash, infer_rows, load_model,
                                    make_lora, only_adapter_trainable, save_adapter, training_batch, update_step)


def train_adapter(run, training, validation, score_fn, profile, seed, max_tokens, batch_size=8):
    mx.random.seed(seed)
    model, tokenizer = load_model()
    make_lora(model)
    initial_hash = adapter_hash(model)
    optimizer = optim.Adam(learning_rate=1e-4)
    epochs = 1 if profile == "smoke" else 2
    losses, history = [], []
    steps, tokens = 0, 0
    best, selected_epoch, selected_step = -1., None, None
    start = time.perf_counter()
    for epoch in range(epochs):
        order = list(training)
        random.Random(seed + epoch).shuffle(order)
        for offset in range(0, len(order), batch_size):
            batch = training_batch(tokenizer, order[offset:offset + batch_size])
            loss, count = update_step(model, optimizer, batch)
            losses.append(loss)
            tokens += count
            steps += 1
            if steps % 25 == 0:
                print(f"SFT epoch={epoch+1} step={steps} completion_loss={loss:.4f}", flush=True)
        outputs = infer_rows(model, tokenizer, validation, max_tokens=max_tokens)
        metrics = score_fn(validation, outputs)
        history.append({"epoch": epoch + 1, "optimizer_steps": steps, "metrics": metrics})
        if metrics["joint_accuracy"] > best:
            best, selected_epoch, selected_step = metrics["joint_accuracy"], epoch + 1, steps
            save_adapter(model, run / "adapter")
        print(f"SFT validation epoch={epoch+1} joint_accuracy={metrics['joint_accuracy']:.4f}", flush=True)
    selected, selected_tokenizer = load_model(adapter=run / "adapter")
    only_adapter_trainable(selected)
    selected_hash = adapter_hash(selected)
    if initial_hash == selected_hash:
        raise AssertionError("Selected adapter has no parameter update")
    count = sum(p.size for _, p in tree_flatten(selected.trainable_parameters()))
    evidence = {"optimizer": "Adam", "learning_rate": 1e-4, "batch_size": batch_size,
                "max_sequence_length": 512, "completion_only_loss": True,
                "epochs": epochs, "optimizer_steps_executed": steps, "selected_step": selected_step,
                "selected_epoch": selected_epoch, "selected_validation_joint_accuracy": best,
                "selection_rule": "Highest validation joint accuracy; earlier trained epoch wins ties",
                "training_examples": len(training), "completion_tokens_processed": tokens,
                "trainable_parameters": count, "lora_layers": LORA_LAYERS, "lora_config": LORA_CONFIG,
                "initial_adapter_sha256": initial_hash, "selected_adapter_sha256": selected_hash,
                "losses": losses, "validation_history": history,
                "elapsed_seconds": time.perf_counter() - start}
    write_json(run / "training.json", evidence)
    write_json(run / "validation_tuned.json", {"metrics": score_fn(validation, infer_rows(selected, selected_tokenizer, validation, max_tokens=max_tokens))})
    return selected, selected_tokenizer, evidence
