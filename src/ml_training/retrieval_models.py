"""Local transformer paths, device probes, and explicit gradient training steps."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import CrossEncoder, SentenceTransformer

from ml_training.common import ROOT, digest, environment, read_json, write_json


def configure(seed: int, requested: str = "auto") -> str:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = ("mps" if torch.backends.mps.is_available() else "cpu") if requested == "auto" else requested
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable in this process; check GPU access before falling back to CPU")
    return device


def model_record(kind):
    return read_json(ROOT / "configs" / "models.json")["retrieval"][kind]


def model_path(kind):
    item = model_record(kind)
    cache = Path(os.environ.get("ML_DEMO_MODEL_CACHE", str(Path.home() / ".cache/huggingface/hub")))
    path = cache / ("models--" + item["model_id"].replace("/", "--")) / "snapshots" / item["revision"]
    if not (path / "model.safetensors").is_file():
        raise FileNotFoundError(f"Missing pinned model snapshot: {path}. Download its exact revision first.")
    return path


def load_bi(device, path=None):
    model = SentenceTransformer(str(path or model_path("bi_encoder")), device=device, local_files_only=True)
    model.max_seq_length = 160
    return model


def load_cross(device, path=None):
    return CrossEncoder(str(path or model_path("cross_encoder")), device=device, max_length=192, local_files_only=True)


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()


def state_hash(model):
    hasher = hashlib.sha256()
    for name, parameter in model.named_parameters():
        hasher.update(name.encode())
        hasher.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def sample_parameter(model):
    name, parameter = next((n, p) for n, p in model.named_parameters() if p.requires_grad and p.numel() > 100)
    return name, parameter.detach().cpu().clone()


def parameter_delta(model, sample):
    name, before = sample
    after = dict(model.named_parameters())[name].detach().cpu()
    return {
        "parameter": name,
        "max_abs_change": float((after - before).abs().max()),
        "changed_elements": int(torch.count_nonzero(after != before)),
    }


def tensor_inputs(inputs, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def bi_loss(model, queries, positives, negatives, device):
    # Each query's own verified positive and two explicit hard negatives form
    # its contrastive group. Other queries' positives are not false negatives.
    q = F.normalize(model(tensor_inputs(model.preprocess(queries), device))["sentence_embedding"], dim=-1)
    documents = [d for p, ns in zip(positives, negatives) for d in [p, *ns]]
    d = F.normalize(model(tensor_inputs(model.preprocess(documents), device))["sentence_embedding"], dim=-1)
    d = d.reshape(len(queries), 3, -1)
    scores = torch.einsum("bd,bkd->bk", q, d) / 0.05
    return F.cross_entropy(scores, torch.zeros(len(queries), dtype=torch.long, device=device))


def cross_loss(model, pairs, labels, device):
    tokens = model.tokenizer(
        [p[0] for p in pairs], [p[1] for p in pairs], padding=True, truncation=True, max_length=192, return_tensors="pt"
    )
    logits = model.model(**tensor_inputs(tokens, device)).logits.reshape(-1)
    return F.binary_cross_entropy_with_logits(logits, torch.tensor(labels, dtype=torch.float32, device=device))


def embed(model, texts):
    return model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True)


def score_pairs(model, pairs):
    # Explicit raw logits ensure the base and tuned rerankers use the same scale.
    return np.asarray(
        model.predict(
            pairs, batch_size=32, show_progress_bar=False, activation_fn=torch.nn.Identity(), convert_to_numpy=True
        )
    ).reshape(-1)


def probe(requested="auto"):
    device = configure(41, requested)
    folder = ROOT / "artifacts" / "probes" / f"retrieval-{device}"
    folder.mkdir(parents=True, exist_ok=True)
    result = {
        "device": device,
        "environment": environment(),
        "models": {},
        "purpose": "compatibility probe, not evaluation",
    }
    for kind in ["bi_encoder", "cross_encoder"]:
        start = time.perf_counter()
        model = load_bi(device) if kind == "bi_encoder" else load_cross(device)
        parameters = model if kind == "bi_encoder" else model.model
        before = state_hash(parameters)
        sample = sample_parameter(parameters)
        parameters.train()
        optimizer = torch.optim.AdamW(parameters.parameters(), lr=2e-5)
        if kind == "bi_encoder":
            loss = bi_loss(
                model,
                ["Inspect the v2 ticket", "Modify the v1 invoice"],
                ["GET v2 ticket", "PATCH v1 invoice"],
                [["PATCH v2 ticket", "GET v1 ticket"], ["GET v1 invoice", "PATCH v2 invoice"]],
                device,
            )
        else:
            loss = cross_loss(model, [("inspect v2", "GET v2"), ("inspect v2", "PATCH v1")], [1.0, 0.0], device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        synchronize(device)
        after = state_hash(parameters)
        if before == after:
            raise AssertionError("Probe did not update parameters")
        destination = folder / kind
        model.save_pretrained(str(destination))
        loaded = load_bi(device, destination) if kind == "bi_encoder" else load_cross(device, destination)
        reloaded_hash = state_hash(loaded if kind == "bi_encoder" else loaded.model)
        if after != reloaded_hash:
            raise AssertionError("Reloaded weights differ")
        model.eval() if kind == "bi_encoder" else model.model.eval()
        if kind == "bi_encoder":
            expected = embed(model, ["inspect the v2 support ticket"])
            actual = embed(loaded, ["inspect the v2 support ticket"])
        else:
            expected = score_pairs(model, [("inspect v2", "GET v2 ticket")])
            actual = score_pairs(loaded, [("inspect v2", "GET v2 ticket")])
        error = float(np.max(np.abs(expected - actual)))
        if error > 1e-5:
            raise AssertionError("Reloaded predictions differ")
        result["models"][kind] = {
            **model_record(kind),
            "input_weights_sha256": digest(model_path(kind) / "model.safetensors"),
            "parameters": sum(p.numel() for p in parameters.parameters()),
            "before_parameter_sha256": before,
            "after_parameter_sha256": after,
            "reload_parameter_sha256": reloaded_hash,
            "optimizer_steps": 1,
            "loss": float(loss.detach().cpu()),
            "reload_max_abs_error": error,
            "parameter_delta": parameter_delta(parameters, sample),
            "elapsed_seconds": time.perf_counter() - start,
        }
        del model, loaded, optimizer, parameters
        if device == "mps":
            torch.mps.empty_cache()
    write_json(folder / "probe.json", result)
    return folder / "probe.json"
