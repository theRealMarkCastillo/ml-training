"""D08: train a bi-encoder and cross-encoder, then evaluate disjoint retrieval."""

from __future__ import annotations

import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
from rank_bm25 import BM25Okapi
import torch

from ml_training.common import ROOT, benchmark, complete_run, digest, mark_failed, new_run, read_json, require_reproduced, verify_run, write_json
from ml_training import retrieval_data as data
from ml_training.retrieval_metrics import family_bootstrap, metrics, negatives_from_scores, ranking
from ml_training.retrieval_models import (bi_loss, configure, cross_loss, embed, load_bi, load_cross,
                                         model_path, model_record, parameter_delta, sample_parameter,
                                         score_pairs, state_hash, synchronize)


def dense_ranks(model, corpus, queries):
    docs = embed(model, [d["text"] for d in corpus])
    query_vectors = embed(model, [q["text"] for q in queries])
    return ranking(query_vectors @ docs.T), docs


def bm25_ranks(corpus, queries):
    engine = BM25Okapi([data.tokenize(d["text"]) for d in corpus])
    return ranking(np.asarray([engine.get_scores(data.tokenize(q["text"])) for q in queries]))


def rerank(model, corpus, queries, candidate_ranks, pool=10):
    pairs = [(q["text"], corpus[int(i)]["text"]) for q, order in zip(queries, candidate_ranks, strict=True)
             for i in order[:pool]]
    scores = score_pairs(model, pairs).reshape(len(queries), pool)
    orders = ranking(scores)
    return np.asarray([np.concatenate([candidates[:pool][order], candidates[pool:]])
                       for candidates, order in zip(candidate_ranks, orders, strict=True)])


def record_training(run, kind, model, initial_hash, original_sample, steps, losses, seconds, epochs):
    parameters = model if kind == "bi_encoder" else model.model
    after = state_hash(parameters)
    delta = parameter_delta(parameters, original_sample)
    if after == initial_hash or delta["changed_elements"] == 0 or steps < 1:
        raise AssertionError(f"No genuine parameter update for {kind}")
    result = {"optimizer": "AdamW", "learning_rate": 2e-5, "weight_decay": .01,
              "optimizer_steps_executed": steps, "epochs": epochs, "losses": losses,
              "elapsed_seconds": seconds, "before_parameter_sha256": initial_hash,
              "selected_after_parameter_sha256": after, "parameter_delta": delta,
              "precision": "float32", "gradient_clip_norm": 1.,
              "base_model": {**model_record(kind), "files": {
                  str(p.relative_to(model_path(kind))): digest(p) for p in model_path(kind).rglob("*") if p.is_file()}}}
    write_json(run / f"{kind}_training.json", result)
    return result


def train_bi(run, corpus, training_queries, validation_queries, device, seed, profile, base):
    docs_by_id = {d["doc_id"]: d for d in corpus}
    training_docs = [d for d in corpus if d["split"] == "train"]
    scores = embed(base, [q["text"] for q in training_queries]) @ embed(base, [d["text"] for d in training_docs]).T
    mined = negatives_from_scores(training_queries, training_docs, scores)
    write_json(run / "training_negatives.json", mined)
    entries = list(zip(training_queries, mined))
    model = load_bi(device)
    original_hash, original_sample = state_hash(model), sample_parameter(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01)
    history, losses, step = [], [], 0
    best, best_epoch = -1., None
    epochs = 1 if profile == "smoke" else 2
    start = time.perf_counter()
    for epoch in range(epochs):
        random.Random(seed + epoch).shuffle(entries)
        model.train()
        for offset in range(0, len(entries), 16):
            batch = entries[offset:offset + 16]
            loss = bi_loss(model, [q["text"] for q, _ in batch],
                           [docs_by_id[m["positive_id"]]["text"] for _, m in batch],
                           [[docs_by_id[d]["text"] for d in m["negative_ids"]] for _, m in batch], device)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite bi-encoder loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            step += 1
            if step % 20 == 0:
                print(f"bi_encoder epoch={epoch+1} step={step} loss={losses[-1]:.4f}", flush=True)
        ranks, _ = dense_ranks(model, corpus, validation_queries)
        value, _ = metrics(validation_queries, corpus, ranks)
        history.append({"epoch": epoch + 1, "optimizer_steps": step, "validation": value})
        if value["mrr_at_10"] > best:
            best, best_epoch = value["mrr_at_10"], epoch + 1
            model.save_pretrained(str(run / "bi_encoder"))
        print(f"bi_encoder validation epoch={epoch+1} mrr10={value['mrr_at_10']:.4f}", flush=True)
    synchronize(device)
    model = load_bi(device, run / "bi_encoder")
    record = record_training(run, "bi_encoder", model, original_hash, original_sample, step, losses,
                             time.perf_counter() - start, epochs)
    record.update({"selected_epoch": best_epoch, "validation_history": history,
                   "loss": "per-query contrastive cross entropy over one positive and two mined negatives; temperature .05"})
    write_json(run / "bi_encoder_training.json", record)
    return model, mined, record


def train_cross(run, corpus, training_queries, validation_queries, mined, validation_candidates, device, seed, profile):
    by_doc = {d["doc_id"]: d for d in corpus}
    by_query = {q["query_id"]: q for q in training_queries}
    entries = []
    for m in mined:
        query = by_query[m["query_id"]]["text"]
        entries.append(((query, by_doc[m["positive_id"]]["text"]), 1.))
        entries.extend(((query, by_doc[did]["text"]), 0.) for did in m["negative_ids"])
    model = load_cross(device)
    original_hash, original_sample = state_hash(model.model), sample_parameter(model.model)
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=2e-5, weight_decay=.01)
    history, losses, step = [], [], 0
    best, best_epoch = -1., None
    epochs = 1 if profile == "smoke" else 2
    start = time.perf_counter()
    for epoch in range(epochs):
        random.Random(seed + epoch).shuffle(entries)
        model.model.train()
        for offset in range(0, len(entries), 32):
            batch = entries[offset:offset + 32]
            loss = cross_loss(model, [p for p, _ in batch], [label for _, label in batch], device)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite cross-encoder loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            step += 1
            if step % 20 == 0:
                print(f"cross_encoder epoch={epoch+1} step={step} loss={losses[-1]:.4f}", flush=True)
        ranks = rerank(model, corpus, validation_queries, validation_candidates)
        value, _ = metrics(validation_queries, corpus, ranks)
        history.append({"epoch": epoch + 1, "optimizer_steps": step, "validation": value})
        if value["mrr_at_10"] > best:
            best, best_epoch = value["mrr_at_10"], epoch + 1
            model.save_pretrained(str(run / "cross_encoder"))
        print(f"cross_encoder validation epoch={epoch+1} mrr10={value['mrr_at_10']:.4f}", flush=True)
    synchronize(device)
    model = load_cross(device, run / "cross_encoder")
    record = record_training(run, "cross_encoder", model, original_hash, original_sample, step, losses,
                             time.perf_counter() - start, epochs)
    record.update({"selected_epoch": best_epoch, "validation_history": history,
                   "loss": "binary cross entropy on raw pair logits; positives and mined negatives"})
    write_json(run / "cross_encoder_training.json", record)
    return model, record


def evaluate_variants(corpus, queries, base_bi, tuned_bi, base_cross, tuned_cross):
    base_ranks, _ = dense_ranks(base_bi, corpus, queries)
    tuned_ranks, _ = dense_ranks(tuned_bi, corpus, queries)
    ranks = {"bm25": bm25_ranks(corpus, queries), "base_bi": base_ranks, "tuned_bi": tuned_ranks,
             "tuned_bi_base_cross": rerank(base_cross, corpus, queries, tuned_ranks),
             "tuned_bi_tuned_cross": rerank(tuned_cross, corpus, queries, tuned_ranks),
             "fixed_base_pool_base_cross": rerank(base_cross, corpus, queries, base_ranks),
             "fixed_base_pool_tuned_cross": rerank(tuned_cross, corpus, queries, base_ranks)}
    reports, predictions = {}, []
    for variant, order in ranks.items():
        aggregate, rows = metrics(queries, corpus, order)
        reports[variant] = aggregate
        for query, row, indices in zip(queries, rows, order, strict=True):
            predictions.append({**row, "variant": variant, "query": query["text"],
                                "positive_id": query["positive_id"],
                                "ranked_ids": [corpus[int(i)]["doc_id"] for i in indices[:10]]})
    return reports, predictions, {"base": base_ranks, "tuned": tuned_ranks}


def train(profile, seed, requested="auto"):
    device = configure(seed, requested)
    folder = data.prepare(profile, seed)
    corpus, queries, challenge = data.load(folder)
    training_queries = [q for q in queries if q["split"] == "train"]
    validation_queries = [q for q in queries if q["split"] == "validation"]
    test_queries = [q for q in queries if q["split"] == "test"]
    run = new_run("retrieval", profile, seed)
    print(f"Run directory: {run}", flush=True)
    shutil.copyfile(folder / "manifest.json", run / "dataset.json")
    try:
        base_bi, base_cross = load_bi(device), load_cross(device)
        tuned_bi, mined, bi_record = train_bi(run, corpus, training_queries, validation_queries, device, seed, profile, base_bi)
        validation_candidates, _ = dense_ranks(tuned_bi, corpus, validation_queries)
        tuned_cross, cross_record = train_cross(run, corpus, training_queries, validation_queries, mined,
                                               validation_candidates, device, seed, profile)
        validation, _, _ = evaluate_variants(corpus, validation_queries, base_bi, tuned_bi, base_cross, tuned_cross)
        deployable = ["bm25", "base_bi", "tuned_bi", "tuned_bi_base_cross", "tuned_bi_tuned_cross"]
        winner = max(deployable, key=lambda k: validation[k]["mrr_at_10"])
        write_json(run / "selection.json", {"criterion": "maximum validation MRR@10; simpler variant wins ties by listed order",
                                            "variant_order": deployable, "selected_variant": winner,
                                            "validation": validation})
        # Test judgments are consulted only after all checkpoint and pipeline choices are saved.
        results, predictions, candidates = evaluate_variants(corpus, test_queries, base_bi, tuned_bi, base_cross, tuned_cross)
        for name, report in results.items():
            report["mrr_family_interval"] = family_bootstrap([p for p in predictions if p["variant"] == name], seed)
        write_json(run / "metrics.json", results)
        write_json(run / "predictions.json", predictions)
        write_json(run / "failure_examples.json", [p for p in predictions if p["rank"] != 1][:50])
        np.savez_compressed(run / "candidate_pools.npz", **candidates)
        if challenge:
            challenge_metrics, challenge_predictions, _ = evaluate_variants(corpus, challenge, base_bi, tuned_bi, base_cross, tuned_cross)
            write_json(run / "challenge.json", {"metrics": challenge_metrics, "predictions": challenge_predictions})
        docs = embed(tuned_bi, [d["text"] for d in corpus])
        example = test_queries[0]["text"]
        pool = candidates["tuned"][0, :10]
        timings = {
            "tuned_bi_query_and_exact_search": benchmark(lambda: embed(tuned_bi, [example]) @ docs.T, warmups=5, samples=30),
            "tuned_cross_ten_pairs": benchmark(lambda: score_pairs(tuned_cross, [(example, corpus[int(i)]["text"]) for i in pool]), warmups=5, samples=30),
            "query_batch_size": 1, "candidate_pool_size": 10, "corpus_size": len(corpus),
            "device": device, "precision": "float32", "torch_cpu_threads": torch.get_num_threads(),
            "timing_scope": "Warm query inference on fixed input; corpus embeddings precomputed; tensor-to-CPU output synchronizes GPU",
        }
        write_json(run / "benchmarks.json", timings)
        # Both saved checkpoints are loaded again and compared on unseen queries.
        again_bi, again_cross = load_bi(device, run / "bi_encoder"), load_cross(device, run / "cross_encoder")
        probe_queries = [q["text"] for q in test_queries[:8]]
        probe_pairs = [(q, corpus[0]["text"]) for q in probe_queries]
        bi_error = float(np.max(np.abs(embed(tuned_bi, probe_queries) - embed(again_bi, probe_queries))))
        cross_error = float(np.max(np.abs(score_pairs(tuned_cross, probe_pairs) - score_pairs(again_cross, probe_pairs))))
        if max(bi_error, cross_error) > 1e-5:
            raise AssertionError("Reload inference parity failed")
        write_report(run, results, winner, bi_record, cross_record, timings)
        complete_run(run, dataset_path=str(folder.relative_to(ROOT)), dataset_sha256=digest(folder / "manifest.json"),
                     device=device, synthetic=True, selected_variant=winner, candidate_pool_size=10,
                     bi_encoder_optimizer_steps=bi_record["optimizer_steps_executed"],
                     cross_encoder_optimizer_steps=cross_record["optimizer_steps_executed"],
                     reload_max_abs_error={"bi_encoder": bi_error, "cross_encoder": cross_error},
                     training_counts={"queries": len(training_queries), "cross_pairs": len(mined) * 3},
                     evaluation_queries=len(test_queries), challenge_queries=len(challenge),
                     replication="single-seed pilot; no broad superiority claim")
    except BaseException as error:
        mark_failed(run, error)
        raise
    return run


def write_report(run, results, winner, bi, cross, timings):
    lines = ["# D08: Version-aware retrieval and reranking", "",
             f"Validation selected **{winner}**. Both encoders were actually fine-tuned and reloaded.", "",
             f"Bi-encoder: {bi['optimizer_steps_executed']} optimizer steps, selected epoch {bi['selected_epoch']}. "
             f"Cross-encoder: {cross['optimizer_steps_executed']} optimizer steps, selected epoch {cross['selected_epoch']}.", "",
             "| Variant | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Top-1 | Wrong version |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, value in results.items():
        lines.append(f"| {name} | {value['recall_at_5']:.4f} | {value['recall_at_10']:.4f} | {value['mrr_at_10']:.4f} | {value['ndcg_at_10']:.4f} | {value['top1_accuracy']:.4f} | {value['wrong_version_rate']:.4f} |")
    lines += ["", "Base and tuned bi-encoder Recall@10 are the candidate-recall ceilings for their respective rerankers. "
              "The two fixed-base-pool rows use identical candidates to isolate the reranker effect. "
              "Wrong-version rate counts any top result with a different explicit API version; resource and operation errors are separately recorded in metrics.json.",
              "", "Training and evaluation use disjoint resource and query-template families. Hard-negative mining uses training queries and documents only; the full corpus is available during inference. Relevance is defined by a single explicit resource/version/operation tuple. The 12 independently authored challenge queries are separate from checkpoint selection.",
              "", "This is one seeded synthetic pilot. Family-bootstrap intervals have only six held-out families and cannot support a broad superiority claim. BM25 is a serious baseline when queries include exact version identifiers. Failure to beat it is a useful result, not grounds to retune against the test set.",
              "", f"Warm timing conditions and measurements:\n```json\n{json.dumps(timings, indent=2)}\n```",
              "", "Training provenance records optimizer settings, losses, parameter hashes before and after, changed parameter elements, pinned input model revisions and hashes, and the validation checkpoint-selection history. The run manifest hashes every saved model/tokenizer file recursively."]
    (run / "report.md").write_text("\n".join(lines) + "\n")


def reevaluate(run: Path, requested="auto", tolerance: float = 0.):
    verify_run(run)
    info = read_json(run / "run.json")
    device = configure(info["seed"], requested)
    folder = ROOT / info["dataset_path"]
    if digest(folder / "manifest.json") != info["dataset_sha256"]:
        raise ValueError("Dataset manifest changed")
    corpus, queries, _ = data.load(folder)
    queries = [q for q in queries if q["split"] == "test"]
    base_bi, base_cross = load_bi(device), load_cross(device)
    bi, cross = load_bi(device, run / "bi_encoder"), load_cross(device, run / "cross_encoder")
    for kind, model in [("bi_encoder", bi), ("cross_encoder", cross.model)]:
        expected = read_json(run / f"{kind}_training.json")["selected_after_parameter_sha256"]
        if state_hash(model) != expected:
            raise AssertionError(f"Checkpoint parameter hash mismatch: {kind}")
    actual, predictions, candidates = evaluate_variants(corpus, queries, base_bi, bi, base_cross, cross)
    expected = read_json(run / "metrics.json")
    saved = read_json(run / "predictions.json")
    require_reproduced([p["ranked_ids"] for p in predictions], [p["ranked_ids"] for p in saved], "D08 top-10 rankings")
    original_candidates = np.load(run / "candidate_pools.npz")
    for name, value in candidates.items():
        require_reproduced(value[:, :10].tolist(), original_candidates[name][:, :10].tolist(), f"D08 {name} candidate pool")
    # The original check allowed 1e-10 metric noise; keep that as the floor.
    require_reproduced(actual, {name: {metric: expected[name][metric] for metric in values} for name, values in actual.items()},
                       "D08 test metrics", max(tolerance, 1e-10))
    return {"status": "reproduced", "run": str(run), "device": device,
            "variants": len(actual), "queries": len(queries), "saved_parameter_hashes": "verified"}


def search(run: Path, query: str, requested="auto", variant="selected"):
    verify_run(run)
    if not query or len(query) > 4000:
        raise ValueError("Query must contain 1-4000 characters")
    info = read_json(run / "run.json")
    device = configure(info["seed"], requested)
    corpus, _, _ = data.load(ROOT / info["dataset_path"])
    variant = info["selected_variant"] if variant == "selected" else variant
    if variant not in ["bm25", "base_bi", "tuned_bi", "tuned_bi_base_cross", "tuned_bi_tuned_cross"]:
        raise ValueError("Unknown retrieval variant")
    item = [{"text": query}]
    if variant == "bm25":
        ranks = bm25_ranks(corpus, item)
    else:
        bi = load_bi(device, run / "bi_encoder" if variant != "base_bi" else None)
        ranks, _ = dense_ranks(bi, corpus, item)
        if variant.endswith("cross"):
            cross = load_cross(device, run / "cross_encoder" if variant.endswith("tuned_cross") else None)
            ranks = rerank(cross, corpus, item, ranks)
    return {"query": query, "variant": variant, "run": str(run),
            "top_five": [corpus[int(i)] for i in ranks[0, :5]]}
