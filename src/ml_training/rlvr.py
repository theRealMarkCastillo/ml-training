"""D03 actual TRL GRPO, paired SFT control, and public-versus-hidden verification."""

from collections import Counter
import json
from pathlib import Path
import random
import shutil
import subprocess
import time

import numpy as np
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from ml_training.common import ROOT, complete_run, digest, new_run, read_json, verify_run, write_json
from ml_training import rlvr_data as data
from ml_training import rlvr_models as models


def group_metrics(rows, groups):
    first, selected, any_hidden, public, selected_outputs, exploits = [], [], [], [], [], []
    for row, group in zip(rows, groups, strict=True):
        public_results = [data.verify(o["text"], row, hidden=False) for o in group]
        hidden_results = [data.verify(o["text"], row, hidden=True) for o in group]
        index = next((i for i, r in enumerate(public_results) if r["passed"]), 0)
        first.append(hidden_results[0]["passed"])
        selected.append(hidden_results[index]["passed"])
        any_hidden.append(any(r["passed"] for r in hidden_results))
        public.append(public_results[index]["passed"])
        exploits.append(public_results[index]["passed"] and not hidden_results[index]["passed"])
        selected_outputs.append({"id": row["id"], "input": row["user"], "expected": row["expected"], "outputs": group,
                                 "public_checks": public_results, "hidden_checks": hidden_results, "selected_index": index})
    return {"rows": len(rows), "sampled_pass_at_1": float(np.mean(first)), "verifier_selected_success_at_n": float(np.mean(selected)),
            "empirical_any_success_at_n": float(np.mean(any_hidden)), "public_selected_pass_rate": float(np.mean(public)),
            "public_pass_hidden_failure_rate": float(np.mean(exploits)), "n": len(groups[0]),
            "mean_generation_tokens": float(np.mean([o["generation_tokens"] for group in groups for o in group]))}, selected_outputs


def train_grpo(run, rows, profile, seed, device):
    model, tokenizer = models.load(run / "shared_sft", device, trainable=True, seed=seed)
    fusion = read_json(run / "shared_sft/fusion_evidence.json")
    transfer_outputs = [group[0] for group in models.generate(model, tokenizer, fusion["validation_rows"], seed=seed)]
    from ml_training.schema_data import score as schema_score
    write_json(run / "checkpoint_transfer.json", {
        "backend": "PyTorch float32 loaded from fused MLX SFT weights, with an initially zero-output new adapter",
        "exact_generation_match_to_mlx_fused": [o["text"] for o in transfer_outputs] == [o["text"] for o in fusion["after"]],
        "validation_metrics": schema_score(fusion["validation_rows"], transfer_outputs), "outputs": transfer_outputs,
        "note": "Backend/fusion numerical differences are recorded; both RL and SFT control start at these identical Torch weights."})
    before = models.adapter_hash(model)
    by_id = {row["id"]: row for row in rows}
    records = []
    def executable_reward(completions, completion_ids, row_id, trainer_state, **kwargs):
        texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
        rewards = [data.reward(text, by_id[key]) for text, key in zip(texts, row_id, strict=True)]
        entry = {"step_before_update": trainer_state.global_step, "row_ids": row_id, "outputs": texts,
                 "rewards": rewards, "reward_std": float(np.std(rewards)),
                 "completion_tokens": [len(ids) for ids in completion_ids]}
        records.append(entry)
        with (run / "rollouts.jsonl").open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        return rewards
    dataset = Dataset.from_list([{"prompt": [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}],
                                  "row_id": r["id"]} for r in rows])
    config = GRPOConfig(output_dir=str(run / "trainer"), per_device_train_batch_size=4, num_generations=4,
                        max_steps=2 if profile == "smoke" else 50, max_completion_length=160,
                        learning_rate=2e-5, temperature=1., top_p=1., top_k=0, bf16=False, fp16=False,
                        gradient_checkpointing=False, optim="adamw_torch", beta=0., weight_decay=0.,
                        loss_type="grpo", scale_rewards="group", epsilon=.2, num_iterations=1,
                        report_to="none", save_strategy="no", logging_steps=1, use_cpu=device == "cpu",
                        dataloader_pin_memory=False, seed=seed, disable_tqdm=True)
    trainer = GRPOTrainer(model=model, args=config, reward_funcs=executable_reward, train_dataset=dataset, processing_class=tokenizer)
    started = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter()-started
    after = models.adapter_hash(model)
    model.save_pretrained(run / "rl_adapter")
    write_json(run / "grpo_config.json", config.to_dict())
    evidence = {"trainer": "trl.GRPOTrainer", "trainer_version": "1.8.0", "optimizer_steps": trainer.state.global_step,
                "initial_adapter_sha256": before, "final_adapter_sha256": after, "parameters_changed": before != after,
                "nonconstant_reward_groups": sum(r["reward_std"] > 0 for r in records),
                "constant_reward_groups": sum(r["reward_std"] == 0 for r in records),
                "training_pool_size": len(rows), "generated_groups": len(records),
                "unique_training_prompts_sampled": len({key for r in records for key in r["row_ids"]}),
                "sampled_public_reward_mean": float(np.mean([reward for r in records for reward in r["rewards"]])),
                "generation_tokens": sum(sum(r["completion_tokens"]) for r in records),
                "elapsed_training_seconds": elapsed, "device": str(trainer.accelerator.device),
                "log_history": trainer.state.log_history,
                "checkpoint_selection": "Final fixed-budget step; no evaluation-based early stopping"}
    write_json(run / "grpo_training.json", evidence)
    del model, tokenizer, trainer
    models.release()
    if before == after:
        raise ValueError("GRPO produced no parameter change; run remains incomplete, inspect reward variance")
    return evidence


def train_sft_control(run, rows, seed, device, seconds):
    """Same SFT initialization and adapter, matched by measured training wall time."""
    model, tokenizer = models.load(run / "shared_sft", device, trainable=True, seed=seed)
    before = models.adapter_hash(model)
    expected = read_json(run / "grpo_training.json")["initial_adapter_sha256"]
    if before != expected:
        raise AssertionError("RL and SFT controls did not start with identical adapter parameters")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-5, weight_decay=0.)
    order, losses, tokens = list(rows), [], 0
    random.Random(seed).shuffle(order)
    model.train()
    start = time.perf_counter()
    step = 0
    while time.perf_counter() - start < seconds or step == 0:
        examples = [order[(step*4+i) % len(order)] for i in range(4)]
        sequences, labels = [], []
        for row in examples:
            prefix = tokenizer.encode(models.prompt(tokenizer, row), add_special_tokens=False)
            target = tokenizer.encode(row["answer"], add_special_tokens=False) + [tokenizer.eos_token_id]
            if len(prefix)+len(target) > 512:
                raise ValueError("SFT control exceeds sequence budget")
            sequences.append(prefix+target)
            labels.append([-100]*len(prefix)+target)
            tokens += len(target)
        length = max(map(len, sequences))
        inputs = torch.tensor([s+[tokenizer.pad_token_id]*(length-len(s)) for s in sequences], device=device)
        targets = torch.tensor([s+[-100]*(length-len(s)) for s in labels], device=device)
        masks = torch.tensor([[1]*len(s)+[0]*(length-len(s)) for s in sequences], device=device)
        loss = model(input_ids=inputs, attention_mask=masks, labels=targets).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        step += 1
        if step % 25 == 0:
            print(f"SFT wall-time control step={step} loss={losses[-1]:.4f}", flush=True)
    elapsed = time.perf_counter()-start
    after = models.adapter_hash(model)
    model.save_pretrained(run / "sft_control_adapter")
    evidence = {"optimizer": "AdamW", "learning_rate": 2e-5, "learning_rate_schedule": "constant",
                "weight_decay": 0., "max_grad_norm": 1., "batch_size": 4,
                "optimizer_steps": step, "completion_tokens": tokens, "losses": losses,
                "initial_adapter_sha256": before, "final_adapter_sha256": after, "parameters_changed": before != after,
                "target_training_seconds": seconds, "elapsed_training_seconds": elapsed,
                "budget_rule": "Same host/device and measured GRPO training wall time, including its sampling; SFT stops after first step crossing budget",
                "limitations": "Wall time is a resource proxy, not matched FLOPs, token count, optimizer steps, or quality. SFT receives oracle labels; RL receives binary public rewards."}
    write_json(run / "sft_control_training.json", evidence)
    del model, tokenizer, optimizer
    models.release()
    return evidence


def evaluate_variants(run, test, challenge, seed, device):
    results, challenges, predictions = {}, {}, {}
    variants = {"base": (models.original_path(), None), "sft": (run / "shared_sft", None),
                "rlvr": (run / "shared_sft", run / "rl_adapter"),
                "sft_control": (run / "shared_sft", run / "sft_control_adapter")}
    for variant, (source, adapter) in variants.items():
        model, tokenizer = models.load(source, device, adapter=adapter)
        if adapter:
            key = "grpo_training.json" if variant == "rlvr" else "sft_control_training.json"
            if models.adapter_hash(model) != read_json(run / key)["final_adapter_sha256"]:
                raise AssertionError("Reloaded RLVR/control checkpoint hash mismatch")
        variant_predictions = {}
        for split, rows, destination in [("test", test, results), ("challenge", challenge, challenges)]:
            greedy = models.generate(model, tokenizer, rows, n=1, temperature=0., seed=seed)
            sampled = models.generate(model, tokenizer, rows, n=4, temperature=1., seed=seed)
            greedy_metrics, greedy_records = group_metrics(rows, greedy)
            sampled_metrics, sampled_records = group_metrics(rows, sampled)
            destination[variant] = {"greedy_pass_at_1": greedy_metrics["sampled_pass_at_1"], **sampled_metrics,
                                    "greedy_mean_generation_tokens": greedy_metrics["mean_generation_tokens"]}
            variant_predictions[split] = {"greedy": greedy_records, "sampled": sampled_records}
        predictions[variant] = variant_predictions
        print(f"D03 {variant} greedy_test={results[variant]['greedy_pass_at_1']:.4f} selected_at_4={results[variant]['verifier_selected_success_at_n']:.4f}", flush=True)
        del model, tokenizer
        models.release()
    return results, challenges, predictions


def train(source_run, profile, seed, device):
    verify_run(source_run)
    torch.set_num_threads(4)
    if device == "auto":
        device = "cpu"  # Full MPS runs emitted unraised Metal command-buffer errors on the measured host.
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable in this process")
    folder = data.prepare(profile, seed, source_run)
    rows, challenge = data.load(folder)
    training = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    run = new_run("rlvr", profile, seed)
    print(f"Run directory: {run}", flush=True)
    try:
        shutil.copyfile(folder / "manifest.json", run / "dataset.json")
        subprocess.run([str(ROOT / "environments/mlx/.venv/bin/python"), str(ROOT / "scripts/export_sft.py"),
                        str(source_run.resolve()), str(run / "shared_sft")], check=True)
        grpo = train_grpo(run, training, profile, seed, device)
        control = train_sft_control(run, training, seed, device, grpo["elapsed_training_seconds"])
        metrics, challenges, predictions = evaluate_variants(run, test, challenge, seed, device)
        write_json(run / "metrics.json", metrics)
        write_json(run / "challenge.json", challenges)
        write_json(run / "predictions.json", predictions)
        failures = [{"variant": variant, "split": split, "mode": mode, **row}
                    for variant, splits in predictions.items() for split, modes in splits.items()
                    for mode, records in modes.items() for row in records
                    if not row["hidden_checks"][row["selected_index"]]["passed"]]
        write_json(run / "failure_examples.json", failures)
        report = ["# D03: Actual executable-reward GRPO pilot", "", "| Variant | Greedy hidden pass@1 | Sampled hidden pass@1 | Public-verifier selected at 4 |", "| --- | --- | --- | --- |"]
        for name, value in metrics.items():
            report.append(f"| {name} | {value['greedy_pass_at_1']:.4f} | {value['sampled_pass_at_1']:.4f} | {value['verifier_selected_success_at_n']:.4f} |")
        report += ["", "This is one synthetic-data pilot, not evidence of general reasoning or self-correction. "
                   "GRPO uses binary execution rewards on public tables, four stochastic completions per task, and a fixed optimizer-step budget. "
                   "Hidden randomized fixtures never enter training rewards. The verifier chooses the first publicly passing output, or the first output if none passes. "
                   "Greedy and sampled results are distinct; all variants receive the same four-sample evaluation budget. "
                   "The continued-SFT control begins at the identical fused SFT checkpoint with identical initial adapter weights and runs for the measured GRPO training wall time. "
                   "This matches a host-time resource budget, not FLOPs or optimizer steps. SFT has oracle labels, whereas GRPO has public verifier rewards.",
                   "", f"Nonconstant reward groups: {grpo['nonconstant_reward_groups']}; constant groups: {grpo['constant_reward_groups']}. "
                   f"Actual GRPO optimizer steps: {grpo['optimizer_steps']}; control steps: {control['optimizer_steps']}."]
        (run / "report.md").write_text("\n".join(report)+"\n")
        complete_run(run, dataset_path=str(folder.relative_to(ROOT)), dataset_sha256=digest(folder / "manifest.json"),
                     source_run=str(source_run.resolve()), source_manifest_sha256=digest(source_run / "run.json"),
                     real_reward_driven_updates=True, synthetic=True, device=device, optimizer_steps=grpo["optimizer_steps"],
                     test_examples=len(test), challenge_examples=len(challenge), replication="single-seed fixed-budget pilot")
    except BaseException as error:
        info = read_json(run / "run.json")
        info.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(run / "run.json", info)
        raise
    return run


def reevaluate(run, device):
    torch.set_num_threads(4)
    verify_run(run)
    info = read_json(run / "run.json")
    if device == "auto":
        device = info["device"]
    folder = ROOT / info["dataset_path"]
    if digest(folder / "manifest.json") != info["dataset_sha256"]:
        raise ValueError("RLVR dataset manifest changed")
    rows, challenge = data.load(folder)
    test = [r for r in rows if r["split"] == "test"]
    metrics, challenges, predictions = evaluate_variants(run, test, challenge, info["seed"], device)
    if metrics != read_json(run / "metrics.json") or challenges != read_json(run / "challenge.json"):
        raise AssertionError("RLVR hidden metrics did not reproduce")
    original = read_json(run / "predictions.json")
    for variant, splits in predictions.items():
        for split, modes in splits.items():
            for mode, records in modes.items():
                current = [[o["text"] for o in r["outputs"]] for r in records]
                saved = [[o["text"] for o in r["outputs"]] for r in original[variant][split][mode]]
                if current != saved:
                    raise AssertionError(f"Sampled outputs differ for {variant}/{split}/{mode}")
    return {"status": "reproduced", "run": str(run), "examples": len(test), "challenge_examples": len(challenge),
            "variants": list(metrics), "exact_outputs_matched": True}


def infer_query(run, query, variant, device):
    torch.set_num_threads(4)
    verify_run(run)
    from ml_training.schema_data import SYSTEM
    from ml_training.structured import gate
    if device == "auto":
        device = read_json(run / "run.json")["device"]
    # No production promotion is inferred from the experimental comparison.
    if variant == "selected":
        variant = "rlvr"
    adapters = {"base": None, "sft": None, "rlvr": run / "rl_adapter", "sft_control": run / "sft_control_adapter"}
    if variant not in adapters:
        raise ValueError(f"Choose {list(adapters)}")
    source = models.original_path() if variant == "base" else run / "shared_sft"
    model, tokenizer = models.load(source, device, adapter=adapters[variant])
    output = models.generate(model, tokenizer, [{"system": SYSTEM, "user": query}])[0][0]
    return {"variant": variant, "input": query, "output": output, "runtime": gate("schema_dsl", output["text"]),
            "note": "Runtime validates syntax; semantic correctness requires the user's task specification and test fixtures."}
