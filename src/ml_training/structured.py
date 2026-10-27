"""D02 and D04: task-specific SFT, strict runtime gates, and held-out comparisons."""

from copy import deepcopy
import json
import shutil
import time

import numpy as np

from ml_training.common import (
    ROOT,
    complete_run,
    digest,
    mark_failed,
    new_run,
    read_json,
    require_reproduced,
    verify_run,
    write_json,
)


def task(demo):
    if demo == "schema_dsl":
        from ml_training import schema_data

        return schema_data, 160
    if demo == "edge":
        from ml_training import device_data

        return device_data, 64
    raise ValueError(f"Unknown structured demo: {demo}")


# The D02 typed_slot baseline knows only training wording, like the trained models.
# The reference variant knows every split's wording, so it is reported but never
# selected. Runs recorded before this field existed used all wordings for typed_slot.
TYPED_SLOT_WORDINGS = "training"
TYPED_SLOT_REFERENCE = "typed_slot_all_wordings"


def typed_wordings(variant, run_wordings=TYPED_SLOT_WORDINGS):
    return "all" if variant == TYPED_SLOT_REFERENCE else run_wordings


def typed_outputs(rows, wordings="all"):
    from ml_training.schema_data import ALL_STYLES, TRAINING_STYLES, typed_slot

    styles = {"training": TRAINING_STYLES, "all": ALL_STYLES}[wordings]
    outputs = []
    for row in rows:
        try:
            outputs.append({"text": json.dumps(typed_slot(row["user"], styles), separators=(",", ":"))})
        except ValueError as error:
            outputs.append({"text": "", "rejection": str(error)})
    return outputs


def load_variant(run, variant):
    from ml_training.mlx_models import load_model

    if variant == "quantized_tuned":
        from mlx_lm import load

        return load(str(run / "quantized"))
    return load_model(adapter=run / "adapter" if variant == "tuned" else None)


def gate(demo, text, state=None):
    """Validate every output before the bounded interpreter or dispatcher runs."""
    data, _ = task(demo)
    if demo == "schema_dsl":
        from ml_training.query_dsl import parse

        try:
            return {"accepted": True, "ast": parse(text)}
        except ValueError as error:
            return {"accepted": False, "reason": str(error)}
    before = deepcopy(state if state is not None else {"celsius": 20, "jobs": []})
    try:
        command = data.parse_output(text)
        after = data.dispatch(command, before)
        return {
            "accepted": True,
            "command": command,
            "before": before,
            "after": after,
            "simulated": True,
            "explicit_reject": command["action"] == "reject",
        }
    except (ValueError, TypeError, RecursionError) as error:
        return {"accepted": False, "reason": str(error), "before": before, "after": before, "simulated": True}


def benchmark(model, tokenizer, rows, demo, variant):
    from ml_training.mlx_models import infer

    _, max_tokens = task(demo)
    values, tokens, memory = [], [], []
    for i, row in enumerate(rows[:13]):
        start = time.perf_counter()
        output = (
            typed_outputs([row], typed_wordings(variant))[0]
            if variant.startswith("typed_slot")
            else infer(model, tokenizer, row["system"], row["user"], max_tokens=max_tokens)
        )
        gate(demo, output["text"])
        if i:  # One warmup; matched validation rows, batch 1 for each variant.
            values.append(time.perf_counter() - start)
            tokens.append(output.get("generation_tokens", 0))
            memory.append(output.get("peak_memory_gb", 0))
    return {
        "p50_seconds": float(np.percentile(values, 50)),
        "p95_seconds": float(np.percentile(values, 95)),
        "p99_seconds": float(np.percentile(values, 99)),
        "samples": len(values),
        "warmups": 1,
        "concurrency": 1,
        "batch_size": 1,
        "max_tokens": max_tokens,
        "mean_generation_tokens": float(np.mean(tokens)),
        "process_mlx_peak_memory_gb": max(memory),
        "scope": "Warm in-process prompt/tokenization, generation and runtime validation; process memory high-water mark is not isolated per variant",
    }


def export_quantized(run):
    from mlx.utils import tree_unflatten
    from mlx_lm import load
    from mlx_lm.utils import quantize_model, save
    from ml_training.mlx_models import path, release

    model, tokenizer, config = load(str(path()), adapter_path=str(run / "adapter"), return_config=True)
    fused = [(name, module.fuse()) for name, module in model.named_modules() if hasattr(module, "fuse")]
    if not fused:
        raise AssertionError("No trained LoRA modules to fuse")
    model.update_modules(tree_unflatten(fused))
    model, config = quantize_model(model, config, group_size=64, bits=4)
    save(run / "quantized", path(), model, tokenizer, config, donate_model=False)
    del model, tokenizer
    release()
    return {
        "method": "Fuse selected task adapter, then affine 4-bit weight quantization",
        "bits": 4,
        "group_size": 64,
        "fused_modules": len(fused),
        "model_bytes": sum(p.stat().st_size for p in (run / "quantized").glob("*.safetensors")),
        "base_model_bytes": sum(p.stat().st_size for p in path().glob("*.safetensors")),
    }


def train(demo, profile, seed):
    from ml_training.mlx_models import adapter_hash, infer_rows, only_adapter_trainable, path, record, release
    from ml_training.mlx_training import train_adapter

    data, max_tokens = task(demo)
    folder = data.prepare(profile, seed)
    rows, challenge = data.load(folder)
    training, validation, test = ([r for r in rows if r["split"] == split] for split in ["train", "validation", "test"])
    run = new_run(demo, profile, seed)
    print(f"Run directory: {run}", flush=True)
    try:
        shutil.copyfile(folder / "manifest.json", run / "dataset.json")
        base, tokenizer = load_variant(run, "base")
        validation_results = {
            "base": data.score(validation, infer_rows(base, tokenizer, validation, max_tokens=max_tokens))
        }
        timings = {"base": benchmark(base, tokenizer, validation, demo, "base")}
        del base, tokenizer
        release()
        tuned, tokenizer, evidence = train_adapter(
            run, training, validation, data.score, profile, seed, max_tokens, batch_size=4
        )
        validation_results["tuned"] = read_json(run / "validation_tuned.json")["metrics"]
        timings["tuned"] = benchmark(tuned, tokenizer, validation, demo, "tuned")
        del tuned, tokenizer
        release()
        variants = ["base", "tuned"]
        if demo == "schema_dsl":
            variants.append("typed_slot")
            for name in ["typed_slot", TYPED_SLOT_REFERENCE]:
                validation_results[name] = data.score(validation, typed_outputs(validation, typed_wordings(name)))
                timings[name] = benchmark(None, None, validation, demo, name)
        else:
            write_json(run / "quantization.json", export_quantized(run))
            variants.append("quantized_tuned")
            quantized, tokenizer = load_variant(run, "quantized_tuned")
            validation_results["quantized_tuned"] = data.score(
                validation, infer_rows(quantized, tokenizer, validation, max_tokens=max_tokens)
            )
            timings["quantized_tuned"] = benchmark(quantized, tokenizer, validation, demo, "quantized_tuned")
            del quantized, tokenizer
            release()
        # Freeze the choice before opening final test and challenge results.
        selected = min(
            variants, key=lambda name: (-validation_results[name]["joint_accuracy"], timings[name]["p95_seconds"])
        )
        write_json(
            run / "selection.json",
            {
                "selected_variant": selected,
                "validation": validation_results,
                "rule": "Highest validation semantic/command accuracy, then lowest measured warm p95; educational selection, no production floor",
                "candidates": variants,
                "reference_only": [TYPED_SLOT_REFERENCE] if demo == "schema_dsl" else [],
            },
        )
        write_json(run / "benchmarks.json", timings)
        metrics, challenges, predictions, failures = {}, {}, [], []
        for variant in variants + ([TYPED_SLOT_REFERENCE] if demo == "schema_dsl" else []):
            if variant.startswith("typed_slot"):
                wordings = typed_wordings(variant)
                outputs, extra = typed_outputs(test, wordings), typed_outputs(challenge, wordings)
            else:
                model, tokenizer = load_variant(run, variant)
                if variant == "tuned":
                    only_adapter_trainable(model)
                    if adapter_hash(model) != evidence["selected_adapter_sha256"]:
                        raise AssertionError("Reloaded task adapter hash mismatch")
                outputs = infer_rows(model, tokenizer, test, max_tokens=max_tokens)
                extra = infer_rows(model, tokenizer, challenge, max_tokens=max_tokens)
                del model, tokenizer
                release()
            metrics[variant], challenges[variant] = data.score(test, outputs), data.score(challenge, extra)
            for split, part, generated in [("test", test, outputs), ("challenge", challenge, extra)]:
                for row, output in zip(part, generated, strict=True):
                    item = {
                        "variant": variant,
                        "split": split,
                        "id": row["id"],
                        "input": row["user"],
                        "expected": row["expected"],
                        "output": output,
                        "runtime": gate(demo, output["text"]),
                        "correct": bool(data.score([row], [output])["joint_accuracy"]),
                    }
                    predictions.append(item)
                    if not item["correct"]:
                        failures.append(item)
            print(f"{demo} {variant} test_accuracy={metrics[variant]['joint_accuracy']:.4f}", flush=True)
        write_json(run / "metrics.json", metrics)
        write_json(run / "challenge.json", challenges)
        write_json(run / "predictions.json", predictions)
        write_json(run / "failure_examples.json", failures)
        write_report(run, demo, metrics, challenges, selected, sum(bool(r.get("novel_composition")) for r in test))
        complete_run(
            run,
            dataset_path=str(folder.relative_to(ROOT)),
            dataset_sha256=digest(folder / "manifest.json"),
            model=record(),
            model_input_hashes={p.name: digest(p) for p in path().glob("*.safetensors")},
            synthetic=True,
            selected_variant=selected,
            optimizer_steps=evidence["optimizer_steps_executed"],
            selected_checkpoint_step=evidence["selected_step"],
            test_examples=len(test),
            challenge_examples=len(challenge),
            replication="single-seed pilot",
            offline_validation="Separate enforced-network-denial run required" if demo == "edge" else None,
            typed_slot_wordings=TYPED_SLOT_WORDINGS if demo == "schema_dsl" else None,
        )
    except BaseException as error:
        mark_failed(run, error)
        raise
    return run


def write_report(run, demo, metrics, challenges, selected, novel_tests=0):
    lines = [
        f"# {demo}: Actual local task-adapter comparison",
        "",
        f"Validation-selected variant: **{selected}**.",
        "",
        "| Variant | Schema valid | Semantic/command accuracy |",
        "| --- | --- | --- |",
    ]
    for variant, result in metrics.items():
        lines.append(f"| {variant} | {result['schema_validity']:.4f} | {result['joint_accuracy']:.4f} |")
    lines += [
        "",
        "## Independent challenge results",
        "",
        "```json",
        json.dumps(challenges, indent=2),
        "```",
        "",
        "All data is synthetic and this is one training seed. Phrasing families are separated before training. "
        "The best trained epoch is selected on validation only. Invalid outputs remain failures; the runtime gate does not repair or infer missing fields.",
    ]
    if demo == "schema_dsl":
        lines += [
            "",
            "Semantic correctness executes the candidate and oracle AST on randomized, boundary-witness and empty tables. "
            "The typed-slot baseline is a deterministic parser, not constrained neural decoding. Like the trained models, it knows only "
            "the training wording, so held-out phrasing tests whether rules generalize. "
            + (
                f"`{TYPED_SLOT_REFERENCE}` was written knowing every split's wording, including the test wording; it is reported "
                "to show what that hand-coding buys and is never selected. "
                if TYPED_SLOT_REFERENCE in metrics
                else ""
            )
            + "The independent challenge includes wording neither parser supports. "
            f"{novel_tests} of {metrics[selected]['rows']} final test examples combine fields not paired during training.",
        ]
    else:
        lines += [
            "",
            "Device actions affect a deterministic simulator only. Predictions contain before/after states. "
            "Range and action validation rejects unsafe payloads independently of model training, but cannot establish that a schema-valid action matches the user's intent. "
            "The quantized variant fuses the trained adapter before 4-bit export and is reloaded before evaluation. "
            "HF offline flags are not proof of blocked networking; enforced-offline evidence is a separate validation artifact.",
        ]
    (run / "report.md").write_text("\n".join(lines) + "\n")


def reevaluate(run, tolerance=0.0):
    from ml_training.mlx_models import adapter_hash, infer_rows, only_adapter_trainable, release

    verify_run(run)
    info = read_json(run / "run.json")
    demo = info["demo"]
    data, max_tokens = task(demo)
    folder = ROOT / info["dataset_path"]
    if digest(folder / "manifest.json") != info["dataset_sha256"]:
        raise ValueError("Dataset manifest changed")
    rows, challenge = data.load(folder)
    test = [r for r in rows if r["split"] == "test"]
    expected, expected_challenge = read_json(run / "metrics.json"), read_json(run / "challenge.json")
    saved = read_json(run / "predictions.json")
    run_wordings = info.get("typed_slot_wordings") or "all"
    for variant in expected:
        if variant.startswith("typed_slot"):
            wordings = typed_wordings(variant, run_wordings)
            outputs, extra = typed_outputs(test, wordings), typed_outputs(challenge, wordings)
        else:
            model, tokenizer = load_variant(run, variant)
            if variant == "tuned":
                only_adapter_trainable(model)
                if adapter_hash(model) != read_json(run / "training.json")["selected_adapter_sha256"]:
                    raise AssertionError("Adapter hash mismatch")
            outputs = infer_rows(model, tokenizer, test, max_tokens=max_tokens)
            extra = infer_rows(model, tokenizer, challenge, max_tokens=max_tokens)
            del model, tokenizer
            release()
        for split, part, generated in [("test", test, outputs), ("challenge", challenge, extra)]:
            originals = [p["output"]["text"] for p in saved if p["variant"] == variant and p["split"] == split]
            require_reproduced([p["text"] for p in generated], originals, f"{demo} {variant}/{split} generated text")
            target = expected[variant] if split == "test" else expected_challenge[variant]
            require_reproduced(data.score(part, generated), target, f"{demo} {variant}/{split} metrics", tolerance)
    return {
        "status": "reproduced",
        "run": str(run),
        "variants": list(expected),
        "examples": len(test),
        "challenge_examples": len(challenge),
        "exact_outputs_matched": True,
    }


def infer_query(run, query, variant="selected"):
    from ml_training.mlx_models import infer

    verify_run(run)
    info = read_json(run / "run.json")
    demo = info["demo"]
    data, max_tokens = task(demo)
    if variant == "selected":
        variant = read_json(run / "selection.json")["selected_variant"]
    allowed = ["base", "tuned", *(read_json(run / "metrics.json") if demo == "schema_dsl" else ["quantized_tuned"])]
    allowed = list(dict.fromkeys(allowed))
    if variant not in allowed:
        raise ValueError(f"Choose a variant from {allowed}")
    if variant.startswith("typed_slot"):
        output = typed_outputs([{"user": query}], typed_wordings(variant, info.get("typed_slot_wordings") or "all"))[0]
    else:
        model, tokenizer = load_variant(run, variant)
        output = infer(model, tokenizer, data.SYSTEM, query, max_tokens=max_tokens)
    return {"variant": variant, "input": query, "output": output, "runtime": gate(demo, output["text"])}
