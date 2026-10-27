"""D01: curated local-teacher distillation against untuned and classical baselines."""

from __future__ import annotations

from collections import Counter
import json
import re
import shutil
import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

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
from ml_training import routing_data as data


def curate(folder, rows):
    from ml_training.mlx_models import infer_rows, load_model, record, release

    training = [r for r in rows if r["split"] == "train"]
    cache = folder / "teacher"
    cache.mkdir(exist_ok=True)
    provenance = {
        "dataset_sha256": digest(folder / "manifest.json"),
        "teacher": record("teacher"),
        "system": data.SYSTEM,
        "max_tokens": 64,
        "temperature": 0.0,
    }
    metadata = cache / "metadata.json"
    if metadata.exists() and read_json(metadata) != provenance:
        raise ValueError("Teacher cache provenance changed")
    write_json(metadata, provenance)
    raw_file = cache / "outputs.jsonl"
    if (cache / "summary.json").exists() and digest(raw_file) != read_json(cache / "summary.json")[
        "cache_output_sha256"
    ]:
        raise ValueError("Completed teacher cache was modified")
    records = [json.loads(line) for line in raw_file.read_text().splitlines()] if raw_file.exists() else []
    if [r["id"] for r in records] != [r["id"] for r in training[: len(records)]]:
        raise ValueError("Teacher cache is not an ordered training prefix")
    for cached, source in zip(records, training, strict=False):
        try:
            parsed = data.parse_output(cached["teacher_output"]["text"])
        except (ValueError, TypeError, RecursionError):
            parsed = None
        if (
            cached["expected"] != source["expected"]
            or cached["parsed"] != parsed
            or cached["accepted"] != (parsed == source["expected"])
        ):
            raise ValueError("Teacher cache curation does not match the task oracle")
    started = time.perf_counter()
    if len(records) < len(training):
        teacher, tokenizer = load_model("teacher")
        with raw_file.open("a") as stream:
            for offset in range(len(records), len(training), 8):
                batch = training[offset : offset + 8]
                outputs = infer_rows(teacher, tokenizer, batch, max_tokens=64)
                for row, output in zip(batch, outputs, strict=True):
                    try:
                        parsed = data.parse_output(output["text"])
                    except (ValueError, TypeError, RecursionError):
                        parsed = None
                    item = {
                        "id": row["id"],
                        "teacher_output": output,
                        "parsed": parsed,
                        "expected": row["expected"],
                        "accepted": parsed == row["expected"],
                    }
                    records.append(item)
                    stream.write(json.dumps(item) + "\n")
                stream.flush()
                if len(records) % 64 == 0 or len(records) == len(training):
                    print(
                        f"Teacher curation {len(records)}/{len(training)} accepted={sum(r['accepted'] for r in records)}",
                        flush=True,
                    )
        del teacher, tokenizer
        release()
    accepted = {r["id"]: r for r in records if r["accepted"]}
    curated = [
        {**r, "answer": json.dumps(accepted[r["id"]]["parsed"], separators=(",", ":"))}
        for r in training
        if r["id"] in accepted
    ]
    counts = Counter(r["expected"]["intent"] for r in curated)
    if any(counts[i] < (2 if len(training) < 100 else 50) for i in data.INTENTS):
        raise ValueError("Teacher curation left insufficient examples in at least one intent")
    summary = {
        "examples": len(training),
        "accepted": len(curated),
        "rejected": len(training) - len(curated),
        "accepted_by_intent": dict(counts),
        # Acceptance requires exact agreement with the oracle, so accepted answers ARE
        # the gold labels: the teacher selects which examples train the student.
        "teacher_role": "filter: an answer is accepted only if identical to the synthetic oracle label",
        "accepted_answers_equal_oracle": all(accepted[r["id"]]["parsed"] == r["expected"] for r in curated),
        "rejected_by_family": dict(Counter(r["family"] for r in training if r["id"] not in accepted)),
        "source": "Actual local teacher outputs; exact synthetic-oracle validation",
        "cache_output_sha256": digest(raw_file),
        "generation_elapsed_this_invocation": time.perf_counter() - started,
        "generation_batch_seconds_total": sum(
            r["teacher_output"]["batch_elapsed_seconds"] / r["teacher_output"]["batch_size"] for r in records
        ),
        "teacher_generation_tokens": sum(r["teacher_output"]["generation_tokens"] for r in records),
    }
    write_json(cache / "summary.json", summary)
    return curated, cache


def classical_outputs(model, rows):
    intents = model.predict([r["user"] for r in rows])
    result = []
    for row, intent in zip(rows, intents, strict=True):
        ids = re.findall(r"\b[A-Z][0-9]{3,6}\b", row["user"])
        account = ids[0] if len(set(ids)) == 1 else None
        result.append({"text": json.dumps({"intent": intent, "account_id": account}, separators=(",", ":"))})
    return result


def latency(model, tokenizer, rows, kind):
    from ml_training.mlx_models import infer

    values, tokens = [], []
    if kind == "classical":
        classical_outputs(model, rows[:1])
    else:
        infer(model, tokenizer, rows[0]["system"], rows[0]["user"], max_tokens=64)
    measured_rows = [
        row for intent in data.INTENTS for row in [r for r in rows if r["expected"]["intent"] == intent][:2]
    ]
    for row in measured_rows:
        start = time.perf_counter()
        if kind == "classical":
            output = classical_outputs(model, [row])[0]
        else:
            output = infer(model, tokenizer, row["system"], row["user"], max_tokens=64)
        values.append(time.perf_counter() - start)
        tokens.append(output.get("generation_tokens", 0))
    return {
        "p50_seconds": float(np.percentile(values, 50)),
        "p95_seconds": float(np.percentile(values, 95)),
        "samples": len(values),
        "warmups": 1,
        "concurrency": 1,
        "batch_size": 1,
        "mean_generation_tokens": float(np.mean(tokens)),
        "scope": "warm in-process prompt/tokenization and generation, or classical prediction and JSON serialization; output validation excluded; matched validation inputs",
    }


def train(profile, seed):
    from ml_training.mlx_models import (
        adapter_hash,
        infer_rows,
        load_model,
        only_adapter_trainable,
        path,
        record,
        release,
    )
    from ml_training.mlx_training import train_adapter

    folder = data.prepare(profile, seed)
    rows, challenge = data.load(folder)
    validation = [r for r in rows if r["split"] == "validation"]
    test = [r for r in rows if r["split"] == "test"]
    run = new_run("distillation", profile, seed)
    print(f"Run directory: {run}", flush=True)
    try:
        training, cache = curate(folder, rows)
        shutil.copytree(cache, run / "teacher_curation")
        shutil.copyfile(folder / "manifest.json", run / "dataset.json")
        model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1), LogisticRegression(max_iter=500, random_state=seed)
        )
        model.fit([r["user"] for r in training], [r["expected"]["intent"] for r in training])
        joblib.dump(model, run / "classical.joblib")
        model = joblib.load(run / "classical.joblib")
        validation_metrics = {"classical": data.score(validation, classical_outputs(model, validation))}
        timings = {"classical": latency(model, None, validation, "classical")}
        # Teacher and base validation comparisons precede any final test evaluation.
        for kind in ["teacher", "base_student"]:
            net, tokenizer = load_model("teacher" if kind == "teacher" else "student")
            outputs = infer_rows(net, tokenizer, validation, max_tokens=64)
            validation_metrics[kind] = data.score(validation, outputs)
            timings[kind] = latency(net, tokenizer, validation, kind)
            write_json(run / f"validation_{kind}.json", {"metrics": validation_metrics[kind], "outputs": outputs})
            del net, tokenizer
            release()
        tuned, tokenizer, training_evidence = train_adapter(run, training, validation, data.score, profile, seed, 64)
        validation_metrics["tuned_student"] = read_json(run / "validation_tuned.json")["metrics"]
        timings["tuned_student"] = latency(tuned, tokenizer, validation, "student")
        floor = read_json(folder / "manifest.json")["quality_floor"]
        eligible = [
            name
            for name, values in validation_metrics.items()
            if all(values[key] >= value for key, value in floor.items())
        ]
        winner = (
            min(eligible, key=lambda name: timings[name]["p95_seconds"]) if eligible else "none_meets_quality_floor"
        )
        write_json(
            run / "selection.json",
            {
                "validation": validation_metrics,
                "quality_floor": floor,
                "eligible": eligible,
                "selected_variant": winner,
                "rule": "Lowest measured validation p95 among variants meeting predefined quality floor",
            },
        )
        write_json(run / "benchmarks.json", timings)
        del tuned, tokenizer
        release()
        reports, challenge_reports, predictions = {}, {}, []
        for kind in ["classical", "teacher", "base_student", "tuned_student"]:
            if kind == "classical":
                outputs = classical_outputs(model, test)
                challenge_outputs = classical_outputs(model, challenge)
            else:
                net, tokenizer = load_model(
                    "teacher" if kind == "teacher" else "student",
                    adapter=run / "adapter" if kind == "tuned_student" else None,
                )
                if kind == "tuned_student":
                    only_adapter_trainable(net)
                    if adapter_hash(net) != training_evidence["selected_adapter_sha256"]:
                        raise AssertionError("Reloaded adapter hash differs")
                outputs = infer_rows(net, tokenizer, test, max_tokens=64)
                challenge_outputs = infer_rows(net, tokenizer, challenge, max_tokens=64)
                del net, tokenizer
                release()
            reports[kind] = data.score(test, outputs)
            challenge_reports[kind] = data.score(challenge, challenge_outputs)
            predictions.extend(
                {
                    "variant": kind,
                    "split": split,
                    "id": r["id"],
                    "input": r["user"],
                    "expected": r["expected"],
                    "output": output,
                }
                for split, part, out in [("test", test, outputs), ("challenge", challenge, challenge_outputs)]
                for r, output in zip(part, out, strict=True)
            )
            print(f"D01 {kind} test_joint_accuracy={reports[kind]['joint_accuracy']:.4f}", flush=True)
        write_json(run / "metrics.json", reports)
        write_json(run / "challenge.json", challenge_reports)
        write_json(run / "predictions.json", predictions)
        failures = []
        for name in reports:
            for p in [p for p in predictions if p["variant"] == name]:
                try:
                    correct = data.parse_output(p["output"]["text"]) == p["expected"]
                except (ValueError, TypeError, RecursionError):
                    correct = False
                if not correct:
                    failures.append(p)
        write_json(run / "failure_examples.json", failures)
        curation = read_json(run / "teacher_curation" / "summary.json")
        savings = timings["teacher"]["p95_seconds"] - timings["tuned_student"]["p95_seconds"]
        costs = {
            "hosted_api_charge_usd": 0,
            "local_energy_or_depreciation_cost": "not measured",
            "teacher_labeling_seconds": curation["generation_batch_seconds_total"],
            "student_training_and_validation_seconds": training_evidence["elapsed_seconds"],
            "teacher_label_tokens": curation["teacher_generation_tokens"],
            "time_only_break_even_requests": (
                curation["generation_batch_seconds_total"] + training_evidence["elapsed_seconds"]
            )
            / savings
            if savings > 0
            else None,
            "break_even_note": "Illustrative time-only ratio using batch labeling/training versus warm batch-1 p95 savings; not monetary cost or a throughput forecast. Only relevant if quality requirements are met.",
        }
        write_json(run / "costs.json", costs)
        write_report(run, reports, challenge_reports, winner, curation, costs)
        complete_run(
            run,
            dataset_path=str(folder.relative_to(ROOT)),
            dataset_sha256=digest(folder / "manifest.json"),
            models={"student": record(), "teacher": record("teacher")},
            model_input_hashes={
                kind: {p.name: digest(p) for p in path(kind).glob("*.safetensors")} for kind in ["student", "teacher"]
            },
            synthetic=True,
            real_teacher_generated=True,
            selected_variant=winner,
            teacher_examples_accepted=len(training),
            optimizer_steps=training_evidence["optimizer_steps_executed"],
            selected_checkpoint_step=training_evidence["selected_step"],
            test_examples=len(test),
            challenge_examples=len(challenge),
            replication="single-seed pilot",
        )
    except BaseException as error:
        mark_failed(run, error)
        raise
    return run


def write_report(run, results, challenge, winner, curation, costs):
    lines = [
        "# D01: Local teacher-to-student distillation",
        "",
        f"Validation/resource selection: **{winner}**. Accepted {curation['accepted']} of {curation['examples']} actual teacher responses.",
        "",
        "| Variant | Macro-F1 | Arguments exact | JSON schema valid | Joint accuracy | Routed coverage |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['macro_f1']:.4f} | {r['argument_exact_match']:.4f} | {r['schema_validity']:.4f} | {r['joint_accuracy']:.4f} | {r['routing_coverage']:.4f} |"
        )
    models = read_json(ROOT / "configs" / "models.json")["language_models"]
    challenge_rows = next(iter(challenge.values()))["rows"]
    lines += [
        "",
        "The classical baseline learns TF-IDF/logistic routing and uses a documented regex for account extraction. "
        f"The teacher is {models['teacher']['model_id']} and the student {models['student']['model_id']}, both run locally. A teacher answer is kept only when it exactly matches the synthetic oracle, "
        "so the student's training targets are the gold labels and the teacher acts as a filter on which examples are used. "
        f"It removed {curation['rejected']} of {curation['examples']} examples"
        + (
            f", concentrated in {', '.join(sorted(curation['rejected_by_family']))}"
            if curation.get("rejected_by_family")
            else ""
        )
        + "; those are cases the teacher found hard, so the student sees fewer of them. "
        "No private reasoning traces or teacher-generated evaluation labels are used. Invalid JSON remains a failure and is not repaired after generation.",
        "",
        f"Independent challenge metrics:\n```json\n{json.dumps(challenge, indent=2)}\n```",
        "",
        f"Cost and amortization boundaries:\n```json\n{json.dumps(costs, indent=2)}\n```",
        "",
        "All data is synthetic. Test phrasing families and account identifiers are held out. "
        f"The {challenge_rows} independent challenge cases include negation, ambiguity and embedded instructions. "
        "A high score on templates does not establish robustness on real support traffic. This is one training seed. "
        "Compare benchmarks.json for matched batch-1 validation latencies; batch generation timings in prediction records are not per-request latency.",
    ]
    (run / "report.md").write_text("\n".join(lines) + "\n")


def reevaluate(run, tolerance=0.0):
    from ml_training.mlx_models import adapter_hash, infer_rows, load_model, only_adapter_trainable, release

    verify_run(run)
    info = read_json(run / "run.json")
    folder = ROOT / info["dataset_path"]
    if digest(folder / "manifest.json") != info["dataset_sha256"]:
        raise ValueError("Dataset manifest changed")
    rows, challenge = data.load(folder)
    test = [r for r in rows if r["split"] == "test"]
    expected = read_json(run / "metrics.json")
    expected_challenge = read_json(run / "challenge.json")
    saved = read_json(run / "predictions.json")
    for kind in expected:
        if kind == "classical":
            classifier = joblib.load(run / "classical.joblib")
            outputs, extra = classical_outputs(classifier, test), classical_outputs(classifier, challenge)
        else:
            model, tokenizer = load_model(
                "teacher" if kind == "teacher" else "student",
                adapter=run / "adapter" if kind == "tuned_student" else None,
            )
            if kind == "tuned_student":
                only_adapter_trainable(model)
                if adapter_hash(model) != read_json(run / "training.json")["selected_adapter_sha256"]:
                    raise AssertionError("Adapter hash mismatch")
            outputs = infer_rows(model, tokenizer, test, max_tokens=64)
            extra = infer_rows(model, tokenizer, challenge, max_tokens=64)
            del model, tokenizer
            release()
        for split, generated in [("test", outputs), ("challenge", extra)]:
            originals = [p["output"]["text"] for p in saved if p["variant"] == kind and p["split"] == split]
            require_reproduced([o["text"] for o in generated], originals, f"D01 {kind}/{split} generated text")
        require_reproduced(data.score(test, outputs), expected[kind], f"D01 {kind} test metrics", tolerance)
        require_reproduced(
            data.score(challenge, extra), expected_challenge[kind], f"D01 {kind} challenge metrics", tolerance
        )
    return {
        "status": "reproduced",
        "run": str(run),
        "variants": list(expected),
        "examples": len(test),
        "challenge_examples": len(challenge),
        "exact_outputs_matched": True,
    }


def infer_query(run, query, variant="selected"):
    from ml_training.mlx_models import infer, load_model

    verify_run(run)
    selection = read_json(run / "selection.json")
    if variant == "selected":
        variant = selection["selected_variant"]
        if variant == "none_meets_quality_floor":
            variant = "tuned_student"
    if variant not in ["classical", "teacher", "base_student", "tuned_student"]:
        raise ValueError("Choose classical, teacher, base_student, or tuned_student")
    if variant == "classical":
        output = classical_outputs(joblib.load(run / "classical.joblib"), [{"user": query}])[0]
    else:
        model, tokenizer = load_model(
            "teacher" if variant == "teacher" else "student",
            adapter=run / "adapter" if variant == "tuned_student" else None,
        )
        output = infer(model, tokenizer, data.SYSTEM, query, max_tokens=64)
    try:
        parsed = data.parse_output(output["text"])
    except (ValueError, TypeError, RecursionError):
        parsed = None
    return {
        "variant": variant,
        "input": query,
        "output": output,
        "parsed": parsed,
        "meets_frozen_validation_floor": variant in selection["eligible"],
        "experiment_selection": selection["selected_variant"],
    }
