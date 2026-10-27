"""D03 tasks and rewards, with no evaluation fixtures in the training reward."""

import hashlib
import json

import numpy as np

from ml_training.common import ROOT, digest, new_dataset_folder, read_json, write_json
from ml_training import schema_data
from ml_training.query_dsl import equivalent


def ast_key(ast):
    return hashlib.sha256(json.dumps(ast, sort_keys=True).encode()).hexdigest()


def build(profile, seed, excluded):
    counts = (
        {"train": 500, "validation": 64, "test": 96} if profile == "local" else {"train": 8, "validation": 4, "test": 4}
    )
    used = {ast_key(row["expected"]) for row in excluded}
    selected = {split: [] for split in counts}
    # Draw from the same declared grammar, with new ASTs and random seeds. Never
    # select using a model's score or difficulty on the held-out data.
    for attempt in range(100):
        candidates = schema_data.build("local", seed + 991 + attempt * 71)
        for row in candidates:
            split = row["split"]
            key = ast_key(row["expected"])
            if key in used or len(selected[split]) >= counts[split]:
                continue
            # Preserve the explicit unseen-composition slice in the smaller test.
            if split == "test":
                want_novel = len(selected[split]) >= counts[split] * 0.75
                if bool(row["novel_composition"]) != want_novel:
                    continue
            i = len(selected[split])
            row = {**row, "id": f"rl-{split}-{i}", "source_seed": seed + 991 + attempt * 71}
            selected[split].append(row)
            used.add(key)
        if all(len(selected[split]) == count for split, count in counts.items()):
            return [row for split in counts for row in selected[split]]
    raise ValueError("Insufficient distinct executable tasks")


# v1 (runs before this version was recorded) graded with tables(hidden=True), which shares
# its boundary-witness and empty tables with the public reward; v2 shares nothing.
HIDDEN_FIXTURES = "v2"


def verify(text, row, hidden=False, hidden_version=HIDDEN_FIXTURES):
    expected, seed = row["expected"], row["fixture_seed"]
    if not hidden:
        fixtures = schema_data.tables(expected, seed, hidden=False)
    elif hidden_version == "v1":
        fixtures = schema_data.tables(expected, seed, hidden=True)
    elif hidden_version == "v2":
        fixtures = schema_data.hidden_tables(expected, seed)
    else:
        raise ValueError(f"Unknown hidden fixture version: {hidden_version}")
    return equivalent(text, expected, fixtures)


def reward(text, row):
    # Binary all-fixture execution reward. No format bonus, partial credit for
    # empty tables, answer-string comparison, or hidden-case access.
    return float(verify(text, row, hidden=False)["passed"])


def group_metrics(rows, groups, hidden_version=HIDDEN_FIXTURES):
    """Grade sampled groups on hidden fixtures; best-of-N selection sees only public checks."""
    first, selected, any_hidden, public, selected_outputs, exploits = [], [], [], [], [], []
    for row, group in zip(rows, groups, strict=True):
        public_results = [verify(o["text"], row, hidden=False) for o in group]
        hidden_results = [verify(o["text"], row, hidden=True, hidden_version=hidden_version) for o in group]
        index = next((i for i, r in enumerate(public_results) if r["passed"]), 0)
        first.append(hidden_results[0]["passed"])
        selected.append(hidden_results[index]["passed"])
        any_hidden.append(any(r["passed"] for r in hidden_results))
        public.append(public_results[index]["passed"])
        exploits.append(public_results[index]["passed"] and not hidden_results[index]["passed"])
        selected_outputs.append(
            {
                "id": row["id"],
                "input": row["user"],
                "expected": row["expected"],
                "outputs": group,
                "public_checks": public_results,
                "hidden_checks": hidden_results,
                "selected_index": index,
            }
        )
    return {
        "rows": len(rows),
        "sampled_pass_at_1": float(np.mean(first)),
        "verifier_selected_success_at_n": float(np.mean(selected)),
        "empirical_any_success_at_n": float(np.mean(any_hidden)),
        "public_selected_pass_rate": float(np.mean(public)),
        "public_pass_hidden_failure_rate": float(np.mean(exploits)),
        "n": len(groups[0]),
        "mean_generation_tokens": float(np.mean([o["generation_tokens"] for group in groups for o in group])),
    }, selected_outputs


def prepare(profile, seed, source_run):
    source_info = read_json(source_run / "run.json")
    if source_info["demo"] != "schema_dsl" or source_info["status"] not in ["evaluated", "smoke"]:
        raise ValueError("RLVR requires an evaluated D02 SFT artifact")
    source_folder = ROOT / source_info["dataset_path"]
    source_rows, source_challenge = schema_data.load(source_folder)
    folder = ROOT / "data/generated" / f"rlvr-v1-{profile}-s{seed}-{digest(source_folder / 'manifest.json')[:10]}"
    if (folder / "manifest.json").exists():
        load(folder)
        return folder
    new_dataset_folder(folder)
    rows = build(profile, seed, source_rows + source_challenge)
    write_json(folder / "rows.json", rows)
    write_json(folder / "challenge.json", source_challenge)
    write_json(
        folder / "manifest.json",
        {
            "version": "rlvr-v1",
            "synthetic": True,
            "seed": seed,
            "sft_dataset_sha256": digest(source_folder / "manifest.json"),
            "split_policy": "Distinct ASTs across RL partitions and all SFT examples/challenges; disjoint phrasing families",
            "reward": "1 iff schema-valid output passes all public execution fixtures, otherwise 0",
            "hidden_evaluation": "Fixtures disjoint from the public reward (see schema_data.hidden_tables); never used by training reward",
            "counts": {s: sum(r["split"] == s for r in rows) for s in ["train", "validation", "test"]},
            "files": {p.name: digest(p) for p in folder.glob("*.json")},
        },
    )
    return folder


def load(folder):
    for name, sha in read_json(folder / "manifest.json")["files"].items():
        if digest(folder / name) != sha:
            raise ValueError("Frozen RLVR dataset changed")
    return read_json(folder / "rows.json"), read_json(folder / "challenge.json")
