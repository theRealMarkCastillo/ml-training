"""D03 tasks and rewards, with no evaluation fixtures in the training reward."""

import hashlib
import json

from ml_training.common import ROOT, digest, new_dataset_folder, read_json, write_json
from ml_training import schema_data
from ml_training.query_dsl import equivalent


def ast_key(ast):
    return hashlib.sha256(json.dumps(ast, sort_keys=True).encode()).hexdigest()


def build(profile, seed, excluded):
    counts = {"train": 500, "validation": 64, "test": 96} if profile == "local" else {"train": 8, "validation": 4, "test": 4}
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
                want_novel = len(selected[split]) >= counts[split] * .75
                if bool(row["novel_composition"]) != want_novel:
                    continue
            i = len(selected[split])
            row = {**row, "id": f"rl-{split}-{i}", "source_seed": seed + 991 + attempt * 71}
            selected[split].append(row)
            used.add(key)
        if all(len(selected[split]) == count for split, count in counts.items()):
            return [row for split in counts for row in selected[split]]
    raise ValueError("Insufficient distinct executable tasks")


def verify(text, row, hidden=False):
    return equivalent(text, row["expected"], schema_data.tables(row["expected"], row["fixture_seed"], hidden=hidden))


def reward(text, row):
    # Binary all-fixture execution reward. No format bonus, partial credit for
    # empty tables, answer-string comparison, or hidden-case access.
    return float(verify(text, row, hidden=False)["passed"])


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
    write_json(folder / "manifest.json", {"version": "rlvr-v1", "synthetic": True, "seed": seed,
                                         "sft_dataset_sha256": digest(source_folder / "manifest.json"),
                                         "split_policy": "Distinct ASTs across RL partitions and all SFT examples/challenges; disjoint phrasing families",
                                         "reward": "1 iff schema-valid output passes all public execution fixtures, otherwise 0",
                                         "hidden_evaluation": "Different random-table seed plus boundary and empty fixtures; never used by training reward",
                                         "counts": {s: sum(r["split"] == s for r in rows) for s in ["train", "validation", "test"]},
                                         "files": {p.name: digest(p) for p in folder.glob("*.json")}})
    return folder


def load(folder):
    for name, sha in read_json(folder / "manifest.json")["files"].items():
        if digest(folder / name) != sha:
            raise ValueError("Frozen RLVR dataset changed")
    return read_json(folder / "rows.json"), read_json(folder / "challenge.json")
