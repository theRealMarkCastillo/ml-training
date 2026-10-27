"""D02 executable query tasks, held-out compositions, and a typed-slot baseline."""

from itertools import product
import json
import random
import re

import numpy as np

from ml_training.common import ROOT, digest, new_dataset_folder, read_json, write_json
from ml_training.query_dsl import InvalidQuery, equivalent, execute, parse, validate

SYSTEM = (
    'Return only compact JSON with exactly filter and aggregate, no markdown. '
    'A filter is {"field":FIELD,"op":OP,"value":VALUE}, {"all":[FILTERS]}, or {"any":[FILTERS]}. '
    'Fields: priority (low,normal,urgent), status (open,pending,closed), team (billing,support,security), '
    'age_hours and amount (numbers). Operators: eq,ne; numbers also gt,ge,lt,le. '
    'Aggregate: {"op":"count"}, {"op":"ids"}, or {"op":"sum"|"mean","field":"age_hours"|"amount"}. '
    'Use all for AND and any for OR. Interpret the request; do not execute code.'
)
VALUES = {"priority": ["low", "normal", "urgent"], "status": ["open", "pending", "closed"],
          "team": ["billing", "support", "security"]}
OP_WORDS = {"eq": "equals", "ne": "does not equal", "gt": "is greater than", "ge": "is at least", "lt": "is less than", "le": "is at most"}
KNOWN_PAIRS = [("status", "age_hours"), ("team", "amount"), ("priority", "age_hours")]
NOVEL_PAIRS = [("priority", "amount"), ("status", "team")]


def predicate(rng, field):
    if field in VALUES:
        return {"field": field, "op": rng.choice(["eq", "ne"]), "value": rng.choice(VALUES[field])}
    return {"field": field, "op": rng.choice(["gt", "ge", "lt", "le"]), "value": rng.randint(3, 97)}


def condition_text(node):
    if "all" in node or "any" in node:
        key = "all" if "all" in node else "any"
        return (" AND " if key == "all" else " OR ").join(condition_text(n) for n in node[key])
    return f"{node['field']} {OP_WORDS[node['op']]} {node['value']}"


def task_text(ast, style):
    aggregate = ast["aggregate"]
    op = aggregate["op"]
    verbs = {
        "count": ["Count tickets", "Return the number of tickets", "How many tickets are there", "Find the count of tickets"],
        "ids": ["List ticket IDs", "Return ticket IDs", "Which ticket IDs apply", "Find ticket IDs"],
        "sum": [f"Sum {aggregate.get('field')}", f"Return the total {aggregate.get('field')}", f"Calculate total {aggregate.get('field')}", f"Find the sum of {aggregate.get('field')}"],
        "mean": [f"Average {aggregate.get('field')}", f"Return the mean {aggregate.get('field')}", f"Calculate average {aggregate.get('field')}", f"Find the mean of {aggregate.get('field')}"],
    }
    return f"{verbs[op][style]} where {condition_text(ast['filter'])}."


def build(profile, seed):
    rows = []
    sizes = {"train": 1000, "validation": 100, "test": 160} if profile == "local" else {"train": 24, "validation": 8, "test": 8}
    for split_index, (split, count) in enumerate(sizes.items()):
        rng = random.Random(seed + split_index * 10000)
        for i in range(count):
            novel = split == "test" and i >= count * .75
            if novel:
                fields = rng.choice(NOVEL_PAIRS)
            elif i % 2:
                fields = rng.choice(KNOWN_PAIRS)
            else:
                fields = [rng.choice(list(VALUES) + ["age_hours", "amount"])]
            predicates = [predicate(rng, field) for field in fields]
            condition = predicates[0] if len(predicates) == 1 else {rng.choice(["all", "any"]): predicates}
            aggregate = {"op": rng.choice(["count", "ids", "sum", "mean"])}
            if aggregate["op"] in ["sum", "mean"]:
                aggregate["field"] = rng.choice(["age_hours", "amount"])
            ast = {"filter": condition, "aggregate": aggregate}
            style = rng.choice([0, 1]) if split == "train" else 2 if split == "validation" else 3
            rows.append({"id": f"{split}-{i}", "split": split, "family": f"phrasing-{style}",
                         "novel_composition": novel, "system": SYSTEM,
                         "user": task_text(ast, style) + f" Request reference {split}-{i}.",
                         "expected": ast, "answer": json.dumps(ast, separators=(",", ":")),
                         "fixture_seed": seed * 100000 + split_index * 10000 + i})
    return rows


def tables(ast, seed, hidden=True):
    """Independent randomized tables plus boundary witnesses and empty input."""
    rng = random.Random(seed + (901337 if hidden else 0))
    def random_table():
        return [{"id": f"r{i}", "priority": rng.choice(VALUES["priority"]), "status": rng.choice(VALUES["status"]),
                 "team": rng.choice(VALUES["team"]), "age_hours": rng.randint(0, 110), "amount": rng.randint(0, 110)} for i in range(48)]
    thresholds = {"age_hours": [], "amount": []}
    def walk(node):
        if "all" in node or "any" in node:
            for child in node.get("all", node.get("any")):
                walk(child)
        elif node["field"] in thresholds:
            thresholds[node["field"]].append(node["value"])
    walk(ast["filter"])
    choices = {}
    for field, values in thresholds.items():
        # Dataset grammar has at most one predicate per numeric field.
        pivot = values[0] if values else 24
        choices[field] = sorted({0, max(0, pivot-1), pivot, pivot+1, 110})
    witness = []
    for i, values in enumerate(product(VALUES["priority"], VALUES["status"], VALUES["team"], choices["age_hours"], choices["amount"])):
        witness.append(dict(zip(["priority", "status", "team", "age_hours", "amount"], values)) | {"id": f"w{i}"})
    return [random_table(), witness, []]


def typed_slot(text):
    """A deterministic bounded-language parser; not constrained neural decoding."""
    lower = re.sub(r"\s+request reference [a-z0-9-]+\.\s*$", "", text.lower().strip())
    if lower.count(" where ") != 1:
        raise InvalidQuery("Outside typed-slot predicate grammar")
    head, condition = lower.split(" where ")
    if re.fullmatch(r"(?:list ticket ids|return ticket ids|which ticket ids apply|find ticket ids)", head):
        aggregate = {"op": "ids"}
    elif re.fullmatch(r"(?:count tickets|return the number of tickets|how many tickets are there|find the count of tickets)", head):
        aggregate = {"op": "count"}
    else:
        total = re.fullmatch(r"(?:sum(?: of)?|return the total|calculate total|find the sum of) (age_hours|amount)", head)
        mean = re.fullmatch(r"(?:average(?: of)?|return the mean|calculate average|find the mean of) (age_hours|amount)", head)
        if not total and not mean:
            raise InvalidQuery("Outside typed-slot aggregate grammar")
        aggregate = {"op":"sum" if total else "mean", "field":(total or mean)[1]}
    condition = condition.rstrip(". ")
    if " and " in condition and " or " in condition:
        raise InvalidQuery("Mixed precedence is not supported by this baseline")
    joiner = " and " if " and " in condition else " or " if " or " in condition else None
    parts = condition.split(joiner) if joiner else [condition]
    predicates = []
    words_to_op = {v: k for k, v in OP_WORDS.items()}
    for part in parts:
        match = re.fullmatch(r"(priority|status|team|age_hours|amount) (equals|does not equal|is greater than|is at least|is less than|is at most) ([a-z]+|[0-9]+)", part)
        if not match:
            raise InvalidQuery("Outside typed-slot predicate grammar")
        field, op, value = match.groups()
        predicates.append({"field": field, "op": words_to_op[op], "value": int(value) if field not in VALUES else value})
    condition_ast = {"all" if joiner == " and " else "any": predicates} if joiner else predicates[0]
    ast = {"filter": condition_ast, "aggregate": aggregate}
    validate(ast)
    return ast


def score(rows, outputs):
    syntax, schema, semantic, exact = [], [], [], []
    for row, output in zip(rows, outputs, strict=True):
        text = output["text"]
        try:
            json.loads(text)
            syntax.append(True)
        except (ValueError, TypeError, RecursionError):
            syntax.append(False)
        result = equivalent(text, row["expected"], tables(row["expected"], row["fixture_seed"]))
        schema.append(result["valid"])
        semantic.append(result["passed"])
        try:
            exact.append(parse(text) == row["expected"])
        except InvalidQuery:
            exact.append(False)
    novel = [i for i, row in enumerate(rows) if row.get("novel_composition")]
    return {"rows": len(rows), "json_parse_rate": float(np.mean(syntax)), "schema_validity": float(np.mean(schema)),
            "semantic_accuracy": float(np.mean(semantic)), "joint_accuracy": float(np.mean(semantic)),
            "ast_exact_match": float(np.mean(exact)),
            "novel_composition_accuracy": float(np.mean([semantic[i] for i in novel])) if novel else None}


def prepare(profile, seed):
    folder = ROOT / "data" / "generated" / f"schema-v1-{profile}-s{seed}"
    if (folder / "manifest.json").exists():
        load(folder)
        return folder
    new_dataset_folder(folder)
    rows = build(profile, seed)
    write_json(folder / "rows.json", rows)
    challenge = read_json(ROOT / "data" / "fixtures" / "schema_challenge.json")
    for i, row in enumerate(challenge):
        row.update(system=SYSTEM, fixture_seed=99900+i, answer=json.dumps(row["expected"], separators=(",", ":")))
    write_json(folder / "challenge.json", challenge)
    write_json(folder / "manifest.json", {"version": "schema-v1", "synthetic": True, "seed": seed,
                                         "split_policy": "Disjoint wording families; 25% of test queries use held-out field combinations; primitive ASTs can recur.",
                                         "counts": {s: sum(r["split"] == s for r in rows) for s in ["train", "validation", "test"]},
                                         "files": {p.name: digest(p) for p in folder.glob("*.json")}})
    return folder


def load(folder):
    manifest = read_json(folder / "manifest.json")
    for name, sha in manifest["files"].items():
        if digest(folder / name) != sha:
            raise ValueError("Frozen schema data changed")
    rows = read_json(folder / "rows.json")
    for row in rows:
        validate(row["expected"])
    return rows, read_json(folder / "challenge.json")
