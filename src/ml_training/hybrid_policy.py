"""Runtime guards and validation-only escalation policy, without test oracles."""

import json
import re

from ml_training.device_data import parse_output, dispatch
from ml_training.query_dsl import parse, equivalent, execute
from ml_training.schema_data import typed_slot, tables


def calibrate(rows, predictions):
    candidates = sorted({0.0, 1.000001, *[p["sequence_confidence"] for p in predictions]})
    choices = []
    for threshold in candidates:
        kept = [
            (row, p)
            for row, p in zip(rows, predictions, strict=True)
            if p["parsed"] is not None and p["parsed"]["intent"] != "escalate" and p["sequence_confidence"] >= threshold
        ]
        accuracy = sum(p["parsed"]["intent"] == row["intent"] for row, p in kept) / len(kept) if kept else None
        if not kept or accuracy >= 0.95:
            choices.append({"threshold": threshold, "retained": len(kept), "retained_accuracy": accuracy})
    selected = min(choices, key=lambda x: (-x["retained"], x["threshold"]))
    return {
        **selected,
        "validation_rows": len(rows),
        "coverage": selected["retained"] / len(rows),
        "rule": "Maximize retained non-escalated validation requests with at least 95% empirical routing accuracy; no retained examples is allowed",
        "limitation": "Twelve synthetic calibration cases; sequence probability is not calibrated correctness",
        "candidates": choices,
    }


def document_guard(doc, request):
    return all(
        doc[key] == request[public]
        for key, public in [("version", "version"), ("family", "resource"), ("operation", "operation")]
    )


def reference(doc):
    # Parse bounded facts from the retrieved corpus page. Text is data, never a
    # source of execution permission or instructions to the assistant.
    endpoint = re.search(r"\b(GET|PATCH) (/v[123]/[a-z-]+/\{resource_id\})", doc["text"])
    field = re.search(r"Responses contain ([a-z_]+) and", doc["text"])
    return {
        "doc_id": doc["doc_id"],
        "method": endpoint[1] if endpoint else None,
        "path": endpoint[2] if endpoint else None,
        "response_field": field[1] if field else "resource_or_operation_id",
        "source_text": doc["text"],
    }


def checked_query(text, user_text, seed=713):
    ast = parse(text)
    # An independent bounded parser can derive a public specification for its
    # supported grammar. No expected answer from the scenario is available here.
    try:
        specification = typed_slot(user_text)
    except ValueError:
        specification = None
    if specification is not None:
        result = equivalent(text, specification, tables(specification, seed, hidden=False))
        if not result["passed"]:
            raise ValueError("Candidate disagrees with the independently parsed public specification")
    from ml_training.schema_data import VALUES

    fixture = [
        {
            "id": f"ticket-{i}",
            "priority": VALUES["priority"][i % 3],
            "status": VALUES["status"][(i // 3) % 3],
            "team": VALUES["team"][(i // 9) % 3],
            "age_hours": i * 3,
            "amount": i * 7,
        }
        for i in range(40)
    ]
    return {
        "ast": ast,
        "result": execute(ast, fixture),
        "rows_examined": len(fixture),
        "semantic_check": "Public typed specification and execution fixtures"
        if specification
        else "Unavailable for this wording; schema/bounds only",
        "source": "Synthetic local support-ticket table",
    }


def checked_device(text, request):
    command = parse_output(text)
    mutation = command["action"] in ["set_temperature", "schedule"]
    if mutation and (request["role"] != "operator" or request["operation"] != "update"):
        raise PermissionError("Requested API context does not authorize a device mutation")
    before = {"celsius": 20, "jobs": []}
    return {"command": command, "before": before, "after": dispatch(command, before), "simulated": True}


def checked_clarification(text):
    value = json.loads(text)
    if (
        not isinstance(value, dict)
        or set(value) != {"question"}
        or not isinstance(value["question"], str)
        or not 4 <= len(value["question"]) <= 600
    ):
        raise ValueError("Invalid clarification response")
    return value
