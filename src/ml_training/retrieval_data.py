"""Frozen synthetic versioned API corpus with disjoint resource/query families."""

from __future__ import annotations

import re
from pathlib import Path

from ml_training.common import ROOT, digest, new_dataset_folder, read_json, write_json

RESOURCES = [
    ("support-ticket", "support ticket", "customer case"),
    ("invoice", "invoice", "billing statement"),
    ("refund", "refund", "payment reversal"),
    ("workspace", "workspace", "team environment"),
    ("member", "member", "team participant"),
    ("role", "role", "permission bundle"),
    ("api-key", "API key", "service credential"),
    ("session-token", "session token", "login credential"),
    ("subscription", "subscription", "recurring plan"),
    ("quota", "quota", "usage allowance"),
    ("attachment", "attachment", "uploaded file"),
    ("comment", "comment", "case note"),
    ("notification", "notification", "delivery alert"),
    ("queue", "queue", "pending work list"),
    ("agent-job", "agent job", "automated task"),
    ("event-stream", "event stream", "activity feed"),
    ("integration", "integration", "external connection"),
    ("contact", "contact", "customer record"),
    ("organization", "organization", "company account"),
    ("report", "report", "analytics summary"),
    ("dashboard", "dashboard", "metrics view"),
    ("payment-method", "payment method", "billing instrument"),
    ("schedule", "schedule", "execution timetable"),
    ("retention-policy", "retention policy", "record lifespan setting"),
    ("access-grant", "access grant", "authorization assignment"),
    ("upload-session", "upload session", "file transfer reservation"),
    ("search-index", "search index", "document lookup collection"),
    ("feature-flag", "feature flag", "release toggle"),
    ("service-account", "service account", "machine identity"),
    ("routing-rule", "routing rule", "dispatch condition"),
    ("billing-profile", "billing profile", "invoice preferences"),
    ("usage-counter", "usage counter", "consumption meter"),
    ("backup-job", "backup job", "recovery copy task"),
    ("incident", "incident", "service interruption record"),
    ("automation-rule", "automation rule", "workflow trigger"),
    ("device-thermostat", "device thermostat", "temperature controller"),
    ("audit-export", "audit export", "compliance event archive"),
    ("webhook-secret", "webhook secret", "callback signing credential"),
    ("session-lease", "session lease", "temporary login reservation"),
    ("deployment-slot", "deployment slot", "application release position"),
]
TEMPLATES = {
    "train": [
        "For {version}, how do I {verb} a {name}?",
        "Find the {version} endpoint to {verb} the {alias}.",
        "Which {name} API in {version} lets me {verb} its configuration?",
        "Our client uses {version}; I need to {verb} the {alias} by ID.",
        "Show the {version} reference for {verb} operations on {name}.",
    ],
    "validation": [
        "I am integrating {alias} in {version}. Point me to the operation that will {verb} it.",
        "Locate documentation for {verb} of {name} using {version}, not another release.",
        "What is the supported route for {verb} on {name}, version {number}?",
    ],
    "test": [
        "The SDK is pinned to version {number}. I want to {verb} a {alias}; which page applies?",
        "We must {verb} our {name} through {version}. Please pick the relevant operation reference.",
        "A colleague sent the wrong release's instructions for {name}. Locate {version} so I can {verb} it.",
        "Does the {version} API document how to {verb} a {alias}? Retrieve that exact page.",
    ],
}


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def build(profile: str):
    resources = RESOURCES if profile == "local" else [*RESOURCES[:4], RESOURCES[28], RESOURCES[34]]
    train_count, val_count = (28, 6) if profile == "local" else (4, 1)
    corpus, queries = [], []
    for index, (slug, name, alias) in enumerate(resources):
        split = "train" if index < train_count else "validation" if index < train_count + val_count else "test"
        for version in ["v1", "v2", "v3"]:
            for operation, verb, method in [("read", "inspect", "GET"), ("update", "modify", "PATCH")]:
                doc_id = f"{slug}-{version}-{operation}"
                behavior = {
                    "v1": "Legacy API. Authenticate using X-Harbor-Key. Address the record with id. Responses contain data and a numeric status. This version is still supported for pinned clients.",
                    "v2": "Stable API. Authenticate with an OAuth bearer token. Address the record with resource_id. Responses contain result and request_id. Use an idempotency key for updates.",
                    "v3": "Preview API. Authenticate with a scoped bearer token. Address the record with resource_id and revision. Updates return an asynchronous operation_id; reads return the current resource plus revision.",
                }[version]
                action = (
                    "Read current settings and status without changing state. This operation cannot edit or change the record."
                    if operation == "read"
                    else "Change selected settings on an existing record. This operation modifies state; use GET to inspect without changing it."
                )
                text = (
                    f"Harbor Support {version} API reference: {name} / {alias}. "
                    f"Operation: {verb} ({operation}). {method} /{version}/{slug}/{{resource_id}}. "
                    f"{action} {behavior} This page applies only to {version}; consult a different version's page for another client release."
                )
                corpus.append(
                    {
                        "doc_id": doc_id,
                        "family": slug,
                        "split": split,
                        "version": version,
                        "operation": operation,
                        "text": text,
                    }
                )
                for template_id, template in enumerate(TEMPLATES[split]):
                    query = template.format(version=version, number=version[-1], verb=verb, name=name, alias=alias)
                    queries.append(
                        {
                            "query_id": f"{doc_id}-q{template_id}",
                            "family": slug,
                            "split": split,
                            "template_family": f"{split}-{template_id}",
                            "version": version,
                            "operation": operation,
                            "text": query,
                            "positive_id": doc_id,
                        }
                    )
    return corpus, queries


def audit(corpus, queries):
    docs = {d["doc_id"]: d for d in corpus}
    if len(docs) != len(corpus) or len({q["query_id"] for q in queries}) != len(queries):
        raise ValueError("Duplicate identifiers")
    if len({q["text"] for q in queries}) != len(queries):
        raise ValueError("Duplicate query text")
    seen_families, seen_templates = set(), set()
    report = {}
    for split in ["train", "validation", "test"]:
        selected = [q for q in queries if q["split"] == split]
        families = {q["family"] for q in selected}
        templates = {q["template_family"] for q in selected}
        if families & seen_families or templates & seen_templates:
            raise ValueError("Query/document family leakage")
        seen_families |= families
        seen_templates |= templates
        for q in selected:
            positive = docs[q["positive_id"]]
            if any(positive[k] != q[k] for k in ["family", "split", "version", "operation"]):
                raise ValueError("Invalid relevance judgment")
        report[split] = {
            "queries": len(selected),
            "families": sorted(families),
            "documents": sum(d["split"] == split for d in corpus),
            "templates": sorted(templates),
        }
    return report


def prepare(profile: str, seed: int) -> Path:
    folder = ROOT / "data" / "generated" / f"retrieval-v1-{profile}-s{seed}"
    if (folder / "manifest.json").exists():
        load(folder)
        return folder
    new_dataset_folder(folder)
    corpus, queries = build(profile)
    checks = audit(corpus, queries)
    write_json(folder / "corpus.json", corpus)
    write_json(folder / "queries.json", queries)
    # These were authored separately from the generator; smoke only uses available families.
    fixtures = read_json(ROOT / "data" / "fixtures" / "retrieval_challenge.json")
    doc_ids = {d["doc_id"] for d in corpus}
    fixtures = [q for q in fixtures if q["positive_id"] in doc_ids]
    write_json(folder / "challenge.json", fixtures)
    write_json(
        folder / "manifest.json",
        {
            "version": "retrieval-v1",
            "synthetic": True,
            "profile": profile,
            "seed": seed,
            "audit": checks,
            "seed_note": "Corpus is fixed; seed controls model training and sampling.",
            "relevance": "One correct document per explicit resource, version, operation tuple.",
            "files": {p.name: digest(p) for p in folder.glob("*.json")},
        },
    )
    return folder


def load(folder):
    manifest = read_json(folder / "manifest.json")
    for filename, sha in manifest["files"].items():
        if digest(folder / filename) != sha:
            raise ValueError(f"Frozen retrieval data changed: {filename}")
    corpus, queries = read_json(folder / "corpus.json"), read_json(folder / "queries.json")
    audit(corpus, queries)
    return corpus, queries, read_json(folder / "challenge.json")
