"""Prepare, train, evaluate and query the lab's demos.

`run` trains a demo, or with --run-dir executes a fresh request against a saved run.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from ml_training.common import ROOT, environment, environments, read_json, verify_run, write_json

DEMOS = {"distillation": "D01", "schema_dsl": "D02", "rlvr": "D03", "edge": "D04",
         "tabular": "D05", "fast_classifier": "D06", "explainability": "D07",
         "retrieval": "D08", "hybrid": "D09"}
ACTIONS = ["list", "doctor", "prepare", "train", "evaluate", "run", "verify", "report"]


def required(args, fail, field, message):
    value = getattr(args, field)
    if not value:
        fail(message)
    return value


def run_dir(args, fail):
    return required(args, fail, "run_dir", f"{args.action} {args.demo} requires --run-dir")


def query(args, fail):
    return required(args, fail, "query", f"run {args.demo} --run-dir requires --query")


def feature_edits(args, fail):
    changes = {}
    for assignment in args.set:
        key, separator, value = assignment.partition("=")
        if not separator:
            fail("--set expects FEATURE=VALUE")
        changes[key] = value if key == "merchant" else float(value)
    return changes


# Each demo maps actions to handlers taking (args, fail). "infer" serves `run --run-dir`;
# plain `run` is an alias for "train". Imports stay inside handlers so each demo only
# loads its own environment's libraries.

def tabular_handlers():
    from ml_training import data, tabular
    return {"prepare": lambda a, f: data.prepare(a.profile, a.seed),
            "train": lambda a, f: tabular.train(a.profile, a.seed),
            "evaluate": lambda a, f: tabular.reevaluate(run_dir(a, f), a.tolerance)}


def fast_classifier_handlers():
    from ml_training import fast_classifier
    return {"prepare": lambda a, f: fast_classifier.prepare(a.profile, a.seed),
            "train": lambda a, f: fast_classifier.train(a.profile, a.seed),
            "evaluate": lambda a, f: fast_classifier.reevaluate(run_dir(a, f), a.tolerance)}


def explainability_handlers():
    from ml_training import explainability
    source = "explainability consumes a trained D05 run: pass --source-run (new report) or --run-dir (fresh feature edit)"
    return {"train": lambda a, f: explainability.run(required(a, f, "source_run", source).resolve(), a.row_index,
                                                     feature_edits(a, f) or None),
            "infer": lambda a, f: explainability.infer_edit(a.run_dir, a.row_index, feature_edits(a, f)),
            "evaluate": lambda a, f: explainability.reevaluate(run_dir(a, f), a.tolerance)}


def distillation_handlers():
    def prepare(a, f):
        from ml_training.routing_data import prepare
        return prepare(a.profile, a.seed)
    def train(a, f):
        from ml_training.distillation import train
        return train(a.profile, a.seed)
    def evaluate(a, f):
        from ml_training.distillation import reevaluate
        return reevaluate(run_dir(a, f), a.tolerance)
    def infer(a, f):
        from ml_training.distillation import infer_query
        return infer_query(a.run_dir, query(a, f), a.variant)
    return {"prepare": prepare, "train": train, "evaluate": evaluate, "infer": infer}


def structured_handlers():
    from ml_training import structured
    return {"prepare": lambda a, f: structured.task(a.demo)[0].prepare(a.profile, a.seed),
            "train": lambda a, f: structured.train(a.demo, a.profile, a.seed),
            "evaluate": lambda a, f: structured.reevaluate(run_dir(a, f), a.tolerance),
            "infer": lambda a, f: structured.infer_query(a.run_dir, query(a, f), a.variant)}


def rlvr_handlers():
    from ml_training import rlvr
    source = "RLVR requires --source-run pointing to a completed D02 SFT run"
    def prepare(a, f):
        from ml_training.rlvr_data import prepare
        return prepare(a.profile, a.seed, required(a, f, "source_run", source))
    return {"prepare": prepare,
            "train": lambda a, f: rlvr.train(required(a, f, "source_run", source), a.profile, a.seed, a.device),
            "evaluate": lambda a, f: rlvr.reevaluate(run_dir(a, f), a.device, a.tolerance),
            "infer": lambda a, f: rlvr.infer_query(a.run_dir, query(a, f), a.variant, a.device)}


def retrieval_handlers():
    def prepare(a, f):
        from ml_training.retrieval_data import prepare
        return prepare(a.profile, a.seed)
    def train(a, f):
        from ml_training.retrieval import train
        return train(a.profile, a.seed, a.device)
    def evaluate(a, f):
        from ml_training.retrieval import reevaluate
        return reevaluate(run_dir(a, f), a.device, a.tolerance)
    def infer(a, f):
        from ml_training.retrieval import search
        return search(a.run_dir, query(a, f), a.device, a.variant)
    return {"prepare": prepare, "train": train, "evaluate": evaluate, "infer": infer}


def hybrid_handlers():
    from ml_training import hybrid
    source = "The hybrid comparison requires --source-run pointing to a completed D03 RLVR run"
    def prepare(a, f):
        if not a.auto_select:
            return hybrid.data.prepare()
        from ml_training.run_selection import auto_select
        return {"selected_runs": auto_select(a.source_run), "dataset": str(hybrid.data.prepare())}
    def evaluate(a, f):
        if not a.run_dir or not a.output_dir:
            f("Hybrid replay requires --run-dir and a new --output-dir")
        return hybrid.reevaluate(a.run_dir, a.output_dir, a.clean_workers, a.tolerance)
    def infer(a, f):
        request = required(a, f, "request_json", "Fresh hybrid execution requires --request-json with public context")
        return hybrid.infer_request(a.run_dir, read_json(request), a.variant)
    return {"prepare": prepare, "evaluate": evaluate, "infer": infer,
            "train": lambda a, f: hybrid.run(required(a, f, "source_run", source), a.profile, a.seed)}


HANDLERS = {"tabular": tabular_handlers, "fast_classifier": fast_classifier_handlers,
            "explainability": explainability_handlers, "distillation": distillation_handlers,
            "schema_dsl": structured_handlers, "edge": structured_handlers, "rlvr": rlvr_handlers,
            "retrieval": retrieval_handlers, "hybrid": hybrid_handlers}


def list_demos():
    config = environments()
    for name, identifier in DEMOS.items():
        env = config["demos"][name]
        project = ROOT / config["environments"][env]["project"]
        installed = "installed" if (project / ".venv" / "bin" / "python").is_file() else "not installed"
        evaluated = 0
        for path in (ROOT / "artifacts" / name).glob("local-*/run.json"):
            try:
                evaluated += read_json(path)["status"] == "evaluated"
            except (OSError, ValueError, KeyError):
                pass
        print(f"{identifier} {name:<16} environment={env} ({installed}), evaluated local runs: {evaluated}")


def doctor():
    result = environment()
    config = environments()["environments"]
    result["lab_environments"] = {name: (ROOT / item["project"] / ".venv" / "bin" / "python").is_file()
                                  for name, item in config.items()}
    if sys.platform == "darwin":
        try:
            raw = subprocess.run(["system_profiler", "SPHardwareDataType", "-json"],
                                 capture_output=True, text=True, check=True, timeout=45)
            hardware = json.loads(raw.stdout)["SPHardwareDataType"][0]
            result["hardware"] = {key: hardware.get(key) for key in
                                  ["chip_type", "physical_memory", "number_processors"]}
        except (subprocess.SubprocessError, KeyError, ValueError) as error:
            result["hardware_probe_error"] = str(error)
    write_json(ROOT / "artifacts" / "environment.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=ACTIONS)
    parser.add_argument("demo", nargs="?", choices=[*DEMOS, "all"])
    parser.add_argument("--profile", choices=["smoke", "local"], default="smoke")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--source-run", type=Path, help="D05 for explanations, D02 SFT for RLVR, or D03 RLVR for the hybrid")
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--set", action="append", default=[], metavar="FEATURE=VALUE")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--query", help="Request text for a fresh run against --run-dir")
    parser.add_argument("--variant", default="selected")
    parser.add_argument("--request-json", type=Path, help="Public request/context for fresh hybrid execution")
    parser.add_argument("--output-dir", type=Path, help="New directory for hybrid replay evidence")
    parser.add_argument("--clean-workers", action="store_true", help="Replay hybrid with locked .venv-repro model environments")
    parser.add_argument("--tolerance", type=float, default=0.,
                        help="evaluate: accept numeric drift up to this absolute difference (default: exact)")
    parser.add_argument("--auto-select", action="store_true",
                        help="prepare hybrid: bind the latest consistent evaluated component runs")
    args = parser.parse_args()
    fail = parser.error
    if args.action == "list":
        return list_demos()
    if args.action == "doctor":
        return print(json.dumps(doctor(), indent=2))
    if args.action == "report":
        from ml_training.reporting import report_all
        return print(report_all())
    if args.action == "verify":
        if args.run_dir:
            verify_run(args.run_dir)
            return print("Artifact checksums verified")
        if args.demo not in [None, "all"]:
            fail(f"verify {args.demo} checks one saved run: pass --run-dir. Use `verify all` for the contract test suite.")
        raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "-q"], cwd=ROOT))
    if args.demo is None:
        fail(f"{args.action} needs a demo: choose from {', '.join(DEMOS)}")
    if args.demo == "all":
        fail(f"{args.action} all is not supported: demos depend on each other's runs. "
             "Follow the order in docs/REPRODUCTION.md.")
    try:
        handlers = HANDLERS[args.demo]()
        action = args.action
        if action == "run":
            action = "infer" if args.run_dir and "infer" in handlers else "train"
        if action not in handlers:
            supported = [name for name in ["prepare", "train", "evaluate"] if name in handlers] + ["run"]
            fail(f"{DEMOS[args.demo]} {args.demo} does not support {args.action}; it supports: {', '.join(supported)}")
        result = handlers[action](args, fail)
    except ModuleNotFoundError as error:
        if (error.name or "").startswith("ml_training"):
            raise
        env = environments()["demos"][args.demo]
        project = environments()["environments"][env]["project"]
        fail(f"{args.demo} runs in the {env} environment, and {error.name} is not installed in this one. "
             f"Use {Path(project) / '.venv/bin/ml-demo'} (install with: python3 scripts/bootstrap.py {env}).")
    print(result if isinstance(result, (str, Path)) else json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
