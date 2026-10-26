"""Command entry point. Unimplemented tracks fail explicitly."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from ml_training.common import ROOT, environment, read_json, verify_run, write_json

DEMOS = {"distillation": "D01", "schema_dsl": "D02", "rlvr": "D03", "edge": "D04",
         "tabular": "D05", "fast_classifier": "D06", "explainability": "D07",
         "retrieval": "D08", "hybrid": "D09"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["list", "doctor", "prepare", "train", "evaluate", "run", "verify", "report"])
    parser.add_argument("demo", nargs="?", choices=[*DEMOS, "all"])
    parser.add_argument("--profile", choices=["smoke", "local"], default="smoke")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--source-run", type=Path, help="D05 for explanations, D02 SFT for RLVR, or D03 RLVR for the hybrid")
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--set", action="append", default=[], metavar="FEATURE=VALUE")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--query", help="Search a trained retrieval demo")
    parser.add_argument("--variant", default="selected")
    parser.add_argument("--request-json", type=Path, help="Public request/context for fresh hybrid execution")
    parser.add_argument("--output-dir", type=Path, help="New directory for hybrid replay evidence")
    parser.add_argument("--clean-workers", action="store_true", help="Replay hybrid with locked .venv-repro model environments")
    args = parser.parse_args()
    if args.action == "list":
        for name, identifier in DEMOS.items():
            print(f"{identifier} {name}: implemented (evaluation status: docs/PROGRESS.md)")
    elif args.action == "doctor":
        result = environment()
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
        print(json.dumps(result, indent=2))
    elif args.action == "verify":
        if args.run_dir:
            verify_run(args.run_dir)
            print("Artifact checksums verified")
        else:
            raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "-q"], cwd=ROOT))
    elif args.action == "report":
        from ml_training.reporting import report_all
        print(report_all())
    elif args.demo == "tabular":
        from ml_training import data, tabular
        if args.action == "prepare":
            print(data.prepare(args.profile, args.seed))
        elif args.action in ["train", "run"]:
            print(tabular.train(args.profile, args.seed))
        elif args.action == "evaluate":
            if not args.run_dir:
                parser.error("evaluate requires --run-dir")
            print(json.dumps(tabular.reevaluate(args.run_dir), indent=2))
    elif args.demo == "fast_classifier":
        from ml_training import fast_classifier
        if args.action == "prepare":
            print(fast_classifier.prepare(args.profile, args.seed))
        elif args.action in ["train", "run"]:
            print(fast_classifier.train(args.profile, args.seed))
        elif args.action == "evaluate":
            if not args.run_dir:
                parser.error("evaluate requires --run-dir")
            print(json.dumps(fast_classifier.reevaluate(args.run_dir), indent=2))
    elif args.demo == "distillation":
        if args.action == "run" and args.run_dir:
            if not args.query:
                parser.error("run with --run-dir requires --query")
            from ml_training.distillation import infer_query
            print(json.dumps(infer_query(args.run_dir, args.query, args.variant), indent=2))
        elif args.action == "prepare":
            from ml_training.routing_data import prepare
            print(prepare(args.profile, args.seed))
        elif args.action in ["train", "run"]:
            from ml_training.distillation import train
            print(train(args.profile, args.seed))
        elif args.action == "evaluate":
            if not args.run_dir:
                parser.error("evaluate requires --run-dir")
            from ml_training.distillation import reevaluate
            print(json.dumps(reevaluate(args.run_dir), indent=2))
    elif args.demo in ["schema_dsl", "edge"]:
        from ml_training import structured
        if args.action == "prepare":
            print(structured.task(args.demo)[0].prepare(args.profile, args.seed))
        elif args.action == "run" and args.run_dir:
            if not args.query:
                parser.error("run with --run-dir requires --query")
            print(json.dumps(structured.infer_query(args.run_dir, args.query, args.variant), indent=2))
        elif args.action in ["train", "run"]:
            print(structured.train(args.demo, args.profile, args.seed))
        elif args.action == "evaluate":
            if not args.run_dir:
                parser.error("evaluate requires --run-dir")
            print(json.dumps(structured.reevaluate(args.run_dir), indent=2))
    elif args.demo == "rlvr":
        from ml_training import rlvr
        if args.action == "run" and args.run_dir:
            if not args.query:
                parser.error("run with --run-dir requires --query")
            print(json.dumps(rlvr.infer_query(args.run_dir, args.query, args.variant, args.device), indent=2))
        elif args.action in ["train", "run", "prepare"]:
            if not args.source_run:
                parser.error("RLVR requires --source-run pointing to a completed D02 SFT run")
            if args.action == "prepare":
                from ml_training.rlvr_data import prepare
                print(prepare(args.profile, args.seed, args.source_run))
            else:
                print(rlvr.train(args.source_run, args.profile, args.seed, args.device))
        elif args.action == "evaluate":
            if not args.run_dir:
                parser.error("evaluate requires --run-dir")
            print(json.dumps(rlvr.reevaluate(args.run_dir, args.device), indent=2))
    elif args.demo == "hybrid":
        from ml_training import hybrid
        if args.action == "prepare":
            print(hybrid.data.prepare())
        elif args.action == "run" and args.run_dir:
            if not args.request_json:
                parser.error("Fresh hybrid execution requires --request-json with public context")
            print(json.dumps(hybrid.infer_request(args.run_dir,read_json(args.request_json),args.variant),indent=2))
        elif args.action in ["train","run"]:
            if not args.source_run:
                parser.error("Hybrid comparison requires --source-run pointing to a completed D03 RLVR run")
            print(hybrid.run(args.source_run,args.profile,args.seed))
        elif args.action == "evaluate":
            if not args.run_dir or not args.output_dir:
                parser.error("Hybrid replay requires --run-dir and a new --output-dir")
            print(json.dumps(hybrid.reevaluate(args.run_dir,args.output_dir,args.clean_workers),indent=2))
    elif args.demo == "retrieval":
        if args.action == "run" and args.run_dir:
            if not args.query:
                parser.error("run retrieval --run-dir requires --query")
            from ml_training.retrieval import search
            print(json.dumps(search(args.run_dir, args.query, args.device, args.variant), indent=2))
        elif args.action == "prepare":
            from ml_training.retrieval_data import prepare
            print(prepare(args.profile, args.seed))
        elif args.action in ["train", "run"]:
            from ml_training.retrieval import train
            print(train(args.profile, args.seed, args.device))
        elif args.action == "evaluate":
            if not args.run_dir:
                parser.error("evaluate requires --run-dir")
            from ml_training.retrieval import reevaluate
            print(json.dumps(reevaluate(args.run_dir, args.device), indent=2))
    elif args.demo == "explainability":
        from ml_training import explainability
        if args.action == "run":
            if not args.source_run and not args.run_dir:
                parser.error("run explainability requires --source-run (new report) or --run-dir (fresh feature edit)")
            changes = {}
            for assignment in args.set:
                key, separator, value = assignment.partition("=")
                if not separator:
                    parser.error("--set expects FEATURE=VALUE")
                changes[key] = value if key == "merchant" else float(value)
            if args.run_dir:
                print(json.dumps(explainability.infer_edit(args.run_dir,args.row_index,changes),indent=2))
            else:
                print(explainability.run(args.source_run.resolve(), args.row_index, changes or None))
        elif args.action == "evaluate":
            if not args.run_dir:
                parser.error("evaluate requires --run-dir")
            print(json.dumps(explainability.reevaluate(args.run_dir), indent=2))
        else:
            parser.error("Explainability consumes trained D05 models; use run or evaluate")
    else:
        parser.error(f"{args.demo or 'A demo'} is not implemented yet; see docs/PROGRESS.md")


if __name__ == "__main__":
    main()
