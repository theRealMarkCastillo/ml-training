"""Install the lab's locked environments listed in configs/environments.json.

Standard library only, so it runs before any environment exists:

    python3 scripts/bootstrap.py                 # every environment
    python3 scripts/bootstrap.py cpu notebooks   # only the ones you need
    python3 scripts/bootstrap.py --repro cpu mlx # second installation for replay checks
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT / "configs" / "environments.json").read_text())["environments"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", metavar="ENVIRONMENT",
                        help=f"Any of: {', '.join(config)}. Default: all.")
    parser.add_argument("--repro", action="store_true", help="Install into .venv-repro instead of .venv")
    args = parser.parse_args()
    if unknown := sorted(set(args.names) - set(config)):
        parser.error(f"unknown environment {', '.join(unknown)}; choose from {', '.join(config)}")
    if shutil.which("uv") is None:
        raise SystemExit("uv is required: https://docs.astral.sh/uv/getting-started/installation/")
    names = args.names or list(config)
    environment = os.environ | ({"UV_PROJECT_ENVIRONMENT": ".venv-repro"} if args.repro else {})
    for name in names:
        project = config[name]["project"]
        print(f"==> {name}: {config[name]['purpose']}", flush=True)
        command = ["uv", "sync", "--frozen"] + ([] if project == "." else ["--project", project])
        # A relative UV_PROJECT_ENVIRONMENT resolves inside each selected project.
        if subprocess.run(command, cwd=ROOT, env=environment).returncode:
            raise SystemExit(f"Installing {name} failed; the uv output above names the cause.")
    if sys.platform == "darwin" and "cpu" in names:
        print("XGBoost on macOS also needs OpenMP: brew install libomp")


if __name__ == "__main__":
    main()
