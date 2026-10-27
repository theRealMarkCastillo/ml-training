"""Fuse a verified D02 adapter into a locally reloadable, unquantized checkpoint."""

import argparse
from pathlib import Path

from mlx.utils import tree_unflatten
from mlx_lm import load
from mlx_lm.utils import save

from ml_training.common import ROOT, digest, read_json, verify_run, write_json
from ml_training.mlx_models import infer_rows, path
from ml_training.schema_data import load as load_data

parser = argparse.ArgumentParser()
parser.add_argument("source_run", type=Path)
parser.add_argument("destination", type=Path)
args = parser.parse_args()
verify_run(args.source_run)
if args.destination.exists():
    raise ValueError("Refusing to overwrite a fused checkpoint")
model, tokenizer, config = load(str(path()), adapter_path=str(args.source_run / "adapter"), return_config=True)
rows, _ = load_data(ROOT / read_json(args.source_run / "run.json")["dataset_path"])
validation = [r for r in rows if r["split"] == "validation"][:8]
before = infer_rows(model, tokenizer, validation, max_tokens=160)
fused = [(name, module.fuse()) for name, module in model.named_modules() if hasattr(module, "fuse")]
if not fused:
    raise AssertionError("Missing task adapter")
model.update_modules(tree_unflatten(fused))
save(args.destination, path(), model, tokenizer, config, donate_model=False)
restored, restored_tokenizer = load(str(args.destination))
after = infer_rows(restored, restored_tokenizer, validation, max_tokens=160)
write_json(
    args.destination / "fusion_evidence.json",
    {
        "source_run": str(args.source_run.resolve()),
        "source_manifest_sha256": digest(args.source_run / "run.json"),
        "fused_modules": len(fused),
        "validation_rows": validation,
        "before": before,
        "after": after,
        "exact_generation_match": [x["text"] for x in before] == [x["text"] for x in after],
        "note": "Fusion changes floating-point operation order; D03 baselines and both training controls use this identical saved fused checkpoint.",
    },
)
print(args.destination)
