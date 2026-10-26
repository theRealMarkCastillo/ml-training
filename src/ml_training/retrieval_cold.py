"""Fresh-process first inference for the two saved retrieval checkpoints."""

import json
from pathlib import Path
import sys

from ml_training.common import ROOT, read_json
from ml_training import retrieval_data
from ml_training.retrieval_models import configure, embed, load_bi, load_cross, score_pairs


def main():
    run, kind, requested = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    info = read_json(run / "run.json")
    device = configure(info["seed"], requested)
    corpus, queries, _ = retrieval_data.load(ROOT / info["dataset_path"])
    query = next(q for q in queries if q["split"] == "test")["text"]
    if kind == "bi_encoder":
        model = load_bi(device, run / kind)
        result = embed(model, [query]).shape
    elif kind == "cross_encoder":
        model = load_cross(device, run / kind)
        result = score_pairs(model, [(query, corpus[i]["text"]) for i in range(10)]).tolist()
    else:
        raise ValueError("Unknown model kind")
    print(json.dumps({"kind": kind, "result": result}))


if __name__ == "__main__":
    main()
