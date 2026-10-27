"""Child process used to measure fresh-process startup and first inference."""

import json
import sys

import joblib


def main():
    backend, model_path, input_path = sys.argv[1:]
    x = joblib.load(input_path)
    if backend == "joblib":
        result = joblib.load(model_path).predict_proba(x)[:, 1]
    elif backend == "onnx":
        from ml_training.onnx_runtime import probabilities, session
        result = probabilities(session(model_path), x)
    else:
        raise ValueError("Unknown backend")
    print(json.dumps(result.tolist()))


if __name__ == "__main__":
    main()
