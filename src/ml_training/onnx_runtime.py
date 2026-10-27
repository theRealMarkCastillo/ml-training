"""Minimal ONNX inference path: imports only onnxruntime and numpy.

Kept separate from the training module so fresh-process startup measures what
a deployed scorer would load, not scikit-learn, skl2onnx or polars.
"""

import numpy as np
import onnxruntime as ort


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def probabilities(runtime, x):
    return runtime.run(None, {"features": np.asarray(x, dtype=np.float32)})[1][:, 1]
