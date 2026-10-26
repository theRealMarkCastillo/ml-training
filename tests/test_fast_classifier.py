import numpy as np
import pytest

from ml_training.fast_classifier import validate_request


@pytest.mark.parametrize("body", [None, {}, {"features": []}, {"features": [[1] * 7]},
                                  {"features": [[float("nan")] * 8]}, {"features": [[True] * 8]},
                                  {"features": [["1"] * 8]}, {"features": [[1] * 8], "extra": 1}])
def test_reject_bad_requests(body):
    with pytest.raises(ValueError):
        validate_request(body)


def test_accept_finite_fixed_size_vectors():
    actual = validate_request({"features": [[1, .5, 1, 2, 3, 0, 1, 2]]})
    assert actual.shape == (1, 8)
    assert actual.dtype == np.float32
