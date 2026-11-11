"""Kiểm thử Table 2 và các thước đo."""

import numpy as np

from darp.metrics import COST, acc_cost, bootstrap_ci, init_reliability, return_to_label


def test_cost_matrix_and_initial_reliability() -> None:
    assert np.allclose(COST, COST.T)
    assert init_reliability([2], [4])[0] == -0.25


def test_perfect_predictions_have_unit_accuracy() -> None:
    assert acc_cost([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0


def test_return_to_label_boundaries() -> None:
    values = [-0.1001, -0.1, -1e-12, 0, 0.1, 0.2]
    assert return_to_label(values).tolist() == [1, 2, 2, 3, 4, 5]


def test_bootstrap_ci_is_reproducible() -> None:
    first = bootstrap_ci([1, 2, 3], [1, 2, 4], n_boot=20, seed=4)
    assert first == bootstrap_ci([1, 2, 3], [1, 2, 4], n_boot=20, seed=4)
