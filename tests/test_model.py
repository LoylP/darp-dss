"""Kiểm thử tối ưu hóa DARP và ví dụ minh họa."""

import numpy as np
from scipy import sparse

from darp.model import darp_fit, solve_c, toy_example, update_eta


def test_theorem_one_signed_laplacian_is_positive_semidefinite() -> None:
    rng = np.random.default_rng(42)
    n = 12
    i, j = np.triu_indices(n, 1)
    rho = rng.uniform(0.01, 1, len(i))
    delta = rng.choice([-1.0, 1.0], len(i))
    weights = sparse.coo_matrix((rho * delta, (i, j)), shape=(n, n)).tocsr()
    weights = weights + weights.T
    absolute = sparse.coo_matrix((rho, (i, j)), shape=(n, n)).tocsr()
    absolute = absolute + absolute.T
    laplacian = sparse.diags(np.asarray(absolute.sum(axis=1)).ravel()) - weights

    assert np.linalg.eigvalsh(laplacian.toarray()).min() >= -1e-9


def test_theorem_two_eta_normalization_and_fit_history() -> None:
    i, j = np.triu_indices(5, 1)
    features = np.ones((len(i), 3))
    delta = np.ones(len(i))
    lab = np.array([True, True, False, False, False])
    initial = np.array([1.0, -0.25])
    c = solve_c(5, i, j, np.ones(len(i)), delta, lab, initial)
    eta, _ = update_eta(features, c, i, j, delta)
    fitted, _, history = darp_fit(5, i, j, features, delta, lab, initial)

    assert np.isclose(np.exp(-eta).sum(), 1)
    assert np.all(np.diff(history) <= 1e-8)
    assert np.all((-1 <= fitted) & (fitted <= 1))


def test_slide_fifteen_example_matches_reference() -> None:
    confidence, majority, darp_prediction = toy_example()

    assert np.allclose(confidence[4:], [0.362, -0.075, -0.058], atol=1e-3)
    assert majority == 2
    assert darp_prediction == 4
