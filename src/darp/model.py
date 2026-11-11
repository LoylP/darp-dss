"""Tối ưu hóa độ tin cậy DARP."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def solve_c(
    n: int,
    i: np.ndarray,
    j: np.ndarray,
    rho: np.ndarray,
    delta: np.ndarray,
    lab: np.ndarray,
    c_lab: np.ndarray,
    ridge: float = 1e-6,
) -> np.ndarray:
    """Giải bước c của QP lồi; nghiệm tự thỏa ràng buộc hộp.

    Với rho không âm và delta thuộc {-1, +1}, mỗi cạnh đóng góp
    rho * (c_i - delta*c_j)^2 nên Laplacian D-W nửa xác định dương.
    Điều kiện tối ưu trên nút chưa gán nhãn là hệ tuyến tính thưa; nghiệm
    là trung bình có trọng số của các nút kề đã gán, nên thuộc [-1, 1].
    Lựa chọn cài đặt (bài báo không nêu): thêm ridge để ổn định hệ tuyến tính.
    """
    labeled = np.asarray(lab, dtype=bool)
    values = np.asarray(c_lab, dtype=float)
    if labeled.shape != (n,) or values.shape != (int(labeled.sum()),):
        raise ValueError("lab phải có n phần tử và c_lab phải ứng với nút đã gán.")
    if ridge < 0 or np.any(rho < 0) or not np.isin(delta, [-1, 1]).all():
        raise ValueError("ridge, rho hoặc delta nằm ngoài miền hợp lệ.")
    weights = sparse.coo_matrix((rho * delta, (i, j)), shape=(n, n)).tocsr()
    weights = weights + weights.T
    absolute_weights = sparse.coo_matrix((rho, (i, j)), shape=(n, n)).tocsr()
    absolute_weights = absolute_weights + absolute_weights.T
    degree = np.asarray(absolute_weights.sum(axis=1)).ravel()
    laplacian = sparse.diags(degree) - weights
    confidence = np.zeros(n)
    confidence[labeled] = values
    unknown = np.flatnonzero(~labeled)
    if len(unknown) == 0:
        return confidence
    labeled_ids = np.flatnonzero(labeled)
    scale = max(1.0, float(degree[unknown].mean())) if len(unknown) else 1.0
    system = laplacian[unknown][:, unknown] + sparse.identity(len(unknown)) * ridge * scale
    rhs = weights[unknown][:, labeled_ids] @ values
    confidence[unknown] = np.clip(spsolve(system.tocsc(), rhs), -1, 1)
    return confidence


def update_eta(
    F: np.ndarray,
    c: np.ndarray,
    i: np.ndarray,
    j: np.ndarray,
    delta: np.ndarray,
    active: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cập nhật eta theo nghiệm đóng Theorem 2 và trả Jp."""
    features = np.asarray(F, dtype=float)
    enabled = np.ones(features.shape[1], dtype=bool) if active is None else np.asarray(active, dtype=bool)
    if enabled.shape != (features.shape[1],) or not enabled.any():
        raise ValueError("active phải bật ít nhất một đặc trưng.")
    residual_squared = (c[i] - delta * c[j]) ** 2
    Jp = np.maximum(features.T @ residual_squared, 1e-12)
    active_cost = np.where(enabled, Jp, 0.0)
    eta = np.where(enabled, -np.log(active_cost / active_cost.sum() + 1e-300), np.inf)
    eta = np.minimum(eta, 50)
    return eta, Jp


def darp_fit(
    n: int,
    i: np.ndarray,
    j: np.ndarray,
    F: np.ndarray,
    delta: np.ndarray,
    lab: np.ndarray,
    c_lab: np.ndarray,
    max_iter: int = 30,
    tol: float = 1e-6,
    active: np.ndarray | None = None,
    ridge: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Chạy Algorithm 1 với cập nhật luân phiên c và eta.

    Lựa chọn cài đặt (bài báo không nêu): `max_iter` và `tol` giới hạn
    thời gian tối ưu hóa cho cấu hình dữ liệu mô phỏng.
    """
    features = np.asarray(F, dtype=float)
    enabled = np.ones(features.shape[1], dtype=bool) if active is None else np.asarray(active, dtype=bool)
    if enabled.shape != (features.shape[1],) or not enabled.any():
        raise ValueError("active phải bật ít nhất một đặc trưng.")
    if max_iter < 1 or tol < 0:
        raise ValueError("max_iter phải dương và tol phải không âm.")
    eta = np.where(enabled, np.log(int(enabled.sum())), np.inf)
    history: list[float] = []
    previous: float | None = None
    for _ in range(max_iter):
        eta_for_weights = np.where(enabled, eta, 0.0)
        rho = features @ eta_for_weights
        keep = rho > 0
        c = solve_c(n, i[keep], j[keep], rho[keep], delta[keep], lab, c_lab, ridge)
        residual_squared = (c[i] - delta * c[j]) ** 2
        Jp = np.maximum(features.T @ residual_squared, 1e-12)
        objective = float(eta_for_weights @ Jp)
        history.append(objective)
        eta, _ = update_eta(features, c, i, j, delta, enabled)
        if previous is not None and abs(previous - objective) <= tol * max(1.0, abs(previous)):
            break
        previous = objective
    return c, np.where(enabled, eta, np.nan), history


def solve_c_fixed_eta(
    n: int,
    i: np.ndarray,
    j: np.ndarray,
    F: np.ndarray,
    delta: np.ndarray,
    lab: np.ndarray,
    c_lab: np.ndarray,
    eta: np.ndarray,
    ridge: float = 1e-6,
) -> np.ndarray:
    """Suy luận c một lần với eta đã học và nhãn train cố định."""
    rho = np.asarray(F, dtype=float) @ np.asarray(eta, dtype=float)
    keep = rho > 0
    return solve_c(n, i[keep], j[keep], rho[keep], delta[keep], lab, c_lab, ridge)


def DARP(
    n: int,
    i: np.ndarray,
    j: np.ndarray,
    F: np.ndarray,
    delta: np.ndarray,
    lab: np.ndarray,
    c_lab: np.ndarray,
    **kwargs: object,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Tên tương thích của bộ ước lượng DARP."""
    return darp_fit(n, i, j, F, delta, lab, c_lab, **kwargs)


def _toy_data() -> tuple[list[tuple[int, str, str, int, int, int | None]], np.ndarray]:
    rows = [
        (0, "a1", "X", 7372, 5, 5),
        (0, "a4", "Z", 7372, 3, 5),
        (1, "a2", "Y", 7371, 2, 4),
        (1, "a4", "Z", 7371, 4, 4),
        (2, "a1", "X", 7373, 4, None),
        (2, "a2", "Y", 7373, 2, None),
        (2, "a3", "Y", 7373, 2, None),
    ]
    events = np.array([row[0] for row in rows])
    return rows, events


def toy_example() -> tuple[np.ndarray, int, int]:
    """Tính ví dụ slide 15, trả về c, dự đoán MV và dự đoán DARP."""
    from darp.metrics import init_reliability

    rows, _ = _toy_data()
    ratings = np.array([row[4] for row in rows])
    truth = np.array([np.nan if row[5] is None else row[5] for row in rows])
    labeled = ~np.isnan(truth)
    c_labeled = init_reliability(ratings[labeled], truth[labeled].astype(int))
    i, j = np.triu_indices(len(rows), 1)
    features = np.stack(
        [
            np.array([rows[left][1] == rows[right][1] for left, right in zip(i, j)], dtype=float),
            np.array([rows[left][2] == rows[right][2] for left, right in zip(i, j)], dtype=float),
            np.array([rows[left][3] // 100 == rows[right][3] // 100 for left, right in zip(i, j)], dtype=float),
            np.array([rows[left][3] // 10 == rows[right][3] // 10 for left, right in zip(i, j)], dtype=float),
            np.array([rows[left][3] == rows[right][3] for left, right in zip(i, j)], dtype=float),
            np.zeros(len(i)),
        ],
        axis=1,
    )
    delta = np.where(
        np.array([rows[left][0] == rows[right][0] and ratings[left] != ratings[right] for left, right in zip(i, j)]),
        -1.0,
        1.0,
    )
    confidence, _, _ = darp_fit(
        len(rows), i, j, features, delta, labeled, c_labeled, active=np.array([1, 1, 1, 1, 1, 0])
    )
    current = ratings[4:]
    votes, counts = np.unique(current, return_counts=True)
    majority = int(votes[np.flatnonzero(counts == counts.max())[0]])
    darp_prediction = int(current[np.argmax(confidence[4:])])
    return confidence, majority, darp_prediction
