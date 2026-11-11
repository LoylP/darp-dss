"""Các thước đo hiệu quả dự đoán."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

_TABLE_2 = np.array(
    [
        [1, 1 / 16, 1 / 4, 1 / 2, 1],
        [1 / 16, 1 / 2, 1 / 16, 1 / 4, 1 / 2],
        [1 / 4, 1 / 16, 1 / 16, 1 / 16, 1 / 4],
        [1 / 2, 1 / 4, 1 / 16, 1 / 2, 1 / 16],
        [1, 1 / 2, 1 / 4, 1 / 16, 1],
    ],
    dtype=float,
)
COST: NDArray[np.float64] = _TABLE_2[::-1, ::-1].copy()


def return_to_label(v: ArrayLike) -> NDArray[np.int_]:
    """Ánh xạ lợi suất sang nhãn Sell (1) đến Strong Buy (5)."""
    returns = np.asarray(v, dtype=float)
    return np.select(
        [returns >= 0.2, returns >= 0.1, returns >= 0.0, returns >= -0.1],
        [5, 4, 3, 2],
        default=1,
    ).astype(int)


def init_reliability(x: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
    """Khởi tạo c bằng +C[x,y] nếu đúng, ngược lại bằng -C[x,y]."""
    rating = np.asarray(x, dtype=int)
    truth = np.asarray(y, dtype=int)
    if np.any((rating < 1) | (rating > 5) | (truth < 1) | (truth > 5)):
        raise ValueError("Nhãn x và y phải nằm trong khoảng 1..5.")
    costs = COST[rating - 1, truth - 1]
    return np.where(rating == truth, costs, -costs)


def acc_cost(pred: ArrayLike, actual: ArrayLike) -> float:
    """Tính ACC có trọng số theo ma trận chi phí Table 2."""
    prediction = np.asarray(pred, dtype=int)
    truth = np.asarray(actual, dtype=int)
    if prediction.shape != truth.shape:
        raise ValueError("pred và actual phải có cùng hình dạng.")
    if np.any((prediction < 1) | (prediction > 5) | (truth < 1) | (truth > 5)):
        raise ValueError("pred và actual phải nằm trong khoảng 1..5.")
    confusion = np.zeros((5, 5))
    np.add.at(confusion, (prediction.ravel() - 1, truth.ravel() - 1), 1)
    weighted = confusion * COST
    denominator = weighted.sum()
    return float(np.trace(weighted) / denominator) if denominator > 0 else float("nan")


def bootstrap_ci(
    pred: ArrayLike, actual: ArrayLike, n_boot: int = 300, seed: int = 0
) -> tuple[float, float]:
    """Tính khoảng tin cậy percentile 95% bằng bootstrap."""
    prediction = np.asarray(pred)
    truth = np.asarray(actual)
    if prediction.ndim != 1 or prediction.shape != truth.shape or prediction.size == 0:
        raise ValueError("pred và actual phải là vector không rỗng cùng kích thước.")
    if n_boot < 1:
        raise ValueError("n_boot phải lớn hơn 0.")
    rng = np.random.default_rng(seed)
    n = len(prediction)
    values = [
        acc_cost(prediction[idx], truth[idx])
        for idx in (rng.integers(0, n, n) for _ in range(n_boot))
    ]
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)
