"""Các phương pháp dự đoán đối chứng."""

from __future__ import annotations

import numpy as np
import pandas as pd


def predict_mv(x: np.ndarray, rng: np.random.Generator) -> int:
    """Dự đoán theo majority vote; dùng RNG truyền vào để xử lý hòa."""
    values, counts = np.unique(x, return_counts=True)
    return int(rng.choice(values[counts == counts.max()]))


def analyst_accuracy(nodes: pd.DataFrame) -> pd.Series:
    """Tính tỷ lệ đúng của từng chuyên gia trên các nút đã gán nhãn."""
    labeled = nodes.loc[nodes.y.notna()]
    return (labeled.x == labeled.y).groupby(labeled.analyst).mean()


def predict_wm(x: np.ndarray, weights: np.ndarray, rng: np.random.Generator) -> int:
    """Chọn rating có trọng số WM lớn nhất, xử lý hòa bằng RNG."""
    best = np.flatnonzero(weights == weights.max())
    return int(x[rng.choice(best)])


def predict_darp(x: np.ndarray, c: np.ndarray, fallback: int | None = None) -> int:
    """Chọn rating của nút có c cao nhất; fallback khi không có lan truyền."""
    if np.allclose(c, 0):
        if fallback is not None:
            return int(fallback)
        values, counts = np.unique(x, return_counts=True)
        return int(values[np.flatnonzero(counts == counts.max())[0]])
    return int(x[np.argmax(c)])


def predict_topk(x: np.ndarray, c: np.ndarray, k: int) -> int:
    """Bỏ phiếu top-k với trọng số (c+1)/2."""
    if k < 1:
        raise ValueError("k phải lớn hơn 0.")
    if np.allclose(c, 0):
        values, counts = np.unique(x, return_counts=True)
        return int(values[np.flatnonzero(counts == counts.max())[0]])
    order = np.argsort(-c)[:k]
    score = np.zeros(6)
    np.add.at(score, x[order], (c[order] + 1) / 2)
    return int(np.argmax(score))
