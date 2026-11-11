"""Đặc trưng liên quan và đồ thị khuyến nghị."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy import sparse

from darp.dataset import Dataset

FEATS = ("f1", "f2", "f3", "f4", "f5", "f6")


def jaccard_matrix(
    frame: pd.DataFrame, row_key: str, col_key: str, row_index: Mapping[str, int]
) -> np.ndarray:
    """Tính Jaccard giữa các hàng dựa trên tập giá trị cột chung."""
    mapped_rows = frame[row_key].map(row_index)
    usable = frame.loc[mapped_rows.notna()]
    row_ids = mapped_rows.loc[mapped_rows.notna()].astype(int).to_numpy()
    _, column_ids = np.unique(usable[col_key].to_numpy(), return_inverse=True)
    matrix = sparse.csr_matrix(
        (np.ones(len(row_ids)), (row_ids, column_ids)),
        shape=(len(row_index), len(np.unique(usable[col_key].to_numpy()))),
    )
    matrix.data[:] = 1
    matrix = (matrix > 0).astype(float)
    intersections = (matrix @ matrix.T).toarray()
    sizes = np.asarray(matrix.sum(axis=1)).ravel()
    unions = sizes[:, None] + sizes[None, :] - intersections
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(unions > 0, intersections / unions, 0.0)


def node_arrays(nodes: pd.DataFrame, ds: Dataset) -> dict[str, np.ndarray]:
    """Chuyển nút sang chỉ số số nguyên phục vụ tính đặc trưng theo cặp."""
    sic = ds.sic_of.loc[nodes.ticker].to_numpy()
    return {
        "a": nodes.analyst.map(ds.analyst_idx).to_numpy(dtype=int),
        "b": nodes.broker.map(ds.broker_idx).to_numpy(dtype=int),
        "s": nodes.ticker.map(ds.stock_idx).to_numpy(dtype=int),
        "sic2": sic // 100,
        "sic3": sic // 10,
        "sic4": sic,
        "ev": nodes.event_id.to_numpy(dtype=int),
        "x": nodes.x.to_numpy(dtype=int),
    }


def pair_features(
    arrays: dict[str, np.ndarray],
    i: np.ndarray,
    j: np.ndarray,
    JA: np.ndarray,
    JS: np.ndarray,
) -> np.ndarray:
    """Tính f1–f6 theo thứ tự định nghĩa trong bài."""
    return np.stack(
        [
            JA[arrays["a"][i], arrays["a"][j]],
            (arrays["b"][i] == arrays["b"][j]).astype(float),
            (arrays["sic2"][i] == arrays["sic2"][j]).astype(float),
            (arrays["sic3"][i] == arrays["sic3"][j]).astype(float),
            (arrays["sic4"][i] == arrays["sic4"][j]).astype(float),
            JS[arrays["s"][i], arrays["s"][j]],
        ],
        axis=1,
    )


def build_graph(
    arrays: dict[str, np.ndarray],
    JA: np.ndarray,
    JS: np.ndarray,
    knn: int = 30,
    use: np.ndarray | None = None,
    random_cross: bool = False,
    rng: np.random.Generator | None = None,
    chunk: int = 600,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tạo cạnh nội sự kiện đầy đủ và cạnh khác sự kiện theo kNN.

    Lựa chọn cài đặt (bài báo không nêu): đồ thị khác sự kiện thưa, giới hạn
    số láng giềng theo `knn` và chia khối theo `chunk` để giảm bộ nhớ.
    """
    if knn < 1 or chunk < 1:
        raise ValueError("knn và chunk phải lớn hơn 0.")
    if random_cross and rng is None:
        raise ValueError("random_cross=True cần truyền một Generator.")
    selected = np.ones(6, dtype=bool) if use is None else np.asarray(use, dtype=bool)
    if selected.shape != (6,) or not selected.any():
        raise ValueError("use phải chọn ít nhất một trong sáu đặc trưng.")
    n_nodes = len(arrays["ev"])
    if n_nodes < 2:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.empty((0, 6), dtype=float),
            np.array([], dtype=float),
        )
    event_ids = arrays["ev"]
    edge_i: list[np.ndarray] = []
    edge_j: list[np.ndarray] = []
    boundaries = np.r_[0, np.flatnonzero(np.diff(event_ids)) + 1, n_nodes]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        indices = np.arange(start, end)
        left, right = np.triu_indices(len(indices), 1)
        edge_i.append(indices[left])
        edge_j.append(indices[right])

    all_nodes = np.arange(n_nodes)
    for row_start in range(0, n_nodes, chunk):
        rows = np.arange(row_start, min(n_nodes, row_start + chunk))
        if random_cross:
            assert rng is not None
            candidates = rng.integers(0, n_nodes, (len(rows), knn))
            valid = event_ids[candidates] != event_ids[rows][:, None]
            edge_i.append(np.repeat(rows, knn)[valid.ravel()])
            edge_j.append(candidates.ravel()[valid.ravel()])
            continue
        row_grid, column_grid = np.meshgrid(rows, all_nodes, indexing="ij")
        relevance = pair_features(
            arrays, row_grid.ravel(), column_grid.ravel(), JA, JS
        )[:, selected].sum(axis=1).reshape(row_grid.shape)
        relevance[event_ids[rows][:, None] == event_ids[None, :]] = -1
        k = min(knn, n_nodes - 1)
        top = np.argpartition(-relevance, k, axis=1)[:, :k]
        scores = np.take_along_axis(relevance, top, axis=1)
        valid = scores > 0
        edge_i.append(np.repeat(rows, k)[valid.ravel()])
        edge_j.append(top.ravel()[valid.ravel()])

    left = np.concatenate(edge_i)
    right = np.concatenate(edge_j)
    low, high = np.minimum(left, right), np.maximum(left, right)
    unique = np.unique(low.astype(np.int64) * n_nodes + high)
    left = (unique // n_nodes).astype(int)
    right = (unique % n_nodes).astype(int)
    features = pair_features(arrays, left, right, JA, JS)
    features[:, ~selected] = 0.0
    same_event = event_ids[left] == event_ids[right]
    delta = np.where(
        same_event & (arrays["x"][left] != arrays["x"][right]), -1.0, 1.0
    )
    return left, right, features, delta
