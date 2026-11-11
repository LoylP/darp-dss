"""Đánh giá cuốn chiếu theo tháng."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from darp.baselines import predict_darp, predict_mv, predict_topk, predict_wm
from darp.config import Config
from darp.dataset import Dataset
from darp.metrics import init_reliability
from darp.model import darp_fit
from darp.relevance import build_graph, jaccard_matrix, node_arrays

MODELS = ("MV", "WM", "DARP")


@dataclass
class MonthContext:
    """Đồ thị huấn luyện và các chỉ số của một tháng đánh giá."""

    month: pd.Period
    nodes: pd.DataFrame
    i: np.ndarray
    j: np.ndarray
    F: np.ndarray
    delta: np.ndarray
    eta: np.ndarray
    JA: np.ndarray
    JS: np.ndarray
    w_acc: pd.Series


def fit_month(
    m: pd.Period | str,
    events: pd.DataFrame,
    nodes: pd.DataFrame,
    ds: Dataset,
    cfg: Config,
    use: np.ndarray | None = None,
    random_cross: bool = False,
    rng: np.random.Generator | None = None,
) -> MonthContext | None:
    """Huấn luyện DARP cho tháng m bằng TP tháng trước đó.

    Lựa chọn cài đặt (bài báo không nêu): Jaccard dùng cửa sổ lịch sử giới
    hạn bởi `coverage_lookback_months`.
    """
    month = pd.Period(m, freq="M")
    training_months = [month - offset for offset in range(cfg.eval.TP, 0, -1)]
    selected_nodes = (
        nodes.loc[nodes.month.isin(training_months + [month])]
        .sort_values(["event_id", "rec_id"])
        .reset_index(drop=True)
    )
    if (
        selected_nodes.empty
        or not (selected_nodes.month == month).any()
        or not (selected_nodes.month != month).any()
    ):
        return None

    history_start = (month - cfg.model.coverage_lookback_months).start_time
    history = ds.recs.loc[
        (ds.recs.date >= history_start) & (ds.recs.date < month.start_time)
    ]
    JA = jaccard_matrix(history, "analyst", "ticker", ds.analyst_idx)
    JS = jaccard_matrix(history, "ticker", "analyst", ds.stock_idx)
    arrays = node_arrays(selected_nodes, ds)
    i, j, features, delta = build_graph(
        arrays,
        JA,
        JS,
        knn=cfg.model.knn,
        use=use,
        random_cross=random_cross,
        rng=rng,
    )
    labeled = (selected_nodes.month != month).to_numpy()
    c_labeled = init_reliability(
        selected_nodes.loc[labeled, "x"].to_numpy(),
        selected_nodes.loc[labeled, "y"].to_numpy(),
    )
    active = use
    if random_cross:
        features = np.ones((len(i), 1))
        active = None
    confidence, eta, _ = darp_fit(
        len(selected_nodes),
        i,
        j,
        features,
        delta,
        labeled,
        c_labeled,
        max_iter=cfg.model.max_iter,
        tol=cfg.model.tol,
        active=active,
        ridge=cfg.model.ridge,
    )
    selected_nodes["c"] = confidence
    selected_nodes["is_train"] = labeled
    training = selected_nodes.loc[labeled]
    weights = (training.x == training.y).groupby(training.analyst).mean()
    return MonthContext(month, selected_nodes, i, j, features, delta, eta, JA, JS, weights)


def run_month(
    m: pd.Period | str,
    events: pd.DataFrame,
    nodes: pd.DataFrame,
    ds: Dataset,
    cfg: Config,
    use: np.ndarray | None = None,
    random_cross: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, int, int, int] | None:
    """Dự đoán các sự kiện của tháng và trả bảng độ tin cậy từng nút."""
    context = fit_month(m, events, nodes, ds, cfg, use, random_cross, rng)
    if context is None:
        return None
    prediction_rng = rng if rng is not None else np.random.default_rng(cfg.data.seed)
    test_nodes = context.nodes.loc[~context.nodes.is_train]
    output: list[dict[str, object]] = []
    for event_id, group in test_nodes.groupby("event_id", sort=False):
        x = group.x.to_numpy(dtype=int)
        confidence = group.c.to_numpy(dtype=float)
        weights = group.analyst.map(context.w_acc).fillna(0).to_numpy(dtype=float)
        majority = predict_mv(x, prediction_rng)
        no_info = bool(np.allclose(confidence, 0))
        output.append(
            {
                "event_id": event_id,
                "y": int(group.y.iloc[0]),
                "n_recs": len(group),
                "MV": majority,
                "WM": predict_wm(x, weights, prediction_rng),
                "DARP": predict_darp(x, confidence, majority),
                "DARP_T3": majority if no_info else predict_topk(x, confidence, 3),
                "DARP_T5": majority if no_info else predict_topk(x, confidence, 5),
                "no_info": no_info,
            }
        )
    predictions = pd.DataFrame(output)
    predictions["month"] = context.month
    predictions = predictions.merge(
        events[["event_id", "year", "division", "sic"]], on="event_id", how="left"
    )
    reliability_columns = [
        "event_id",
        "rec_id",
        "analyst",
        "x",
        "y",
        "c",
        "true_q",
        "year",
        "month",
        "ticker",
        "division",
        "sic",
    ]
    reliability = test_nodes.drop(
        columns=[column for column in ("division", "sic") if column in test_nodes]
    ).copy()
    reliability["sic"] = reliability.ticker.map(ds.sic_of)
    reliability["division"] = reliability.ticker.map(ds.division_of)
    reliability = reliability[reliability_columns]
    return (
        predictions,
        reliability,
        context.eta,
        len(context.nodes),
        len(context.i),
        0,
    )


def run_rolling(
    events: pd.DataFrame,
    nodes: pd.DataFrame,
    ds: Dataset,
    cfg: Config,
    months: list[pd.Period | str] | None = None,
    use: np.ndarray | None = None,
    random_cross: bool = False,
    label: str = "DARP",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Đánh giá tuần tự tháng; một RNG được dùng xuyên suốt các tháng."""
    rng = np.random.default_rng(cfg.data.seed)
    all_months = sorted(events.month.unique())
    valid_months = all_months[cfg.eval.TP :]
    selected_months = (
        valid_months
        if months is None
        else [pd.Period(month, freq="M") for month in months if pd.Period(month, freq="M") in valid_months]
    )
    predictions: list[pd.DataFrame] = []
    reliabilities: list[pd.DataFrame] = []
    etas: list[np.ndarray] = []
    started = perf_counter()
    for index, month in enumerate(selected_months):
        result = run_month(
            month, events, nodes, ds, cfg, use=use, random_cross=random_cross, rng=rng
        )
        if result is None:
            continue
        prediction, reliability, eta, n_nodes, n_edges, _ = result
        predictions.append(prediction)
        reliabilities.append(reliability)
        etas.append(eta)
        if verbose and (index % 12 == 0 or index == len(selected_months) - 1):
            elapsed = perf_counter() - started
            print(
                f"  [{label}] {month}  nút={n_nodes:5d}  cạnh={n_edges:7d}  "
                f"({elapsed:5.1f}s)"
            )
    if not predictions:
        raise ValueError("Không có tháng nào đủ dữ liệu train và test.")
    return (
        pd.concat(predictions, ignore_index=True),
        pd.concat(reliabilities, ignore_index=True),
        np.asarray(etas),
    )
