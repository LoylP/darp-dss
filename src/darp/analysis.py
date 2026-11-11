"""Bảng tổng hợp và phân tích kết quả."""

from __future__ import annotations

import numpy as np
import pandas as pd

from darp.evaluation import MODELS
from darp.metrics import acc_cost
from darp.model import toy_example as _toy_example


def _accuracy(frame: pd.DataFrame, models: tuple[str, ...] = MODELS) -> dict[str, float]:
    return {name: acc_cost(frame[name], frame.y) for name in models}


def acc_overall(predictions: pd.DataFrame) -> pd.DataFrame:
    """ACC toàn bộ dự đoán."""
    return pd.DataFrame.from_dict(_accuracy(predictions), orient="index", columns=["ACC"])


def acc_by_year(predictions: pd.DataFrame) -> pd.DataFrame:
    """ACC theo năm."""
    rows = {year: _accuracy(group) for year, group in predictions.groupby("year")}
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def acc_by_sector(predictions: pd.DataFrame) -> pd.DataFrame:
    """ACC theo ngành; chỉ giữ ngành có ít nhất 100 sự kiện."""
    rows = {
        division: {**_accuracy(group), "n": len(group)}
        for division, group in predictions.groupby("division")
    }
    frame = pd.DataFrame.from_dict(rows, orient="index")
    return frame.loc[frame.n >= 100].sort_values("DARP", ascending=False)


def acc_by_nrecs(predictions: pd.DataFrame) -> pd.DataFrame:
    """ACC theo số khuyến nghị trong sự kiện."""
    bins = pd.cut(
        predictions.n_recs,
        [2, 3, 4, 5, 6, 100],
        labels=["3", "4", "5", "6", "7+"],
    )
    rows: dict[str, dict[str, float]] = {}
    for name in ["3", "4", "5", "6", "7+"]:
        group = predictions.loc[bins == name]
        if not group.empty:
            rows[name] = {**_accuracy(group), "n": len(group)}
    return pd.DataFrame.from_dict(rows, orient="index")


def topk_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """ACC cho MV, WM, DARP top-5, top-3 và DARP."""
    columns = ["MV", "WM", "DARP_T5", "DARP_T3", "DARP"]
    return pd.DataFrame(
        {"ACC": {column: acc_cost(predictions[column], predictions.y) for column in columns}}
    )


def reliability_by_year(reliability: pd.DataFrame) -> pd.DataFrame:
    """Thống kê c theo năm."""
    return reliability.groupby("year").c.agg(["mean", "std", "count"])


def reliability_by_sector(reliability: pd.DataFrame) -> pd.DataFrame:
    """Thống kê c theo ngành."""
    return reliability.groupby("division").c.agg(["mean", "std", "count"]).sort_values(
        "mean", ascending=False
    )


def reliability_heatmap(reliability: pd.DataFrame) -> pd.DataFrame:
    """Bảng trung bình c theo ngành và năm."""
    return reliability.pivot_table(
        index="division", columns="year", values="c", aggfunc="mean"
    )


def reliability_quintiles(reliability: pd.DataFrame) -> pd.DataFrame:
    """Tỷ lệ đúng và q thật theo ngũ phân vị độ tin cậy."""
    ranked = reliability.c.rank(method="first")
    quintile = pd.qcut(
        ranked,
        5,
        labels=["Q1 (thấp)", "Q2", "Q3", "Q4", "Q5 (cao)"],
    )
    frame = reliability.assign(
        correct=(reliability.x == reliability.y).astype(int), quintile=quintile
    )
    return frame.groupby("quintile", observed=False).agg(
        c_mean=("c", "mean"),
        true_q=("true_q", "mean"),
        ty_le_dung=("correct", "mean"),
    )


def spearman_c_true_q(reliability: pd.DataFrame) -> float:
    """Tương quan Spearman giữa c học được và q ẩn trong dữ liệu mô phỏng."""
    return float(reliability[["c", "true_q"]].corr(method="spearman").iloc[0, 1])


def toy_example(rows: list[tuple[object, ...]] | None = None) -> dict[str, object]:
    """Trả kết quả ví dụ slide 15; đầu vào tùy chỉnh chưa được hỗ trợ."""
    if rows is not None:
        raise NotImplementedError("Ví dụ tùy chỉnh được xử lý ở lớp API.")
    confidence, majority, darp_prediction = _toy_example()
    return {
        "c": {
            "a1": float(confidence[4]),
            "a2": float(confidence[5]),
            "a3": float(confidence[6]),
        },
        "MV": majority,
        "DARP": darp_prediction,
    }
