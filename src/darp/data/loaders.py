"""Nạp dữ liệu IBES/CRSP theo schema của package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from darp.data.synthetic import sic_division


def _read_table(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = Path(source)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def validate_schema(
    frame: pd.DataFrame, required: tuple[str, ...], source_name: str
) -> None:
    """Báo thiếu cột schema thay vì âm thầm bỏ qua dữ liệu."""
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source_name} thiếu cột bắt buộc: {', '.join(missing)}")


def load_ibes(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Đổi dữ liệu khuyến nghị IBES chuẩn hóa sang schema recs của package.

    Cần các cột `analyst`, `broker`, `ticker`, `date`, `IRECCD`.
    """
    raw = _read_table(source)
    validate_schema(raw, ("analyst", "broker", "ticker", "date", "IRECCD"), "IBES")
    recs = raw[["analyst", "broker", "ticker", "date", "IRECCD"]].copy()
    recs["date"] = pd.to_datetime(recs.date, errors="raise")
    recs["rating"] = 6 - pd.to_numeric(recs.IRECCD, errors="raise").astype(int)
    recs["true_q"] = np.nan
    recs = recs.drop(columns="IRECCD").sort_values(
        ["date", "ticker", "analyst"]
    ).reset_index(drop=True)
    recs.insert(0, "rec_id", np.arange(len(recs)))
    validate_schema(recs, ("rec_id", "analyst", "broker", "ticker", "date", "rating", "true_q"), "IBES")
    if not recs.rating.between(1, 5).all():
        raise ValueError("IBES IRECCD phải ánh xạ thành rating trong khoảng 1..5.")
    return recs[["rec_id", "analyst", "broker", "ticker", "date", "rating", "true_q"]]


def load_stocks(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Nạp bảng ticker/SIC và thêm phân ngành SIC."""
    stocks = _read_table(source)
    validate_schema(stocks, ("ticker", "sic"), "stocks")
    stocks = stocks[["ticker", "sic"]].drop_duplicates("ticker").copy()
    stocks["sic"] = pd.to_numeric(stocks.sic, errors="raise").astype(int)
    stocks["division"] = stocks.sic.map(sic_division)
    return stocks.reset_index(drop=True)


def load_crsp(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Nạp giá CRSP dạng bảng rộng, lấy trị tuyệt đối của PRC."""
    raw = _read_table(source)
    if {"date", "ticker", "PRC"}.issubset(raw.columns):
        raw["date"] = pd.to_datetime(raw.date, errors="raise")
        raw["PRC"] = pd.to_numeric(raw.PRC, errors="raise").abs()
        prices = raw.pivot(index="date", columns="ticker", values="PRC")
    elif {"date", "ticker", "prc"}.issubset(raw.columns):
        raw["date"] = pd.to_datetime(raw.date, errors="raise")
        raw["prc"] = pd.to_numeric(raw.prc, errors="raise").abs()
        prices = raw.pivot(index="date", columns="ticker", values="prc")
    else:
        validate_schema(raw, ("date",), "CRSP")
        prices = raw.set_index(pd.to_datetime(raw.pop("date"), errors="raise"))
        prices = prices.apply(pd.to_numeric, errors="raise").abs()
    prices.index.name = "date"
    prices.columns.name = None
    return prices.sort_index()
