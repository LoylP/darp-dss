"""Dựng các sự kiện từ khuyến nghị và chuỗi giá."""

from __future__ import annotations

import numpy as np
import pandas as pd

from darp.dataset import Dataset
from darp.metrics import return_to_label


def build_events(
    ds: Dataset, OB: int, PD: int, min_recs: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dựng sự kiện ticker/ngày và nút khuyến nghị mới nhất trong cửa sổ OB."""
    if OB < 0 or PD < 1 or min_recs < 1:
        raise ValueError("OB phải không âm; PD và min_recs phải lớn hơn 0.")
    recs = ds.recs
    prices = ds.prices
    business_days = prices.index
    price_values = prices.to_numpy()
    price_column = {ticker: column for column, ticker in enumerate(prices.columns)}
    event_rows: list[tuple[int, str, pd.Timestamp, float, int]] = []
    node_rows: list[tuple[int, int, str, str, int, float, pd.Timestamp]] = []
    event_id = 0

    for ticker, group in recs.groupby("ticker", sort=False):
        ordered = group.sort_values("date")
        dates = ordered.date.to_numpy()
        column = price_column[ticker]
        for date in np.unique(dates):
            low = date - np.timedelta64(OB, "D")
            window = ordered[(dates > low) & (dates <= date)]
            window = window.drop_duplicates("analyst", keep="last")
            if len(window) < min_recs:
                continue
            price_index = business_days.searchsorted(date)
            if price_index + PD >= len(business_days):
                continue
            ret = price_values[price_index + PD, column] / price_values[price_index, column] - 1
            event_date = pd.Timestamp(date)
            event_rows.append((event_id, ticker, event_date, ret, len(window)))
            for row in window.itertuples(index=False):
                node_rows.append(
                    (
                        event_id,
                        int(row.rec_id),
                        row.analyst,
                        row.broker,
                        int(row.rating),
                        float(row.true_q) if pd.notna(row.true_q) else np.nan,
                        pd.Timestamp(row.date),
                    )
                )
            event_id += 1

    events = pd.DataFrame(
        event_rows, columns=["event_id", "ticker", "date", "v", "n_recs"]
    )
    if events.empty:
        events = events.astype(
            {"event_id": "int64", "ticker": "object", "date": "datetime64[ns]", "v": "float64", "n_recs": "int64"}
        )
        events["y"] = pd.Series(dtype="int64")
        events["month"] = pd.Series(dtype="period[M]")
        events["year"] = pd.Series(dtype="int64")
    else:
        events["y"] = return_to_label(events.v.to_numpy())
        events["month"] = events.date.dt.to_period("M")
        events["year"] = events.date.dt.year
    events["sic"] = events.ticker.map(ds.sic_of)
    events["division"] = events.ticker.map(ds.division_of)

    nodes = pd.DataFrame(
        node_rows,
        columns=["event_id", "rec_id", "analyst", "broker", "x", "true_q", "rec_date"],
    )
    nodes = nodes.merge(
        events[
            ["event_id", "ticker", "date", "y", "month", "year", "sic", "division"]
        ],
        on="event_id",
        how="left",
    )
    nodes = nodes.sort_values(["event_id", "rec_id"]).reset_index(drop=True)
    return events, nodes
