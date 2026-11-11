"""Kiểm thử cửa sổ quan sát và nhãn sự kiện."""

import numpy as np
import pandas as pd

from darp.dataset import Dataset
from darp.events import build_events


def test_latest_recommendation_per_analyst_and_forward_label() -> None:
    days = pd.bdate_range("2020-01-01", periods=5)
    recs = pd.DataFrame(
        [
            (0, "A1", "B1", "T1", days[0], 1, np.nan),
            (1, "A1", "B1", "T1", days[2], 4, np.nan),
            (2, "A2", "B2", "T1", days[2], 3, np.nan),
            (3, "A3", "B3", "T1", days[2], 2, np.nan),
        ],
        columns=["rec_id", "analyst", "broker", "ticker", "date", "rating", "true_q"],
    )
    stocks = pd.DataFrame({"ticker": ["T1"], "sic": [7372]})
    prices = pd.DataFrame({"T1": [100.0, 101.0, 102.0, 120.0, 121.0]}, index=days)

    events, nodes = build_events(Dataset(recs, stocks, prices), OB=3, PD=1, min_recs=3)

    event = events.loc[events.date == days[2]].iloc[0]
    assert event.y == 4
    event_nodes = nodes.loc[nodes.event_id == event.event_id]
    assert len(event_nodes) == 3
    assert event_nodes.loc[event_nodes.analyst == "A1", "x"].item() == 4


def test_observation_window_excludes_old_recommendations() -> None:
    days = pd.bdate_range("2020-01-01", periods=5)
    recs = pd.DataFrame(
        [
            (0, "A1", "B1", "T1", days[0], 3, np.nan),
            (1, "A2", "B2", "T1", days[0], 3, np.nan),
            (2, "A3", "B3", "T1", days[0], 3, np.nan),
            (3, "A4", "B4", "T1", days[4], 3, np.nan),
        ],
        columns=["rec_id", "analyst", "broker", "ticker", "date", "rating", "true_q"],
    )
    ds = Dataset(
        recs,
        pd.DataFrame({"ticker": ["T1"], "sic": [7372]}),
        pd.DataFrame({"T1": [100.0] * len(days)}, index=days),
    )

    events, _ = build_events(ds, OB=1, PD=1, min_recs=3)

    assert events.date.tolist() == [days[0]]
