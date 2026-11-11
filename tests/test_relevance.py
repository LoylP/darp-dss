"""Kiểm thử đặc trưng và dựng đồ thị."""

import numpy as np
import pandas as pd

from darp.dataset import Dataset
from darp.relevance import build_graph, jaccard_matrix, node_arrays, pair_features


def test_jaccard_identity_and_pair_features() -> None:
    frame = pd.DataFrame({"analyst": ["a1", "a1", "a2"], "ticker": ["t1", "t2", "t1"]})
    jac = jaccard_matrix(frame, "analyst", "ticker", {"a1": 0, "a2": 1})

    assert jac[0, 0] == 1
    assert jac[0, 1] == 0.5


def test_graph_has_complete_event_edges_and_signed_delta() -> None:
    recs = pd.DataFrame(
        {
            "rec_id": [0, 1],
            "analyst": ["a1", "a2"],
            "broker": ["b1", "b2"],
            "ticker": ["t1", "t1"],
            "date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "rating": [5, 2],
            "true_q": [np.nan, np.nan],
        }
    )
    ds = Dataset(
        recs,
        pd.DataFrame({"ticker": ["t1"], "sic": [7372]}),
        pd.DataFrame({"t1": [1.0]}, index=pd.to_datetime(["2020-01-01"])),
    )
    nodes = pd.DataFrame(
        {"event_id": [1, 1], "analyst": ["a1", "a2"], "broker": ["b1", "b2"], "ticker": ["t1", "t1"], "x": [5, 2]}
    )
    arrays = node_arrays(nodes, ds)
    JA = np.eye(2)
    JS = np.ones((1, 1))
    features = pair_features(arrays, np.array([0]), np.array([1]), JA, JS)
    i, j, graph_features, delta = build_graph(arrays, JA, JS, knn=1)

    assert features[0, 2:].tolist() == [1, 1, 1, 1]
    assert len(i) == len(set(zip(i.tolist(), j.tolist()))) == 1
    assert graph_features.shape == (1, 6)
    assert delta.tolist() == [-1.0]
