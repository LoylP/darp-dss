"""Kiểm thử hồi quy giữa package và kết quả đã sinh từ package."""

import json
from pathlib import Path

import pandas as pd
import pytest

from darp.config import load_config
from darp.dataset import Dataset
from darp.events import build_events
from darp.evaluation import run_rolling

pytestmark = pytest.mark.slow


def test_first_six_months_match_package_predictions() -> None:
    cfg = load_config("configs/default.yaml")
    ds = Dataset.synthetic(cfg)
    events, nodes = build_events(ds, cfg.eval.OB, cfg.eval.PD, cfg.eval.min_recs)
    months = sorted(events.month.unique())[cfg.eval.TP : cfg.eval.TP + 6]
    actual, _, _ = run_rolling(events, nodes, ds, cfg, months=months, verbose=False)
    expected = pd.read_csv("tests/fixtures/predictions.csv")
    expected = expected.loc[expected.month.isin([str(month) for month in months])]
    actual = actual.sort_values("event_id").reset_index(drop=True)
    expected = expected.sort_values("event_id").reset_index(drop=True)
    actual["month"] = actual["month"].astype(str)

    columns = [
        "event_id",
        "y",
        "n_recs",
        "MV",
        "WM",
        "DARP",
        "DARP_T3",
        "DARP_T5",
        "no_info",
        "month",
        "year",
        "division",
        "sic",
    ]
    pd.testing.assert_frame_equal(
        actual[columns].astype(expected[columns].dtypes.to_dict()),
        expected[columns],
        check_dtype=False,
    )


def test_full_summary_matches_package_output() -> None:
    fixture = json.loads(Path("tests/fixtures/summary.json").read_text(encoding="utf-8"))
    result_path = Path("artifacts/results.json")
    assert result_path.is_file(), "Chạy `make precompute` trước kiểm thử hồi quy đầy đủ."
    actual = json.loads(result_path.read_text(encoding="utf-8"))

    assert actual.keys() == fixture.keys()
    _assert_package_equal(fixture, actual)


def _assert_package_equal(expected: object, actual: object, path: str = "") -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert expected.keys() == actual.keys(), path
        for key, value in expected.items():
            _assert_package_equal(value, actual[key], f"{path}.{key}".strip("."))
        return
    if isinstance(expected, list):
        assert isinstance(actual, list) and len(expected) == len(actual), path
        for index, (left, right) in enumerate(zip(expected, actual)):
            _assert_package_equal(left, right, f"{path}[{index}]")
        return
    if isinstance(expected, (float, int)) and not isinstance(expected, bool):
        assert isinstance(actual, (float, int)) and not isinstance(actual, bool), path
        assert abs(expected - actual) <= 1e-12, path
        return
    assert expected == actual, path
