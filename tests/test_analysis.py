"""Kiểm thử các phép tổng hợp kết quả."""

import pandas as pd

from darp.analysis import acc_by_nrecs, acc_overall, toy_example


def test_accuracy_tables_and_slide_example() -> None:
    predictions = pd.DataFrame(
        {
            "MV": [1, 2, 3],
            "WM": [1, 2, 4],
            "DARP": [1, 2, 5],
            "DARP_T3": [1, 2, 3],
            "DARP_T5": [1, 2, 3],
            "y": [1, 2, 3],
            "n_recs": [3, 4, 5],
        }
    )
    overall = acc_overall(predictions)
    by_nrecs = acc_by_nrecs(predictions)
    example = toy_example()

    assert overall.loc["MV", "ACC"] == 1
    assert by_nrecs.index.tolist() == ["3", "4", "5"]
    assert example["MV"] == 2 and example["DARP"] == 4
