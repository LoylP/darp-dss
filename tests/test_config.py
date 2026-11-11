"""Kiểm thử tải và ghi đè cấu hình."""

from pathlib import Path

import pytest

from darp.config import Config, load_config


def test_load_default_config() -> None:
    config = load_config(Path("configs/default.yaml"))

    assert config.data.seed == 2024
    assert config.data.year_shock == {2017: 0.0, 2018: 0.1, 2019: 0.0, 2020: -0.3, 2021: -0.9}
    assert config.eval.OB == 14
    assert config.model.knn == 30
    assert config.experiment.sensitivity == {"TP": [1, 2, 3], "OB": [7, 14, 30], "PD": [1, 3, 5]}


def test_with_overrides_returns_new_config() -> None:
    original = Config()
    changed = original.with_overrides(**{"eval.OB": 7, "data.seed": 17})

    assert original.eval.OB == 14
    assert changed.eval.OB == 7
    assert changed.data.seed == 17


def test_with_overrides_rejects_unknown_field() -> None:
    with pytest.raises(ValueError, match="không hợp lệ"):
        Config().with_overrides(**{"eval.unknown": 1})
