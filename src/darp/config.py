"""Cấu hình có kiểu dữ liệu cho pipeline DARP."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    """Thông số sinh dữ liệu."""

    seed: int = 2024
    start: str = "2017-01-01"
    end: str = "2021-12-31"
    n_brokers: int = 40
    n_analysts: int = 350
    n_stocks: int = 400
    earnings_per_year: int = 4
    p_issue: float = 0.33
    sporadic_per_stock_year: int = 2
    move_std: float = 0.15
    skill_offset: float = -1.0
    herd_prob: float = 0.7
    year_shock: dict[int, float] | None = None


@dataclass(frozen=True)
class EvalConfig:
    """Thông số đánh giá theo thời gian."""

    TP: int = 1
    OB: int = 14
    PD: int = 1
    min_recs: int = 3


@dataclass(frozen=True)
class ModelConfig:
    """Thông số thuật toán và đồ thị."""

    knn: int = 30
    coverage_lookback_months: int = 12
    max_iter: int = 30
    tol: float = 1e-6
    ridge: float = 1e-6


@dataclass(frozen=True)
class ExperimentConfig:
    """Thông số thí nghiệm phân tích."""

    fast: bool = True
    sensitivity: dict[str, list[int]] | None = None


@dataclass(frozen=True)
class Config:
    """Cấu hình đầy đủ; các nhóm con bất biến để tránh thay đổi ngầm."""

    data: DataConfig = DataConfig(
        year_shock={2017: 0.0, 2018: 0.1, 2019: 0.0, 2020: -0.3, 2021: -0.9}
    )
    eval: EvalConfig = EvalConfig()
    model: ModelConfig = ModelConfig()
    experiment: ExperimentConfig = ExperimentConfig(
        sensitivity={"TP": [1, 2, 3], "OB": [7, 14, 30], "PD": [1, 3, 5]}
    )

    def with_overrides(self, **overrides: Any) -> Config:
        """Tạo cấu hình mới; khóa ghi đè dùng dạng `nhóm.trường`."""
        groups: dict[str, Any] = {
            "data": self.data,
            "eval": self.eval,
            "model": self.model,
            "experiment": self.experiment,
        }
        updates: dict[str, dict[str, Any]] = {name: {} for name in groups}
        for key, value in overrides.items():
            if "." not in key:
                raise ValueError(f"Khóa ghi đè phải có dạng 'nhóm.trường': {key}")
            group, field = key.split(".", maxsplit=1)
            if group not in groups or field not in {item.name for item in fields(groups[group])}:
                raise ValueError(f"Trường cấu hình không hợp lệ: {key}")
            updates[group][field] = value
        replaced = {
            name: replace(group, **updates[name]) if updates[name] else group
            for name, group in groups.items()
        }
        return Config(**replaced)


def load_config(path: str | Path) -> Config:
    """Đọc YAML thành cấu hình có kiểu; lỗi cấu trúc được báo rõ."""
    with Path(path).open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("Tệp cấu hình phải chứa một mapping YAML.")
    expected = {"data": DataConfig, "eval": EvalConfig, "model": ModelConfig, "experiment": ExperimentConfig}
    unknown = set(raw) - set(expected)
    if unknown:
        raise ValueError(f"Nhóm cấu hình không hợp lệ: {', '.join(sorted(unknown))}")
    values = {name: cls(**raw.get(name, {})) for name, cls in expected.items()}
    return Config(**values)
