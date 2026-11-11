"""Đối tượng dữ liệu thống nhất cho các bước mô hình."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from darp.config import Config
from darp.data.loaders import load_crsp, load_ibes, load_stocks
from darp.data.synthetic import make_synthetic, sic_division


@dataclass
class Dataset:
    """Dữ liệu nguồn và các chỉ mục ổn định dùng trong đồ thị."""

    recs: pd.DataFrame
    stocks: pd.DataFrame
    prices: pd.DataFrame
    analyst_idx: dict[str, int] = field(init=False)
    broker_idx: dict[str, int] = field(init=False)
    stock_idx: dict[str, int] = field(init=False)
    sic_of: pd.Series = field(init=False)
    division_of: pd.Series = field(init=False)

    def __post_init__(self) -> None:
        self.recs = self.recs.copy()
        self.stocks = self.stocks.copy()
        self.prices = self.prices.copy()
        self.recs["date"] = pd.to_datetime(self.recs.date, errors="raise")
        self.prices.index = pd.to_datetime(self.prices.index, errors="raise")
        if "division" not in self.stocks:
            self.stocks["division"] = self.stocks.sic.map(sic_division)
        self.analyst_idx = {
            analyst: index for index, analyst in enumerate(sorted(self.recs.analyst.unique()))
        }
        self.broker_idx = {
            broker: index for index, broker in enumerate(sorted(self.recs.broker.unique()))
        }
        self.stock_idx = {
            ticker: index for index, ticker in enumerate(self.stocks.ticker.tolist())
        }
        self.sic_of = self.stocks.set_index("ticker").sic
        self.division_of = self.stocks.set_index("ticker").division

    @classmethod
    def synthetic(cls, cfg: Config) -> Dataset:
        """Tạo Dataset mô phỏng với một RNG cục bộ bắt đầu từ seed cấu hình."""
        import numpy as np

        recs, stocks, prices = make_synthetic(cfg, np.random.default_rng(cfg.data.seed))
        return cls(recs, stocks, prices)

    @classmethod
    def from_files(
        cls,
        ibes: str | Path | pd.DataFrame,
        stocks: str | Path | pd.DataFrame,
        crsp: str | Path | pd.DataFrame,
    ) -> Dataset:
        """Nạp IBES, bảng SIC và CRSP từ tệp hoặc DataFrame."""
        return cls(load_ibes(ibes), load_stocks(stocks), load_crsp(crsp))
