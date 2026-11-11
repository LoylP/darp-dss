"""Sinh dữ liệu mô phỏng theo cấu hình tái lập."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from darp.config import Config, DataConfig
from darp.metrics import return_to_label

SIC_DIVISIONS: dict[str, tuple[int, int]] = {
    "Mining": (1000, 1499),
    "Construction": (1500, 1799),
    "Manufacturing": (2000, 3999),
    "Transportation & Utilities": (4000, 4999),
    "Wholesale": (5000, 5199),
    "Retail": (5200, 5999),
    "Finance & Real Estate": (6000, 6799),
    "Services": (7000, 8999),
}
DIV_NAMES = list(SIC_DIVISIONS)


def sic_division(sic: int) -> str:
    """Trả về phân ngành SIC; các mã ngoài Table 4 được xếp Other."""
    for name, (low, high) in SIC_DIVISIONS.items():
        if low <= sic <= high:
            return name
    return "Other"


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Hàm logistic dùng để sinh xác suất chuyên gia có thông tin."""
    return 1 / (1 + np.exp(-z))


def _data_values(cfg: Config | DataConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(cfg, Config):
        data = cfg.data
        return {key: getattr(data, key) for key in data.__dataclass_fields__}
    if isinstance(cfg, DataConfig):
        return {key: getattr(cfg, key) for key in cfg.__dataclass_fields__}
    return dict(cfg)


def make_synthetic(
    cfg: Config | DataConfig | Mapping[str, Any], rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sinh recs, stocks và giá với thứ tự gọi RNG cố định."""
    values = _data_values(cfg)
    bdays = pd.bdate_range(values["start"], pd.Timestamp(values["end"]) + pd.Timedelta(days=45))
    n_days = len(bdays)
    year_of_day = bdays.year.values
    n_stocks, n_analysts, n_brokers = (
        values["n_stocks"],
        values["n_analysts"],
        values["n_brokers"],
    )
    prior = np.array([0.06, 0.12, 0.42, 0.28, 0.12])

    division_weights = np.array([0.05, 0.05, 0.30, 0.10, 0.05, 0.10, 0.20, 0.15])
    stock_division = rng.choice(len(DIV_NAMES), n_stocks, p=division_weights)
    sic = np.zeros(n_stocks, int)
    sic2_pool: dict[int, np.ndarray] = {}
    for division, name in enumerate(DIV_NAMES):
        low, high = SIC_DIVISIONS[name]
        n_sic2 = max(2, min(6, (high - low + 1) // 100))
        sic2_pool[division] = rng.choice(
            np.arange(low // 100, high // 100 + 1), n_sic2, replace=False
        )
    for stock in range(n_stocks):
        sic2 = rng.choice(sic2_pool[stock_division[stock]])
        sic[stock] = sic2 * 100 + rng.integers(0, 3) * 10 + rng.integers(0, 2)
    tickers = np.array([f"S{stock:04d}" for stock in range(n_stocks)])

    broker_q = rng.normal(0, 0.8, n_brokers)
    broker_home = np.array([rng.choice(len(DIV_NAMES), 2, replace=False) for _ in range(n_brokers)])
    analyst_broker = rng.choice(n_brokers, n_analysts, p=np.r_[rng.dirichlet(np.ones(n_brokers) * 2)])
    analyst_base = rng.normal(0, 0.5, n_analysts)
    affinity = rng.normal(-0.5, 0.25, (n_analysts, len(DIV_NAMES)))
    analyst_home = np.zeros((n_analysts, 2), int)
    for analyst in range(n_analysts):
        home = broker_home[analyst_broker[analyst]].copy()
        if rng.random() < 0.3:
            home[1] = rng.integers(len(DIV_NAMES))
        analyst_home[analyst] = home
        affinity[analyst, home] = rng.normal(1.0, 0.25, 2)

    coverage: list[np.ndarray] = []
    for analyst in range(n_analysts):
        count = rng.integers(8, 21)
        home_stocks = np.where(np.isin(stock_division, analyst_home[analyst]))[0]
        n_home = min(len(home_stocks), int(round(0.8 * count)))
        covered = (
            set(rng.choice(home_stocks, n_home, replace=False)) if n_home > 0 else set()
        )
        while len(covered) < count:
            covered.add(int(rng.integers(n_stocks)))
        coverage.append(np.array(sorted(covered)))
    coverers: list[list[int]] = [[] for _ in range(n_stocks)]
    for analyst in range(n_analysts):
        for stock in coverage[analyst]:
            coverers[stock].append(analyst)

    sigma = rng.uniform(0.015, 0.03, n_stocks)
    logret = rng.normal(0, 1, (n_days, n_stocks)) * sigma
    drift = np.zeros((n_days, n_stocks))
    years = len(np.unique(year_of_day[bdays <= pd.Timestamp(values["end"])]))
    rec_rows: list[tuple[int, int, int, int]] = []
    last_day = np.searchsorted(bdays, pd.Timestamp(values["end"]), side="right") - 15
    for stock in range(n_stocks):
        n_events = values["earnings_per_year"] * years
        event_days = np.sort(rng.choice(np.arange(5, last_day), n_events, replace=False))
        for start_day in event_days:
            move = rng.normal(0, values["move_std"])
            horizon = 15
            drift[start_day : start_day + horizon, stock] += move * 0.95 ** np.arange(
                min(horizon, n_days - start_day)
            )
        for start_day in event_days:
            herd = int(rng.choice(np.arange(1, 6), p=prior))
            for analyst in coverers[stock]:
                if rng.random() < values["p_issue"]:
                    delay = min(int(rng.geometric(0.3)) - 1, 9)
                    rec_rows.append((analyst, stock, start_day + delay, herd))
        for _ in range(rng.poisson(values["sporadic_per_stock_year"] * years)):
            if coverers[stock]:
                rec_rows.append(
                    (
                        int(rng.choice(coverers[stock])),
                        stock,
                        int(rng.integers(5, last_day)),
                        int(rng.choice(np.arange(1, 6), p=prior)),
                    )
                )
    logret += drift
    close = 50 * np.exp(np.cumsum(logret, axis=0))
    prices = pd.DataFrame(close, index=bdays, columns=tickers)

    recs = (
        pd.DataFrame(rec_rows, columns=["a", "s", "t", "herd"])
        .drop_duplicates(["a", "s", "t"])
        .reset_index(drop=True)
    )
    years_for_recs = year_of_day[recs.t.values]
    year_shock = values["year_shock"] or {}
    z = (
        analyst_base[recs.a]
        + broker_q[analyst_broker[recs.a]]
        + affinity[recs.a, stock_division[recs.s]]
        + np.array([year_shock.get(int(year), 0.0) for year in years_for_recs])
        + values["skill_offset"]
    )
    true_q = sigmoid(z)
    true_signal = np.expm1(drift[np.minimum(recs.t.values + 1, n_days - 1), recs.s.values])
    informed = rng.random(len(recs)) < true_q
    rating_informed = return_to_label(true_signal + rng.normal(0, 0.01, len(recs)))
    rating_random = rng.choice(np.arange(1, 6), len(recs), p=prior)
    rating_uninformed = np.where(
        rng.random(len(recs)) < values["herd_prob"], recs.herd.values, rating_random
    )
    recs["rating"] = np.where(informed, rating_informed, rating_uninformed)
    recs["true_q"] = true_q
    recs["date"] = bdays[recs.t.values]
    recs = recs[recs.date <= pd.Timestamp(values["end"])].copy()
    recs["analyst"] = recs.a.map(lambda analyst: f"A{analyst:04d}")
    recs["broker"] = analyst_broker[recs.a.values]
    recs["broker"] = recs.broker.map(lambda broker: f"B{broker:03d}")
    recs["ticker"] = tickers[recs.s.values]
    recs = recs.sort_values(["date", "ticker", "analyst"]).reset_index(drop=True)
    recs["rec_id"] = np.arange(len(recs))
    recs = recs[["rec_id", "analyst", "broker", "ticker", "date", "rating", "true_q"]]
    stocks = pd.DataFrame({"ticker": tickers, "sic": sic})
    return recs, stocks, prices
