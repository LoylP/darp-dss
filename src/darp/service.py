"""Dịch vụ quyết định trên các artifacts đã tính trước."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from darp.baselines import predict_darp, predict_mv, predict_topk, predict_wm
from darp.config import Config, load_config
from darp.dataset import Dataset
from darp.evaluation import MonthContext, fit_month, run_rolling
from darp.metrics import bootstrap_ci, init_reliability
from darp.model import solve_c_fixed_eta
from darp.relevance import build_graph, node_arrays

RATING_NAME = {1: "Sell", 2: "Under", 3: "Hold", 4: "Buy", 5: "Strong Buy"}


class DecisionService:
    """Phục vụ quyết định từ artifacts đã sinh trước, không tạo dữ liệu lúc khởi động."""

    def __init__(
        self,
        artifacts_dir: str | Path = "artifacts",
        config_path: str | Path = "configs/default.yaml",
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        required = (
            "events.parquet",
            "nodes.parquet",
            "recs.parquet",
            "stocks.parquet",
            "prices.parquet",
            "results.json",
            "predictions.parquet",
        )
        missing = [name for name in required if not (self.artifacts_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Thiếu artifacts bắt buộc trong {self.artifacts_dir}: {', '.join(missing)}"
            )
        self.config: Config = load_config(config_path)
        self.events = pd.read_parquet(self.artifacts_dir / "events.parquet")
        self.nodes = pd.read_parquet(self.artifacts_dir / "nodes.parquet")
        self.recs = pd.read_parquet(self.artifacts_dir / "recs.parquet")
        self.stocks = pd.read_parquet(self.artifacts_dir / "stocks.parquet")
        self.prices = pd.read_parquet(self.artifacts_dir / "prices.parquet")
        self.predictions = pd.read_parquet(self.artifacts_dir / "predictions.parquet")
        self.results = pd.read_json(self.artifacts_dir / "results.json", typ="series").to_dict()
        self.dataset = Dataset(self.recs, self.stocks, self.prices)
        self.events["month"] = self.events.month.map(lambda value: pd.Period(value, freq="M"))
        self.nodes["month"] = self.nodes.month.map(lambda value: pd.Period(value, freq="M"))
        self.predictions["month"] = self.predictions.month.map(
            lambda value: pd.Period(value, freq="M")
        )
        self._event_rows = self.events.set_index("event_id", drop=False)
        self._prediction_rows = self.predictions.set_index("event_id", drop=False)

    def months(self) -> list[str]:
        """Trả các tháng có dự đoán."""
        return [str(month) for month in sorted(self.predictions.month.unique())]

    def meta(self) -> dict[str, Any]:
        """Trả danh sách lựa chọn và cấu hình cho giao diện/API."""
        analysts = self.recs[["analyst", "broker"]].drop_duplicates().sort_values(
            ["analyst", "broker"]
        )
        stocks = self.stocks[["ticker", "sic", "division"]].sort_values("ticker")
        return {
            "config": self.results.get("config", {}),
            "months": self.months(),
            "analysts": analysts.to_dict(orient="records"),
            "brokers": sorted(self.recs.broker.unique().tolist()),
            "tickers": stocks.to_dict(orient="records"),
            "rating_names": RATING_NAME,
        }

    @lru_cache(maxsize=12)
    def context(self, month: str) -> MonthContext:
        """Huấn luyện và lưu đệm đồ thị một tháng."""
        if month not in self.months():
            raise KeyError(f"Không tìm thấy tháng: {month}")
        context = fit_month(
            month,
            self.events,
            self.nodes,
            self.dataset,
            self.config,
        )
        if context is None:
            raise KeyError(f"Tháng không có đủ dữ liệu train/test: {month}")
        return context

    def list_events(
        self, month: str, only_disagree: bool = False, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Liệt kê sự kiện test, có thể chỉ giữ các sự kiện MV/DARP khác nhau."""
        self._require_month(month)
        if limit < 1:
            raise ValueError("limit phải lớn hơn 0.")
        frame = self.predictions.loc[self.predictions.month == pd.Period(month, freq="M")]
        if only_disagree:
            frame = frame.loc[frame.MV != frame.DARP]
        frame = frame.head(min(limit, 500)).merge(
            self.events[["event_id", "ticker"]], on="event_id", how="left"
        )
        frame = frame[
            ["event_id", "ticker", "month", "year", "division", "sic", "y", "n_recs", "MV", "WM", "DARP"]
        ].copy()
        frame["month"] = frame.month.astype(str)
        return frame.to_dict(orient="records")

    def event_detail(self, event_id: int) -> dict[str, Any]:
        """Trả dự đoán, nút và láng giềng của một sự kiện đã quan sát."""
        if event_id not in self._event_rows.index or event_id not in self._prediction_rows.index:
            raise KeyError(f"Không tìm thấy event_id={event_id}")
        event = self._event_rows.loc[event_id]
        prediction = self._prediction_rows.loc[event_id]
        context = self.context(str(event.month))
        event_nodes = context.nodes.loc[context.nodes.event_id == event_id]
        nodes_out = [
            self._node_output(node_index, row, context)
            for node_index, row in event_nodes.iterrows()
        ]
        y = int(event.y)
        decisions = {
            model: int(prediction[model])
            for model in ("MV", "WM", "DARP", "DARP_T3", "DARP_T5")
        }
        return {
            "event_id": int(event_id),
            "ticker": str(event.ticker),
            "date": pd.Timestamp(event.date).date().isoformat(),
            "division": str(event.division),
            "y": y,
            "decisions": decisions,
            "correct": {model: choice == y for model, choice in decisions.items()},
            "nodes": nodes_out,
            "eta": self._eta_dict(context.eta),
        }

    def decide(
        self, month: str, ticker: str, recommendations: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Đánh giá một sự kiện ảo, không truy cập nhãn tương lai."""
        self._require_month(month)
        if ticker not in self.dataset.stock_idx:
            raise KeyError(f"Không tìm thấy ticker: {ticker}")
        if len(recommendations) < 2:
            raise ValueError("Cần ít nhất hai khuyến nghị.")
        analysts = [str(item["analyst"]) for item in recommendations]
        if len(set(analysts)) != len(analysts):
            raise ValueError("Không được lặp analyst trong cùng sự kiện.")
        unknown = set(analysts) - set(self.dataset.analyst_idx)
        if unknown:
            raise ValueError(f"Analyst không tồn tại: {', '.join(sorted(unknown))}")
        if any(int(item["rating"]) not in range(1, 6) for item in recommendations):
            raise ValueError("rating phải nằm trong khoảng 1..5.")

        context = self.context(month)
        training = context.nodes.loc[context.nodes.is_train].copy()
        new_event_id = int(self.nodes.event_id.max()) + 1
        analyst_broker = self.recs.drop_duplicates("analyst").set_index("analyst").broker
        date = pd.Period(month, freq="M").start_time
        virtual = pd.DataFrame(
            [
                {
                    "event_id": new_event_id,
                    "rec_id": int(self.recs.rec_id.max()) + index + 1,
                    "analyst": analyst,
                    "broker": analyst_broker.loc[analyst],
                    "x": int(item["rating"]),
                    "true_q": np.nan,
                    "rec_date": date,
                    "ticker": ticker,
                    "date": date,
                    "y": np.nan,
                    "month": pd.Period(month, freq="M"),
                    "year": date.year,
                    "sic": int(self.dataset.sic_of.loc[ticker]),
                    "division": self.dataset.division_of.loc[ticker],
                    "c": 0.0,
                    "is_train": False,
                }
                for index, (analyst, item) in enumerate(zip(analysts, recommendations))
            ]
        )
        combined = pd.concat([training, virtual], ignore_index=True)
        arrays = node_arrays(combined, self.dataset)
        i, j, features, delta = build_graph(
            arrays,
            context.JA,
            context.JS,
            knn=self.config.model.knn,
        )
        labeled = combined.is_train.to_numpy(dtype=bool)
        c_labeled = init_reliability(
            combined.loc[labeled, "x"].to_numpy(),
            combined.loc[labeled, "y"].to_numpy(),
        )
        confidence = solve_c_fixed_eta(
            len(combined),
            i,
            j,
            features,
            delta,
            labeled,
            c_labeled,
            np.nan_to_num(context.eta, nan=0.0),
            self.config.model.ridge,
        )
        virtual_indices = np.flatnonzero(~labeled)
        virtual["c"] = confidence[virtual_indices]
        x = virtual.x.to_numpy(dtype=int)
        c = virtual.c.to_numpy(dtype=float)
        w = virtual.analyst.map(context.w_acc).fillna(0).to_numpy(dtype=float)
        rng = np.random.default_rng(self.config.data.seed)
        majority = predict_mv(x, rng)
        no_info = bool(np.allclose(c, 0))
        decisions = {
            "MV": majority,
            "WM": predict_wm(x, w, rng),
            "DARP": predict_darp(x, c, majority),
            "DARP_T3": majority if no_info else predict_topk(x, c, 3),
            "DARP_T5": majority if no_info else predict_topk(x, c, 5),
        }
        virtual["c"] = c
        rho = features @ np.nan_to_num(context.eta, nan=0.0)
        node_outputs = []
        for offset, row in enumerate(virtual.itertuples(index=False)):
            analyst_train = training.loc[
                training.analyst == row.analyst, "rec_id"
            ].nunique()
            node_outputs.append(
                {
                    "analyst": row.analyst,
                    "broker": row.broker,
                    "rating": int(row.x),
                    "rating_name": RATING_NAME[int(row.x)],
                    "c": float(row.c),
                    "wm_weight": float(context.w_acc.get(row.analyst, 0.0)),
                    "n_train": int(analyst_train),
                    "true_q": None,
                    "influences": self._edge_influences(
                        len(training) + offset,
                        combined,
                        i,
                        j,
                        rho,
                        delta,
                        k=5,
                    ),
                }
            )
        return {
            "event_id": None,
            "ticker": ticker,
            "date": None,
            "division": str(self.dataset.division_of.loc[ticker]),
            "y": None,
            "decisions": decisions,
            "correct": None,
            "nodes": node_outputs,
            "eta": self._eta_dict(context.eta),
        }

    def influences(self, node: dict[str, Any], k: int = 5) -> list[dict[str, Any]]:
        """Xếp hạng nút train theo |rho * delta * c_j|."""
        context = self.context(str(node["month"]))
        node_index = int(node["node_index"])
        if not 0 <= node_index < len(context.nodes):
            return []
        rho = context.F @ np.nan_to_num(context.eta, nan=0.0)
        return self._ranked_influences(
            node_index, context.nodes, context.i, context.j, rho, context.delta, k
        )

    def run_experiment(
        self, TP: int, OB: int, PD: int, n_months: int
    ) -> dict[str, Any]:
        """Chạy đánh giá trên tối đa 12 tháng cuối với tham số mới."""
        if not 1 <= n_months <= 12:
            raise ValueError("n_months phải nằm trong khoảng 1..12.")
        config = self.config.with_overrides(
            **{"eval.TP": TP, "eval.OB": OB, "eval.PD": PD}
        )
        events = self.events
        nodes = self.nodes
        if OB != self.config.eval.OB or PD != self.config.eval.PD:
            from darp.events import build_events

            events, nodes = build_events(
                self.dataset, OB, PD, config.eval.min_recs
            )
        months = sorted(events.month.unique())[-n_months:]
        started = perf_counter()
        predictions, _, _ = run_rolling(
            events, nodes, self.dataset, config, months=months, verbose=False
        )
        from darp.metrics import acc_cost

        models = ("MV", "WM", "DARP")
        return {
            "TP": TP,
            "OB": OB,
            "PD": PD,
            "n_months": n_months,
            "seconds": perf_counter() - started,
            "n_test_events": int(len(predictions)),
            "ACC": {
                model: acc_cost(predictions[model], predictions.y)
                for model in models
            },
            "ACC_CI95": {
                model: bootstrap_ci(
                    predictions[model],
                    predictions.y,
                    seed=self.config.data.seed + index,
                )
                for index, model in enumerate(models)
            },
        }

    def _require_month(self, month: str) -> None:
        if month not in self.months():
            raise KeyError(f"Không tìm thấy tháng: {month}")

    @staticmethod
    def _eta_dict(eta: np.ndarray) -> dict[str, float]:
        return {
            f"f{index + 1}": float(value)
            for index, value in enumerate(eta)
            if np.isfinite(value)
        }

    def _node_output(
        self, node_index: int, details: pd.Series, context: MonthContext
    ) -> dict[str, Any]:
        node = {
            "analyst": str(details["analyst"]),
            "broker": str(details["broker"]),
            "rating": int(details["x"]),
            "rating_name": RATING_NAME[int(details["x"])],
            "c": float(details["c"]),
            "wm_weight": float(context.w_acc.get(details["analyst"], 0.0)),
            "n_train": int(
                context.nodes.loc[
                    context.nodes.is_train
                    & (context.nodes.analyst == details["analyst"]),
                    "rec_id",
                ].nunique()
            ),
            "true_q": float(details["true_q"])
            if pd.notna(details["true_q"])
            else None,
            "month": str(context.month),
            "node_index": node_index,
        }
        node["influences"] = self.influences(node, 5)
        return node

    @staticmethod
    def _edge_influences(
        node_index: int,
        nodes: pd.DataFrame,
        i: np.ndarray,
        j: np.ndarray,
        rho: np.ndarray,
        delta: np.ndarray,
        k: int,
    ) -> list[dict[str, Any]]:
        """Tạo lời giải thích theo các cạnh của đồ thị đang được suy luận."""
        return DecisionService._ranked_influences(
            node_index, nodes, i, j, rho, delta, k
        )

    @staticmethod
    def _ranked_influences(
        node_index: int,
        nodes: pd.DataFrame,
        i: np.ndarray,
        j: np.ndarray,
        rho: np.ndarray,
        delta: np.ndarray,
        k: int,
    ) -> list[dict[str, Any]]:
        grouped: dict[int, dict[str, Any]] = {}
        for edge in np.flatnonzero((i == node_index) | (j == node_index)):
            other = int(j[edge] if i[edge] == node_index else i[edge])
            if not bool(nodes.iloc[other].is_train):
                continue
            row = nodes.iloc[other]
            rec_id = int(row.rec_id)
            weight = float(rho[edge])
            sign = float(delta[edge])
            contribution = weight * sign * float(row.c)
            if rec_id not in grouped:
                grouped[rec_id] = {
                    "rec_id": rec_id,
                    "analyst": row.analyst,
                    "ticker": row.ticker,
                    "date": pd.Timestamp(row.rec_date).date().isoformat(),
                    "x": int(row.x),
                    "y": int(row.y),
                    "c_values": [],
                    "contribution": 0.0,
                    "edges": [],
                }
            grouped[rec_id]["c_values"].append(float(row.c))
            grouped[rec_id]["contribution"] += contribution
            grouped[rec_id]["edges"].append(
                {
                    "rho": weight,
                    "delta": sign,
                    "c": float(row.c),
                    "contribution": contribution,
                }
            )
        for item in grouped.values():
            item["c"] = float(np.mean(item.pop("c_values")))
        ranked = sorted(
            grouped.values(),
            key=lambda item: (-abs(item["contribution"]), item["rec_id"]),
        )
        return ranked[: max(0, k)]
