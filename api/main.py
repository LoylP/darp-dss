"""FastAPI cho hệ thống hỗ trợ quyết định DARP."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import DecideRequest, EventDecision, ExperimentRequest, ToyRequest
from darp.metrics import init_reliability
from darp.model import darp_fit, toy_example
from darp.service import DecisionService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Nạp artifacts đúng một lần trong suốt vòng đời ứng dụng."""
    app.state.service = DecisionService(
        Path("artifacts"), Path("configs/default.yaml")
    )
    yield


app = FastAPI(title="darp-dss", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _service(request: Request) -> DecisionService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="DecisionService chưa sẵn sàng.")
    return service


def _not_found(error: KeyError) -> HTTPException:
    return HTTPException(status_code=404, detail=str(error).strip("'"))


@app.get("/health")
def health() -> dict[str, str]:
    """Kiểm tra tình trạng API."""
    return {"status": "ok"}


@app.get("/meta")
def meta(request: Request) -> dict[str, Any]:
    """Cấu hình, tháng, analyst, broker và ticker."""
    return _service(request).meta()


@app.get("/results")
def results(request: Request) -> dict[str, Any]:
    """Kết quả precompute đầy đủ."""
    return _service(request).results


@app.get("/events")
def events(
    request: Request,
    month: str,
    only_disagree: bool = False,
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict[str, Any]]:
    """Danh sách sự kiện dự đoán trong một tháng."""
    try:
        return _service(request).list_events(month, only_disagree, limit)
    except KeyError as error:
        raise _not_found(error) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/events/{event_id}", response_model=EventDecision)
def event_detail(event_id: int, request: Request) -> dict[str, Any]:
    """Chi tiết quyết định đã đánh giá."""
    try:
        return _service(request).event_detail(event_id)
    except KeyError as error:
        raise _not_found(error) from error


@app.post("/decide", response_model=EventDecision)
def decide(payload: DecideRequest, request: Request) -> dict[str, Any]:
    """Đánh giá sự kiện mới mà không cần biết nhãn thật."""
    service = _service(request)
    try:
        return service.decide(
            payload.month,
            payload.ticker,
            [recommendation.model_dump() for recommendation in payload.recommendations],
        )
    except KeyError as error:
        raise _not_found(error) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/toy")
def get_toy() -> dict[str, Any]:
    """Ví dụ slide 15 mặc định."""
    confidence, majority, darp = toy_example()
    return {
        "c": {"a1": float(confidence[4]), "a2": float(confidence[5]), "a3": float(confidence[6])},
        "MV": majority,
        "DARP": darp,
    }


@app.post("/toy")
def post_toy(payload: ToyRequest) -> dict[str, Any]:
    """Tính DARP cho bảng toy tùy chỉnh."""
    rows = [row.model_dump() for row in payload.rows]
    event_ids = np.array([row["ev"] for row in rows])
    ratings = np.array([row["x"] for row in rows])
    truth = np.array([np.nan if row["y"] is None else row["y"] for row in rows])
    labeled = ~np.isnan(truth)
    if not labeled.any() or labeled.all():
        raise HTTPException(
            status_code=422, detail="Ví dụ cần có cả nút train và nút chưa gán nhãn."
        )
    c_labeled = init_reliability(ratings[labeled], truth[labeled].astype(int))
    i, j = np.triu_indices(len(rows), 1)
    features = np.stack(
        [
            np.array([rows[left]["analyst"] == rows[right]["analyst"] for left, right in zip(i, j)], dtype=float),
            np.array([rows[left]["broker"] == rows[right]["broker"] for left, right in zip(i, j)], dtype=float),
            np.array([rows[left]["sic"] // 100 == rows[right]["sic"] // 100 for left, right in zip(i, j)], dtype=float),
            np.array([rows[left]["sic"] // 10 == rows[right]["sic"] // 10 for left, right in zip(i, j)], dtype=float),
            np.array([rows[left]["sic"] == rows[right]["sic"] for left, right in zip(i, j)], dtype=float),
            np.zeros(len(i)),
        ],
        axis=1,
    )
    delta = np.where(
        (event_ids[i] == event_ids[j]) & (ratings[i] != ratings[j]), -1.0, 1.0
    )
    confidence, _, _ = darp_fit(
        len(rows),
        i,
        j,
        features,
        delta,
        labeled,
        c_labeled,
        active=np.array([True, True, True, True, True, False]),
    )
    current_event = int(event_ids.max())
    current = np.flatnonzero((event_ids == current_event) & ~labeled)
    if len(current) == 0:
        raise HTTPException(
            status_code=422, detail="Sự kiện mới nhất phải chứa nút chưa gán nhãn."
        )
    votes, counts = np.unique(ratings[current], return_counts=True)
    majority = int(votes[np.flatnonzero(counts == counts.max())[0]])
    darp = int(ratings[current[np.argmax(confidence[current])]])
    return {
        "nodes": [
            {**rows[index], "c": float(confidence[index])} for index in range(len(rows))
        ],
        "MV": majority,
        "DARP": darp,
    }


@app.post("/experiment")
def experiment(payload: ExperimentRequest, request: Request) -> dict[str, Any]:
    """Chạy đánh giá với tham số do người dùng chọn."""
    try:
        return _service(request).run_experiment(
            payload.TP, payload.OB, payload.PD, payload.n_months
        )
    except KeyError as error:
        raise _not_found(error) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
