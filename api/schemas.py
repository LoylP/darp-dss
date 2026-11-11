"""Mô hình yêu cầu API và kiểm tra đầu vào."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Recommendation(BaseModel):
    """Khuyến nghị của một chuyên gia."""

    model_config = ConfigDict(extra="forbid")

    analyst: str = Field(min_length=1)
    rating: int = Field(ge=1, le=5)


class DecideRequest(BaseModel):
    """Yêu cầu đánh giá một sự kiện giả định."""

    model_config = ConfigDict(extra="forbid")

    month: str
    ticker: str = Field(min_length=1)
    recommendations: list[Recommendation] = Field(min_length=2)

    @field_validator("month")
    @classmethod
    def validate_month(cls, value: str) -> str:
        """Kiểm tra hình dạng YYYY-MM."""
        try:
            parsed = date.fromisoformat(f"{value}-01")
        except ValueError as error:
            raise ValueError("month phải có định dạng YYYY-MM.") from error
        if parsed.strftime("%Y-%m") != value:
            raise ValueError("month phải có định dạng YYYY-MM.")
        return value

    @field_validator("recommendations")
    @classmethod
    def validate_unique_analysts(
        cls, value: list[Recommendation]
    ) -> list[Recommendation]:
        """Không cho phép lặp chuyên gia trong cùng yêu cầu."""
        analysts = [recommendation.analyst for recommendation in value]
        if len(analysts) != len(set(analysts)):
            raise ValueError("Không được lặp analyst trong cùng sự kiện.")
        return value


class NodeOut(BaseModel):
    """Thông tin một nút khuyến nghị trong kết quả."""

    analyst: str
    broker: str
    rating: int
    rating_name: str
    c: float
    wm_weight: float
    n_train: int
    true_q: float | None
    influences: list[dict[str, object]]


class EventDecision(BaseModel):
    """Kết quả quyết định của một sự kiện."""

    event_id: int | None
    ticker: str
    date: str | None
    division: str
    y: int | None
    decisions: dict[str, int]
    correct: dict[str, bool] | None
    nodes: list[NodeOut]
    eta: dict[str, float]



class ToyRow(BaseModel):
    """Một dòng trong ví dụ toy."""

    model_config = ConfigDict(extra="forbid")

    ev: int
    analyst: str
    broker: str
    sic: int = Field(ge=1000, le=9999)
    x: int = Field(ge=1, le=5)
    y: int | None = Field(default=None, ge=1, le=5)


class ToyRequest(BaseModel):
    """Dữ liệu tùy chỉnh cho ví dụ toy."""

    model_config = ConfigDict(extra="forbid")

    rows: list[ToyRow] = Field(min_length=3)


class ExperimentRequest(BaseModel):
    """Tham số thay đổi cho thực nghiệm."""

    model_config = ConfigDict(extra="forbid")

    TP: int = Field(default=1, ge=1, le=3)
    OB: int = Field(default=14, ge=1)
    PD: int = Field(default=1, ge=1, le=5)
    n_months: int = Field(default=12, ge=1, le=12)
