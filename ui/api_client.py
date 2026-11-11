"""HTTP client mỏng cho API DARP."""

from __future__ import annotations

import os
from typing import Any

import requests


class APIError(RuntimeError):
    """Lỗi gọi API với thông báo phù hợp cho giao diện."""


class APIClient:
    """Gọi các endpoint API; giao diện không nạp package mô hình."""

    def __init__(self, base_url: str | None = None, timeout: float = 30.0) -> None:
        self.base_url = (base_url or os.getenv("API_URL", "http://localhost:8000")).rstrip("/")
        self.timeout = timeout

    def _request(
        self, method: str, path: str, **kwargs: Any
    ) -> Any:
        try:
            response = requests.request(
                method,
                f"{self.base_url}{path}",
                timeout=self.timeout,
                **kwargs,
            )
            response.raise_for_status()
        except requests.Timeout as error:
            raise APIError("API phản hồi quá chậm. Vui lòng thử lại sau.") from error
        except requests.ConnectionError as error:
            raise APIError("Không kết nối được API. Kiểm tra API_URL và trạng thái dịch vụ.") from error
        except requests.HTTPError as error:
            detail = response.text[:500]
            raise APIError(f"API trả lỗi {response.status_code}: {detail}") from error
        return response.json()

    def health(self) -> dict[str, Any]:
        """Kiểm tra tình trạng API."""
        return self._request("GET", "/health")

    def meta(self) -> dict[str, Any]:
        """Lấy lựa chọn tháng, chuyên gia và mã cổ phiếu."""
        return self._request("GET", "/meta")

    def results(self) -> dict[str, Any]:
        """Lấy kết quả thí nghiệm đã tính sẵn."""
        return self._request("GET", "/results")

    def events(
        self, month: str, only_disagree: bool = False, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Liệt kê sự kiện theo tháng."""
        return self._request(
            "GET",
            "/events",
            params={"month": month, "only_disagree": only_disagree, "limit": limit},
        )

    def event_detail(self, event_id: int) -> dict[str, Any]:
        """Lấy chi tiết sự kiện."""
        return self._request("GET", f"/events/{event_id}")

    def decide(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Đánh giá sự kiện giả định."""
        return self._request("POST", "/decide", json=payload)

    def toy(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Lấy ví dụ slide 15 hoặc đánh giá bảng tùy chỉnh."""
        if payload is None:
            return self._request("GET", "/toy")
        return self._request("POST", "/toy", json=payload)

    def experiment(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Chạy thử nghiệm với tham số tùy chỉnh."""
        return self._request("POST", "/experiment", json=payload)
