"""Kiểm thử dịch vụ ra quyết định."""

import pytest

from darp.service import DecisionService


@pytest.fixture(scope="module")
def service() -> DecisionService:
    return DecisionService("artifacts", "configs/default.yaml")


def test_artifacts_load_and_meta(service: DecisionService) -> None:
    meta = service.meta()

    assert service.months()
    assert meta["analysts"]
    assert meta["tickers"]


def test_context_is_cached(service: DecisionService) -> None:
    month = service.months()[0]
    service.context.cache_clear()
    first = service.context(month)
    second = service.context(month)

    assert first is second
    assert service.context.cache_info().hits == 1


def test_event_detail_and_virtual_decision(service: DecisionService) -> None:
    month = service.months()[0]
    event = service.list_events(month, limit=1)[0]
    detail = service.event_detail(event["event_id"])
    meta = service.meta()
    analysts = [item["analyst"] for item in meta["analysts"][:2]]
    decision = service.decide(
        month,
        meta["tickers"][0]["ticker"],
        [{"analyst": analyst, "rating": 3 + index} for index, analyst in enumerate(analysts)],
    )

    assert detail["y"] in range(1, 6)
    assert detail["nodes"]
    assert any(node["n_train"] == 0 for node in detail["nodes"])
    assert decision["y"] is None
    assert set(decision["decisions"]) == {"MV", "WM", "DARP", "DARP_T3", "DARP_T5"}
    assert all(-1 <= node["c"] <= 1 for node in decision["nodes"])
    assert all(node["n_train"] >= 0 for node in decision["nodes"])
    assert all(
        len({influence["rec_id"] for influence in node["influences"]})
        == len(node["influences"])
        for node in detail["nodes"]
    )
    for node in detail["nodes"]:
        for influence in node["influences"]:
            expected = sum(
                edge["rho"] * edge["delta"] * edge["c"]
                for edge in influence["edges"]
            )
            assert influence["contribution"] == pytest.approx(expected)
