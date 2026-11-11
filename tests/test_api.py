"""Kiểm thử contract endpoint FastAPI."""

from fastapi.testclient import TestClient

from api.main import app
from api.schemas import ExperimentRequest


def test_all_api_endpoints_and_validation() -> None:
    assert ExperimentRequest().n_months == 12
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        meta = client.get("/meta")
        assert meta.status_code == 200
        choices = meta.json()
        month = choices["months"][0]
        ticker = choices["tickers"][0]["ticker"]
        analysts = [item["analyst"] for item in choices["analysts"][:2]]

        assert client.get("/results").status_code == 200
        events = client.get("/events", params={"month": month, "limit": 1})
        assert events.status_code == 200
        event_id = events.json()[0]["event_id"]
        assert client.get(f"/events/{event_id}").status_code == 200
        assert client.get("/toy").status_code == 200
        assert client.post(
            "/toy",
            json={
                "rows": [
                    {"ev": 0, "analyst": "a1", "broker": "X", "sic": 7372, "x": 5, "y": 5},
                    {"ev": 1, "analyst": "a2", "broker": "Y", "sic": 7371, "x": 2, "y": 4},
                    {"ev": 2, "analyst": "a3", "broker": "Y", "sic": 7373, "x": 4},
                ]
            },
        ).status_code == 200
        decision = client.post(
            "/decide",
            json={
                "month": month,
                "ticker": ticker,
                "recommendations": [
                    {"analyst": analysts[0], "rating": 3},
                    {"analyst": analysts[1], "rating": 4},
                ],
            },
        )
        assert decision.status_code == 200
        assert decision.json()["y"] is None
        assert all("n_train" in node for node in decision.json()["nodes"])
        assert client.post(
            "/decide",
            json={
                "month": month,
                "ticker": ticker,
                "recommendations": [
                    {"analyst": "unknown", "rating": 3},
                    {"analyst": analysts[1], "rating": 4},
                ],
            },
        ).status_code == 422
        assert client.get("/events", params={"month": "1900-01"}).status_code == 404
        experiment = client.post(
            "/experiment",
            json={"TP": 1, "OB": 14, "PD": 1, "n_months": 1},
        )
        assert experiment.status_code == 200
        assert experiment.json()["n_test_events"] > 0
        assert set(experiment.json()["ACC_CI95"]) == {"MV", "WM", "DARP"}
