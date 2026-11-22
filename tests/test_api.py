import pytest
from fastapi.testclient import TestClient

FRAUDY = {"step": 3, "type": "TRANSFER", "amount": 950000.0, "nameOrig": "C1", "oldbalanceOrg": 950000.0,
          "newbalanceOrig": 0.0, "nameDest": "C2", "oldbalanceDest": 0.0, "newbalanceDest": 0.0}
NORMAL = {"step": 14, "type": "PAYMENT", "amount": 1200.0, "nameOrig": "C3", "oldbalanceOrg": 50000.0,
          "newbalanceOrig": 48800.0, "nameDest": "M4", "oldbalanceDest": 0.0, "newbalanceDest": 0.0}


@pytest.fixture(scope="module")
def client(trained):
    from auditlens.api import main
    main.bundle.cache_clear(); main.explainer.cache_clear(); main.rag.cache_clear()
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["model"]["loaded"]


def test_predict_scores_and_explains(client):
    r = client.post("/predict", json={"transactions": [FRAUDY, NORMAL], "explain": True, "narrate": True})
    assert r.status_code == 200, r.text
    p = r.json()["predictions"]
    assert p[0]["fraud_score"] > p[1]["fraud_score"]
    assert len(p[0]["reasons"]) == 4
    if p[0]["flagged"]:
        assert p[0]["narrative"]


def test_predict_validation(client):
    bad = dict(NORMAL, type="WIRE")
    assert client.post("/predict", json={"transactions": [bad]}).status_code == 422


def test_forecast(client):
    r = client.post("/forecast", json={"group": "CASH_OUT", "horizon": 5})
    assert r.status_code == 200 and len(r.json()["forecast"]) == 5
    assert client.post("/forecast", json={"group": "NOPE"}).status_code == 404


def test_ask(client):
    r = client.post("/ask", json={"question": "How long is audit documentation retained?"}).json()
    assert r["citations"] and "seven years" in r["answer"]


def test_request_log_written(client, trained):
    client.get("/health")
    log = (trained["tmp"] / "requests.jsonl").read_text()
    assert '"path": "/health"' in log
