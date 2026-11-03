import pytest
from fastapi.testclient import TestClient

from horus_ai.api.main import app

VALID = dict(
    tower_id="OVS1",
    ts_wall=1.793321e9,
    master_offset_ns=-85046,
    freq_ppb=120.5,
    path_delay_ns=45000,
    servo_state="s2",
)


@pytest.fixture
def client():
    with TestClient(app) as c:  # the "with" makes the startup code run
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_valid_sample(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    assert r.json()["status"] == "warming_up"  # the window is not full yet


def test_predict_missing_field(client):
    bad = {k: v for k, v in VALID.items() if k != "freq_ppb"}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_bad_servo_state(client):
    assert client.post("/predict", json={**VALID, "servo_state": "s9"}).status_code == 422


def test_timing_header(client):
    r = client.get("/health")
    assert "x-process-time-ms" in r.headers