import io

import pandas as pd


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and set(body["models"]) == {"load", "solar", "wind"}


def test_forecast_with_explanations(client):
    r = client.post("/forecast", json={"signal": "load", "origin_utc": "2018-06-04T09:00:00Z", "explain": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["points"]) == 48 and body["points"][0]["horizon_h"] == 1
    assert body["points"][0]["explanation"]["contributions"]
    assert body["mae_vs_actual_mw"] is not None


def test_forecast_validation_errors(client):
    assert client.post("/forecast", json={"signal": "coal", "origin_utc": "2018-06-04T09:00:00Z"}).status_code == 422
    assert client.post("/forecast", json={"signal": "load", "origin_utc": "2018-06-04T09:30:00Z"}).status_code == 422
    assert client.post("/forecast", json={"signal": "load", "origin_utc": "2018-06-04T09:00:00Z",
                                          "horizon_hours": 72}).status_code == 422
    r = client.post("/forecast", json={"signal": "load", "origin_utc": "2030-01-01T00:00:00Z"})
    assert r.status_code == 400 and "origin" in r.json()["detail"]


def test_custom_forecast(client):
    s = client.get("/data/series", params={"signal": "wind", "start_utc": "2018-02-01T00:00:00Z",
                                           "end_utc": "2018-02-20T23:00:00Z"}).json()
    hist = [{"time_utc": t, "value_mw": v} for t, v in zip(s["time_utc"], s["actual_mw"])]
    r = client.post("/forecast/custom", json={"signal": "wind", "history": hist, "horizon_hours": 24})
    assert r.status_code == 200, r.text
    assert len(r.json()["points"]) == 24


def test_scan(client):
    r = client.post("/anomaly/scan", json={"start_utc": "2018-05-01T00:00:00Z", "end_utc": "2018-05-10T23:00:00Z",
                                           "include_hourly": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["summary"]) == {"load", "solar", "wind"}
    assert len(body["hourly"]["load"]) == 240
    assert any(e["start_utc"].startswith("2018-05-06T10") and e["signal"] == "load" for e in body["events"])


def test_scan_rejects_bad_window(client):
    assert client.post("/anomaly/scan", json={"start_utc": "2018-05-10T00:00:00Z",
                                              "end_utc": "2018-05-01T00:00:00Z"}).status_code == 422
    r = client.post("/anomaly/scan", json={"start_utc": "2015-01-05T00:00:00Z", "end_utc": "2015-01-06T00:00:00Z"})
    assert r.status_code == 400


def test_upload_csv(client):
    s = client.get("/data/series", params={"signal": "solar", "start_utc": "2018-04-01T00:00:00Z",
                                           "end_utc": "2018-05-10T23:00:00Z"}).json()
    df = pd.DataFrame({"time_utc": s["time_utc"], "value_mw": s["actual_mw"]})
    df.loc[37 * 24 + 10: 37 * 24 + 13, "value_mw"] = 0.0          # solar zero at midday
    r = client.post("/anomaly/upload", data={"signal": "solar"},
                    files={"file": ("x.csv", io.BytesIO(df.to_csv(index=False).encode()), "text/csv")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["events"], "the injected zero output should raise an alert"


def test_upload_rejects_bad_file(client):
    r = client.post("/anomaly/upload", data={"signal": "solar"},
                    files={"file": ("x.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")})
    assert r.status_code == 400


def test_alerts_and_business(client):
    a = client.get("/alerts/2018", params={"signal": "solar", "min_severity": "CRITICAL"}).json()
    assert a and all(x["severity"] == "CRITICAL" and x["signal"] == "solar" for x in a)
    b = client.get("/business").json()
    assert {"forecast_cost_MEUR", "alert_workload"} <= set(b)
    assert client.get("/info").status_code == 200
    assert len(client.get("/metrics/models", params={"split": "validation"}).json()) > 20
