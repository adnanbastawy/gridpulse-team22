"""The service must reproduce the notebook numbers exactly (same rules, features and models)."""
import numpy as np
import pandas as pd

from gridpulse.quality import trust_rules

SIGNALS = ("load", "solar", "wind")


def test_forecast_parity_with_notebook(bundle, services):
    store, fc, _ = services
    nb = pd.read_parquet(bundle.path("parity_forecast.parquet"))
    nb["time_utc"] = pd.to_datetime(nb["time_utc"], utc=True)
    for s in SIGNALS:
        part = nb[nb["signal"] == s].set_index("time_utc")["forecast_mw"]
        usable = fc.usable(store.actual(s), s)
        ours = fc.predict_hours(s, usable, pd.DatetimeIndex(part.index))
        assert ours.notna().all(), s
        assert np.abs(ours.to_numpy() - part.to_numpy()).max() < 0.05, s


def test_trust_rules_counts(bundle, services):
    store, _, _ = services
    counts = {s: int(trust_rules(store.actual(s), s, bundle.config["trust_rules"])[0].sum()) for s in SIGNALS}
    assert counts == {"load": 36, "solar": 93, "wind": 18}


def test_anomaly_parity_on_2018(bundle, services):
    store, _, an = services
    nb = pd.read_parquet(bundle.path("parity_anomaly_test.parquet"))
    nb["time_utc"] = pd.to_datetime(nb["time_utc"], utc=True)
    start, end = nb["time_utc"].min(), nb["time_utc"].max()
    res = an.scan(start, end)
    for s in SIGNALS:
        ref = nb[nb["signal"] == s].set_index("time_utc")
        ours = res.hourly[s].reindex(ref.index)
        assert np.nanmax(np.abs(ours["nowcast_mw"] - ref["nowcast_mw"])) < 0.05, s
        for col in ("rules", "band_forecast", "band_nowcast", "combined"):
            assert (ours[col].astype(bool) == ref[col].astype(bool)).all(), (s, col)
    alerts = pd.read_csv(bundle.path("alerts_test_2018.csv"), parse_dates=["start_utc", "end_utc"])
    merged = res.events.merge(alerts, on=["signal", "start_utc", "end_utc"], how="outer", indicator=True, suffixes=("", "_nb"))
    assert (merged["_merge"] == "both").all(), "every 2018 alert must match the notebook"
    assert (merged["severity"] == merged["severity_nb"]).all()
    assert (merged["evidence"] == merged["evidence_nb"]).all()
