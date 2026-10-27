"""Injected faults must raise alerts; clean data must stay mostly quiet."""
import pandas as pd


def _window(store, signal, start, days=40):
    s = pd.Timestamp(start, tz="UTC")
    return store.actual(signal).loc[s: s + pd.Timedelta(days=days)].copy()


def test_injected_wind_dropout_is_caught(services):
    store, _, an = services
    v = _window(store, "wind", "2018-03-01")
    t0 = v.index[0] + pd.Timedelta(days=37, hours=10)
    v.loc[t0: t0 + pd.Timedelta(hours=3)] = 0.0
    res = an.check("wind", v)
    flagged = res.hourly["wind"].loc[t0: t0 + pd.Timedelta(hours=3), "combined"]
    assert flagged.any()


def test_frozen_load_sensor_is_a_data_error(services):
    store, _, an = services
    v = _window(store, "load", "2018-09-01")
    t0 = v.index[0] + pd.Timedelta(days=36, hours=5)
    v.loc[t0: t0 + pd.Timedelta(hours=9)] = v.loc[t0]
    res = an.check("load", v)
    h = res.hourly["load"].loc[t0: t0 + pd.Timedelta(hours=9)]
    assert h["data_error"].all() and h["reasons"].str.contains("flatline").all()
    assert (res.events["evidence"] == "data_quality_rule").any()


def test_clean_month_alert_rate_is_low(services):
    _, _, an = services
    res = an.scan(pd.Timestamp("2018-03-01", tz="UTC"), pd.Timestamp("2018-03-31 23:00", tz="UTC"))
    for s, h in res.hourly.items():
        assert h["combined"].mean() < 0.06, s
