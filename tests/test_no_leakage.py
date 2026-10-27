"""A forecast made at an origin must not change when data after the origin changes or disappears."""
import numpy as np
import pandas as pd


def test_future_data_cannot_change_forecast(services):
    store, fc, _ = services
    origin = pd.Timestamp("2018-06-04 09:00", tz="UTC")
    base = fc.forecast("load", origin, 48).table["forecast_mw"]
    original = store.df["load_actual"].copy()
    try:
        store.df.loc[store.df.index > origin, "load_actual"] = 1e9      # corrupt the future
        corrupted = fc.forecast("load", origin, 48).table["forecast_mw"]
    finally:
        store.df["load_actual"] = original
    assert np.allclose(base.to_numpy(), corrupted.to_numpy(), equal_nan=True)


def test_inputs_are_at_least_48h_old(bundle):
    assert min(bundle.config["forecast"]["lags_h"]) >= 48


def test_explanations_add_up(services):
    _, fc, _ = services
    res = fc.forecast("solar", pd.Timestamp("2018-07-10 09:00", tz="UTC"), 48, explain=True, top_k=11)
    for ts, e in res.explanations.items():
        total = e["base_value_mw"] + sum(c["contribution_mw"] for c in e["contributions"])
        assert abs(total - res.table.loc[ts, "forecast_mw"]) < 0.01
