"""GridPulse dispatcher app (Streamlit). Talks to the FastAPI service only."""
from __future__ import annotations

import io
import os
from datetime import date, datetime, time, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.environ.get("GRIDPULSE_API_URL", "http://localhost:8000").rstrip("/")
SIGNALS = ["load", "solar", "wind"]
LABEL = {"load": "Load (demand)", "solar": "Solar generation", "wind": "Wind generation"}
SEV_COLOR = {"CRITICAL": "#c0392b", "HIGH": "#e67e22", "MEDIUM": "#d4ac0d"}
C_ACTUAL, C_MODEL, C_TSO, C_BAND = "#1f4e79", "#e4572e", "#7f8c8d", "rgba(31,78,121,0.13)"

st.set_page_config(page_title="GridPulse", page_icon="⚡", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 3.2rem;}
div[data-testid="stMetricValue"] {font-size: 1.45rem;}
.small {color: #6b7280; font-size: 0.86rem;}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ API client
class ApiError(RuntimeError):
    pass


def api(method: str, path: str, **kw):
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=120, **kw)
    except requests.RequestException as exc:
        raise ApiError(f"API not reachable at {API_URL}: {exc}") from exc
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail")
        except ValueError:
            detail = r.text
        raise ApiError(f"{r.status_code}: {detail}")
    return r.json()


@st.cache_data(ttl=300, show_spinner=False)
def cached(path: str):
    return api("GET", path)


def utc_iso(d: date, h: int) -> str:
    return datetime.combine(d, time(h), tzinfo=timezone.utc).isoformat()


def layout(fig, title, height=380):
    fig.update_layout(title=dict(text=title, y=0.97, yanchor="top"), height=height,
                      margin=dict(l=10, r=10, t=56, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=-0.14, x=0, yanchor="top"), yaxis_title="MW",
                      plot_bgcolor="white")
    fig.update_xaxes(gridcolor="#eef0f3")
    fig.update_yaxes(gridcolor="#eef0f3")
    return fig


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## ⚡ GridPulse")
    st.caption("48-h forecasts and anomaly alerts for Spain's load, solar and wind (Team 22)")
    try:
        health = cached("/health")
        st.success("API online")
        st.markdown(f"<span class='small'>Dataset `{health['dataset_version']}` · {health['artifact_files_checked']} "
                    f"artifact files verified</span>", unsafe_allow_html=True)
        st.markdown("**Production models**")
        for s, m in health["models"].items():
            st.markdown(f"- {LABEL[s]}: `{m}`")
        lo, hi = (pd.Timestamp(x) for x in health["data_range_utc"])
        st.markdown(f"<span class='small'>Data: {lo:%Y-%m-%d} → {hi:%Y-%m-%d} (UTC)</span>", unsafe_allow_html=True)
    except ApiError as e:
        st.error(str(e))
        st.stop()
    st.divider()
    st.markdown("<span class='small'>Forecasts use only data at least 48 h old. "
                "Anomaly bands were calibrated on 2017 and applied unchanged to 2018.</span>", unsafe_allow_html=True)

tab_over, tab_fc, tab_an, tab_up, tab_bus = st.tabs(["Overview", "Forecast", "Anomalies", "Check your data", "Business"])

# ------------------------------------------------------------------ overview
with tab_over:
    info = cached("/info")
    st.subheader("How good are the forecasts? (test year 2018, 48 h ahead)")
    prod = pd.DataFrame(info["production"]).set_index("signal")
    cols = st.columns(3)
    for c, s in zip(cols, SIGNALS):
        r = prod.loc[s]
        c.metric(f"{LABEL[s]} · {r['production_model']}", f"{r['test_MAE']:,.0f} MW MAE",
                 f"{r['improvement_vs_naive_pct']:.0f}% better than naive", delta_color="normal")
        c.markdown(f"<span class='small'>nMAE {r['test_nMAE_pct']:.1f}% · operator (TSO) {r['tso_test_MAE']:,.0f} MW · "
                   f"naive {r['naive_test_MAE']:,.0f} MW</span>", unsafe_allow_html=True)

    test = pd.DataFrame(cached("/metrics/models?split=test"))
    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        for s in SIGNALS:
            d = test[test["signal"] == s].sort_values("nMAE_pct")
            fig.add_bar(name=LABEL[s], x=d["model"], y=d["nMAE_pct"])
        layout(fig, "All compared models: test nMAE (% of mean, lower is better)").update_yaxes(title="nMAE %")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        shap_df = pd.DataFrame(info["shap_mean_abs_test"])
        fig = go.Figure()
        for s in SIGNALS:
            fig.add_bar(name=LABEL[s], y=shap_df.index, x=shap_df[s], orientation="h")
        layout(fig, "What drives the forecast (mean |SHAP|, MW)").update_layout(barmode="group", yaxis_title=None,
                                                                                  xaxis_title="MW")
        st.plotly_chart(fig, use_container_width=True)
    tun = pd.DataFrame(info["tuning"]).T[["team_cv_MAE", "best_cv_MAE"]].astype(float).round(1)
    tun.columns = ["team LightGBM (CV MAE)", "Optuna-tuned (CV MAE)"]
    st.markdown("**Optuna tuning** (time-ordered folds inside 2016, MW)")
    st.dataframe(tun, use_container_width=True)
    st.info("The operator's own day-ahead forecast (TSO) is far more accurate: it uses weather forecasts this dataset "
            "does not contain. Our models are an independent forecast and the expectation behind the anomaly alerts.")

# ------------------------------------------------------------------ forecast
with tab_fc:
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 0.8, 1])
    sig = c1.selectbox("Signal", SIGNALS, format_func=LABEL.get, key="fc_sig")
    d0 = c2.date_input("Forecast made on (UTC)", date(2018, 7, 30), min_value=date(2015, 2, 1), max_value=date(2018, 12, 29))
    h0 = c3.number_input("at hour", 0, 23, 9)
    horizon = c4.slider("Horizon (h)", 1, 48, 48)
    try:
        fc = api("POST", "/forecast", json={"signal": sig, "origin_utc": utc_iso(d0, h0), "horizon_hours": horizon,
                                             "explain": True, "top_k": 8})
    except ApiError as e:
        st.error(str(e))
        st.stop()
    pts = pd.DataFrame(fc["points"])
    pts["t"] = pd.to_datetime(pts["target_time_utc"])
    fig = go.Figure()
    fig.add_scatter(x=pts["t"], y=pts["actual_mw"], name="actual", line=dict(color=C_ACTUAL, width=2.5))
    fig.add_scatter(x=pts["t"], y=pts["forecast_mw"], name=f"our forecast ({fc['model']})", line=dict(color=C_MODEL, width=2))
    fig.add_scatter(x=pts["t"], y=pts["tso_mw"], name="operator (TSO)", line=dict(color=C_TSO, dash="dot"))
    st.plotly_chart(layout(fig, f"{LABEL[sig]}: forecast made {fc['origin_utc'][:16].replace('T', ' ')} UTC"),
                    use_container_width=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE of this forecast", f"{fc['mae_vs_actual_mw']:,.0f} MW" if fc["mae_vs_actual_mw"] else "n/a")
    tso_mae = (pts["tso_mw"] - pts["actual_mw"]).abs().mean()
    m2.metric("MAE of the operator forecast", f"{tso_mae:,.0f} MW" if pd.notna(tso_mae) else "n/a")
    m3.metric("Hours forecast", f"{pts['forecast_mw'].notna().sum()} / {len(pts)}")
    for w in fc["warnings"]:
        st.warning(w)

    st.markdown("#### Why this number? (SHAP)")
    with_exp = pts[pts["explanation"].notna()]
    if not with_exp.empty:
        pick = st.select_slider("Target hour", options=list(with_exp["t"]), value=with_exp["t"].iloc[min(23, len(with_exp) - 1)],
                                format_func=lambda t: f"{t:%a %d %b %H:00} UTC")
        row = with_exp[with_exp["t"] == pick].iloc[0]
        exp = row["explanation"]
        con = pd.DataFrame(exp["contributions"]).iloc[::-1]
        fig = go.Figure(go.Bar(x=con["contribution_mw"], y=[f"{f} = {v:,.2f}" for f, v in zip(con["feature"], con["value"])],
                               orientation="h", marker_color=np.where(con["contribution_mw"] >= 0, C_MODEL, C_ACTUAL)))
        layout(fig, f"Start from the average {exp['base_value_mw']:,.0f} MW, each input pushes the forecast up (red) or "
                    f"down (blue) → {row['forecast_mw']:,.0f} MW", height=360).update_layout(yaxis_title=None, xaxis_title="MW")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("lag_Nh = the same signal N hours before the target hour (always at least 48 h old).")

# ------------------------------------------------------------------ anomalies
def anomaly_chart(sig: str, hourly: pd.DataFrame, events: pd.DataFrame):
    fig = go.Figure()
    fig.add_scatter(x=hourly["t"], y=hourly["nowcast_mw"], name="expected now (nowcast)", line=dict(color=C_TSO, dash="dot"))
    fig.add_scatter(x=hourly["t"], y=hourly["forecast_mw"], name="48-h forecast", line=dict(color=C_MODEL, width=1), opacity=0.6)
    fig.add_scatter(x=hourly["t"], y=hourly["actual_mw"], name="actual", line=dict(color=C_ACTUAL, width=2))
    for _, e in events.iterrows():
        fig.add_vrect(x0=e["start_utc"], x1=pd.Timestamp(e["end_utc"]) + pd.Timedelta(hours=1),
                      fillcolor=SEV_COLOR[e["severity"]], opacity=0.25, line_width=0)
    de = hourly[hourly["data_error"]]
    fig.add_scatter(x=de["t"], y=de["actual_mw"], mode="markers", name="data error", marker=dict(symbol="x", size=8, color="black"))
    return layout(fig, f"{LABEL[sig]} - shaded = alert (red critical, orange high, yellow medium)")


with tab_an:
    c1, c2, c3 = st.columns([1.6, 1.4, 1])
    sigs = c1.multiselect("Signals", SIGNALS, SIGNALS, format_func=LABEL.get)
    rng = c2.date_input("Window (UTC)", (date(2018, 5, 1), date(2018, 5, 14)), min_value=date(2015, 2, 20),
                        max_value=date(2018, 12, 31))
    sev = c3.selectbox("Minimum severity", ["MEDIUM", "HIGH", "CRITICAL"])
    if len(sigs) and isinstance(rng, tuple) and len(rng) == 2:
        try:
            res = api("POST", "/anomaly/scan", json={"start_utc": utc_iso(rng[0], 0), "end_utc": utc_iso(rng[1], 23),
                                                     "signals": sigs, "min_severity": sev, "include_hourly": True})
        except ApiError as e:
            st.error(str(e))
            st.stop()
        ev = pd.DataFrame(res["events"])
        cols = st.columns(len(sigs))
        for c, s in zip(cols, sigs):
            sm = res["summary"][s]
            c.metric(LABEL[s], f"{sum(sm['events'].values())} alerts")
            c.markdown(f"<span class='small'>{sm['events']['CRITICAL']} critical · {sm['events']['HIGH']} high · "
                       f"{sm['flagged_hours']} of {sm['hours_checked']} hours flagged</span>", unsafe_allow_html=True)
        for s in sigs:
            h = pd.DataFrame(res["hourly"][s])
            h["t"] = pd.to_datetime(h["time_utc"])
            e = ev[ev["signal"] == s] if not ev.empty else ev
            st.plotly_chart(anomaly_chart(s, h, e), use_container_width=True)
        if not ev.empty:
            st.markdown("#### Alerts and the evidence behind them")
            show = ev[["signal", "start_utc", "end_utc", "hours", "severity", "direction", "peak_dev_nowcast_MW",
                       "peak_ratio_to_band", "evidence", "other_signals_alerting", "trust_layer_reasons"]]
            st.dataframe(show.style.map(lambda v: f"color: {SEV_COLOR.get(v, 'inherit')}; font-weight: 600"
                                        if v in SEV_COLOR else "", subset=["severity"]), use_container_width=True, hide_index=True)
            st.caption("Evidence: data_quality_rule = a trust-layer rule fired; simultaneous_alert_other_signal = another "
                       "signal alerted at the same time (often a reporting error); public_holiday (load); "
                       "operator_forecast_also_far_off; unexplained_review = needs a human look.")
        else:
            st.success("No alerts in this window at this severity.")

# ------------------------------------------------------------------ upload
with tab_up:
    st.markdown("Upload hourly data (CSV with `time_utc,value_mw`, optional `wx_temp, wx_humidity, wx_wind_speed, "
                "wx_pressure, wx_clouds_all`). The first **35 days** are history; later hours are checked.")
    c1, c2 = st.columns([1, 2])
    up_sig = c1.selectbox("Signal", SIGNALS, format_func=LABEL.get, key="up_sig")
    with c2.expander("No file? Build a demo file from real 2018 data with an injected fault"):
        d1, d2, d3 = st.columns(3)
        fault = d1.selectbox("Fault", ["drop 50% for 6 h", "frozen sensor 10 h", "zero output 4 h", "none"])
        start_demo = d2.date_input("Data starts", date(2018, 4, 1), min_value=date(2015, 2, 1), max_value=date(2018, 11, 1))
        fault_day = d3.number_input("Fault on day", 36, 40, 38)
        if st.button("Build demo file"):
            s0 = datetime.combine(start_demo, time(0), tzinfo=timezone.utc)
            data = api("GET", "/data/series", params={"signal": up_sig, "start_utc": s0.isoformat(),
                                                       "end_utc": (s0 + timedelta(days=41)).isoformat()})
            df = pd.DataFrame({"time_utc": data["time_utc"], "value_mw": data["actual_mw"]})
            f0 = (fault_day - 1) * 24 + 11
            if fault.startswith("drop"):
                df.loc[f0:f0 + 5, "value_mw"] *= 0.5
            elif fault.startswith("frozen"):
                df.loc[f0:f0 + 9, "value_mw"] = df.loc[f0, "value_mw"]
            elif fault.startswith("zero"):
                df.loc[f0:f0 + 3, "value_mw"] = 0.0
            st.session_state["demo_csv"] = df.to_csv(index=False).encode()
            st.session_state["demo_fault"] = f"{fault} starting {df.loc[f0, 'time_utc'][:16]} UTC"
        if "demo_csv" in st.session_state:
            st.download_button("Download demo CSV", st.session_state["demo_csv"], "gridpulse_demo.csv", "text/csv")
            st.caption(f"Injected: {st.session_state['demo_fault']}")
    file = st.file_uploader("CSV file", type=["csv"])
    payload = file.getvalue() if file else st.session_state.get("demo_csv")
    if payload and st.button("Check for anomalies", type="primary"):
        try:
            res = api("POST", "/anomaly/upload", data={"signal": up_sig, "include_hourly": "true"},
                      files={"file": ("data.csv", io.BytesIO(payload), "text/csv")})
        except ApiError as e:
            st.error(str(e))
            st.stop()
        for w in res["warnings"]:
            st.info(w)
        h = pd.DataFrame(res["hourly"][up_sig])
        h["t"] = pd.to_datetime(h["time_utc"])
        ev = pd.DataFrame(res["events"])
        st.plotly_chart(anomaly_chart(up_sig, h, ev if not ev.empty else pd.DataFrame(columns=["start_utc", "end_utc", "severity"])),
                        use_container_width=True)
        if ev.empty:
            st.success("No anomalies found.")
        else:
            st.dataframe(ev, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ business
with tab_bus:
    b = cached("/business")
    st.subheader("What is the forecast worth? (2018, proxy)")
    st.caption(f"Assumption: each MWh of forecast error costs that hour's {b['assumptions']['cost_basis']}. "
               f"Real settlement rules differ, so read these as orders of magnitude. "
               f"Review cost: {b['assumptions']['review_minutes_per_alert']} min per alert.")
    cost = pd.DataFrame(b["forecast_cost_MEUR"]).set_index("signal")
    c = st.columns(3)
    for col, s in zip(c, SIGNALS):
        r = cost.loc[s]
        col.metric(LABEL[s], f"€{r['saving_vs_naive_MEUR']:,.1f} M saved vs naive")
        col.markdown(f"<span class='small'>closes {r['gap_to_tso_closed_pct']:.0f}% of the gap between naive and the "
                     f"operator forecast</span>", unsafe_allow_html=True)
    fig = go.Figure()
    for name, colr in [("naive_MEUR", "#bdc3c7"), ("ours_MEUR", C_MODEL), ("tso_MEUR", C_ACTUAL)]:
        fig.add_bar(name={"naive_MEUR": "naive", "ours_MEUR": "our model", "tso_MEUR": "operator (TSO)"}[name],
                    x=[LABEL[s] for s in cost.index], y=cost[name], marker_color=colr)
    st.plotly_chart(layout(fig, "Cost of forecast errors in 2018 (million EUR, imbalance-spread proxy)").update_yaxes(title="M EUR"),
                    use_container_width=True)
    st.subheader("Alert stream")
    wl = pd.DataFrame(b["alert_workload"]).set_index("signal")
    st.dataframe(wl.rename(columns={"alerts_2018": "alerts in 2018", "high_or_critical": "high + critical",
                                    "review_hours_all": "review hours (all)",
                                    "review_hours_high_critical": "review hours (high + critical)",
                                    "data_issue_evidence": "alerts pointing to data issues"}), use_container_width=True)
    rec = pd.DataFrame(b["injected_fault_recall"]).set_index("signal")
    st.markdown("**Share of injected faults caught by the alert policy** (30 cases per fault and size)")
    st.dataframe(rec.style.format("{:.0%}", na_rep="-"), use_container_width=True)
