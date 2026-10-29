"""GridPulse dispatcher app (Streamlit). Talks to the FastAPI service only."""
from __future__ import annotations

import io
import os
import re
from datetime import date, datetime, time, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Theme is set from code, so no separate .streamlit/config.toml file is needed.
from streamlit import config as _cfg

for _k, _v in {"theme.base": "dark", "theme.primaryColor": "#84a98c", "theme.backgroundColor": "#182126",
               "theme.secondaryBackgroundColor": "#2f3e46", "theme.textColor": "#ffffff"}.items():
    try:
        _cfg.set_option(_k, _v)
    except Exception:
        pass

API_URL = os.environ.get("GRIDPULSE_API_URL", "http://localhost:8000").rstrip("/")
SIGNALS = ["load", "solar", "wind"]
LABEL = {"load": "Load (demand)", "solar": "Solar generation", "wind": "Wind generation"}
SEV_COLOR = {"CRITICAL": "#e74c3c", "HIGH": "#f39c12", "MEDIUM": "#f4d03f"}
# Palette: White, Muted Teal, Deep Teal, Dark Slate Grey, Charcoal Blue
WHITE, TEAL, DEEP, SLATE, CHAR = "#ffffff", "#84a98c", "#52796f", "#354f52", "#2f3e46"
C_ACTUAL, C_MODEL, C_TSO, C_BAND = CHAR, DEEP, TEAL, "rgba(47,62,70,0.10)"  # chart colours (charts have a white background)
BG, GRID = "#182126", "#e3e9e6"

st.set_page_config(page_title="GridPulse", page_icon="⚡", layout="wide")

def inject_css(css: str):
    """Minify (no comments / blank lines) so Streamlit's markdown parser can't break the style block."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    css = re.sub(r"\s+", " ", css).strip()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


GP_CSS = """
html {color-scheme: dark;}
/* ---- base + animated background ---- */
html, body, .stApp, [data-testid="stApp"] {background-color: #182126 !important;}
[data-testid="stMain"] {position: relative; z-index: 1;}
[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"] {background: transparent !important;}
.stApp::before {content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background-image: radial-gradient(circle, rgba(132,169,140,.55) 1.4px, transparent 2px),
                    radial-gradient(circle, rgba(255,255,255,.30) 1.1px, transparent 1.7px),
                    radial-gradient(circle, rgba(82,121,111,.60) 1.2px, transparent 1.9px);
  background-size: 52px 52px, 91px 91px, 137px 137px;
  background-position: 0 0, 23px 41px, 70px 12px;
  animation: gp-drift 70s linear infinite, gp-twinkle 6s ease-in-out infinite alternate;
  -webkit-mask-image: radial-gradient(ellipse at 50% 35%, #000 25%, transparent 90%);
          mask-image: radial-gradient(ellipse at 50% 35%, #000 25%, transparent 90%);}
.stApp::after {content: ""; position: fixed; inset: -20%; z-index: 0; pointer-events: none;
  background: radial-gradient(520px circle at 18% 22%, rgba(82,121,111,.22), transparent 70%),
              radial-gradient(620px circle at 82% 72%, rgba(53,79,82,.35), transparent 70%),
              radial-gradient(420px circle at 60% 10%, rgba(132,169,140,.12), transparent 70%);
  animation: gp-float 22s ease-in-out infinite alternate;}
@keyframes gp-drift {to {background-position: 52px 104px, -68px 132px, -67px 149px;}}
@keyframes gp-twinkle {from {opacity: .55;} to {opacity: 1;}}
@keyframes gp-float {from {transform: translate3d(-2%, -1%, 0) scale(1);} to {transform: translate3d(3%, 2%, 0) scale(1.08);}}
@media (prefers-reduced-motion: reduce) {.stApp::before, .stApp::after {animation: none;}}

/* ---- text ---- */
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp span,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-testid="stCaptionContainer"] {color: #ffffff;}
.stApp h2, .stApp h3, .stApp h4 {color: #ffffff;}
.stApp code {background: #354f52 !important; color: #ffffff !important; border: 1px solid #52796f; border-radius: 6px; padding: 1px 6px;}
.small {color: #ffffff !important; font-size: 0.88rem;}
.block-container {padding-top: 3.2rem;}
div[data-testid="stMetricValue"] {font-size: 1.45rem;}

/* ---- sidebar ---- */
[data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {background-color: #2f3e46 !important; border-right: 1px solid #354f52;}

/* ---- tabs ---- */
.stTabs [data-baseweb="tab"] {color: #ffffff !important;}
.stTabs [aria-selected="true"] {color: #ffffff !important;}
.stTabs [data-baseweb="tab-highlight"] {background-color: #84a98c !important;}
.stTabs [data-baseweb="tab-border"] {background-color: #354f52 !important;}

/* ---- buttons ---- */
[data-testid^="stBaseButton-secondary"], button[kind="secondary"], [data-testid="stDownloadButton"] button {
  background: #354f52 !important; color: #ffffff !important; border: 1px solid #52796f !important;}
[data-testid^="stBaseButton-secondary"]:hover, button[kind="secondary"]:hover {border-color: #84a98c !important; color: #ffffff !important;}
[data-testid^="stBaseButton-primary"], button[kind="primary"] {
  background: #84a98c !important; border: 1px solid #84a98c !important; color: #182126 !important; font-weight: 700;}
[data-testid^="stBaseButton-primary"] *, button[kind="primary"] * {color: #182126 !important;}
[data-testid^="stBaseButton-primary"]:hover, button[kind="primary"]:hover {background: #9cc2a4 !important;}

/* ---- inputs / selects / dates ---- */
[data-testid="stSelectbox"] div[data-baseweb="select"] div, [data-testid="stMultiSelect"] div[data-baseweb="select"] div,
[data-testid="stDateInput"] div[data-baseweb="input"] div, [data-testid="stDateInput"] div[data-baseweb="base-input"],
[data-testid="stNumberInput"] div[data-baseweb="input"] div, [data-testid="stNumberInput"] div[data-baseweb="base-input"],
[data-testid="stTextInput"] div[data-baseweb="input"] div, [data-testid="stTextInput"] div[data-baseweb="base-input"] {
  background-color: #2f3e46 !important; color: #ffffff !important;}
div[data-baseweb="select"] > div, div[data-baseweb="input"] {border: 1px solid #52796f !important; border-radius: 8px !important;}
[data-testid="stSelectbox"] *, [data-testid="stMultiSelect"] input, [data-testid="stDateInput"] input,
[data-testid="stNumberInput"] input, [data-testid="stTextInput"] input {color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;}
input::placeholder {color: #ffffff !important; -webkit-text-fill-color: #ffffff !important;}
div[data-baseweb="select"] svg, [data-testid="stNumberInput"] svg, [data-testid="stDateInput"] svg {fill: #ffffff !important; color: #ffffff !important;}
[data-testid="stNumberInput"] button {background: #354f52 !important; color: #ffffff !important; border-color: #52796f !important;}
span[data-baseweb="tag"] {background-color: #52796f !important;}
span[data-baseweb="tag"] * {color: #ffffff !important; background: transparent !important;}
div[data-baseweb="popover"] div, div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li, ul[role="listbox"] {
  background-color: #2f3e46 !important; color: #ffffff !important;}
div[data-baseweb="popover"] li:hover, div[data-baseweb="popover"] li[aria-selected="true"], li[role="option"]:hover {background-color: #354f52 !important;}
div[data-baseweb="calendar"] [aria-selected="true"], div[data-baseweb="calendar"] [aria-selected="true"] * {
  background-color: #84a98c !important; color: #182126 !important;}
/* ---- charts: white cards ---- */
[data-testid="stPlotlyChart"] {background: #ffffff; border-radius: 14px; overflow: hidden; padding: 4px; margin-bottom: 1rem;}
[data-testid="stPlotlyChart"] span, [data-testid="stPlotlyChart"] p {color: #2f3e46;}
/* ---- file uploader ---- */
[data-testid="stFileUploaderDropzone"] {background: #2f3e46 !important; border: 1px dashed #52796f !important;}
[data-testid="stFileUploaderDropzone"] * {color: #ffffff !important;}
[data-testid="stFileUploaderDropzone"] button {background: #354f52 !important; border: 1px solid #52796f !important;}

/* ---- expander, alerts, slider ---- */
[data-testid="stExpander"] {background: rgba(47,62,70,.70); border: 1px solid #354f52 !important; border-radius: 10px;}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {color: #ffffff !important;}
[data-testid="stSlider"] [data-testid="stTickBar"] * {color: #ffffff !important;}
[data-testid="stAlert"] * {color: #ffffff !important;}

/* ---- dark tables (replace the white dataframes) ---- */
.dt {overflow-x: auto; border: 1px solid #354f52; border-radius: 12px; background: rgba(47,62,70,.60); margin-bottom: 1rem;}
.dt table {border-collapse: collapse; width: 100%; font-size: .9rem;}
.dt th {background: #354f52; color: #ffffff !important; text-align: left; padding: 9px 14px; font-weight: 700; border-bottom: 1px solid #52796f; white-space: nowrap;}
.dt td {color: #ffffff; padding: 8px 14px; border-bottom: 1px solid #354f52; text-align: right;}
.dt td:first-child, .dt th:first-child {text-align: left;}
.dt tr:hover td {background: rgba(53,79,82,.35);}
"""
inject_css(GP_CSS)


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


def dark_chart(fig, **kw):
    kw.setdefault("use_container_width", True)
    kw["theme"] = None  # use our own dark template, not Streamlit's light one
    return st.plotly_chart(fig, **kw)


def show_table(df, styler=None, index=True):
    """Dark HTML table (the built-in dataframe widget is white unless Streamlit's own theme is dark)."""
    sty = styler if styler is not None else df.style.format(precision=2, na_rep="-")
    if not index:
        sty = sty.hide(axis="index")
    html = " ".join(line.strip() for line in sty.to_html().splitlines() if line.strip())
    st.markdown(f'<div class="dt">{html}</div>', unsafe_allow_html=True)


def utc_iso(d: date, h: int) -> str:
    return datetime.combine(d, time(h), tzinfo=timezone.utc).isoformat()


def layout(fig, title, height=380):
    fig.update_layout(title=dict(text=title, y=0.97, yanchor="top", font=dict(color=CHAR, size=16)), height=height,
                      margin=dict(l=14, r=14, t=60, b=14), hovermode="x unified",
                      legend=dict(orientation="h", y=-0.14, x=0, yanchor="top", font=dict(color=CHAR)), yaxis_title="MW",
                      template="plotly_white",
                      paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                      font=dict(color=CHAR),
                      hoverlabel=dict(bgcolor="#ffffff", bordercolor=DEEP, font=dict(color=CHAR, size=13)),
                      colorway=[CHAR, TEAL, DEEP, SLATE, "#b7c9c0"])
    fig.update_xaxes(gridcolor=GRID, zerolinecolor="#c9d6d0", linecolor="#c9d6d0", tickfont=dict(color=CHAR),
                     title_font=dict(color=CHAR))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="#c9d6d0", linecolor="#c9d6d0", tickfont=dict(color=CHAR),
                     title_font=dict(color=CHAR))
    return fig


# ------------------------------------------------------------------ landing page
if "started" not in st.session_state:
    st.session_state["started"] = False


def _start():
    st.session_state["started"] = True


def _home():
    st.session_state["started"] = False


def landing():
    inject_css("""
[data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"] {display: none;}
.block-container {max-width: 1100px; padding-top: 2rem;}
.hero {text-align: center; padding: 2.2rem 0 0.5rem;}
.hero .tag {display: inline-block; padding: 4px 14px; border: 1px solid #52796f; border-radius: 999px;
            color: #84a98c; font-size: 0.8rem; letter-spacing: .12em; text-transform: uppercase;}
.hero h1 {font-size: 4.2rem; font-weight: 800; margin: 0.6rem 0 0.2rem; line-height: 1.05;
          background: linear-gradient(90deg, #ffffff, #b9d8c1 45%, #84a98c);
          -webkit-background-clip: text; background-clip: text; color: transparent !important;}
.hero p.sub {font-size: 1.25rem; color: #ffffff !important; max-width: 640px; margin: 0.4rem auto 0;}
.hero p.sub.q {font-size: 1.8rem; font-weight: 700; color: #ffffff !important; max-width: 760px; margin-top: 1rem;}
.hero p.sub.facts {font-size: 1.1rem; color: #ffffff !important; max-width: 820px;}
.hero p.sub.facts span {display: inline-block; margin: 2px 8px; color: #ffffff;}
.pulse {width: 100%; max-width: 900px; margin: 1.2rem auto 0; display: block;}
.pulse .line {fill: none; stroke: #ffffff; stroke-width: 3; stroke-linecap: round; stroke-linejoin: round;
              stroke-dasharray: 1400; stroke-dashoffset: 1400; animation: draw 3.2s ease-out forwards;}
.pulse .fc {fill: none; stroke: #84a98c; stroke-width: 2.5; stroke-dasharray: 7 7; opacity: 0;
            animation: fade 1s ease-out 2.6s forwards;}
.pulse .band {fill: #84a98c; opacity: 0; animation: fadeb 1s ease-out 2.6s forwards;}
.pulse .dot {fill: #e74c3c; opacity: 0; animation: fade 0.4s ease-out 3s forwards;}
.pulse .ring {fill: none; stroke: #e74c3c; stroke-width: 2; opacity: 0; transform-box: fill-box;
              transform-origin: center; animation: ring 2s ease-out 3.2s infinite;}
.pulse text {fill: #ffffff; font-size: 13px; opacity: 0; animation: fade 1s ease-out 3.2s forwards;}
@keyframes draw {to {stroke-dashoffset: 0;}}
@keyframes fade {to {opacity: 1;}}
@keyframes fadeb {to {opacity: .14;}}
@keyframes ring {0% {transform: scale(.4); opacity: .9;} 100% {transform: scale(2.6); opacity: 0;}}
.chips {text-align: center; margin: .4rem 0 1.2rem;}
.chips span {display: inline-block; margin: 4px 6px; padding: 6px 16px; border-radius: 999px;
             background: #2f3e46; color: #ffffff; font-size: .92rem;}
.step {background: #2f3e46; border: 1px solid #354f52; border-radius: 16px; padding: 1.3rem 1.3rem 1.1rem; height: 100%;}
.step .n {font-size: 2.4rem; font-weight: 800; color: #52796f; line-height: 1;}
.step h4 {margin: .4rem 0 .3rem; color: #ffffff !important;}
.step p {color: #ffffff !important; font-size: .95rem; margin: 0;}
.how {text-align: center; margin: 2.6rem 0 1rem; letter-spacing: .08em;}
.foot {text-align: center; color: #ffffff; font-size: .85rem; margin-top: 2.5rem;}
""")

    st.markdown("""<div class="hero">
<span class="tag">Spain's power grid</span>
<h1>GridPulse</h1>
<p class="sub q">So why should we wait for problems to happen?</p>
<p class="sub">The grid doesn't wait for you to be ready.</p>
<p class="sub facts"><span>Demand changes.</span> <span>Renewable generation fluctuates.</span> <span>Anomalies happen.</span></p>
<svg class="pulse" viewBox="0 0 900 200" xmlns="http://www.w3.org/2000/svg">
<path class="band" d="M 560 100 C 600 60, 640 50, 680 78 S 760 120, 840 72 L 840 112 C 760 160, 720 132, 680 118 S 600 104, 560 138 Z"/>
<path class="line" d="M 0 120 C 40 110, 60 70, 110 78 S 170 140, 220 122 S 290 56, 340 70 S 400 150, 450 128 S 510 70, 540 92 L 560 118 L 575 40 L 592 150 L 610 100 C 650 70, 690 100, 720 92"/>
<path class="fc" d="M 560 120 C 600 82, 640 84, 680 98 S 760 128, 840 94"/>
<circle class="dot" cx="575" cy="40" r="6"/>
<circle class="ring" cx="575" cy="40" r="10"/>
<text x="590" y="36">anomaly</text>
<text x="700" y="176">48 h forecast</text>
</svg>
</div>
<div class="chips"><span>⚡ Load · ☀️ Solar · 🌬️ Wind</span><span>🔮 48-hour forecasts</span>
<span>🚨 Anomaly alerts</span><span>🔍 Every number explained</span></div>""", unsafe_allow_html=True)

    _, mid, _ = st.columns([2, 1.2, 2])
    mid.button("Start  →", type="primary", use_container_width=True, on_click=_start)

    st.markdown("<h3 class='how'>HOW IT WORKS</h3>", unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.markdown("""<div class="step"><div class="n">01</div><h4>Forecast</h4>
<p>Pick a signal and a moment. GridPulse predicts the next 48 hours using only data that is at least 48 hours old, then compares itself with the operator's own forecast.</p></div>""", unsafe_allow_html=True)
    b.markdown("""<div class="step"><div class="n">02</div><h4>Explain</h4>
<p>Every forecast comes with the reasons behind it: which past hours and weather inputs pushed the number up or down.</p></div>""", unsafe_allow_html=True)
    c.markdown("""<div class="step"><div class="n">03</div><h4>Alert</h4>
<p>When reality drifts far from what was expected, an alert is raised with its severity and evidence, so a human knows whether it is a real event or a bad sensor.</p></div>""", unsafe_allow_html=True)
    st.markdown("<div class='foot'>You can also upload your own hourly data and check it for faults.</div>", unsafe_allow_html=True)


if not st.session_state["started"]:
    landing()
    st.stop()


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## ⚡ GridPulse")
    st.button("← Home", on_click=_home, use_container_width=True)
    st.caption("48-h forecasts and anomaly alerts for Spain's load, solar and wind")
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
        dark_chart(fig, use_container_width=True)
    with right:
        shap_df = pd.DataFrame(info["shap_mean_abs_test"])
        fig = go.Figure()
        for s in SIGNALS:
            fig.add_bar(name=LABEL[s], y=shap_df.index, x=shap_df[s], orientation="h")
        layout(fig, "What drives the forecast (mean |SHAP|, MW)").update_layout(barmode="group", yaxis_title=None,
                                                                                  xaxis_title="MW")
        dark_chart(fig, use_container_width=True)
    tun = pd.DataFrame(info["tuning"]).T[["team_cv_MAE", "best_cv_MAE"]].astype(float).round(1)
    tun.columns = ["team LightGBM (CV MAE)", "Optuna-tuned (CV MAE)"]
    st.markdown("**Optuna tuning** (time-ordered folds inside 2016, MW)")
    show_table(tun)
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
    dark_chart(layout(fig, f"{LABEL[sig]}: forecast made {fc['origin_utc'][:16].replace('T', ' ')} UTC"),
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
                               orientation="h", marker_color=np.where(con["contribution_mw"] >= 0, TEAL, CHAR)))
        layout(fig, f"Start from the average {exp['base_value_mw']:,.0f} MW, each input pushes the forecast up (light green) or "
                    f"down (dark) → {row['forecast_mw']:,.0f} MW", height=360).update_layout(yaxis_title=None, xaxis_title="MW")
        dark_chart(fig, use_container_width=True)
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
    fig.add_scatter(x=de["t"], y=de["actual_mw"], mode="markers", name="data error", marker=dict(symbol="x", size=9, color="#c0392b"))
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
            dark_chart(anomaly_chart(s, h, e), use_container_width=True)
        if not ev.empty:
            st.markdown("#### Alerts and the evidence behind them")
            show = ev[["signal", "start_utc", "end_utc", "hours", "severity", "direction", "peak_dev_nowcast_MW",
                       "peak_ratio_to_band", "evidence", "other_signals_alerting", "trust_layer_reasons"]]
            show_table(show, show.style.format(precision=2, na_rep="-").map(
                lambda v: f"color: {SEV_COLOR.get(v, 'inherit')}; font-weight: 700" if v in SEV_COLOR else "",
                subset=["severity"]), index=False)
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
        dark_chart(anomaly_chart(up_sig, h, ev if not ev.empty else pd.DataFrame(columns=["start_utc", "end_utc", "severity"])),
                        use_container_width=True)
        if ev.empty:
            st.success("No anomalies found.")
        else:
            show_table(ev, index=False)

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
    for name, colr in [("naive_MEUR", "#b7c9c0"), ("ours_MEUR", C_MODEL), ("tso_MEUR", C_ACTUAL)]:
        fig.add_bar(name={"naive_MEUR": "naive", "ours_MEUR": "our model", "tso_MEUR": "operator (TSO)"}[name],
                    x=[LABEL[s] for s in cost.index], y=cost[name], marker_color=colr)
    dark_chart(layout(fig, "Cost of forecast errors in 2018 (million EUR, imbalance-spread proxy)").update_yaxes(title="M EUR"),
                    use_container_width=True)
    st.subheader("Alert stream")
    wl = pd.DataFrame(b["alert_workload"]).set_index("signal")
    show_table(wl.rename(columns={"alerts_2018": "alerts in 2018", "high_or_critical": "high + critical",
                                    "review_hours_all": "review hours (all)",
                                    "review_hours_high_critical": "review hours (high + critical)",
                                    "data_issue_evidence": "alerts pointing to data issues"}))
    rec = pd.DataFrame(b["injected_fault_recall"]).set_index("signal")
    st.markdown("**Share of injected faults caught by the alert policy** (30 cases per fault and size)")
    show_table(rec, rec.style.format("{:.0%}", na_rep="-"))
