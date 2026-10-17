import json, os, uuid
from datetime import datetime
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from core.data import generate, DAYS
from core.ml import run_models
from core.rules import build_alerts, enrich, quality_ok
from core.i18n import LANGS, tr
from core.imis import push

st.set_page_config(page_title="Jal Mitra – FHTC Monitor", page_icon="💧", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:1.2rem;max-width:1250px}
.hero{background:linear-gradient(120deg,#023e8a,#0077b6 55%,#00b4d8);color:#fff;padding:20px 26px;border-radius:18px;margin-bottom:14px}
.hero h1{margin:0;font-size:2rem;color:#fff}.hero p{margin:4px 0 0;opacity:.92}
.big{border-radius:18px;padding:22px;text-align:center;font-size:1.6rem;font-weight:700;color:#fff;margin:8px 0}
.ok{background:#22a05a}.warn{background:#f5a623}.crit{background:#d63031}
div[data-testid="stMetric"]{background:rgba(0,119,182,.07);border:1px solid rgba(0,119,182,.18);padding:12px 14px;border-radius:14px}
.stTabs [data-baseweb="tab"]{font-size:1rem;padding:10px 14px}
.stButton>button,.stFormSubmitButton>button{border-radius:12px;font-weight:600;min-height:48px}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Training AI models…")
def load():
    hh, hist = generate(42)
    scores, imp, auc = run_models(hist)
    last = hist[hist.day == DAYS - 1].drop(columns=["day", "date"])
    latest = hh.merge(last, on="fhtc_id").merge(scores, on="fhtc_id")
    alerts = build_alerts(latest)
    latest = enrich(latest, alerts)
    return hh, hist, latest, alerts, imp, auc


hh, hist, latest, alerts, imp, auc = load()

# ---------- sidebar ----------
lang_name = st.sidebar.selectbox("🌐 Language / भाषा", list(LANGS))
L = LANGS[lang_name]
t = lambda k: tr(L, k)
mode = st.sidebar.radio(t("mode"), ["citizen", "officer"], format_func=t)
if st.sidebar.button(t("refresh")):
    st.cache_data.clear(); st.rerun()
st.sidebar.caption(t("demo_note"))

ST = {"ok": t("st_ok"), "warn": t("st_warn"), "crit": t("st_crit")}
ATYPE = lambda k: t("a_" + k) if k != "-" else "-"
ISSUES = ["i_no_water", "i_low_pressure", "i_dirty", "i_smell", "i_leak", "i_other"]

# ---------- feedback store ----------
FB_COLS = ["ticket", "time", "gp", "fhtc_id", "sujal_id", "category", "text", "lat", "lon", "photo", "status"]
if "fb" not in st.session_state:
    rng = np.random.default_rng(7)
    rows = []
    for i, r in latest[latest.status != "ok"].sample(8, random_state=3).iterrows():
        rows.append(["T" + uuid.uuid4().hex[:6].upper(), datetime.now().strftime("%d %b %H:%M"), r.gp, r.fhtc_id,
                     r.sujal_id, rng.choice(ISSUES[:5]), "", round(r.lat, 5), round(r.lon, 5),
                     bool(rng.random() > .6), rng.choice(["Open", "Open", "In progress"])])
    st.session_state.fb = pd.DataFrame(rows, columns=FB_COLS)
    st.session_state.sync_log = []

st.markdown(f"<div class='hero'><h1>💧 {t('title')}</h1><p>{t('subtitle')}</p></div>", unsafe_allow_html=True)


# =====================================================================
#  CITIZEN VIEW
# =====================================================================
def citizen():
    tap, rep = st.tabs([t("tab_tap"), t("tab_report")])
    c1, c2 = st.columns(2)
    gp = c1.selectbox(t("village"), sorted(hh.gp.unique()), key="c_gp")
    fid = c2.selectbox(t("fhtc_id"), sorted(latest[latest.gp == gp].fhtc_id), key="c_id")
    r = latest[latest.fhtc_id == fid].iloc[0]
    with tap:
        if r.supply_hours < 1: cls, msg = "crit", t("water_none")
        elif r.supply_hours < 3.5 or r.pressure_bar < .5: cls, msg = "warn", t("water_low")
        else: cls, msg = "ok", t("water_coming")
        st.markdown(f"<div class='big {cls}'>{msg}</div>", unsafe_allow_html=True)
        m = st.columns(2)
        m[0].metric(t("hours_today"), f"{r.supply_hours:.1f} h")
        m[1].metric(t("pressure"), f"{r.pressure_bar:.2f}")
        q = bool(quality_ok(latest[latest.fhtc_id == fid]).iloc[0])
        (st.success if q else st.error)(("✅ " + t("safe_drink")) if q else ("⚠️ " + t("not_safe")))
        h = hist[hist.fhtc_id == fid].tail(7).set_index("date")[["supply_hours"]]
        st.caption(t("daily_supply")); st.bar_chart(h, height=180)
    with rep:
        st.subheader(t("report_title"))
        with st.form("report"):
            cat = st.radio(t("issue_type"), ISSUES, format_func=t, horizontal=True)
            txt = st.text_area(t("describe"))
            photo = st.file_uploader(t("photo"), type=["jpg", "jpeg", "png"])
            st.markdown(f"**📍 {t('geotag')}**")
            g1, g2 = st.columns(2)
            la = g1.number_input(t("lat"), value=float(round(r.lat, 5)), format="%.5f")
            lo = g2.number_input(t("lon"), value=float(round(r.lon, 5)), format="%.5f")
            ok = st.form_submit_button("✅ " + t("submit"), use_container_width=True)
        if ok:
            tk = "T" + uuid.uuid4().hex[:6].upper()
            new = pd.DataFrame([[tk, datetime.now().strftime("%d %b %H:%M"), r.gp, fid, r.sujal_id, cat, txt, la, lo,
                                 photo is not None, "Open"]], columns=FB_COLS)
            st.session_state.fb = pd.concat([new, st.session_state.fb], ignore_index=True)
            st.success(f"{t('thanks')}\n\n**{t('ticket')}: {tk}**")


# =====================================================================
#  OFFICER VIEW
# =====================================================================
def gp_summary():
    g = latest.groupby("gp").agg(households=("fhtc_id", "count"), functional=("supply_hours", lambda s: (s >= 1).mean() * 100),
                                 avg_supply=("supply_hours", "mean"), alerts=("n_alerts", "sum"),
                                 high_risk=("fail_prob", lambda s: int((s > .6).sum())),
                                 crit=("status", lambda s: (s == "crit").mean())).reset_index()
    g["quality"] = latest.assign(q=quality_ok(latest)).groupby("gp").q.mean().values * 100
    g["status"] = np.where(g.crit > .25, ST["crit"], np.where(g.crit > .08, ST["warn"], ST["ok"]))
    return g


def dashboard():
    k = st.columns(5)
    k[0].metric(t("k_total"), len(latest))
    k[1].metric(t("k_func"), f"{(latest.supply_hours >= 1).mean() * 100:.1f}%")
    k[2].metric(t("k_alerts"), len(alerts), f"{(alerts.severity == 'critical').sum()} critical", delta_color="inverse")
    k[3].metric(t("k_quality"), f"{quality_ok(latest).mean() * 100:.1f}%")
    k[4].metric(t("k_risk"), int((latest.fail_prob > .6).sum()))
    st.subheader("🗺️ " + t("map"))
    d = latest.assign(label=latest.status.map(ST))
    layer = pdk.Layer("ScatterplotLayer", data=d[["lon", "lat", "color", "fhtc_id", "gp", "label"]],
                      get_position="[lon, lat]", get_fill_color="color", get_radius=60, pickable=True)
    st.pydeck_chart(pdk.Deck(layers=[layer], map_style="light", initial_view_state=pdk.ViewState(
        latitude=d.lat.mean(), longitude=d.lon.mean(), zoom=10.2),
        tooltip={"text": "{fhtc_id}\n{gp}\n{label}"}), height=430)
    st.subheader("🏘️ " + t("gp_table"))
    g = gp_summary().sort_values("crit", ascending=False)
    st.dataframe(g.drop(columns="crit"), hide_index=True, use_container_width=True, column_config={
        "gp": t("gp") if False else "Gram Panchayat", "households": "FHTCs",
        "functional": st.column_config.ProgressColumn(t("k_func"), min_value=0, max_value=100, format="%.0f%%"),
        "quality": st.column_config.ProgressColumn(t("k_quality"), min_value=0, max_value=100, format="%.0f%%"),
        "avg_supply": st.column_config.NumberColumn(t("supply_hours"), format="%.1f"),
        "alerts": t("k_alerts"), "high_risk": t("k_risk"), "status": t("severity")})
    sel = st.selectbox(t("select_gp"), sorted(hh.gp.unique()), index=2)
    h = hist.merge(hh[["fhtc_id", "gp"]], on="fhtc_id")
    h = h[h.gp == sel].groupby("date")[["supply_hours", "pressure_bar", "chlorine_mgl", "turbidity_ntu"]].mean()
    st.caption("📈 " + t("trends"))
    cols = st.columns(4)
    for c, (col, lab) in zip(cols, [("supply_hours", "supply_hours"), ("pressure_bar", "pressure"),
                                    ("chlorine_mgl", "chlorine"), ("turbidity_ntu", "turbidity")]):
        c.markdown(f"**{t(lab)}**"); c.line_chart(h[col], height=150)


def alerts_page():
    c = st.columns(3)
    sev = c[0].selectbox(t("severity"), ["all", "critical", "warning"], format_func=lambda x: t("all") if x == "all" else x)
    gps = c[1].multiselect("Gram Panchayat", sorted(hh.gp.unique()))
    types = c[2].multiselect("Type", sorted(alerts.type.unique()), format_func=ATYPE)
    a = alerts.copy()
    if sev != "all": a = a[a.severity == sev]
    if gps: a = a[a.gp.isin(gps)]
    if types: a = a[a.type.isin(types)]
    st.bar_chart(a.type.map(ATYPE).value_counts(), height=200)
    v = a.assign(type=a.type.map(ATYPE))[["severity", "type", "gp", "fhtc_id", "sujal_id", "value", "action"]]
    st.dataframe(v, hide_index=True, use_container_width=True, height=300)
    st.download_button(t("download"), v.to_csv(index=False).encode(), "alerts.csv", "text/csv")
    st.subheader("🔧 " + t("priority_title")); st.caption(t("priority_help"))
    p = latest.sort_values("priority", ascending=False).head(15).copy()
    p["status"] = p.status.map(ST); p["top_alert"] = p.top_alert.map(ATYPE); p["fail_prob"] = p.fail_prob * 100
    st.dataframe(p[["fhtc_id", "gp", "status", "top_alert", "fail_prob", "anomaly", "priority"]], hide_index=True,
                 use_container_width=True, column_config={
                     "fail_prob": st.column_config.ProgressColumn(t("risk72"), min_value=0, max_value=100, format="%.0f%%"),
                     "anomaly": st.column_config.NumberColumn(t("anomaly"), format="%.2f"),
                     "priority": st.column_config.ProgressColumn("Priority", min_value=0, max_value=60, format="%.1f")})
    with st.expander("🧠 " + t("drivers")):
        st.bar_chart(imp.head(8), height=220)
        st.caption(f"{t('model_auc')}: {auc:.2f}" if not np.isnan(auc) else "")


def household():
    gp = st.selectbox(t("village"), sorted(hh.gp.unique()), key="o_gp", index=2)
    fid = st.selectbox(t("hh_pick"), latest[latest.gp == gp].sort_values("priority", ascending=False).fhtc_id)
    r = latest[latest.fhtc_id == fid].iloc[0]
    st.markdown(f"<div class='big {r.status}'>{fid} · {ST[r.status]}</div>", unsafe_allow_html=True)
    st.caption(f"Sujal Gaon ID: {r.sujal_id} · IMIS: {r.imis_id}")
    m = st.columns(5)
    for c, (k, lab, f) in zip(m, [("supply_hours", "supply_hours", "{:.1f}"), ("pressure_bar", "pressure", "{:.2f}"),
                                  ("chlorine_mgl", "chlorine", "{:.2f}"), ("turbidity_ntu", "turbidity", "{:.1f}"), ("ph", "ph", "{:.1f}")]):
        c.metric(t(lab), f.format(r[k]))
    a, b = st.columns(2)
    a.metric(t("risk72"), f"{r.fail_prob * 100:.0f}%"); b.metric(t("anomaly"), f"{r.anomaly:.2f}")
    h = hist[hist.fhtc_id == fid].set_index("date")
    c1, c2, c3 = st.columns(3)
    c1.caption(t("supply_hours")); c1.line_chart(h.supply_hours, height=170)
    c2.caption(t("pressure")); c2.line_chart(h.pressure_bar, height=170)
    c3.caption(t("chlorine")); c3.line_chart(h.chlorine_mgl, height=170)
    al = alerts[alerts.fhtc_id == fid]
    if al.empty: st.success(t("no_alerts"))
    for _, x in al.iterrows():
        (st.error if x.severity == "critical" else st.warning)(f"**{ATYPE(x.type)}** ({x.value}) — {t('recommended')}: {x.action}")


def feedback():
    fb = st.session_state.fb
    c = st.columns(3)
    c[0].metric(t("inbox"), len(fb)); c[1].metric(t("open"), int((fb.status == "Open").sum()))
    c[2].metric("📷", int(fb.photo.sum()))
    view = fb.copy(); view["category"] = view.category.map(t)
    edited = st.data_editor(view, hide_index=True, use_container_width=True, disabled=[c for c in FB_COLS if c != "status"],
                            column_config={"status": st.column_config.SelectboxColumn("Status", options=["Open", "In progress", "Resolved"]),
                                           "photo": st.column_config.CheckboxColumn("📷")})
    st.session_state.fb["status"] = edited["status"].values
    st.map(fb.rename(columns={"lon": "longitude", "lat": "latitude"})[["latitude", "longitude"]], height=260)


def gp_payload(g):
    d, a = latest[latest.gp == g], alerts[alerts.gp == g]
    f = st.session_state.fb
    return {"sujal_gaon_id": d.sujal_id.iloc[0], "imis_id": d.imis_id.iloc[0], "gram_panchayat": g,
            "timestamp": datetime.now().isoformat(timespec="seconds"), "fhtc_total": int(len(d)),
            "fhtc_functional": int((d.supply_hours >= 1).sum()), "avg_supply_hours": round(float(d.supply_hours.mean()), 2),
            "quality_compliance_pct": round(float(quality_ok(d).mean() * 100), 1),
            "active_alerts": {k: int(v) for k, v in a.type.value_counts().items()},
            "open_citizen_complaints": int(((f.gp == g) & (f.status != "Resolved")).sum()),
            "high_risk_72h": int((d.fail_prob > .6).sum())}


def sync():
    st.subheader("🔗 " + t("sync_title")); st.caption(t("sync_help"))
    g = st.selectbox("Gram Panchayat", sorted(hh.gp.unique()))
    st.caption(t("payload")); st.json(gp_payload(g))
    if st.button(t("sync_btn"), type="primary", use_container_width=True):
        bar = st.progress(0)
        for i, name in enumerate(sorted(hh.gp.unique())):
            p = gp_payload(name); res = push(p)
            st.session_state.sync_log.insert(0, {"time": p["timestamp"], "gram_panchayat": name, "sujal_gaon_id": p["sujal_gaon_id"],
                                                 "imis_id": p["imis_id"], "mode": res["mode"], "status": res["status"], "ack": res.get("ack", "")})
            bar.progress((i + 1) / len(hh.gp.unique()))
        st.success("✅ " + t("sync_ok"))
    if st.session_state.sync_log:
        st.caption(t("sync_log")); st.dataframe(pd.DataFrame(st.session_state.sync_log), hide_index=True, use_container_width=True)


if mode == "citizen":
    citizen()
else:
    tabs = st.tabs([t("tab_dash"), t("tab_alerts"), t("tab_hh"), t("tab_fb"), t("tab_sync")])
    for tab, fn in zip(tabs, [dashboard, alerts_page, household, feedback, sync]):
        with tab: fn()
