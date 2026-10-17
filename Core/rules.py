"""Threshold rules (BIS IS-10500 / JJM service norms) + status & maintenance priority."""
import pandas as pd

ACTIONS = {
    "no_supply": "Dispatch pump operator / check valve & rising main for this branch.",
    "low_supply": "Review valve schedule; check tank level and tail-end flow.",
    "low_pressure": "Inspect for partial blockage / throttled valves upstream.",
    "low_chlorine": "Re-dose chlorination at source; verify residual at tail end.",
    "high_turbidity": "Check filter / source contamination; advise boiling until cleared.",
    "ph_out": "Collect sample for lab test; check source and dosing chemicals.",
    "leakage": "Send plumber - high flow with low pressure suggests pipe leakage.",
}
COLORS = {"ok": [34, 160, 90, 200], "warn": [245, 166, 35, 210], "crit": [214, 48, 49, 220]}


def quality_ok(df):
    return (df.chlorine_mgl >= .2) & (df.turbidity_ntu <= 5) & df.ph.between(6.5, 8.5)


def build_alerts(l):
    specs = [
        ("no_supply", "critical", l.supply_hours < 1, "supply_hours"),
        ("low_supply", "warning", (l.supply_hours >= 1) & (l.supply_hours < 3.5), "supply_hours"),
        ("low_pressure", "warning", (l.pressure_bar < .5) & (l.supply_hours >= 1), "pressure_bar"),
        ("low_chlorine", "critical", l.chlorine_mgl < .2, "chlorine_mgl"),
        ("high_turbidity", "critical", l.turbidity_ntu > 5, "turbidity_ntu"),
        ("high_turbidity", "warning", (l.turbidity_ntu > 2) & (l.turbidity_ntu <= 5), "turbidity_ntu"),
        ("ph_out", "warning", (l.ph < 6.5) | (l.ph > 8.5), "ph"),
        ("leakage", "warning", (l.lpcd > 120) & (l.pressure_bar < .9), "lpcd"),
    ]
    out = []
    for typ, sev, mask, col in specs:
        s = l[mask]
        out.append(pd.DataFrame(dict(fhtc_id=s.fhtc_id, gp=s.gp, sujal_id=s.sujal_id, type=typ, severity=sev,
                                     value=s[col].round(2), action=ACTIONS[typ])))
    a = pd.concat(out, ignore_index=True)
    a["rank"] = a.severity.map({"critical": 0, "warning": 1})
    return a.sort_values(["rank", "gp"]).drop(columns="rank").reset_index(drop=True)


def enrich(l, alerts):
    l = l.copy()
    crit = set(alerts[alerts.severity == "critical"].fhtc_id)
    anyal = set(alerts.fhtc_id)
    l["status"] = ["crit" if f in crit else "warn" if (f in anyal or p > .6) else "ok"
                   for f, p in zip(l.fhtc_id, l.fail_prob)]
    cnt = alerts.groupby("fhtc_id").size()
    l["n_alerts"] = l.fhtc_id.map(cnt).fillna(0).astype(int)
    l["top_alert"] = l.fhtc_id.map(alerts.drop_duplicates("fhtc_id").set_index("fhtc_id").type).fillna("-")
    l["priority"] = (100 * (.5 * l.fail_prob + .3 * l.anomaly + .2 * (l.n_alerts / 3).clip(upper=1)) *
                     (1 + l.members / 20)).round(1)
    l["color"] = l.status.map(COLORS)
    return l
