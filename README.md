# 💧 Jal Mitra – AI/ML Monitoring of Functional Household Tap Connections (FHTC)

A working Streamlit prototype for real-time monitoring of rural piped water service at **Gram Panchayat and household level**,
aligned with Jal Jeevan Mission (IMIS / Sujal Gaon IDs).

## Features
| Requirement | Where |
|---|---|
| Continuous FHTC monitoring (GP + household) | Dashboard, Household tab |
| Real-time alerts: no supply, low pressure, leakage, chlorine / turbidity / pH | `core/rules.py` → Alerts tab |
| Citizen feedback with photo + geo-tag, **local language UI** (English / हिन्दी / मराठी) | Citizen mode → Report a problem |
| Anomaly detection (Isolation Forest) + 72h failure prediction (Gradient Boosting) + maintenance priority | `core/ml.py` |
| Sync to IMIS / Sujal Gaon IDs | `core/imis.py`, IMIS Sync tab |
| GIS map + multi-level dashboards | Dashboard (pydeck) |
| Two modes: simple **Citizen** view (big status card) & **Officer / VWSC** view | Sidebar |

> ⚠️ Data is **synthetic** (generated in `core/data.py`) with injected scenarios (pump failure, low chlorine, leak, muddy water, low pressure).
> Replace `generate()` with real sensor / SCADA / IMIS feeds.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy free
Push to GitHub → https://share.streamlit.io → New app → select repo → main file `app.py`.

## Connect real IMIS
Set environment variables `IMIS_API_URL` and `IMIS_API_TOKEN`; `core/imis.py` will POST payloads instead of simulating.

## Add a language
Copy the `en` block in `core/i18n.py`, translate values, add to `LANGS`.

## Structure
```
app.py            UI (citizen + officer views)
core/data.py      synthetic sensor data (replace with real feeds)
core/ml.py        anomaly detection + failure prediction
core/rules.py     BIS/JJM thresholds, alerts, status, priority
core/imis.py      IMIS / Sujal Gaon sync adapter
core/i18n.py      translations
```

## Roadmap
MQTT/IoT ingestion · SMS/IVR/WhatsApp complaints · voice input in local languages · persistent DB (PostgreSQL) · role-based login · offline-first PWA.
