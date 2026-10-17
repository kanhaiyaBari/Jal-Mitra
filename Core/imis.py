"""IMIS / Sujal Gaon ID adapter. Simulated unless IMIS_API_URL is set."""
import os
import uuid


def push(payload: dict) -> dict:
    url = os.getenv("IMIS_API_URL")
    if url:
        import requests
        r = requests.post(url, json=payload, timeout=15,
                          headers={"Authorization": f"Bearer {os.getenv('IMIS_API_TOKEN', '')}"})
        return {"ok": r.ok, "status": r.status_code, "mode": "live"}
    return {"ok": True, "status": 202, "mode": "simulated", "ack": uuid.uuid4().hex[:10]}
