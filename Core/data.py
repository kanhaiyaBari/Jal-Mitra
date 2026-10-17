"""Synthetic data generator that mimics IoT / SCADA / field-test feeds.

Replace `generate()` with real feeds (flow meters, pressure sensors, residual-chlorine
sensors, IMIS exports) - the rest of the app only needs the two returned DataFrames.
"""
from __future__ import annotations
from datetime import date, timedelta
import numpy as np
import pandas as pd

DAYS = 14
METRICS = ["supply_hours", "lpcd", "pressure_bar", "chlorine_mgl", "turbidity_ntu", "ph"]

# (Gram Panchayat, lat, lon)  - fictional villages
GPS = [
    ("Rampur", 21.150, 79.080), ("Shivpur", 21.185, 79.130), ("Kharegaon", 21.120, 79.150),
    ("Mohgaon", 21.210, 79.060), ("Pimpalgaon", 21.090, 79.050), ("Dhamna", 21.160, 79.200),
    ("Sawargaon", 21.230, 79.170), ("Tembhurni", 21.070, 79.110), ("Bhosari", 21.250, 79.100),
    ("Lonara", 21.130, 79.020),
]
# injected scenarios: GP -> (problem, onset day)
PROBLEMS = {"Kharegaon": ("pressure", 8), "Dhamna": ("chlorine", 9), "Sawargaon": ("turbidity", 10),
            "Bhosari": ("pump", 11), "Lonara": ("leak", 7)}


def generate(seed: int = 42):
    rng = np.random.default_rng(seed)
    end = date.today()
    dates = [end - timedelta(days=DAYS - 1 - i) for i in range(DAYS)]
    d = np.arange(DAYS)
    hh, frames = [], []
    for gi, (gp, lat, lon) in enumerate(GPS):
        for k in range(int(rng.integers(40, 66))):
            tail = float(rng.beta(2, 3))  # 0 = near source, 1 = tail end
            fid = f"FHTC-{gi + 1:02d}-{k + 1:03d}"
            hh.append(dict(fhtc_id=fid, gp=gp, sujal_id=f"SG-MH-{1001 + gi}", imis_id=f"IMIS-27-{540200 + gi * 7}",
                           lat=lat + rng.normal(0, .006), lon=lon + rng.normal(0, .006), tail=tail,
                           members=int(rng.integers(2, 8))))
            sup = rng.normal(5.8, .5, DAYS) - 1.0 * tail
            pr = 1.25 - .55 * tail + rng.normal(0, .07, DAYS)
            ch = .55 - .25 * tail + rng.normal(0, .05, DAYS)
            tb = np.abs(rng.normal(.7, .25, DAYS))
            ph = rng.normal(7.4, .15, DAYS)
            leak = np.zeros(DAYS)
            if gp in PROBLEMS:
                kind, on = PROBLEMS[gp]
                ramp, w = np.clip(d - on + 1, 0, None), .4 + tail
                if kind == "pressure": pr -= .12 * ramp * w; sup -= .35 * ramp * w
                if kind == "chlorine": ch -= .07 * ramp * w
                if kind == "turbidity": tb += 1.6 * ramp * w
                if kind == "pump": sup -= 1.1 * ramp * w; pr -= .06 * ramp * w
                if kind == "leak": pr -= .08 * ramp * w; leak = 28 * ramp * w
            if rng.random() < .04:  # random tap/pipe failure
                on = int(rng.integers(9, 13)); r2 = np.clip(d - on + 1, 0, None)
                sup -= 1.4 * r2; pr -= .12 * r2
            sup, pr = np.clip(sup, 0, 12), np.clip(pr, 0, 3)
            frames.append(pd.DataFrame(dict(
                fhtc_id=fid, day=d, date=dates, supply_hours=sup,
                lpcd=np.clip(sup / 6 * 55 + rng.normal(0, 4, DAYS) + leak, 0, 300),
                pressure_bar=pr, chlorine_mgl=np.clip(ch, 0, 2.5),
                turbidity_ntu=np.clip(tb, 0, 15), ph=ph)))
    return pd.DataFrame(hh), pd.concat(frames, ignore_index=True)
