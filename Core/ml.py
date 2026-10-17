"""ML layer: (1) IsolationForest -> abnormal service patterns, (2) GradientBoosting -> 72h failure risk."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.metrics import roc_auc_score
from .data import DAYS, METRICS

FEAT_NAMES = [f"{s}_{m}" for s in ("last", "mean3", "slope7", "std7") for m in METRICS]


def _tensor(hist):
    h = hist.sort_values(["fhtc_id", "day"])
    ids = h.fhtc_id.unique()
    return ids, h[METRICS].to_numpy().reshape(len(ids), DAYS, len(METRICS))


def _feats(a, t):
    w = a[:, t - 6:t + 1, :]
    x = np.arange(7) - 3.0
    slope = ((w - w.mean(1, keepdims=True)) * x[None, :, None]).sum(1) / (x ** 2).sum()
    return np.hstack([w[:, -1], w[:, -3:].mean(1), slope, w.std(1)])


def _event(a, t):  # service failure within the next 3 days
    f = a[:, t + 1:t + 4, :]
    return ((f[:, :, 0] < 3) | (f[:, :, 2] < .4) | (f[:, :, 3] < .2) | (f[:, :, 4] > 5)).any(1)


def run_models(hist):
    ids, a = _tensor(hist)
    mk = lambda: GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=0)
    last_t = DAYS - 1
    # hold-out check (train on earlier windows, test on the most recent labelled window)
    tr_t, te_t = list(range(6, 10)), DAYS - 4
    Xtr = np.vstack([_feats(a, t) for t in tr_t]); ytr = np.concatenate([_event(a, t) for t in tr_t])
    Xte, yte = _feats(a, te_t), _event(a, te_t)
    auc = float("nan")
    if len(set(ytr)) > 1 and len(set(yte)) > 1:
        auc = roc_auc_score(yte, mk().fit(Xtr, ytr).predict_proba(Xte)[:, 1])
    # final model on all labelled windows
    ts = list(range(6, DAYS - 3))
    X = np.vstack([_feats(a, t) for t in ts]); y = np.concatenate([_event(a, t) for t in ts])
    clf = mk().fit(X, y)
    Xn = _feats(a, last_t)
    prob = clf.predict_proba(Xn)[:, 1]
    iso = IsolationForest(n_estimators=150, contamination=.06, random_state=0).fit(Xn)
    raw = -iso.score_samples(Xn)
    anom = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    scores = pd.DataFrame(dict(fhtc_id=ids, fail_prob=prob, anomaly=anom, is_anomaly=iso.predict(Xn) == -1))
    imp = pd.Series(clf.feature_importances_, index=FEAT_NAMES).sort_values(ascending=False)
    return scores, imp, auc
