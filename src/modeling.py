"""Baselines, models, metrics. Everything here takes already-split frames."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-4
CAP_H = 24 * 30            # cap for 'hours since' features (30 days)

FEATURE_SETS = {
    # state of the alert process + activity level + time of day
    "base": ["state", "level", "hour"],
    "base+short+elapsed": ["state", "level", "hour", "short", "elapsed"],
    "all(+dow)": ["state", "level", "hour", "short", "elapsed", "dow"],
}


def make_X(df: pd.DataFrame, fset: str) -> pd.DataFrame:
    """Model inputs. Built only from leakage-tested columns of the grid."""
    groups = FEATURE_SETS[fset]
    X = pd.DataFrame(index=df.index)
    if "state" in groups:
        X["no_end_yet"] = df.since_last_end_h.isna().astype(int)
        X["log_since_end"] = np.log1p(df.since_last_end_h.fillna(CAP_H).clip(upper=CAP_H))
        X["log_since_start"] = np.log1p(df.since_last_start_h.fillna(CAP_H).clip(upper=CAP_H))
        X["active_now"] = df.active_now
    if "level" in groups:
        for c in ["starts_24h", "starts_7d", "starts_30d"]:
            X[f"log_{c}"] = np.log1p(df[c])
    if "hour" in groups:   # hour_local comes from tz_convert('Europe/Kyiv'), DST-aware
        a = 2 * np.pi * df.hour_local / 24
        X["h_sin1"], X["h_cos1"] = np.sin(a), np.cos(a)
        X["h_sin2"], X["h_cos2"] = np.sin(2 * a), np.cos(2 * a)
    if "short" in groups:
        for c in ["starts_1h", "starts_3h", "starts_6h"]:
            X[c] = df[c]
    if "elapsed" in groups:
        X["log_elapsed"] = np.log1p(df.active_elapsed_min)
    if "dow" in groups:
        for d in range(1, 7):
            X[f"dow_{d}"] = (df.dow_local == d).astype(int)
    return X


def make_model(kind: str):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=200,
            l2_regularization=1.0, early_stopping=False, random_state=0)
    raise ValueError(kind)


def fit_predict(kind, fset, fit_df, pred_df):
    m = make_model(kind)
    m.fit(make_X(fit_df, fset), fit_df.y)
    return m.predict_proba(make_X(pred_df, fset))[:, 1], m


# ------------------------------------------------------------------ baselines
def baselines(fit_df: pd.DataFrame, pred_df: pd.DataFrame, full_grid: pd.DataFrame,
              hour_histories=("all", "365D", "182D")) -> dict:
    """Probabilistic baselines. Lookup tables are estimated on fit_df only;
    inputs at prediction time are known at t."""
    out = {"constant (train rate)": np.full(len(pred_df), fit_df.y.mean())}

    # persistence: y[t-1] == alert started in [t-1h, t) == starts_1h > 0 (known at t)
    prev_fit, prev = (fit_df.starts_1h > 0).astype(int), (pred_df.starts_1h > 0).astype(int)
    out["persistence  P(y | y[t-1])"] = prev.map(fit_df.y.groupby(prev_fit).mean()).values

    # same hour yesterday: y[t-24] covers [t-24h, t-23h), known at t
    lag = full_grid.y.shift(24)
    lag_fit, lag_pred = lag.reindex(fit_df.index), lag.reindex(pred_df.index)
    table = fit_df.y.groupby(lag_fit).mean()
    out["yesterday  P(y | y[t-24])"] = lag_pred.map(table).fillna(fit_df.y.mean()).values

    end = fit_df.index.max()
    for h in hour_histories:
        f = fit_df if h == "all" else fit_df[fit_df.index > end - pd.Timedelta(h)]
        out[f"hour-of-day rate ({h})"] = pred_df.hour_local.map(f.groupby("hour_local").y.mean()).values

    # current level: starts per hour over the last 30 days (no fitting at all)
    out["last-30d rate"] = (pred_df.starts_30d / 720).clip(EPS, 1 - EPS).values
    return out


# ------------------------------------------------------------------ metrics
def metrics(y, p) -> dict:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y)
    return {"ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p),
            "Brier": brier_score_loss(y, p), "LogLoss": log_loss(y, p, labels=[0, 1]),
            "acc@0.5": float(((p >= 0.5) == y).mean()),
            "mean_pred": p.mean(), "actual": y.mean()}


def block_bootstrap_diff(y, p_a, p_b, index, n=2000, seed=0):
    """Paired bootstrap over whole days: CI of (A - B) for log-loss and ROC-AUC."""
    rng = np.random.default_rng(seed)
    y, p_a, p_b = np.asarray(y), np.clip(p_a, EPS, 1 - EPS), np.clip(p_b, EPS, 1 - EPS)
    day = pd.DatetimeIndex(index).floor("D")
    groups = [np.where(day == d)[0] for d in day.unique()]
    ll, auc = [], []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if y[idx].min() == y[idx].max():
            continue
        ll.append(log_loss(y[idx], p_a[idx], labels=[0, 1]) - log_loss(y[idx], p_b[idx], labels=[0, 1]))
        auc.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    q = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5), float(np.mean(np.array(v) < 0)))
    return {"logloss_diff_CI": q(ll), "auc_diff_CI": q(auc)}


def reliability(y, p, n_bins=5):
    """Quantile bins: mean predicted vs observed rate."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df.p.rank(method="first"), n_bins, labels=False)
    return df.groupby("bin").agg(mean_pred=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
