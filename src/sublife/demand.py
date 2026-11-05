"""What the next twelve months look like: renewals due, expected non-renewals, new signings, swaps.

The renewal calendar is deterministic - every active slot has an expiration date - so the uncertain part
is only the share that will not renew. New-customer signings are forecast from the monthly history of
first contract dates with simple, backtested methods; the series is short and the early years are thinned
by purged rows, so the forecast is deliberately modest and comes with an empirical interval.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .survival import DiscreteHazard


def renewal_calendar(o: pd.DataFrame, c: pd.DataFrame, p: pd.DataFrame, snapshot: pd.Timestamp, months: int = 12) -> pd.DataFrame:
    """Per month: active slots and customers whose term ends, with the hazard model's expected non-renewals.

    Two scenarios bracket the calendar effect: the model at the most recent fitted year's level and at the
    2025 peak level.
    """
    dh = DiscreteHazard().fit(p)
    a = c[c["is_active_now"] & (c["expiration_date"] > snapshot)].merge(
        o[["org_key", "start_date", "fleet_at_start", "active_slots", "state_group", "channel", "email_is_free_domain"]], on="org_key", how="left")
    a = a[(a["expiration_date"] <= snapshot + pd.DateOffset(months=months)) & a["start_date"].notna()]
    ret = c[c["return_date"] > snapshot - pd.Timedelta(365, "D")].groupby("org_key").size()
    q = pd.DataFrame({
        "org_key": a["org_key"].values, "k": ((a["expiration_date"] - a["start_date"]).dt.days / 365).round().clip(lower=1).astype(int).values,
        "fleet_at_decision": a["active_slots"].values, "fleet_at_start": a["fleet_at_start"].values,
        "state_group": a["state_group"].values, "channel": a["channel"].values, "email_is_free_domain": a["email_is_free_domain"].values,
    })
    q["returned_in_term"] = q["org_key"].map(ret).fillna(0).gt(0)
    q["fleet_shrank_by_decision"] = q["fleet_at_decision"] < q["fleet_at_start"]
    years = dh.year_levels
    a["p_recent"] = dh.predict(q.assign(calendar_year=max(years)))
    a["p_peak"] = dh.predict(q.assign(calendar_year=2025 if 2025 in years else max(years)))
    a["month"] = a["expiration_date"].dt.to_period("M").dt.to_timestamp()
    t = a.groupby("month").agg(slots_due=("org_key", "size"), customers_due=("org_key", "nunique"), expected_non_renewals_recent=("p_recent", "sum"), expected_non_renewals_peak=("p_peak", "sum")).reset_index()
    t["expected_renewals_recent"] = t["slots_due"] - t["expected_non_renewals_recent"]
    return t


def signings_per_month(o: pd.DataFrame, start: str = "2021-01", end: pd.Timestamp | None = None) -> pd.DataFrame:
    """New customers per month by first contract date (purged customers restored from the 2024 export included)."""
    s = o[o["start_date"] >= pd.Timestamp(start)]
    if end is not None:
        s = s[s["start_date"] < end]
    t = s.groupby(s["start_date"].dt.to_period("M")).size()
    idx = pd.period_range(start, t.index.max(), freq="M")
    t = t.reindex(idx, fill_value=0)
    out = t.rename("new_customers").reset_index().rename(columns={"index": "month"})
    out["month"] = out["month"].dt.to_timestamp()
    return out


def _seasonal_naive(y: np.ndarray, h: int, season: int = 12) -> np.ndarray:
    return np.array([y[-season + (i % season)] for i in range(h)])


def _mean_last(y: np.ndarray, h: int, window: int = 12) -> np.ndarray:
    return np.full(h, y[-window:].mean())


def _drift(y: np.ndarray, h: int, window: int = 24) -> np.ndarray:
    w = y[-window:]
    slope = (w[-1] - w[0]) / max(len(w) - 1, 1)
    return np.array([y[-1] + slope * (i + 1) for i in range(h)])


def _holt_winters_additive(y: np.ndarray, h: int, season: int = 12, alpha: float = 0.3, beta: float = 0.05, gamma: float = 0.3) -> np.ndarray:
    n = len(y)
    if n < 2 * season:
        return _mean_last(y, h)
    level = y[:season].mean()
    trend = (y[season : 2 * season].mean() - y[:season].mean()) / season
    seas = list(y[:season] - level)
    for t in range(season, n):
        prev_level = level
        level = alpha * (y[t] - seas[t - season]) + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
        seas.append(gamma * (y[t] - level) + (1 - gamma) * seas[t - season])
    return np.array([max(0.0, level + (i + 1) * trend + seas[n - season + (i % season)]) for i in range(h)])


METHODS = {"seasonal_naive": _seasonal_naive, "mean_last_12": _mean_last, "drift_24": _drift, "holt_winters": _holt_winters_additive}


def backtest(series: pd.DataFrame, horizon: int = 3, origins: int = 12, min_train: int = 24) -> pd.DataFrame:
    """Rolling-origin backtest over the last `origins` months: MAE per method at horizons 1..h."""
    y = series["new_customers"].to_numpy(dtype=float)
    rows = []
    for name, f in METHODS.items():
        errs = {k: [] for k in range(1, horizon + 1)}
        for o_ in range(len(y) - origins - horizon + 1, len(y) - horizon + 1):
            if o_ < min_train:
                continue
            fc = f(y[:o_], horizon)
            for k in range(1, horizon + 1):
                errs[k].append(abs(fc[k - 1] - y[o_ + k - 1]))
        for k, e in errs.items():
            rows.append({"method": name, "horizon": k, "mae": float(np.mean(e)) if e else np.nan, "n_origins": len(e)})
    return pd.DataFrame(rows)


def forecast(series: pd.DataFrame, bt: pd.DataFrame, months: int = 12) -> tuple[pd.DataFrame, str]:
    """Forecast with the simplest method whose backtest MAE is within 10 % of the best; interval = +-1.5 x its MAE (clipped at 0).

    The parsimony rule keeps a 24-month drift line from being extrapolated to zero on the strength of a
    marginal backtest win.
    """
    mae_by = bt.groupby("method")["mae"].mean()
    simplicity = ["mean_last_12", "seasonal_naive", "drift_24", "holt_winters"]
    best = next(m for m in simplicity if mae_by[m] <= 1.10 * mae_by.min())
    y = series["new_customers"].to_numpy(dtype=float)
    fc = METHODS[best](y, months)
    mae = float(bt[bt["method"] == best]["mae"].mean())
    last = series["month"].max()
    idx = pd.date_range(last + pd.DateOffset(months=1), periods=months, freq="MS")
    out = pd.DataFrame({"month": idx, "forecast": np.round(fc, 1), "lo": np.clip(np.round(fc - 1.5 * mae, 1), 0, None), "hi": np.round(fc + 1.5 * mae, 1)})
    return out, best
