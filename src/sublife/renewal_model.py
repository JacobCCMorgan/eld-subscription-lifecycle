"""Can the renewal desk tell, 60 days before an anniversary, who will not renew?

Two questions, both answered on time-ordered holdouts so that nothing from the evaluation period leaks into
the fit:

1. First-anniversary model: customers who signed in 2021-2023 train, 2024 signings (anniversaries in 2025)
   are the holdout. Features are what is known at signing plus what the account looked like 60 days before
   the anniversary.
2. All-anniversary model (the discrete-time hazard of survival.py): fitted on anniversaries in 2023-2025,
   evaluated on anniversaries in 2026.

Metrics: ROC-AUC, PR-AUC, Brier score, calibration by predicted-probability decile, and the practical
question - if the desk can call the top 10 %/20 % of accounts, what share of the churners does it reach?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.utils import resample

from .survival import DiscreteHazard, _period_design

FIRST_TRAIN_YEARS = (2021, 2022, 2023)
FIRST_TEST_YEARS = (2024,)
FEATURES_AT_SIGNING = ["log_fleet_at_start", "outside_ca", "channel_other_rep", "channel_unknown", "free_email_domain", "signed_oct_mar", "start_year", "vehicle_model_age", "vehicle_model_age_missing"]
FEATURES_AT_DECISION = ["returned_in_term", "fleet_shrank_by_decision", "log_fleet_at_decision"]


def first_anniversary_frame(p: pd.DataFrame, o: pd.DataFrame) -> pd.DataFrame:
    d = p[(p["k"] == 1) & p["at_risk_at_decision"]].merge(o[["org_key", "start_month", "first_model_year"]], on="org_key", how="left")
    x = pd.DataFrame(index=d.index)
    x["log_fleet_at_start"] = np.log1p(d["fleet_at_start"].clip(lower=1))
    x["outside_ca"] = d["state_group"].eq("other").astype(int)
    x["channel_other_rep"] = d["channel"].eq("other_rep").astype(int)
    x["channel_unknown"] = d["channel"].eq("unknown").astype(int)
    x["free_email_domain"] = d["email_is_free_domain"].astype("boolean").fillna(False).astype(int)
    x["signed_oct_mar"] = d["start_month"].isin([10, 11, 12, 1, 2, 3]).astype(int)
    x["start_year"] = d["start_year"].astype(int)
    age = (d["start_year"] - d["first_model_year"]).clip(lower=-1, upper=30)
    x["vehicle_model_age_missing"] = age.isna().astype(int)
    x["vehicle_model_age"] = age.fillna(age.median())
    x["returned_in_term"] = d["returned_in_term"].astype(int)
    x["fleet_shrank_by_decision"] = d["fleet_shrank_by_decision"].astype(int)
    x["log_fleet_at_decision"] = np.log1p(d["fleet_at_decision"].clip(lower=1))
    x["churn"] = d["churn"].values
    x["org_key"] = d["org_key"].values
    return x


def _metrics(y: np.ndarray, s: np.ndarray) -> dict:
    out = {"n": int(len(y)), "churners": int(y.sum()), "base_rate": float(y.mean()), "roc_auc": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan, "pr_auc": float(average_precision_score(y, s)), "brier": float(brier_score_loss(y, s))}
    order = np.argsort(-s, kind="stable")
    for share in (0.1, 0.2, 0.3):
        k = max(1, int(round(share * len(y))))
        top = y[order[:k]]
        out[f"recall_top{int(share * 100)}"] = float(top.sum() / max(y.sum(), 1))
        out[f"precision_top{int(share * 100)}"] = float(top.mean())
    return out


def _bootstrap_auc(y: np.ndarray, s: np.ndarray, n_boot: int = 300, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):
            vals.append(roc_auc_score(y[idx], s[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def calibration_table(y: np.ndarray, s: np.ndarray, bins: int = 5) -> pd.DataFrame:
    q = pd.qcut(pd.Series(s).rank(method="first"), bins, labels=False)
    t = pd.DataFrame({"bin": q, "pred": s, "obs": y}).groupby("bin").agg(n=("obs", "size"), predicted=("pred", "mean"), observed=("obs", "mean")).reset_index()
    return t


def first_anniversary_models(x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Base rate vs logistic vs gradient boosting, with and without the decision-point features."""
    tr = x[x["start_year"].isin(FIRST_TRAIN_YEARS)]
    te = x[x["start_year"].isin(FIRST_TEST_YEARS)]
    ytr, yte = tr["churn"].values, te["churn"].values
    rows, cal, imp = [], [], []
    specs = {
        "base_rate": None,
        "logistic_signing_only": (LogisticRegression(max_iter=2000, C=1.0), FEATURES_AT_SIGNING),
        "logistic_signing_plus_decision": (LogisticRegression(max_iter=2000, C=1.0), FEATURES_AT_SIGNING + FEATURES_AT_DECISION),
        "boosting_signing_plus_decision": (HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=0), FEATURES_AT_SIGNING + FEATURES_AT_DECISION),
    }
    for name, spec in specs.items():
        if spec is None:
            s = np.full(len(te), ytr.mean())
        else:
            model, feats = spec
            model.fit(tr[feats], ytr)
            s = model.predict_proba(te[feats])[:, 1]
            if name.startswith("logistic"):
                for f, b in zip(feats, model.coef_[0]):
                    imp.append({"model": name, "feature": f, "odds_ratio": float(np.exp(b))})
            else:
                from sklearn.inspection import permutation_importance

                pi = permutation_importance(model, te[feats], yte, scoring="roc_auc", n_repeats=10, random_state=0)
                for f, m in zip(feats, pi.importances_mean):
                    imp.append({"model": name, "feature": f, "permutation_importance_auc": float(m)})
        m = _metrics(yte, s)
        m["model"] = name
        if spec is not None:
            m["roc_auc_lo"], m["roc_auc_hi"] = _bootstrap_auc(yte, s)
        rows.append(m)
        c = calibration_table(yte, s)
        c["model"] = name
        cal.append(c)
    return pd.DataFrame(rows), pd.concat(cal, ignore_index=True), pd.DataFrame(imp)


def hazard_temporal_holdout(p: pd.DataFrame, train_years=(2023, 2024, 2025), test_year: int = 2026) -> tuple[dict, pd.DataFrame]:
    """Fit the discrete-time hazard on earlier anniversaries, score the test year's anniversaries."""
    d = p[p["at_risk_at_decision"]]
    tr = d[d["calendar_year"].isin(train_years)]
    te = d[d["calendar_year"] == test_year]
    dh = DiscreteHazard(year_min=min(train_years)).fit(tr)
    # the test year is not among the fitted calendar dummies: score it at the most recent fitted year's level
    te2 = te.assign(calendar_year=max(train_years))
    s = dh.predict(te2)
    y = te["churn"].values
    m = _metrics(y, s)
    m["roc_auc_lo"], m["roc_auc_hi"] = _bootstrap_auc(y, s)
    m["train_years"] = str(train_years)
    m["test_year"] = test_year
    m["predicted_mean"] = float(s.mean())
    return m, calibration_table(y, s)


def hazard_bootstrap_odds(p: pd.DataFrame, n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """Cluster bootstrap (by organization) of the discrete-time hazard odds ratios."""
    d = p[(p["calendar_year"] >= 2023) & p["at_risk_at_decision"]]
    orgs = d["org_key"].unique()
    base = DiscreteHazard().fit(d)
    cols = base.columns
    draws = []
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        pick = rng.choice(orgs, len(orgs), replace=True)
        b = d.set_index("org_key").loc[pick].reset_index()
        x = _period_design(b, base.k_levels, base.year_levels)[cols]
        m = LogisticRegression(C=1.0, max_iter=2000).fit(x, b["churn"])
        draws.append(np.exp(m.coef_[0]))
    draws = np.array(draws)
    out = base.odds_ratios()[["covariate", "odds_ratio"]].copy()
    out["or_lo"] = np.percentile(draws, 2.5, axis=0)
    out["or_hi"] = np.percentile(draws, 97.5, axis=0)
    return out


def risk_list(o: pd.DataFrame, c: pd.DataFrame, p: pd.DataFrame, snapshot: pd.Timestamp, horizon_days: int = 60) -> pd.DataFrame:
    """Accounts with a term ending in the next `horizon_days`, scored with the discrete-time hazard."""
    dh = DiscreteHazard().fit(p)
    a = o[(o["active_slots"] > 0) & (o["next_expiration"] > snapshot) & (o["next_expiration"] <= snapshot + pd.Timedelta(horizon_days, "D"))].copy()
    k = ((a["next_expiration"] - a["start_date"]).dt.days / 365).round().clip(lower=1).astype(int)
    rows = c[c["org_key"].isin(a["org_key"])]
    ret = rows[rows["return_date"] > snapshot - pd.Timedelta(365, "D")].groupby("org_key").size()
    q = pd.DataFrame({
        "org_key": a["org_key"].values, "k": k.values, "calendar_year": max(dh.year_levels),
        "fleet_at_decision": a["active_slots"].values, "fleet_at_start": a["fleet_at_start"].values,
        "state_group": a["state_group"].values, "channel": a["channel"].values, "email_is_free_domain": a["email_is_free_domain"].values,
    })
    q["returned_in_term"] = q["org_key"].map(ret).fillna(0).gt(0)
    q["fleet_shrank_by_decision"] = q["fleet_at_decision"] < q["fleet_at_start"]
    q["p_churn"] = dh.predict(q)
    out = a[["org_key", "next_expiration", "start_date", "active_slots", "legacy_slots_active", "billing_hold_slots", "n_returns", "state_group", "channel"]].copy()
    out["anniversary_k"] = k.values
    out["returned_device_last_12m"] = q["returned_in_term"].values
    out["p_churn"] = q["p_churn"].values.round(4)
    out["expected_churn_slots"] = (out["p_churn"] * out["active_slots"]).round(4)
    out = out.sort_values(["p_churn", "active_slots"], ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    edges = np.quantile(out["p_churn"], [0.5, 0.8]) if len(out) else [0, 0]
    out["tier"] = np.where(out["p_churn"] >= edges[1], "A", np.where(out["p_churn"] >= edges[0], "B", "C"))
    return out


def resample_check(y: np.ndarray, s: np.ndarray, seed: int = 0) -> float:
    """Smoke-test helper used in tests: AUC on a bootstrap resample should stay close to the full-sample AUC."""
    yb, sb = resample(y, s, random_state=seed)
    return float(roc_auc_score(yb, sb)) if 0 < yb.sum() < len(yb) else np.nan
