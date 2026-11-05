"""Survival and renewal-hazard analysis at the organization level.

Three complementary views:
  * Kaplan-Meier curves of customer tenure (continuous time), by signing cohort and by segment;
  * a Cox proportional-hazards model on attributes known at signing (cohorts 2023+, where the 2024 export
    confirms churn is fully recorded);
  * a discrete-time hazard model on the anniversary table: logit(churn at anniversary k) with tenure
    dummies, calendar-year dummies and segment covariates, so the tenure effect and the period effect are
    separated.

Anniversaries before 2023 are shown descriptively only: the CRM purges old rows, so churn in 2021-2022 is
under-recorded (see validate.py and the report).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from sklearn.linear_model import LogisticRegression

COX_COHORT_MIN = 2023
HAZARD_YEAR_MIN = 2023
SEGMENTS = ["start_year", "fleet_band", "state_group", "channel", "has_dot", "email_is_free_domain"]


def km_table(o: pd.DataFrame, by: str | None = None, horizons=(365, 730, 1095, 1460)) -> pd.DataFrame:
    """Survival probability at the horizons (days since signing) and the number at risk, overall or by group."""
    rows = []
    groups = [("all", o)] if by is None else [(str(k), g) for k, g in o.groupby(by, dropna=False)]
    for name, g in groups:
        if len(g) < 20:
            continue
        kmf = KaplanMeierFitter().fit(g["tenure_days"], g["churned"])
        sf = kmf.survival_function_at_times(list(horizons)).values
        ci = kmf.confidence_interval_survival_function_
        rec = {"group_by": by or "all", "group": name, "customers": len(g), "churned": int(g["churned"].sum()), "median_tenure_days": kmf.median_survival_time_}
        for h, s in zip(horizons, sf):
            rec[f"S_{h}d"] = s
            at = ci.index.get_indexer([h], method="pad")[0]
            rec[f"S_{h}d_lo"] = float(ci.iloc[at, 0]) if at >= 0 else np.nan
            rec[f"S_{h}d_hi"] = float(ci.iloc[at, 1]) if at >= 0 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def km_curves(o: pd.DataFrame, by: str, max_days: int = 1825) -> dict[str, pd.DataFrame]:
    out = {}
    for k, g in o.groupby(by, dropna=False):
        if len(g) < 20:
            continue
        kmf = KaplanMeierFitter().fit(g["tenure_days"], g["churned"])
        sf = kmf.survival_function_.reset_index()
        sf.columns = ["days", "S"]
        out[str(k)] = sf[sf["days"] <= max_days]
    return out


def _design(o: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=o.index)
    x["log_fleet_at_start"] = np.log1p(o["fleet_at_start"].clip(lower=1))
    x["outside_ca"] = o["state_group"].eq("other").astype(int)
    x["channel_other_rep"] = o["channel"].eq("other_rep").astype(int)
    x["channel_unknown"] = o["channel"].eq("unknown").astype(int)
    x["free_email_domain"] = o["email_is_free_domain"].astype("boolean").fillna(False).astype(int)
    x["signed_oct_mar"] = o["start_month"].isin([10, 11, 12, 1, 2, 3]).astype(int)
    return x


def cox_model(o: pd.DataFrame, cohort_min: int = COX_COHORT_MIN) -> tuple[pd.DataFrame, dict]:
    """Cox PH on signing attributes for cohorts whose churn is fully observed; hazard ratios with 95% CI."""
    d = o[o["start_year"] >= cohort_min].copy()
    x = _design(d)
    for y in sorted(d["start_year"].unique())[1:]:
        x[f"cohort_{y}"] = d["start_year"].eq(y).astype(int)
    x["T"] = d["tenure_days"].clip(lower=1)
    x["E"] = d["churned"]
    cph = CoxPHFitter(penalizer=0.01).fit(x, "T", "E")
    s = cph.summary
    hr = pd.DataFrame({"covariate": s.index, "hazard_ratio": s["exp(coef)"].values, "hr_lo": s["exp(coef) lower 95%"].values, "hr_hi": s["exp(coef) upper 95%"].values, "p": s["p"].values})
    ph = proportional_hazard_test(cph, x, time_transform="rank").summary
    meta = {"n": int(len(d)), "events": int(d["churned"].sum()), "concordance": float(cph.concordance_index_), "ph_test_min_p": float(ph["p"].min()), "cohorts": f"{cohort_min}+"}
    return hr, meta


def hazard_by_k(p: pd.DataFrame, year_min: int = HAZARD_YEAR_MIN) -> pd.DataFrame:
    """Observed churn at each anniversary, overall and per calendar year, with Wilson intervals."""
    from .analysis import wilson

    rows = []
    for (k, y), g in p[p["calendar_year"] >= year_min].groupby(["k", "calendar_year"]):
        r, lo, hi = wilson(int(g["churn"].sum()), len(g))
        rows.append({"k": k, "calendar_year": y, "at_risk": len(g), "churned": int(g["churn"].sum()), "churn_rate": r, "lo": lo, "hi": hi})
    for k, g in p[p["calendar_year"] >= year_min].groupby("k"):
        r, lo, hi = wilson(int(g["churn"].sum()), len(g))
        rows.append({"k": k, "calendar_year": "all", "at_risk": len(g), "churned": int(g["churn"].sum()), "churn_rate": r, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def hazard_by_year(p: pd.DataFrame) -> pd.DataFrame:
    from .analysis import wilson

    rows = []
    for y, g in p.groupby("calendar_year"):
        r, lo, hi = wilson(int(g["churn"].sum()), len(g))
        rows.append({"calendar_year": y, "at_risk": len(g), "churned": int(g["churn"].sum()), "churn_rate": r, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def _period_design(p: pd.DataFrame, k_levels, year_levels) -> pd.DataFrame:
    x = pd.DataFrame(index=p.index)
    for k in k_levels[1:]:
        x[f"k_{k}"] = (p["k"].clip(upper=k_levels[-1]) == k).astype(int)
    for y in year_levels[1:]:
        x[f"year_{y}"] = (p["calendar_year"] == y).astype(int)
    x["log_fleet_at_decision"] = np.log1p(p["fleet_at_decision"].clip(lower=1))
    x["fleet_shrank_by_decision"] = p["fleet_shrank_by_decision"].astype(int)
    x["returned_in_term"] = p["returned_in_term"].astype(int)
    x["outside_ca"] = p["state_group"].eq("other").astype(int)
    x["channel_other_rep"] = p["channel"].eq("other_rep").astype(int)
    x["channel_unknown"] = p["channel"].eq("unknown").astype(int)
    x["free_email_domain"] = p["email_is_free_domain"].astype("boolean").fillna(False).astype(int)
    return x


class DiscreteHazard:
    """Logistic discrete-time hazard, fitted at the decision point: P(churn at anniversary k | still a customer 60 days before it)."""

    def __init__(self, year_min: int = HAZARD_YEAR_MIN, k_cap: int = 5):
        self.year_min = year_min
        self.k_cap = k_cap

    def fit(self, p: pd.DataFrame):
        d = p[(p["calendar_year"] >= self.year_min) & p["at_risk_at_decision"]].copy()
        self.k_levels = list(range(1, self.k_cap + 1))
        self.year_levels = sorted(d["calendar_year"].unique())
        x = _period_design(d, self.k_levels, self.year_levels)
        self.columns = list(x.columns)
        self.model = LogisticRegression(C=1.0, max_iter=2000).fit(x, d["churn"])
        self.n, self.events = len(d), int(d["churn"].sum())
        return self

    def predict(self, p: pd.DataFrame) -> np.ndarray:
        x = _period_design(p, self.k_levels, self.year_levels)[self.columns]
        return self.model.predict_proba(x)[:, 1]

    def odds_ratios(self) -> pd.DataFrame:
        return pd.DataFrame({"covariate": self.columns, "odds_ratio": np.exp(self.model.coef_[0])}).assign(reference="k_1, first calendar year, CA, main channel, business e-mail")
