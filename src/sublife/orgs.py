"""Organization-level lifecycle table and the anniversary (person-period) table.

Unit of analysis: the customer organization. Contracts are sold per vehicle slot, but the decision to stay
or leave is taken by the customer, and 90 % of customers run five vehicles or fewer.

Definitions
  start_date       first `contract_date` across the organization's slots (first `activation_date` when no
                   contract date survives);
  coverage_end     latest end of paid coverage across slots (term end, or return date when earlier);
  churned          no active slot and coverage ended more than GRACE_DAYS before the snapshot;
  tenure_days      coverage_end - start_date for churned organizations, snapshot - start_date otherwise;
  anniversary k    start_date + k * 365 days; the organization "renewed at k" when its coverage runs at
                   least RENEW_MARGIN_DAYS past that date, "churned at k" when it did not and k is the
                   last anniversary it reached, and is censored when the anniversary is too recent for the
                   decision to be visible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CURRENT_TYPE, GRACE_DAYS, LEGACY_TYPES, SNAPSHOT, TERM_DAYS

RENEW_MARGIN_DAYS = 180  # coverage must run this far past an anniversary to count as renewed there
START_WINDOW_DAYS = 30  # slots sold within this many days of the first one form the "fleet at start"


def _mode(s: pd.Series):
    s = s.dropna()
    return s.mode().iloc[0] if len(s) else None


def org_table(c: pd.DataFrame, dim_org: pd.DataFrame, snapshot: pd.Timestamp = SNAPSHOT) -> pd.DataFrame:
    """One row per customer organization with dated slots."""
    d = c[c["slot_start"].notna()].copy()
    d["is_legacy_type"] = d["eld_type"].isin(LEGACY_TYPES)
    d["is_current_type"] = d["eld_type"].eq(CURRENT_TYPE)
    first = d.groupby("org_key")["slot_start"].transform("min")
    d["at_start"] = (d["slot_start"] - first).dt.days <= START_WINDOW_DAYS

    g = d.groupby("org_key")
    o = pd.DataFrame({
        "start_date": g["slot_start"].min(),
        "start_source": g["start_source"].agg(lambda s: "contract_date" if (s == "contract_date").any() else "activation_date"),
        "coverage_end": g["row_end"].max(),
        "last_term_end": g["expiration_date"].max(),
        "n_slots": g.size(),
        "n_devices": g["device_key"].nunique(),
        "vehicles_ever": g["vin_key"].nunique(),
        "active_slots": g["is_active_now"].sum(),
        "billing_hold_slots": g["is_billing_hold"].sum(),
        "fleet_at_start": g["at_start"].sum(),
        "renewals_max": g["renewals_so_far"].max(),
        "n_replacements": g["is_replacement"].sum(),
        "n_returns": g["return_date"].count(),
        "first_return": g["return_date"].min(),
        "has_legacy_type": g["is_legacy_type"].any(),
        "has_current_type": g["is_current_type"].any(),
        "legacy_slots_active": g.apply(lambda x: int((x["is_active_now"] & x["is_legacy_type"]).sum()), include_groups=False),
        "vehicle_limit": g["vehicle_limit"].max(),
        "sales_rep_key": g["sales_rep_key"].agg(_mode),
        "next_expiration": g.apply(lambda x: x.loc[x["is_active_now"], "expiration_date"].min(), include_groups=False),
    })
    # generation and vehicle of the slots sold at the start
    s0 = d[d["at_start"]].groupby("org_key")
    o["first_generation"] = s0["eld_type"].agg(_mode)
    o["first_make"] = s0["make"].agg(_mode)
    o["first_model_year"] = s0["model_year"].median()
    o["first_generation"] = o["first_generation"].fillna("unknown")

    o["start_year"] = o["start_date"].dt.year
    o["start_month"] = o["start_date"].dt.month
    end_known = o["coverage_end"].notna()
    lapsed = (o["active_slots"] == 0) & end_known & (o["coverage_end"] < snapshot - pd.Timedelta(GRACE_DAYS, "D"))
    o["churned"] = lapsed.astype(int)
    o["end_date"] = np.where(lapsed, o["coverage_end"], snapshot)
    o["end_date"] = pd.to_datetime(o["end_date"])
    o["tenure_days"] = (o["end_date"] - o["start_date"]).dt.days.clip(lower=0)
    o["lifecycle"] = np.select(
        [o["active_slots"] > 0, lapsed, o["billing_hold_slots"] > 0, ~end_known],
        ["active", "churned", "billing_hold", "undated_end"],
        default="in_grace",
    )
    # how a churned customer left: returned the device(s) before the term ran out, or simply did not renew
    early = lapsed & o["last_term_end"].notna() & (o["coverage_end"] < o["last_term_end"] - pd.Timedelta(14, "D"))
    o["exit_mode"] = np.select([lapsed & early, lapsed], ["early_return", "non_renewal"], default=None)
    o["source"] = "contract_table"
    o["fleet_band"] = pd.cut(o["fleet_at_start"].clip(lower=1), [0, 1, 2, 5, np.inf], labels=["1", "2", "3-5", "6+"]).astype(str)

    dims = dim_org[["org_key", "state", "has_dot", "email_is_free_domain", "is_reseller", "crm_active", "sales_rep_key"]].drop_duplicates("org_key").set_index("org_key")
    o = o.join(dims, rsuffix="_crm")
    o["sales_rep_key"] = o["sales_rep_key"].fillna(o["sales_rep_key_crm"])
    o = o.drop(columns=["sales_rep_key_crm"])
    o["state_group"] = np.where(o["state"].eq("CA"), "CA", np.where(o["state"].isna(), "unknown", "other"))
    top_rep = o["sales_rep_key"].value_counts().idxmax()
    o["channel"] = np.where(o["sales_rep_key"].eq(top_rep), "main_channel", np.where(o["sales_rep_key"].isna(), "unknown", "other_rep"))
    return o.reset_index()


def org_periods(o: pd.DataFrame, c: pd.DataFrame, snapshot: pd.Timestamp = SNAPSHOT, margin_days: int = RENEW_MARGIN_DAYS) -> pd.DataFrame:
    """One row per organization x anniversary reached, with the renewal outcome and time-varying covariates."""
    c = c[c["slot_start"].notna()]
    rows = []
    cutoff = snapshot - pd.Timedelta(GRACE_DAYS, "D")
    for r in o.itertuples(index=False):
        k = 1
        while True:
            a = r.start_date + pd.Timedelta(k * TERM_DAYS, "D")
            if a > cutoff:
                break
            renewed = pd.notna(r.coverage_end) and (r.coverage_end >= a + pd.Timedelta(margin_days, "D"))
            if not renewed and r.active_slots > 0 and pd.notna(r.next_expiration) and r.next_expiration > a:
                renewed = True  # an active slot whose term runs past this anniversary
            # the renewal at k is only observable when the margin has elapsed (or the org already churned)
            observable = (a + pd.Timedelta(margin_days, "D") <= snapshot) or r.churned == 1 or renewed
            if not observable:
                break
            rows.append({"org_key": r.org_key, "k": k, "anniversary": a, "renewed": int(renewed)})
            if not renewed:
                break
            k += 1
    p = pd.DataFrame(rows)
    if p.empty:
        return p
    p["calendar_year"] = p["anniversary"].dt.year
    # slots alive at the anniversary and replacements in the preceding term
    slot = c[["org_key", "slot_start", "row_end", "eld_type"]].copy()
    alive = p.merge(slot, on="org_key", how="left")
    alive["alive"] = (alive["slot_start"] <= alive["anniversary"]) & (alive["row_end"].fillna(snapshot) >= alive["anniversary"] - pd.Timedelta(GRACE_DAYS, "D"))
    fleet = alive.groupby(["org_key", "k"])["alive"].sum().rename("fleet_at_k")
    gen = alive[alive["alive"]].groupby(["org_key", "k"])["eld_type"].agg(_mode).rename("generation_at_k")
    p = p.merge(fleet, on=["org_key", "k"], how="left").merge(gen, on=["org_key", "k"], how="left")
    p["generation_at_k"] = p["generation_at_k"].fillna("unknown")
    p["fleet_at_k"] = p["fleet_at_k"].fillna(0).astype(int)
    static = o[["org_key", "start_year", "fleet_at_start", "fleet_band", "first_generation", "state_group", "channel", "has_dot", "email_is_free_domain"]]
    p = p.merge(static, on="org_key", how="left")
    p["churn"] = 1 - p["renewed"]
    return p


def restore_purged_orgs(snapshot_2024: pd.DataFrame, o: pd.DataFrame, dim_org: pd.DataFrame, snapshot: pd.Timestamp = SNAPSHOT) -> pd.DataFrame:
    """Organizations that were active in the January 2024 export but have no dated contract row today.

    The CRM purges old rows, so these customers would otherwise vanish from the history together with their
    churn. They are rebuilt from the 2024 export: start = earliest contract date there, coverage end = the
    latest expiration there (the last term we can see), churned = yes. Static attributes come from the
    organization list; the hardware generation at the start is unknown for them.
    """
    s = snapshot_2024[~snapshot_2024["is_internal_test"]].copy()
    s = s[~s["org_key"].isin(o["org_key"])]
    s["slot_start"] = s["contract_date"].fillna(s["activation_date"])
    s = s[s["slot_start"].notna()]
    first = s.groupby("org_key")["slot_start"].transform("min")
    s["at_start"] = (s["slot_start"] - first).dt.days <= START_WINDOW_DAYS
    g = s.groupby("org_key")
    r = pd.DataFrame({
        "start_date": g["slot_start"].min(),
        "start_source": "contract_date",
        "coverage_end": g["expiration_date"].max(),
        "last_term_end": g["expiration_date"].max(),
        "n_slots": g.size(),
        "n_devices": g["device_key"].nunique(),
        "vehicles_ever": g["vin_key"].nunique(),
        "active_slots": 0,
        "billing_hold_slots": 0,
        "fleet_at_start": g["at_start"].sum(),
        "renewals_max": np.nan,
        "n_replacements": g["prev_device_key"].count(),
        "n_returns": 0,
        "first_return": pd.NaT,
        "has_legacy_type": True,
        "has_current_type": False,
        "legacy_slots_active": 0,
        "vehicle_limit": np.nan,
        "sales_rep_key": None,
        "next_expiration": pd.NaT,
        "first_generation": "unknown",
        "first_make": s[s["at_start"]].groupby("org_key")["make"].agg(_mode),
        "first_model_year": s[s["at_start"]].groupby("org_key")["model_year"].median(),
    })
    r["start_year"] = r["start_date"].dt.year
    r["start_month"] = r["start_date"].dt.month
    r["churned"] = 1
    r["end_date"] = r["coverage_end"]
    r["tenure_days"] = (r["end_date"] - r["start_date"]).dt.days.clip(lower=0)
    r["lifecycle"] = "churned"
    r["exit_mode"] = "purged_after_2024"
    r["source"] = "snapshot_2024_only"
    r["fleet_band"] = pd.cut(r["fleet_at_start"].clip(lower=1), [0, 1, 2, 5, np.inf], labels=["1", "2", "3-5", "6+"]).astype(str)
    dims = dim_org[["org_key", "state", "has_dot", "email_is_free_domain", "is_reseller", "crm_active", "sales_rep_key"]].drop_duplicates("org_key").set_index("org_key")
    r = r.join(dims, rsuffix="_crm")
    r["sales_rep_key"] = r["sales_rep_key_crm"]
    r = r.drop(columns=["sales_rep_key_crm"])
    r["state_group"] = np.where(r["state"].eq("CA"), "CA", np.where(r["state"].isna(), "unknown", "other"))
    top_rep = o["sales_rep_key"].value_counts().idxmax()
    r["channel"] = np.where(r["sales_rep_key"].eq(top_rep), "main_channel", np.where(r["sales_rep_key"].isna(), "unknown", "other_rep"))
    return r.reset_index()


def org_table_with_restored(c: pd.DataFrame, dim_org: pd.DataFrame, snapshot_2024: pd.DataFrame, snapshot: pd.Timestamp = SNAPSHOT) -> pd.DataFrame:
    o = org_table(c, dim_org, snapshot)
    r = restore_purged_orgs(snapshot_2024, o, dim_org, snapshot)
    out = pd.concat([o, r[o.columns.intersection(r.columns)]], ignore_index=True)
    out["in_snapshot_2024"] = out["org_key"].isin(snapshot_2024["org_key"])
    return out
