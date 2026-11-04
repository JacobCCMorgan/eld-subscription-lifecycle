"""Check the reconstructed histories against the January 2024 export of the same CRM list.

The 2024 export lists every vehicle subscription that was active on 5 Jan 2024 with the term dates as they
stood then. Comparing it with the 2026 export answers three questions the current table alone cannot:

1. does `activation_date` really move at renewal (device-level, same serial in both exports)?
2. how many customers active in January 2024 have no trace left in the 2026 table (purged rows)?
3. what did the customers active in January 2024 actually do at their next anniversary (an outcome that
   does not depend on the current table's survivorship)?
"""
from __future__ import annotations

import pandas as pd

from .data import GRACE_DAYS, SNAPSHOT, TERM_DAYS
from .orgs import RENEW_MARGIN_DAYS


def device_level(c: pd.DataFrame, snap: pd.DataFrame) -> dict:
    s = snap[~snap["is_internal_test"]].dropna(subset=["device_key"]).drop_duplicates("device_key")
    cur = c.dropna(subset=["device_key"]).drop_duplicates("device_key")
    m = s.merge(cur[["device_key", "activation_date", "contract_date", "status"]], on="device_key", suffixes=("_2024", "_2026"))
    d = (m["activation_date_2026"] - m["activation_date_2024"]).dt.days
    return {
        "devices_in_2024_export": int(len(s)),
        "devices_found_in_2026_table": int(len(m)),
        "activation_unchanged": float((d == 0).mean()),
        "activation_moved_one_year": float(d.between(360, 371).mean()),
        "activation_moved_two_years": float(d.between(725, 736).mean()),
        "activation_moved_other": float((~(d.eq(0) | d.between(360, 371) | d.between(725, 736))).mean()),
        "contract_date_unchanged": float((m["contract_date_2026"] == m["contract_date_2024"]).mean()),
    }


def org_level(o: pd.DataFrame, snap: pd.DataFrame, snapshot: pd.Timestamp = SNAPSHOT) -> tuple[dict, pd.DataFrame]:
    """Coverage of the 2024 customers in the reconstructed table, and their realized next-anniversary outcome."""
    s = snap[~snap["is_internal_test"]]
    jan = pd.Timestamp("2024-01-05")
    g = s.groupby("org_key").agg(con_min=("contract_date", "min"), act_min=("activation_date", "min"), exp_max=("expiration_date", "max"), rows=("org_key", "size"))
    g["start_2024"] = g["con_min"].fillna(g["act_min"])
    cur = o[o["source"] == "contract_table"].set_index("org_key")
    m = g.join(cur[["start_date", "coverage_end", "active_slots", "churned"]], how="left")
    present = m["start_date"].notna()
    covered = present & (m["start_date"] <= jan) & (m["coverage_end"].fillna(snapshot) >= jan - pd.Timedelta(GRACE_DAYS, "D"))
    same_start = present & ((m["start_date"] - m["start_2024"]).dt.days.abs() <= 31)
    summary = {
        "orgs_active_2024_01_05": int(len(m)),
        "present_in_2026_table": float(present.mean()),
        "purged_from_2026_table": int((~present).sum()),
        "reconstructed_as_covered_on_2024_01_05": float(covered[present].mean()),
        "start_date_within_31_days_of_2024_export": float(same_start[present].mean()),
    }
    # realized outcome at the first anniversary after the 2024 export, by cohort; purged customers count as churned
    m["cohort"] = m["start_2024"].dt.year
    k = ((jan - m["start_2024"]).dt.days // TERM_DAYS + 1).astype("Int64")
    m["k_next"] = k
    m["anniv_next"] = m["start_2024"] + pd.to_timedelta(k.astype(float) * TERM_DAYS, unit="D")
    observable = m["anniv_next"] + pd.Timedelta(RENEW_MARGIN_DAYS, "D") <= snapshot
    renewed = present & ((m["coverage_end"] >= m["anniv_next"] + pd.Timedelta(RENEW_MARGIN_DAYS, "D")) | (m["active_slots"] > 0))
    m["renewed_next"] = renewed.astype(int)
    mm = m[observable & m["cohort"].between(2019, 2023)]
    by = mm.groupby(["cohort", "k_next"]).agg(customers=("renewed_next", "size"), churn_rate=("renewed_next", lambda x: 1 - x.mean()), purged=("start_date", lambda x: int(x.isna().sum()))).reset_index()
    by["churn_rate_excluding_purged"] = mm[mm["start_date"].notna()].groupby(["cohort", "k_next"])["renewed_next"].apply(lambda x: 1 - x.mean()).values
    return summary, by


def cohort_coverage(o: pd.DataFrame, snap: pd.DataFrame) -> pd.DataFrame:
    """Customers per signing year: active in the 2024 export vs present (any status) in the 2026 table."""
    s = snap[~snap["is_internal_test"]]
    g = s.groupby("org_key").agg(con_min=("contract_date", "min"), act_min=("activation_date", "min"))
    g["start"] = g["con_min"].fillna(g["act_min"])
    a = g["start"].dt.year.value_counts().rename("active_in_2024_export")
    b = o[o["source"] == "contract_table"]["start_year"].value_counts().rename("in_2026_table_any_status")
    out = pd.concat([a, b], axis=1).fillna(0).astype(int).sort_index()
    out.index.name = "signing_year"
    out["shortfall"] = (out["active_in_2024_export"] - out["in_2026_table_any_status"]).clip(lower=0)
    return out.reset_index()
