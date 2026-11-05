"""The ELD 2.0 migration call campaign: who was reached, what they said, and what happened afterwards.

Five call rounds. Rounds 1-3 ran in 2025 (no per-call date in the source), rounds 4-5 in September 2026,
two to four weeks before the CRM export, so only the 2025 rounds have a long enough window to look at
outcomes. Outcome = the customer's state on 28 Sep 2026: migrated (an active 2.0 slot), still on legacy
hardware, or no longer a customer.
"""
from __future__ import annotations

import pandas as pd

from .analysis import suppress, wilson

REACHED = {"agreed_walk_in", "agreed_shipping", "agreed_exchange", "already_on_new_device", "wants_to_cancel", "account_closed", "suspended", "unpaid", "callback_later", "language_barrier", "small_vehicle", "device_lost", "other", "emailed"}
AGREED = {"agreed_walk_in", "agreed_shipping", "agreed_exchange"}
GROUPS = {
    "agreed_walk_in": "agreed to exchange", "agreed_shipping": "agreed to exchange", "agreed_exchange": "agreed to exchange",
    "already_on_new_device": "already migrated", "no_answer_text_sent": "no answer (text sent)", "invalid_number": "no answer (text sent)",
    "wants_to_cancel": "cancel / closed / on hold", "account_closed": "cancel / closed / on hold", "suspended": "cancel / closed / on hold", "unpaid": "cancel / closed / on hold",
    "callback_later": "other", "emailed": "other", "language_barrier": "other", "small_vehicle": "other", "device_lost": "other", "other": "other",
}


def funnel(calls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r, g in calls.groupby("round"):
        contacted = g[g["contacted"]]
        reached = contacted[contacted["result_code"].isin(REACHED)]
        rows.append({
            "round": int(r), "campaign_phase": g["campaign_phase"].iloc[0], "listed_rows": len(g), "listed_customers": g["org_key"].nunique(),
            "contacted_rows": len(contacted), "reached_rows": len(reached), "reach_rate_of_contacted": len(reached) / max(len(contacted), 1),
            "agreed_rows": int(contacted["result_code"].isin(AGREED).sum()), "already_migrated_rows": int(contacted["result_code"].eq("already_on_new_device").sum()),
            "cancel_or_closed_rows": int(contacted["result_code"].isin(["wants_to_cancel", "account_closed", "suspended", "unpaid"]).sum()),
        })
    return pd.DataFrame(rows)


def result_mix(calls: pd.DataFrame) -> pd.DataFrame:
    d = calls[calls["contacted"]].copy()
    d["result_group"] = d["result_code"].map(GROUPS).fillna("other")
    t = d.groupby(["round", "result_group"]).size().rename("rows").reset_index()
    t["share_of_contacted"] = t["rows"] / t.groupby("round")["rows"].transform("sum")
    t["rows"] = t["rows"].map(suppress)
    return t


def outcomes_2025_rounds(calls: pd.DataFrame, o: pd.DataFrame) -> pd.DataFrame:
    """Customers listed in the 2025 rounds: state on the export date by the best result they gave in 2025."""
    d = calls[calls["campaign_phase"] == "2025 outreach"].copy()
    d["result_group"] = d["result_code"].map(GROUPS)
    order = {"agreed to exchange": 0, "already migrated": 1, "cancel / closed / on hold": 2, "other": 3, "no answer (text sent)": 4}
    d["rank"] = d["result_group"].map(order).fillna(5)
    best = d.sort_values("rank").drop_duplicates("org_key")[["org_key", "result_group"]]
    best["result_group"] = best["result_group"].fillna("listed, never contacted")
    m = best.merge(o[["org_key", "lifecycle", "has_current_type", "legacy_slots_active", "active_slots"]], on="org_key", how="left")
    m["state_2026_09"] = "not found"
    m.loc[m["lifecycle"].notna(), "state_2026_09"] = "left"
    m.loc[(m["active_slots"] > 0) & (m["legacy_slots_active"] > 0) & ~m["has_current_type"].fillna(False).astype(bool), "state_2026_09"] = "active, still legacy"
    m.loc[(m["active_slots"] > 0) & m["has_current_type"].fillna(False).astype(bool), "state_2026_09"] = "active, migrated"
    m.loc[(m["active_slots"] > 0) & (m["state_2026_09"] == "left"), "state_2026_09"] = "active, generation unknown"
    rows = []
    for grp, g in m.groupby("result_group"):
        n = len(g)
        mig = int((g["state_2026_09"] == "active, migrated").sum())
        left = int((g["state_2026_09"] == "left").sum())
        pm, lo, hi = wilson(mig, n)
        pl, llo, lhi = wilson(left, n)
        rows.append({"result_in_2025": grp, "customers": n, "migrated_by_2026_09": suppress(mig), "migrated_share": pm, "migrated_lo": lo, "migrated_hi": hi,
                     "left_by_2026_09": suppress(left), "left_share": pl, "left_lo": llo, "left_hi": lhi,
                     "still_legacy": suppress(int((g["state_2026_09"] == "active, still legacy").sum()))})
    return pd.DataFrame(rows).sort_values("customers", ascending=False).reset_index(drop=True)


def repeat_contacts(calls: pd.DataFrame) -> pd.DataFrame:
    """How many 2025-round customers were called in one, two or three rounds, and the reach rate by attempt number."""
    d = calls[(calls["campaign_phase"] == "2025 outreach") & calls["contacted"] & calls["org_key"].notna()].copy()
    d["attempt"] = d.groupby("org_key")["round"].rank(method="first").astype(int)
    d["reached"] = d["result_code"].isin(REACHED)
    t = d.groupby("attempt").agg(customers=("org_key", "nunique"), reached=("reached", "sum")).reset_index()
    t["reach_rate"] = t["reached"] / t["customers"]
    return t
