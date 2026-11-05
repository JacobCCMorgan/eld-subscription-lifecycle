"""Hardware generations: current mix, what replaced what (where the link survives), and what is left to migrate.

The replacement link (`prev_device_key`) resolves to a known generation for only about a fifth of the
replacement rows - old units leave the registry - so the transition table is a sample, not a census.
"""
from __future__ import annotations

import pandas as pd

from .data import CURRENT_TYPE, LEGACY_TYPES, device_generation_map


def generation_mix(c: pd.DataFrame, snap: pd.DataFrame, devices: pd.DataFrame) -> pd.DataFrame:
    """Active vehicle subscriptions by hardware generation: January 2024 export vs September 2026."""
    g = device_generation_map(c, devices)
    s = snap[~snap["is_internal_test"]]
    m24 = s["device_key"].map(g).fillna("unknown").value_counts().rename("active_2024_01")
    m26 = c.loc[c["is_active_now"], "eld_type"].fillna("unknown").value_counts().rename("active_2026_09")
    out = pd.concat([m24, m26], axis=1).fillna(0).astype(int)
    out.index.name = "generation"
    out["share_2024_01"] = out["active_2024_01"] / out["active_2024_01"].sum()
    out["share_2026_09"] = out["active_2026_09"] / out["active_2026_09"].sum()
    out["is_legacy"] = out.index.isin(LEGACY_TYPES)
    return out.reset_index()


def transitions(c: pd.DataFrame, devices: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    g = device_generation_map(c, devices)
    r = c[c["prev_device_key"].notna()].copy()
    r["prev_generation"] = r["prev_device_key"].map(g)
    known = r.dropna(subset=["prev_generation", "eld_type"])
    t = pd.crosstab(known["prev_generation"], known["eld_type"]).rename_axis(index="from_generation", columns="to_generation")
    t = t.reset_index().melt(id_vars="from_generation", var_name="to_generation", value_name="replacements")
    t = t[t["replacements"] > 0].sort_values("replacements", ascending=False).reset_index(drop=True)
    t["same_generation"] = t["from_generation"] == t["to_generation"]
    t["to_current"] = t["to_generation"].eq(CURRENT_TYPE)
    meta = {"replacement_rows": int(len(r)), "previous_generation_known": float(r["prev_generation"].notna().mean()), "both_known": int(len(known)),
            "same_generation_share_of_known": float((known["prev_generation"] == known["eld_type"]).mean()), "to_current_share_of_known": float(known["eld_type"].eq(CURRENT_TYPE).mean())}
    return t, meta


def pending_migration(c: pd.DataFrame, o: pd.DataFrame) -> pd.DataFrame:
    """Active legacy-generation slots still to be swapped, by customer segment."""
    a = c[c["is_active_now"] & c["eld_type"].isin(LEGACY_TYPES)].merge(o[["org_key", "fleet_band", "state_group", "start_year", "next_expiration"]], on="org_key", how="left")
    rows = []
    for by in ["eld_type", "fleet_band", "state_group", "start_year"]:
        t = a.groupby(by).agg(legacy_slots=("org_key", "size"), customers=("org_key", "nunique")).reset_index().rename(columns={by: "value"})
        t.insert(0, "segment", by)
        rows.append(t)
    out = pd.concat(rows, ignore_index=True)
    out["value"] = out["value"].astype(str)
    return out


def returns_per_month(c: pd.DataFrame, o: pd.DataFrame, start: str = "2024-01") -> pd.DataFrame:
    """Device returns per month, split by whether the customer was still covered afterwards (swap) or not (exit)."""
    r = c.dropna(subset=["return_date"]).merge(o[["org_key", "coverage_end", "churned"]], on="org_key", how="left")
    r = r[r["return_date"] >= pd.Timestamp(start)]
    r["kind"] = "swap_or_downsize"
    r.loc[(r["churned"] == 1) & (r["coverage_end"] <= r["return_date"] + pd.Timedelta(14, "D")), "kind"] = "exit"
    r["month"] = r["return_date"].dt.to_period("M").dt.to_timestamp()
    t = r.groupby(["month", "kind"]).size().unstack(fill_value=0).reset_index()
    for k in ["swap_or_downsize", "exit"]:
        if k not in t:
            t[k] = 0
    t["total"] = t["swap_or_downsize"] + t["exit"]
    return t
