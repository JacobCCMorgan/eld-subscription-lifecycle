"""Presentation extracts for a BI tool: three flat CSV files that relate on `org_key`.

The tables under reports/tables are shaped for the report. A dashboard needs something else: one row per
entity, labels instead of codes, yes / no instead of booleans, and no column that needs the code to be
read. Nothing is modelled here that the pipeline has not already modelled; the extracts are a selection
and a relabelling of the organization and anniversary tables, the hazard score of every account with a
term still to end, and the slot-level rows behind the renewal calendar.

customers.csv           one row per customer organization
anniversaries.csv       one row per customer x anniversary reached
subscriptions_due.csv   one row per active subscription whose term ends in the next twelve months

Apart from the key, no column name is used in more than one file, so a tool that relates the three
does not have to disambiguate them.

Risk tiers use the two probability cut points of the published 60-day list (its median and its 80th
percentile), applied to every account, so an account keeps its tier whatever window a dashboard shows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import demand
from . import renewal_model as rm

LIFECYCLE = {"active": "Active", "billing_hold": "Billing hold", "in_grace": "In grace", "undated_end": "Undated end", "churned": "Churned"}
EXIT_MODE = {"non_renewal": "Non-renewal at term end", "early_return": "Early device return", "purged_after_2024": "Purged after 2024"}
STATE_GROUP = {"CA": "California", "other": "Other states", "unknown": "Unknown"}
CHANNEL = {"main_channel": "Main channel", "other_rep": "Other reps", "unknown": "Unknown"}
FLEET_BAND = {"0": "None left", "1": "1 vehicle", "2": "2 vehicles", "3-5": "3-5 vehicles", "6+": "6+ vehicles"}
SOURCE = {"contract_table": "Contract list", "snapshot_2024_only": "Restored from 2024 export"}

# column -> description; the order is the column order of the file
FIELDS = {
    "customers.csv": {
        "org_key": "Customer pseudonym; the key the three files relate on",
        "customer_since": "Date of the first contract",
        "cohort_year": "Year of the first contract",
        "lifecycle": "Active (an active subscription), Churned, Billing hold, Undated end (no end of coverage on record) or In grace (coverage ended within 30 days), on the export date",
        "churned": "1 when the customer has left (no active subscription, coverage ended more than 30 days before the export)",
        "exit_mode": "How a churned customer left; empty otherwise",
        "tenure_days": "Days from the first contract to the end of coverage, or to the export date while still a customer",
        "vehicles_at_signing": "Subscriptions sold within 30 days of the first contract",
        "fleet_size_at_signing": "vehicles_at_signing in four bands",
        "active_subscriptions": "Subscriptions active on the export date",
        "subscriptions_due_next_year": "Active subscriptions whose term ends in the twelve months after the export (the rows of subscriptions_due.csv)",
        "legacy_units_active": "Active subscriptions still on pre-2.0 hardware",
        "device_returns": "Subscriptions with a device return on record, all years",
        "state_group": "California, Other states or Unknown",
        "channel": "Main channel, Other reps or Unknown",
        "record_source": "Contract list, or Restored from 2024 export for customers whose rows were purged",
        "next_expiration": "Earliest end of term among the active subscriptions; empty without one. A date on or before the export date is a term that has ended on a row still marked active",
        "days_to_expiration": "Days from the export date to next_expiration; zero or negative for the overdue rows",
        "next_anniversary": "Which anniversary next_expiration is (1 = first); empty unless next_expiration is after the export date, like the four columns that follow",
        "returned_device_last_12m": "Yes when a device came back in the twelve months before the export",
        "p_non_renewal": "Hazard model probability that the account does not renew at next_expiration, at the most recent year's level",
        "expected_lost_subscriptions": "p_non_renewal x active_subscriptions",
        "risk_tier": "A, B or C by the cut points of the 60-day list (A = its top 20 %, B = the next 30 %)",
    },
    "anniversaries.csv": {
        "org_key": "Customer pseudonym",
        "anniversary": "1 = first anniversary of the first contract, 2 = second, ...",
        "anniversary_date": "Date of that anniversary",
        "anniversary_year": "Calendar year of anniversary_date",
        "decision_date": "60 days before the anniversary, when the renewal desk would act",
        "renewed": "1 when coverage ran at least 180 days past the anniversary",
        "not_renewed": "1 - renewed",
        "still_customer_at_decision": "Yes when at least one subscription was alive on decision_date",
        "vehicles_at_decision": "Subscriptions alive on decision_date",
        "fleet_size_at_decision": "vehicles_at_decision in bands",
        "fleet_shrank_before_decision": "Yes when vehicles_at_decision is below vehicles_at_signing",
        "returned_device_in_term": "Yes when a device came back during the term, before decision_date",
    },
    "subscriptions_due.csv": {
        "due_id": "Row number, ordered by due date",
        "org_key": "Customer pseudonym",
        "due_date": "End of the current term of this subscription",
        "days_until_due": "Days from the export date to due_date",
        "anniversary_due": "Which anniversary of the customer that date is",
        "p_non_renewal_recent": "Hazard model probability of non-renewal at the most recent year's level",
        "p_non_renewal_peak": "The same at the 2025 peak level",
    },
}


def _yes_no(s: pd.Series) -> pd.Series:
    return s.fillna(False).astype(bool).map({True: "Yes", False: "No"})


def tier_edges(o: pd.DataFrame, c: pd.DataFrame, p: pd.DataFrame, snapshot: pd.Timestamp, horizon_days: int = 60) -> tuple[float, float]:
    """The median and 80th percentile of the published list's probabilities: the B and A cut points."""
    rl = rm.risk_list(o, c, p, snapshot, horizon_days)
    lo, hi = np.quantile(rl["p_churn"], [0.5, 0.8]) if len(rl) else (0.0, 0.0)
    return float(lo), float(hi)


def customers(o: pd.DataFrame, c: pd.DataFrame, p: pd.DataFrame, snapshot: pd.Timestamp, due: pd.DataFrame | None = None) -> pd.DataFrame:
    lo, hi = tier_edges(o, c, p, snapshot)
    due = subscriptions_due(o, c, p, snapshot) if due is None else due
    scored = rm.risk_list(o, c, p, snapshot, horizon_days=36500).set_index("org_key")  # every account with a term still to end
    active = o["active_slots"] > 0
    t = pd.DataFrame({
        "org_key": o["org_key"], "customer_since": o["start_date"], "cohort_year": o["start_year"], "lifecycle": o["lifecycle"].map(LIFECYCLE), "churned": o["churned"].astype(int),
        "exit_mode": o["exit_mode"].map(EXIT_MODE), "tenure_days": o["tenure_days"], "vehicles_at_signing": o["fleet_at_start"], "fleet_size_at_signing": o["fleet_band"].map(FLEET_BAND),
        "active_subscriptions": o["active_slots"], "subscriptions_due_next_year": o["org_key"].map(due.groupby("org_key").size()).fillna(0).astype(int), "legacy_units_active": o["legacy_slots_active"], "device_returns": o["n_returns"], "state_group": o["state_group"].map(STATE_GROUP),
        "channel": o["channel"].map(CHANNEL), "record_source": o["source"].map(SOURCE), "next_expiration": o["next_expiration"].where(active),
    })
    t["days_to_expiration"] = (t["next_expiration"] - snapshot).dt.days.astype("Int64")
    s = scored.reindex(t["org_key"])
    t["next_anniversary"] = s["anniversary_k"].astype("Int64").values
    t["returned_device_last_12m"] = s["returned_device_last_12m"].map({True: "Yes", False: "No"}).values
    t["p_non_renewal"] = s["p_churn"].values
    t["expected_lost_subscriptions"] = (t["p_non_renewal"] * t["active_subscriptions"]).round(4)
    t["risk_tier"] = np.where(t["p_non_renewal"].isna(), None, np.where(t["p_non_renewal"] >= hi, "A", np.where(t["p_non_renewal"] >= lo, "B", "C")))
    return t.sort_values("org_key").reset_index(drop=True)[list(FIELDS["customers.csv"])]


def anniversaries(p: pd.DataFrame) -> pd.DataFrame:
    t = pd.DataFrame({
        "org_key": p["org_key"], "anniversary": p["k"], "anniversary_date": p["anniversary"], "anniversary_year": p["calendar_year"], "decision_date": p["decision_date"],
        "renewed": p["renewed"].astype(int), "not_renewed": p["churn"].astype(int), "still_customer_at_decision": _yes_no(p["at_risk_at_decision"]), "vehicles_at_decision": p["fleet_at_decision"],
        "fleet_size_at_decision": p["fleet_at_decision_band"].map(FLEET_BAND), "fleet_shrank_before_decision": _yes_no(p["fleet_shrank_by_decision"]),
        "returned_device_in_term": _yes_no(p["returned_in_term"]),
    })
    return t.sort_values(["org_key", "anniversary"]).reset_index(drop=True)[list(FIELDS["anniversaries.csv"])]


def subscriptions_due(o: pd.DataFrame, c: pd.DataFrame, p: pd.DataFrame, snapshot: pd.Timestamp, months: int = 12) -> pd.DataFrame:
    a = demand.renewal_slots(o, c, p, snapshot, months)
    t = pd.DataFrame({
        "org_key": a["org_key"].values, "due_date": a["expiration_date"].values, "days_until_due": (a["expiration_date"] - snapshot).dt.days.values, "anniversary_due": a["k"].values,
        "p_non_renewal_recent": a["p_recent"].round(4).values, "p_non_renewal_peak": a["p_peak"].round(4).values, "_row": a["contract_row_id"].values,
    })
    t = t.sort_values(["due_date", "org_key", "_row"]).drop(columns="_row").reset_index(drop=True)
    t.insert(0, "due_id", np.arange(1, len(t) + 1))
    return t[list(FIELDS["subscriptions_due.csv"])]


def extracts(o: pd.DataFrame, c: pd.DataFrame, p: pd.DataFrame, snapshot: pd.Timestamp) -> dict[str, pd.DataFrame]:
    due = subscriptions_due(o, c, p, snapshot)
    return {"customers.csv": customers(o, c, p, snapshot, due), "anniversaries.csv": anniversaries(p), "subscriptions_due.csv": due}


def fields_markdown(tables: dict[str, pd.DataFrame], snapshot: pd.Timestamp, edges: tuple[float, float]) -> str:
    lines = ["# Dashboard extracts", "",
             f"Three files that relate on `org_key`, written by `sublife` from the same run as the report. Export date: {snapshot.date()}.",
             f"Risk tier cut points: A at or above {edges[1]:.4f}, B at or above {edges[0]:.4f}, C below.", ""]
    for name, t in tables.items():
        lines += [f"## `{name}` ({len(t):,} rows)", "", "| column | meaning |", "|---|---|"]
        lines += [f"| `{col}` | {FIELDS[name][col]} |" for col in t.columns]
        lines.append("")
    return "\n".join(lines)


def write_extracts(o: pd.DataFrame, c: pd.DataFrame, p: pd.DataFrame, snapshot: pd.Timestamp, out: Path) -> dict[str, pd.DataFrame]:
    out.mkdir(parents=True, exist_ok=True)
    tables = extracts(o, c, p, snapshot)
    for name, t in tables.items():
        t.to_csv(out / name, index=False, date_format="%Y-%m-%d", float_format="%.4f", lineterminator="\n")
    (out / "FIELDS.md").write_text(fields_markdown(tables, snapshot, tier_edges(o, c, p, snapshot)), encoding="ascii")
    return tables
