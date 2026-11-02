"""Build the de-identified input tables in data/input/ from the raw exports.

This is the one step that needs the raw files and the secret key; it is kept in the repository so the
provenance of every input column is explicit, but it cannot be run from the public data alone.

    TDF_SRC=/path/to/telematics-data-foundation/src \
    TDF_DEID_KEY_FILE=/path/to/deid_key.txt \
    python scripts/prepare_inputs.py --foundation /path/to/telematics-data-foundation --raw /path/to/raw

Inputs
  * the public tables of telematics-data-foundation (fact_contract, dim_org, dim_device) - copied as is;
  * the January 2024 "All Users" contract snapshot (xlsx) - pseudonymized with the same key;
  * the ELD 2.0 migration call-campaign workbook (xlsx) - reduced to per-call categories, no text.

The pseudonymizer and the organization-name normalizer are imported from the foundation project so that
the keys agree across repositories.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.environ.get("TDF_SRC", ""))
from tdf.build import extract_dot, flag_internal_org, normalize_org_name, to_datetime, to_number  # noqa: E402
from tdf.deid import Pseudonymizer  # noqa: E402

# the workbook's own sheet and column names are not in English; they are spelled out by code point so the
# repository stays ASCII-only
CJK_RUN = "[" + chr(0x4E00) + "-" + chr(0x9FFF) + "]+"
SMALL_VEHICLE_COL = chr(0x5C0F) + chr(0x8ECA) + "?"  # "small vehicle?" flag column in rounds 4 and 5
PENDING_SHEET = "".join(chr(c) for c in (0x5C1A, 0x672A, 0x66F4, 0x63DB, 0x540D, 0x55AE))  # "not yet replaced" list

RESULT_CODES = {
    "no answer, send text message": "no_answer_text_sent",
    "msg is sent (no ans)": "no_answer_text_sent",
    "voice mail": "no_answer_text_sent",
    "will come to exchange": "agreed_walk_in",
    "will come soon": "agreed_walk_in",
    "need delivery": "agreed_shipping",
    "need shipping": "agreed_shipping",
    "rtha55": "already_on_new_device",
    "changed already": "already_on_new_device",
    "account closed": "account_closed",
    "closed": "account_closed",
    "want to cancel": "wants_to_cancel",
    "want to cxl": "wants_to_cancel",
    "suspend": "suspended",
    "suspended": "suspended",
    "suspendx": "suspended",
    "unpaid $": "unpaid",
    "other (will call back later)": "callback_later",
    "emailed": "emailed",
    "emailed to client": "emailed",
    "not english speaking": "language_barrier",
    "no contact no": "invalid_number",
    "phone is invalid": "invalid_number",
    "tel is invaild": "invalid_number",
    "wrong/ not in use number": "invalid_number",
    "small car": "small_vehicle",
    "device lost": "device_lost",
    "others": "other",
}


def _patch_openpyxl():
    # one sheet carries an autofilter reference openpyxl refuses to parse; relax the check
    import openpyxl.descriptors
    import openpyxl.worksheet.filters as f

    f.AutoFilter.ref = openpyxl.descriptors.String(allow_none=True)


def result_code(value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    s = re.sub(CJK_RUN, "", s).strip()  # drop the operators' non-English annotations (e.g. the small-vehicle tag)
    if s in RESULT_CODES:
        return RESULT_CODES[s]
    if s.startswith("will come before"):
        return "agreed_exchange"  # "will come before <deadline> or need shipping"
    if re.fullmatch(r"[\(\)\d\s\-+]{7,}", s):
        return "invalid_number"
    if s in {"mandy", "justin", "kalley", "grace"}:
        return None  # a staff name typed into the result column
    return "other"


def build_snapshot_2024(path: Path, p: Pseudonymizer) -> pd.DataFrame:
    raw = pd.ExcelFile(path).parse("All Users (01052024)")
    d = pd.DataFrame()
    norm = raw["Org Name"].map(normalize_org_name)
    d["org_key"] = p.series(norm, "org", "ORG-", normalize=lambda x: x)
    d["is_internal_test"] = flag_internal_org(norm)
    d["vin_key"] = p.series(raw["VIN"], "vin", "VIN-")
    d["device_key"] = p.series(raw["ELD SN."], "device", "DEV-")
    d["prev_device_key"] = p.series(raw["Previous ELD"], "device", "DEV-")
    d["make"] = raw["Make"].astype(str).str.strip().replace({"nan": np.nan, "-": np.nan})
    d["model_year"] = pd.to_numeric(raw["Year"], errors="coerce")
    d["contract_date"] = to_datetime(raw["Contract Date"])
    d["activation_date"] = to_datetime(raw["Activation Date"])
    d["expiration_date"] = to_datetime(raw["Expiration Date"])
    d["snapshot_date"] = pd.Timestamp("2024-01-05")
    return d


def _org_from_name_dot(series: pd.Series, p: Pseudonymizer) -> tuple[pd.Series, pd.Series]:
    norm = series.map(normalize_org_name)
    org = p.series(norm, "org", "ORG-", normalize=lambda x: x)
    dot = p.series(series.map(extract_dot), "dot", "DOT-", normalize=lambda x: str(x))
    return org, dot


def build_calls(path: Path, p: Pseudonymizer) -> pd.DataFrame:
    x = pd.ExcelFile(path)
    frames = []

    r1 = x.parse("2.0-1st call")
    d = pd.DataFrame()
    d["org_key"] = p.series(r1["Org Name"].map(normalize_org_name), "org", "ORG-", normalize=lambda v: v)
    d["dot_key"] = p.series(r1["DOT#"].map(lambda v: extract_dot(f"({v})") if pd.notna(v) else None), "dot", "DOT-", normalize=lambda v: str(v))
    d["result_code"] = r1["Action"].map(result_code)
    d["text_sent"] = r1["sent6/25"].notna()
    d["call_date"] = pd.NaT
    d["campaign_phase"] = "2025 outreach"
    d["date_known"] = False
    d["staff_key"] = p.series(r1["Status"], "staff", "STF-", normalize=lambda v: str(v).strip().lower())
    d["sales_rep_key"] = p.series(r1["Sales"], "sales", "REP-", normalize=lambda v: str(v).strip().lower())
    d["eld_type_at_call"] = None
    d["status_at_call"] = None
    d["fleet_flag"] = None
    d.insert(0, "round", 1)
    frames.append(d)

    r2 = x.parse("2.0-2nd call")
    d = pd.DataFrame()
    d["org_key"], d["dot_key"] = _org_from_name_dot(r2["Org Name(DOT#)"], p)
    d["result_code"] = r2["result"].map(result_code)
    d["text_sent"] = d["result_code"].eq("no_answer_text_sent")
    d["call_date"] = pd.NaT
    d["campaign_phase"] = "2025 outreach"
    d["date_known"] = False
    d["staff_key"] = p.series(r2["csr"], "staff", "STF-", normalize=lambda v: str(v).strip().lower())
    d["sales_rep_key"] = p.series(r2["Sales"], "sales", "REP-", normalize=lambda v: str(v).strip().lower())
    d["eld_type_at_call"] = None
    d["status_at_call"] = None
    d["fleet_flag"] = None
    d.insert(0, "round", 2)
    frames.append(d)

    r3 = x.parse("2.0-3rd Call")
    d = pd.DataFrame()
    d["org_key"], d["dot_key"] = _org_from_name_dot(r3["Org Name(DOT#)"], p)
    d["result_code"] = r3["Result"].map(result_code)
    d["text_sent"] = d["result_code"].eq("no_answer_text_sent")
    d["call_date"] = pd.NaT
    d["campaign_phase"] = "2025 outreach"
    d["date_known"] = False
    d["staff_key"] = p.series(r3["Staff"], "staff", "STF-", normalize=lambda v: str(v).strip().lower())
    d["sales_rep_key"] = p.series(r3["Sales"], "sales", "REP-", normalize=lambda v: str(v).strip().lower())
    d["eld_type_at_call"] = None
    d["status_at_call"] = r3["Status"].astype(str).str.strip().str.lower().replace({"nan": None})
    d["fleet_flag"] = np.where(to_number(r3["Vehicle Limit"]) >= 4, "4+", np.where(to_number(r3["Vehicle Limit"]) >= 3, "3", None))
    d.insert(0, "round", 3)
    frames.append(d)

    r4 = x.parse("2.0 -4th Call")
    d = pd.DataFrame()
    d["org_key"], d["dot_key"] = _org_from_name_dot(r4["Org Name(DOT#)"], p)
    d["result_code"] = r4["result"].map(result_code)
    d["text_sent"] = d["result_code"].eq("no_answer_text_sent")
    d["call_date"] = to_datetime(r4["date"])
    d["campaign_phase"] = "2026-09 outreach"
    d["date_known"] = d["call_date"].notna()
    d["staff_key"] = None
    d["sales_rep_key"] = None
    d["eld_type_at_call"] = r4["ELD Type"].astype(str).str.strip().str.lower().replace({"nan": None})
    d["status_at_call"] = r4["Status"].astype(str).str.strip().str.lower().replace({"nan": None})
    big = r4["big co"].astype(str).str.strip().str.lower()
    d["fleet_flag"] = np.where(big.str.startswith("4"), "4+", np.where(big.str.startswith("3"), "3", None))
    d.loc[r4[SMALL_VEHICLE_COL].notna(), "result_code"] = d.loc[r4[SMALL_VEHICLE_COL].notna(), "result_code"].fillna("small_vehicle")
    d.insert(0, "round", 4)
    frames.append(d)

    r5 = x.parse("2.0 -5th Call")
    d = pd.DataFrame()
    d["org_key"], d["dot_key"] = _org_from_name_dot(r5["Org Name(DOT#)"], p)
    d["result_code"] = r5["Result"].map(result_code)
    d["text_sent"] = d["result_code"].eq("no_answer_text_sent")
    d["call_date"] = to_datetime(r5["Date"])
    d["campaign_phase"] = "2026-09 outreach"
    d["date_known"] = d["call_date"].notna()
    d["staff_key"] = None
    d["sales_rep_key"] = None
    d["eld_type_at_call"] = r5["ELD Type"].astype(str).str.strip().str.lower().replace({"nan": None})
    d["status_at_call"] = r5["Status"].astype(str).str.strip().str.lower().replace({"nan": None})
    d["fleet_flag"] = None
    d.insert(0, "round", 5)
    frames.append(d)

    out = pd.concat(frames, ignore_index=True)
    out["listed"] = True
    out["contacted"] = out["result_code"].notna()
    return out


def build_pending_jan2026(path: Path, p: Pseudonymizer) -> pd.DataFrame:
    raw = pd.ExcelFile(path).parse(PENDING_SHEET)
    raw = raw[raw["Org Name"].notna()]
    d = pd.DataFrame()
    d["org_key"] = p.series(raw["Org Name"].map(normalize_org_name), "org", "ORG-", normalize=lambda v: v)
    d["dot_key"] = p.series(raw["DOT#"].map(lambda v: extract_dot(f"({v})") if pd.notna(v) else None), "dot", "DOT-", normalize=lambda v: str(v))
    d["status"] = raw["Status"].astype(str).str.strip().str.lower().replace({"nan": None})
    d["sales_rep_key"] = p.series(raw["Sales"], "sales", "REP-", normalize=lambda v: str(v).strip().lower())
    d["list_date"] = pd.Timestamp("2026-01-15")
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--foundation", required=True, type=Path)
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/input"))
    a = ap.parse_args(argv)
    _patch_openpyxl()
    p = Pseudonymizer.from_env()
    a.out.mkdir(parents=True, exist_ok=True)
    pub = a.foundation / "data" / "public"
    for name in ["fact_contract", "dim_org", "dim_device"]:
        shutil.copy(pub / f"{name}.parquet", a.out / f"{name}.parquet")
    snap = build_snapshot_2024(a.raw / "RX-05_eld12_device_inventory.xlsx", p)
    snap.to_parquet(a.out / "contracts_snapshot_2024-01-05.parquet", index=False)
    calls = build_calls(a.raw / "RX-06_eld12_to_13_migration.xlsx", p)
    calls.to_parquet(a.out / "migration_calls.parquet", index=False)
    pend = build_pending_jan2026(a.raw / "RX-06_eld12_to_13_migration.xlsx", p)
    pend.to_parquet(a.out / "migration_pending_2026-01-15.parquet", index=False)
    print({"snapshot_2024_rows": len(snap), "calls_rows": len(calls), "calls_contacted": int(calls.contacted.sum()), "pending_rows": len(pend)})
    print("result codes:", calls.result_code.value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
