"""Load the de-identified inputs and clean the contract rows.

The CRM list has one row per vehicle subscription slot. At every renewal the term dates are overwritten in
place (`activation_date` moves forward a year, `expiration_date` with it), so a row's `activation_date` is
the start of its *current* term, not the day the device was switched on. `contract_date` keeps the date the
slot was first sold and is inherited when the device is replaced. See data/input/SOURCE.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SNAPSHOT = pd.Timestamp("2026-09-28")  # date of the CRM export
TERM_DAYS = 365
GRACE_DAYS = 30  # a lapse is only called after this many days past the end of coverage
FAR_FUTURE = pd.Timestamp("2030-12-31")  # expirations beyond this are placeholders
LEGACY_TYPES = ("rtps47", "rtel47", "rtps12", "rtoa47", "rtsa12")  # pre-2.0 hardware generations
CURRENT_TYPE = "rtha55"


def load_inputs(path: str | Path = "data/input") -> dict[str, pd.DataFrame]:
    p = Path(path)
    names = ["fact_contract", "dim_org", "dim_device", "contracts_snapshot_2024-01-05", "migration_calls", "migration_pending_2026-01-15"]
    return {n: pd.read_parquet(p / f"{n}.parquet") for n in names}


def clean_contracts(raw: pd.DataFrame, snapshot: pd.Timestamp = SNAPSHOT) -> pd.DataFrame:
    """Customer rows only, placeholder dates removed, renewal count and end of coverage derived per row."""
    c = raw[~raw["is_internal_test"]].copy()
    c["flag_far_future_expiration"] = c["expiration_date"] > FAR_FUTURE
    c.loc[c["flag_far_future_expiration"], "expiration_date"] = pd.NaT
    c["flag_return_before_activation"] = c["return_date"].notna() & c["activation_date"].notna() & (c["return_date"] < c["activation_date"])
    c.loc[c["flag_return_before_activation"], "return_date"] = pd.NaT
    c["flag_negative_vehicle_limit"] = c["vehicle_limit"] < 0
    c.loc[c["flag_negative_vehicle_limit"], "vehicle_limit"] = np.nan
    c["flag_activation_after_snapshot"] = c["activation_date"] > snapshot
    c["flag_active_past_expiration"] = c["status"].eq("active") & (c["expiration_date"] < snapshot)
    c["flag_missing_dates"] = c["contract_date"].isna() & c["activation_date"].isna()

    gap = (c["activation_date"] - c["contract_date"]).dt.days
    c["renewals_so_far"] = (gap / TERM_DAYS).round()
    c.loc[gap < -14, "renewals_so_far"] = np.nan  # activation before the contract date: inconsistent, leave undefined
    c["renewals_so_far"] = c["renewals_so_far"].clip(lower=0)
    c["gap_is_whole_years"] = (gap - c["renewals_so_far"] * TERM_DAYS).abs() <= 7

    c["eld_type"] = c["eld_type"].where(c["eld_type"].isin(LEGACY_TYPES + (CURRENT_TYPE,)), None)
    c["is_replacement"] = c["prev_device_key"].notna()
    c["slot_start"] = c["contract_date"].fillna(c["activation_date"])
    c["start_source"] = np.where(c["contract_date"].notna(), "contract_date", np.where(c["activation_date"].notna(), "activation_date", "none"))

    # end of the row's paid coverage: the term end, or the return date when the device went back earlier
    early_return = c["status"].eq("deleted") & c["return_date"].notna() & (c["return_date"] < c["expiration_date"].fillna(pd.Timestamp.max))
    c["row_end"] = np.where(early_return, c["return_date"], c["expiration_date"])
    c["row_end"] = pd.to_datetime(c["row_end"])
    c["is_active_now"] = c["status"].eq("active") & (c["expiration_date"] >= snapshot - pd.Timedelta(GRACE_DAYS, "D"))
    c["is_billing_hold"] = c["status"].isin(["inactive", "unpaid", "suspended"]) & (c["expiration_date"] >= snapshot)
    return c


def device_generation_map(contracts: pd.DataFrame, devices: pd.DataFrame) -> pd.Series:
    """device_key -> hardware generation (eld_type), from the contract rows first, then the device registry."""
    from_rows = contracts.dropna(subset=["device_key", "eld_type"]).drop_duplicates("device_key").set_index("device_key")["eld_type"]
    reg = devices.dropna(subset=["device_key", "model_code"]).copy()
    reg["gen"] = reg["model_code"].astype(str).str.strip().str.lower()
    reg = reg[reg["gen"].isin(LEGACY_TYPES + (CURRENT_TYPE,))].drop_duplicates("device_key").set_index("device_key")["gen"]
    return pd.concat([from_rows, reg[~reg.index.isin(from_rows.index)]])
