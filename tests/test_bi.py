"""Dashboard extracts: shape, labels, and agreement with the tables the report is written from."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sublife import bi, demand, survival
from sublife import renewal_model as rm
from sublife.data import SNAPSHOT, clean_contracts, load_inputs
from sublife.orgs import org_periods, org_table_with_restored

INPUT = Path(__file__).resolve().parents[1] / "data" / "input"
PUBLISHED = Path(__file__).resolve().parents[1] / "reports" / "tableau"
needs_inputs = pytest.mark.skipif(not (INPUT / "fact_contract.parquet").exists(), reason="published inputs not present")
T = pd.Timestamp


def test_anniversary_extract_relabels_without_changing_the_outcome():
    p = pd.DataFrame({"org_key": ["ORG-B", "ORG-A", "ORG-A"], "k": [1, 2, 1], "anniversary": [T("2025-03-01"), T("2025-01-01"), T("2024-01-01")], "calendar_year": [2025, 2025, 2024],
                      "decision_date": [T("2024-12-31"), T("2024-11-02"), T("2023-11-02")], "renewed": [1, 0, 1], "churn": [0, 1, 0], "at_risk_at_decision": [True, False, True], "fleet_at_decision": [4, 0, 1],
                      "fleet_at_decision_band": ["3-5", "0", "1"], "fleet_shrank_by_decision": [False, True, False], "returned_in_term": [False, True, False]})
    t = bi.anniversaries(p)
    assert list(t.columns) == list(bi.FIELDS["anniversaries.csv"])
    assert t[["org_key", "anniversary"]].values.tolist() == [["ORG-A", 1], ["ORG-A", 2], ["ORG-B", 1]]  # sorted, one row per customer x anniversary
    assert t["not_renewed"].tolist() == [0, 1, 0] and (t["renewed"] + t["not_renewed"] == 1).all()
    assert t["still_customer_at_decision"].tolist() == ["Yes", "No", "Yes"] and t["fleet_size_at_decision"].tolist() == ["1 vehicle", "None left", "3-5 vehicles"]
    assert t["returned_device_in_term"].tolist() == ["No", "Yes", "No"]


def test_every_label_map_is_ascii_and_every_column_is_described():
    for m in (bi.LIFECYCLE, bi.EXIT_MODE, bi.STATE_GROUP, bi.CHANNEL, bi.FLEET_BAND, bi.SOURCE):
        assert all(v.isascii() and (v[0].isupper() or v[0].isdigit()) for v in m.values())
    assert set(bi.FIELDS) == {"customers.csv", "anniversaries.csv", "subscriptions_due.csv"}
    assert all(desc and desc.isascii() for cols in bi.FIELDS.values() for desc in cols.values())


@pytest.fixture(scope="module")
def real():
    inp = load_inputs(INPUT)
    c = clean_contracts(inp["fact_contract"])
    o = org_table_with_restored(c, inp["dim_org"], inp["contracts_snapshot_2024-01-05"])
    p = org_periods(o, c)
    return o, c, p, bi.extracts(o, c, p, SNAPSHOT)


@needs_inputs
def test_extracts_relate_on_org_key(real):
    o, c, p, x = real
    cu, an, du = x["customers.csv"], x["anniversaries.csv"], x["subscriptions_due.csv"]
    for name, t in x.items():
        assert list(t.columns) == list(bi.FIELDS[name])
    assert cu["org_key"].is_unique and len(cu) == len(o)
    assert not an.duplicated(["org_key", "anniversary"]).any() and len(an) == len(p)
    assert du["due_id"].tolist() == list(range(1, len(du) + 1)) and du["expiration_date"].is_monotonic_increasing
    assert an["org_key"].isin(cu["org_key"]).all() and du["org_key"].isin(cu["org_key"]).all()
    assert cu["lifecycle"].notna().all() and cu["state_group"].notna().all() and cu["channel"].notna().all() and cu["fleet_size_at_signing"].notna().all()
    assert not {"state", "sales_rep_key", "first_make"} & set(cu.columns)  # segments only at the level the report publishes


@needs_inputs
def test_sixty_day_window_is_the_published_risk_list(real):
    o, c, p, x = real
    cu = x["customers.csv"]
    rl = rm.risk_list(o, c, p, SNAPSHOT).set_index("org_key")
    w = cu[cu["days_to_expiration"].between(1, 60)].set_index("org_key")
    assert set(w.index) == set(rl.index)
    assert (w["risk_tier"] == rl["tier"].reindex(w.index)).all()
    assert np.allclose(w["p_non_renewal"], rl["p_churn"].reindex(w.index)) and w["active_subscriptions"].sum() == rl["active_slots"].sum()
    assert np.isclose(w["expected_lost_subscriptions"].sum(), rl["expected_churn_slots"].sum(), atol=0.01)
    # scored = every active account with a term still to end; the others have no score and no tier
    scored = cu["p_non_renewal"].notna()
    assert (cu.loc[scored, "days_to_expiration"] > 0).all() and (cu.loc[scored, "lifecycle"] == "Active").all()
    assert cu.loc[~scored, "risk_tier"].isna().all() and cu.loc[scored, "risk_tier"].isin(["A", "B", "C"]).all()
    overdue = cu[cu["days_to_expiration"] <= 0]
    assert len(overdue) > 0 and overdue["p_non_renewal"].isna().all()  # terms already ended on rows still marked active


@needs_inputs
def test_due_rows_add_up_to_the_renewal_calendar(real):
    o, c, p, x = real
    du = x["subscriptions_due.csv"]
    cal = demand.renewal_calendar(o, c, p, SNAPSHOT)
    assert len(du) == cal["slots_due"].sum() and du["org_key"].nunique() == cal.attrs["customers"]
    month = du["expiration_date"].dt.to_period("M").dt.to_timestamp()
    g = du.groupby(month).agg(n=("due_id", "size"), recent=("p_non_renewal_recent", "sum"), peak=("p_non_renewal_peak", "sum"))
    assert (g["n"].values == cal["slots_due"].values).all()
    assert np.allclose(g["recent"].values, cal["expected_non_renewals_recent"].values, atol=0.02) and np.allclose(g["peak"].values, cal["expected_non_renewals_peak"].values, atol=0.02)
    assert (du["p_non_renewal_peak"] >= du["p_non_renewal_recent"]).all()  # the two scenarios differ by the year level only, and 2025 is the higher one
    assert du["days_to_expiration"].between(1, 366).all()


@needs_inputs
def test_rates_from_the_extract_are_the_report_rates(real):
    o, c, p, x = real
    an = x["anniversaries.csv"]
    by_year = survival.hazard_by_year(p).set_index("calendar_year")["churn_rate"]
    got = an.groupby("anniversary_year")["not_renewed"].mean()
    assert np.allclose(got.reindex(by_year.index).values, by_year.values)
    base = p[(p["calendar_year"] >= 2023) & p["at_risk_at_decision"]]
    sel = an[(an["anniversary_year"] >= 2023) & (an["still_customer_at_decision"] == "Yes")]
    assert len(sel) == len(base)
    assert np.isclose(sel.loc[sel["fleet_size_at_decision"] == "1 vehicle", "not_renewed"].mean(), base.loc[base["fleet_at_decision_band"] == "1", "churn"].mean())
    assert np.isclose(sel.loc[sel["returned_device_in_term"] == "Yes", "not_renewed"].mean(), base.loc[base["returned_in_term"], "churn"].mean())


@needs_inputs
def test_committed_extracts_are_current(real, tmp_path):
    o, c, p, _ = real
    bi.write_extracts(o, c, p, SNAPSHOT, tmp_path)
    for name in [*bi.FIELDS, "FIELDS.md"]:
        assert (tmp_path / name).read_bytes() == (PUBLISHED / name).read_bytes(), name
        assert (tmp_path / name).read_bytes().isascii() and b"\r" not in (tmp_path / name).read_bytes()
