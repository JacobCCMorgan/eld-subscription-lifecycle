"""Unit tests on hand-made contract rows, plus invariants on the published inputs."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sublife import analysis, demand, validate
from sublife.data import SNAPSHOT, clean_contracts, load_inputs
from sublife.orgs import DECISION_LEAD_DAYS, RENEW_MARGIN_DAYS, org_periods, org_table
from sublife.renewal_model import FEATURES_AT_DECISION, FEATURES_AT_SIGNING, first_anniversary_frame
from sublife.survival import DiscreteHazard

T = pd.Timestamp


def _row(org, contract, activation, expiration, status="active", device=None, prev=None, eld="rtha55", ret=None, vin=None, rep="REP-a", limit=1.0):
    return {"contract_row_id": 0, "org_key": org, "is_internal_test": False, "dot_key": None, "vin_key": vin or f"VIN-{org}{contract}", "eld_type": eld, "device_key": device, "prev_device_key": prev,
            "make": "Freightliner", "model_year": 2020.0, "status": status, "contract_date": T(contract) if contract else pd.NaT, "activation_date": T(activation) if activation else pd.NaT,
            "expiration_date": T(expiration) if expiration else pd.NaT, "return_date": T(ret) if ret else pd.NaT, "vehicle_limit": limit, "sales_rep_key": rep}


@pytest.fixture
def toy():
    rows = [
        # A: signed 2022-01-01, renewed twice (activation rolled to 2024-01-01), term runs to 2025-01-01, lapsed since -> churn at the 3rd anniversary
        _row("ORG-A", "2022-01-01", "2024-01-01", "2025-01-01", status="inactive", device="DEV-A1"),
        # B: signed 2023-03-01, active, renewed once, current term to 2027-03-01 -> censored, renewed at k=1, k=2
        _row("ORG-B", "2023-03-01", "2025-03-01", "2026-03-01", status="deleted", device="DEV-B1", ret="2025-11-15"),
        _row("ORG-B", "2023-03-01", "2026-03-01", "2027-03-01", status="active", device="DEV-B2", prev="DEV-B1"),
        # C: two vehicles at start, one returned mid-term, the other let to expire -> churned, early return at the first anniversary
        _row("ORG-C", "2024-05-01", "2024-05-01", "2025-05-01", status="deleted", device="DEV-C1", ret="2024-11-01"),
        _row("ORG-C", "2024-05-01", "2024-05-01", "2025-05-01", status="deleted", device="DEV-C2"),
        # D: placeholder expiration and a return before activation (both cleaned), active
        _row("ORG-D", "2025-02-01", "2025-02-01", "2099-01-01", status="active", device="DEV-D1", ret="2024-12-01"),
        # E: internal test organization, must be dropped
        {**_row("ORG-E", "2021-01-01", "2021-01-01", "2022-01-01"), "is_internal_test": True},
        # F: undated legacy row
        _row("ORG-F", None, None, None, status="deleted"),
    ]
    c = pd.DataFrame(rows)
    c["contract_row_id"] = np.arange(len(c))
    dims = pd.DataFrame({"org_key": ["ORG-A", "ORG-B", "ORG-C", "ORG-D", "ORG-F"], "state": ["CA", "TX", "CA", "CA", None], "has_dot": [True] * 5, "email_is_free_domain": [True, False, True, True, None],
                         "is_reseller": [False] * 5, "crm_active": ["Y"] * 5, "sales_rep_key": ["REP-a", "REP-b", "REP-a", "REP-a", None]})
    return c, dims


def test_cleaning_derives_renewals_and_row_end(toy):
    c, _ = toy
    cc = clean_contracts(c)
    assert "ORG-E" not in set(cc["org_key"])
    a = cc[cc["org_key"] == "ORG-A"].iloc[0]
    assert a["renewals_so_far"] == 2 and a["gap_is_whole_years"]
    d = cc[cc["org_key"] == "ORG-D"].iloc[0]
    assert pd.isna(d["expiration_date"]) and d["flag_far_future_expiration"]
    assert pd.isna(d["return_date"]) and d["flag_return_before_activation"]
    c1 = cc[cc["device_key"] == "DEV-C1"].iloc[0]
    assert c1["row_end"] == T("2024-11-01")  # early return ends the row's coverage
    c2 = cc[cc["device_key"] == "DEV-C2"].iloc[0]
    assert c2["row_end"] == T("2025-05-01")
    assert cc[cc["org_key"] == "ORG-F"]["flag_missing_dates"].all()


def test_org_table_lifecycle(toy):
    c, dims = toy
    o = org_table(clean_contracts(c), dims).set_index("org_key")
    assert set(o.index) == {"ORG-A", "ORG-B", "ORG-C", "ORG-D"}  # the undated organization has no row
    assert o.loc["ORG-A", "churned"] == 1 and o.loc["ORG-A", "exit_mode"] == "non_renewal" and o.loc["ORG-A", "end_date"] == T("2025-01-01")
    assert o.loc["ORG-A", "tenure_days"] == (T("2025-01-01") - T("2022-01-01")).days
    assert o.loc["ORG-B", "churned"] == 0 and o.loc["ORG-B", "lifecycle"] == "active" and o.loc["ORG-B", "n_replacements"] == 1
    assert o.loc["ORG-B", "renewals_max"] == 3
    assert o.loc["ORG-C", "fleet_at_start"] == 2 and o.loc["ORG-C", "churned"] == 1 and o.loc["ORG-C", "exit_mode"] == "non_renewal"
    assert o.loc["ORG-D", "lifecycle"] == "active"
    assert o.loc["ORG-B", "state_group"] == "other" and o.loc["ORG-A", "state_group"] == "CA"


def test_anniversary_periods(toy):
    c, dims = toy
    cc = clean_contracts(c)
    o = org_table(cc, dims)
    p = org_periods(o, cc).set_index(["org_key", "k"])
    # A: renewed at k=1 (2023-01-01) and k=2 (2024-01-01), churned at k=3 (2025-01-01)
    assert p.loc[("ORG-A", 1), "renewed"] == 1 and p.loc[("ORG-A", 2), "renewed"] == 1 and p.loc[("ORG-A", 3), "churn"] == 1
    assert ("ORG-A", 4) not in p.index
    assert p.loc[("ORG-A", 3), "at_risk_at_decision"]  # still covered 60 days before the 3rd anniversary
    # B: renewed at k=1 and k=2; k=3 (2026-03-01) renewed because the active term runs past it; k=4 too recent
    assert p.loc[("ORG-B", 1), "renewed"] == 1 and p.loc[("ORG-B", 2), "renewed"] == 1 and p.loc[("ORG-B", 3), "renewed"] == 1
    assert ("ORG-B", 4) not in p.index
    assert p.loc[("ORG-B", 3), "returned_in_term"]  # DEV-B1 came back inside the term before the decision point
    # C: churned at the first anniversary; one of two vehicles was gone by the decision point
    assert p.loc[("ORG-C", 1), "churn"] == 1 and p.loc[("ORG-C", 1), "fleet_at_decision"] == 1 and p.loc[("ORG-C", 1), "fleet_shrank_by_decision"]
    assert p["decision_date"].eq(p["anniversary"] - pd.Timedelta(DECISION_LEAD_DAYS, "D")).all()


def test_margin_and_grace_are_applied(toy):
    c, dims = toy
    cc = clean_contracts(c)
    o = org_table(cc, dims)
    p = org_periods(o, cc, margin_days=RENEW_MARGIN_DAYS)
    # an anniversary whose margin has not elapsed by the snapshot must not appear unless the outcome is already known
    assert (p["anniversary"] + pd.Timedelta(RENEW_MARGIN_DAYS, "D") <= SNAPSHOT).all() | (p["renewed"] == 1).all() | (p["churn"] == 1).all()


def test_feature_lists_do_not_leak_the_outcome(toy):
    c, dims = toy
    cc = clean_contracts(c)
    o = org_table(cc, dims)
    p = org_periods(o, cc)
    x = first_anniversary_frame(p, o)
    for f in FEATURES_AT_SIGNING + FEATURES_AT_DECISION:
        assert f in x.columns
    assert not {"churn", "renewed", "coverage_end", "end_date", "lifecycle"} & set(FEATURES_AT_SIGNING + FEATURES_AT_DECISION)
    # decision-point features are measured 60 days before the anniversary, never after
    assert "fleet_at_k" not in FEATURES_AT_DECISION


def test_discrete_hazard_fits_and_scores():
    rng = np.random.default_rng(0)
    n = 600
    p = pd.DataFrame({
        "org_key": [f"ORG-{i}" for i in range(n)], "k": rng.integers(1, 6, n), "calendar_year": rng.choice([2023, 2024, 2025], n), "fleet_at_decision": rng.integers(1, 6, n),
        "fleet_at_start": rng.integers(1, 6, n), "state_group": rng.choice(["CA", "other"], n), "channel": rng.choice(["main_channel", "other_rep"], n), "email_is_free_domain": rng.choice([True, False], n),
        "returned_in_term": rng.choice([True, False], n, p=[0.1, 0.9]), "at_risk_at_decision": True,
    })
    p["fleet_shrank_by_decision"] = p["fleet_at_decision"] < p["fleet_at_start"]
    logit = -1.5 + 0.8 * p["returned_in_term"] - 0.4 * np.log1p(p["fleet_at_decision"])
    p["churn"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    dh = DiscreteHazard().fit(p)
    s = dh.predict(p)
    assert s.shape == (n,) and (0 <= s).all() and (s <= 1).all()
    ors = dh.odds_ratios().set_index("covariate")["odds_ratio"]
    assert ors["returned_in_term"] > 1 and ors["log_fleet_at_decision"] < 1


def test_wilson_and_suppression():
    p, lo, hi = analysis.wilson(10, 100)
    assert abs(p - 0.1) < 1e-9 and lo < 0.1 < hi and lo > 0.04 and hi < 0.18
    assert analysis.suppress(3) == "<5" and analysis.suppress(5) == "5"


def test_forecast_methods_and_backtest():
    months = pd.date_range("2021-01-01", periods=48, freq="MS")
    y = 30 + 5 * np.sin(np.arange(48) * 2 * np.pi / 12) + np.arange(48) * 0.1
    s = pd.DataFrame({"month": months, "new_customers": np.round(y)})
    bt = demand.backtest(s, horizon=3, origins=6)
    assert set(bt["method"]) == set(demand.METHODS) and bt["mae"].notna().all()
    fc, best = demand.forecast(s, bt, months=12)
    assert len(fc) == 12 and (fc["lo"] <= fc["forecast"]).all() and (fc["forecast"] <= fc["hi"]).all() and best in demand.METHODS
    assert np.allclose(demand._seasonal_naive(np.arange(24.0), 3), [12, 13, 14])


def test_device_level_validation_on_toy():
    cur = pd.DataFrame({"device_key": ["DEV-1", "DEV-2", "DEV-3"], "activation_date": [T("2025-01-01"), T("2024-01-01"), T("2024-06-15")], "contract_date": [T("2023-01-01")] * 3, "status": ["active"] * 3, "is_internal_test": False})
    snap = pd.DataFrame({"device_key": ["DEV-1", "DEV-2", "DEV-3"], "activation_date": [T("2024-01-01"), T("2024-01-01"), T("2024-01-01")], "contract_date": [T("2023-01-01")] * 3, "is_internal_test": False})
    d = validate.device_level(cur, snap)
    assert d["devices_found_in_2026_table"] == 3
    assert abs(d["activation_moved_one_year"] - 1 / 3) < 1e-9 and abs(d["activation_unchanged"] - 1 / 3) < 1e-9 and abs(d["activation_moved_other"] - 1 / 3) < 1e-9


# ---- invariants on the published inputs ------------------------------------------------------------------

INPUT = Path(__file__).resolve().parents[1] / "data" / "input"
pytestmark_inputs = pytest.mark.skipif(not INPUT.exists(), reason="published inputs not present")


@pytestmark_inputs
def test_published_inputs_are_pseudonymized_and_english():
    ident = re.compile(r"\bRT\d{8}\b|@|\d{3}[-) ]\d{3}-\d{4}|\b\d{15}\b|[" + chr(0x4E00) + "-" + chr(0x9FFF) + "]")
    for f in sorted(INPUT.glob("*.parquet")):
        df = pd.read_parquet(f)
        for col in df.select_dtypes("object"):
            v = df[col].dropna().astype(str)
            assert not v.str.contains(ident).any(), f"{f.name}:{col}"
    c = pd.read_parquet(INPUT / "fact_contract.parquet")
    for col, prefix in [("org_key", "ORG-"), ("device_key", "DEV-"), ("prev_device_key", "DEV-"), ("vin_key", "VIN-"), ("sales_rep_key", "REP-"), ("dot_key", "DOT-")]:
        v = c[col].dropna()
        assert v.str.startswith(prefix).all(), col
    calls = pd.read_parquet(INPUT / "migration_calls.parquet")
    allowed = {"no_answer_text_sent", "agreed_walk_in", "agreed_shipping", "agreed_exchange", "already_on_new_device", "account_closed", "wants_to_cancel", "suspended", "unpaid", "callback_later", "emailed", "language_barrier", "invalid_number", "small_vehicle", "device_lost", "other"}
    assert set(calls["result_code"].dropna()) <= allowed
    assert set(calls["round"].unique()) == {1, 2, 3, 4, 5}


@pytestmark_inputs
def test_published_tables_are_consistent():
    inp = load_inputs(INPUT)
    c = clean_contracts(inp["fact_contract"])
    assert len(c) > 13000 and c["is_internal_test"].sum() == 0
    assert c.loc[c["renewals_so_far"].notna(), "gap_is_whole_years"].mean() > 0.9  # the roll-forward signature
    snap = inp["contracts_snapshot_2024-01-05"]
    assert (snap["snapshot_date"] == T("2024-01-05")).all()
    assert snap["org_key"].isin(inp["dim_org"]["org_key"]).mean() > 0.95
