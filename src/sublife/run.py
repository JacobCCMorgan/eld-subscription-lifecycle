"""End-to-end pipeline: inputs -> tables -> models -> figures -> report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis, calls, demand, lineage, report, survival, validate
from . import renewal_model as rm
from .data import SNAPSHOT, clean_contracts, load_inputs
from .orgs import org_periods, org_table_with_restored


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    return str(o)


def _csv(df: pd.DataFrame, path: Path):
    """Published tables carry six decimals, not float noise."""
    df.round(6).to_csv(path, index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Subscription lifecycle analysis.")
    ap.add_argument("--input", type=Path, default=Path("data/input"))
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--bootstrap", type=int, default=200, help="cluster-bootstrap draws for the hazard odds ratios")
    a = ap.parse_args(argv)
    out, tab, fig = a.out, a.out / "tables", a.out / "figures"
    tab.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)

    inp = load_inputs(a.input)
    c = clean_contracts(inp["fact_contract"])
    snap = inp["contracts_snapshot_2024-01-05"]
    o = org_table_with_restored(c, inp["dim_org"], snap)
    p = org_periods(o, c)
    o.to_parquet(tab / "organizations.parquet", index=False)
    p.to_parquet(tab / "anniversaries.parquet", index=False)

    # 1. validation against the 2024 export
    dev = validate.device_level(c, snap)
    vsum, realized = validate.org_level(o, snap)
    cov = validate.cohort_coverage(o, snap)
    _csv(realized, tab / "validation_realized_2024_outcomes.csv")
    _csv(cov, tab / "validation_cohort_coverage.csv")

    # 2. survival
    km_all = survival.km_table(o)
    km_seg = pd.concat([survival.km_table(o, by) for by in ["start_year", "fleet_band", "state_group", "channel", "email_is_free_domain"]], ignore_index=True)
    pd.concat([km_all, km_seg]).to_csv(tab / "km_survival.csv", index=False)
    hr, cox_meta = survival.cox_model(o)
    _csv(hr, tab / "cox_hazard_ratios.csv")
    by_year = survival.hazard_by_year(p)
    by_k = survival.hazard_by_k(p)
    by_k24 = survival.hazard_by_k(p, year_min=2024)
    _csv(by_year, tab / "churn_by_calendar_year.csv")
    _csv(by_k, tab / "churn_by_anniversary.csv")
    k_rates = by_k24[by_k24["calendar_year"] == "all"].set_index("k")["churn_rate"].to_dict()
    odds = rm.hazard_bootstrap_odds(p, n_boot=a.bootstrap)
    _csv(odds, tab / "hazard_odds_ratios.csv")

    # 3. who leaves and how
    exit_t = analysis.exit_mode_table(o)
    _csv(exit_t, tab / "exit_mode.csv")
    atrisk = p[(p["calendar_year"] >= 2023) & p["at_risk_at_decision"]]
    seg_rows = []
    for by in ["fleet_at_decision_band", "fleet_band", "state_group", "channel", "email_is_free_domain", "k", "start_year"]:
        t = analysis.rate_table(atrisk, by, outcome="churn").rename(columns={by: "value", "rate": "churn_rate"})
        t.insert(0, "segment", by)
        t["value"] = t["value"].astype(str)
        seg_rows.append(t)
    segments = pd.concat(seg_rows, ignore_index=True)
    _csv(segments, tab / "churn_at_decision_point_by_segment.csv")
    first_all = p[(p["k"] == 1) & (p["calendar_year"] >= 2023)]
    midterm_share = float(1 - first_all.loc[first_all["churn"] == 1, "at_risk_at_decision"].mean())

    # 4. models
    x = rm.first_anniversary_frame(p, o)
    met, cal, imp = rm.first_anniversary_models(x)
    _csv(met, tab / "model_metrics_first_anniversary.csv")
    _csv(cal, tab / "model_calibration_first_anniversary.csv")
    _csv(imp, tab / "model_features_first_anniversary.csv")
    hold, hold_cal = rm.hazard_temporal_holdout(p)
    _csv(hold_cal, tab / "hazard_holdout_2026_calibration.csv")
    rl = rm.risk_list(o, c, p, SNAPSHOT)
    _csv(rl, out / "renewal_risk_list.csv")

    # 5. demand
    rcal = demand.renewal_calendar(o, c, p, SNAPSHOT)
    _csv(rcal, tab / "renewal_calendar_12m.csv")
    sign = demand.signings_per_month(o, end=SNAPSHOT.to_period("M").to_timestamp())
    bt = demand.backtest(sign)
    fc, best = demand.forecast(sign, bt)
    flat_12m = float(demand.METHODS["mean_last_12"](sign["new_customers"].to_numpy(dtype=float), 12).sum())
    _csv(sign, tab / "signings_per_month.csv")
    _csv(bt, tab / "signings_backtest.csv")
    _csv(fc, tab / "signings_forecast_12m.csv")

    # 6. hardware and calls
    mix = lineage.generation_mix(c, snap, inp["dim_device"])
    trans, tmeta = lineage.transitions(c, inp["dim_device"])
    pend = lineage.pending_migration(c, o)
    rets = lineage.returns_per_month(c, o)
    _csv(mix, tab / "generation_mix.csv")
    _csv(trans, tab / "generation_transitions.csv")
    _csv(pend, tab / "pending_migration.csv")
    _csv(rets, tab / "device_returns_per_month.csv")
    fun = calls.funnel(inp["migration_calls"])
    mixr = calls.result_mix(inp["migration_calls"])
    outc = calls.outcomes_2025_rounds(inp["migration_calls"], o)
    rep = calls.repeat_contacts(inp["migration_calls"])
    _csv(fun, tab / "calls_funnel.csv")
    _csv(mixr, tab / "calls_result_mix.csv")
    _csv(outc, tab / "calls_outcomes_2025_rounds.csv")
    _csv(rep, tab / "calls_reach_by_attempt.csv")

    # figures
    report.fig_hazard(by_year, by_k, realized, fig)
    report.fig_km(survival.km_curves(o, "start_year"), survival.km_curves(o, "fleet_band"), fig)
    report.fig_validation(cov, dev, fig)
    report.fig_exit_and_signals(exit_t, exit_t.attrs["early_return_days_into_term"], odds, fig)
    report.fig_models(met, cal, hold, hold_cal, fig)
    report.fig_demand(rcal, sign, fc, best, fig)
    report.fig_migration(mix, fun, outc, fig)

    r2 = inp["migration_calls"]
    r2 = r2[(r2["round"] == 2) & r2["contacted"]]
    rets26 = rets[(rets["month"] >= "2026-01-01") & (rets["month"] <= "2026-07-01")]
    legacy_pend = pend[pend["segment"] == "eld_type"]
    ctx = {
        "summary": {"snapshot_date": str(SNAPSHOT.date()), "contract_rows": int(len(c)), "organizations": int((o["source"] == "contract_table").sum()), "restored_from_2024": int((o["source"] != "contract_table").sum()),
                    "snapshot_2024_rows": int((~snap["is_internal_test"]).sum()), "call_rows": int(len(inp["migration_calls"])), "calls_made": int(inp["migration_calls"]["contacted"].sum()),
                    "gap_whole_years": float(c.loc[c["renewals_so_far"].notna(), "gap_is_whole_years"].mean()), "risk_horizon_end": str((SNAPSHOT + pd.Timedelta(60, "D")).date())},
        "validation": vsum, "device_validation": dev, "cohort_coverage": cov, "km_all": km_all, "odds": odds, "model_metrics": met, "hazard_holdout": hold, "renewal_calendar": rcal,
        "funnel": fun, "call_outcomes": outc, "hazard_by_year": by_year, "realized_2024": realized, "risk_list": rl, "generation_mix": mix, "transition_meta": tmeta, "segments_first_anniversary": segments,
        "k_rates": k_rates, "midterm_share_first_year": midterm_share, "signings_flat_12m": flat_12m, "exit_mode": exit_t, "early_return_days": exit_t.attrs["early_return_days_into_term"], "cox_meta": cox_meta,
        "signings_forecast": fc, "signings_method": best, "signings_recent_mean": float(sign["new_customers"].tail(12).mean()), "backtest_origins": int(bt["n_origins"].max()),
        "backtest_best_mae": float(bt.groupby("method")["mae"].mean().min()), "backtest_worst_mae": float(bt.groupby("method")["mae"].mean().max()),
        "pending_legacy_slots": int(legacy_pend["legacy_slots"].sum()), "pending_legacy_customers": int(c[c["is_active_now"] & c["eld_type"].isin(lineage.LEGACY_TYPES)]["org_key"].nunique()),
        "returns_recent_mean": float(rets26["total"].mean()), "returns_exit_share": float(rets26["exit"].sum() / max(rets26["total"].sum(), 1)),
        "round2_no_answer": float(r2["result_code"].eq("no_answer_text_sent").mean()), "reach_by_attempt": rep["reach_rate"].tolist(),
    }
    report.write_report(out / "subscription_lifecycle_report.md", ctx)
    summary = {k: v for k, v in ctx["summary"].items()}
    summary.update({"validation": vsum, "device_validation": dev, "k_rates_2024_2026": k_rates, "churn_by_year": by_year.set_index("calendar_year")["churn_rate"].to_dict(),
                    "cox": cox_meta, "hazard_odds": odds.set_index("covariate")["odds_ratio"].round(3).to_dict(), "model_metrics": met.set_index("model")[["roc_auc", "pr_auc", "brier", "recall_top20"]].round(3).to_dict(orient="index"),
                    "hazard_holdout_2026": hold, "risk_list": {"customers": int(len(rl)), "slots": int(rl["active_slots"].sum()), "expected_churn_slots": float(rl["expected_churn_slots"].sum())},
                    "renewal_calendar_12m": rcal[["slots_due", "customers_due", "expected_non_renewals_recent", "expected_non_renewals_peak"]].sum().to_dict(),
                    "signings": {"method": best, "next_12m": float(fc["forecast"].sum()), "lo": float(fc["lo"].sum()), "hi": float(fc["hi"].sum())},
                    "generation_mix_2026": mix.set_index("generation")["share_2026_09"].round(3).to_dict(), "transitions": tmeta,
                    "calls": {"funnel": fun.to_dict(orient="records"), "outcomes_2025": outc[["result_in_2025", "customers", "migrated_share", "left_share"]].round(3).to_dict(orient="records")}})
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"organizations": summary["organizations"], "anniversaries": int(len(p)), "churn_by_year": {k: round(v, 3) for k, v in summary["churn_by_year"].items()}, "risk_list": summary["risk_list"]}, default=_json_default))


if __name__ == "__main__":
    main()
