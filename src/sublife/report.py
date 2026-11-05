"""Figures and the Markdown report."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9", "grey": "#7F7F7F", "yellow": "#F0E442"}
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold"})


def _save(fig, out: Path, name: str):
    fig.tight_layout()
    fig.savefig(out / name)
    plt.close(fig)


def fig_hazard(by_year: pd.DataFrame, by_k: pd.DataFrame, realized: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    y = by_year[by_year["calendar_year"].between(2021, 2026)]
    ax[0].bar(y["calendar_year"].astype(str), y["churn_rate"], color=[C["grey"] if v < 2023 else C["blue"] for v in y["calendar_year"]])
    ax[0].errorbar(y["calendar_year"].astype(str), y["churn_rate"], yerr=[y["churn_rate"] - y["lo"], y["hi"] - y["churn_rate"]], fmt="none", ecolor="black", capsize=3, lw=1)
    r = realized[realized["customers"] >= 50]
    ax[0].scatter(["2024"] * len(r), r["churn_rate"], color=C["orange"], zorder=5, s=28, label="realized 2024 outcome of each signing cohort\nin the Jan-2024 export (purged = churned)")
    ax[0].set_ylabel("share not renewing at their anniversary")
    ax[0].set_title("Churn at anniversaries by calendar year (grey: under-recorded, rows purged)")
    ax[0].set_ylim(0, 0.42)
    ax[0].legend(fontsize=8, loc="upper left", frameon=False)
    for v in ax[0].patches:
        if v.get_height() > 0.05:
            ax[0].annotate(f"{v.get_height():.0%}", (v.get_x() + v.get_width() / 2, v.get_height() / 2), ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
        else:
            ax[0].annotate(f"{v.get_height():.0%}", (v.get_x() + v.get_width() / 2, v.get_height() + 0.025), ha="center", fontsize=8)
    k = by_k[by_k["calendar_year"] != "all"].copy()
    k["calendar_year"] = k["calendar_year"].astype(int)
    for yr, col in zip([2024, 2025, 2026], [C["sky"], C["red"], C["blue"]]):
        d = k[(k["calendar_year"] == yr) & (k["k"] <= 6)]
        ax[1].plot(d["k"], d["churn_rate"], marker="o", color=col, label=f"anniversaries in {yr}")
        ax[1].fill_between(d["k"], d["lo"], d["hi"], color=col, alpha=0.12)
    ax[1].set_xlabel("anniversary number (1 = first renewal)")
    ax[1].set_ylabel("share not renewing")
    ax[1].set_title("Within a year, tenure barely matters")
    ax[1].set_ylim(0, 0.5)
    ax[1].legend(fontsize=8, frameon=False)
    _save(fig, out, "01_churn_by_year_and_anniversary.png")


def fig_km(curves_cohort: dict, curves_fleet: dict, out: Path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    cols = [C["grey"], C["sky"], C["blue"], C["green"], C["orange"], C["red"]]
    for (name, sf), col in zip([(n, s) for n, s in curves_cohort.items() if n in {"2020", "2021", "2022", "2023", "2024", "2025"}], cols):
        ax[0].step(sf["days"] / 365.25, sf["S"], where="post", color=col, label=f"signed {name}")
    ax[0].set_xlabel("years since signing")
    ax[0].set_ylabel("share still a customer")
    ax[0].set_title("Kaplan-Meier by signing cohort (2020-21 flattered by purged rows)")
    ax[0].set_ylim(0, 1.02)
    ax[0].legend(fontsize=8, frameon=False)
    for (name, sf), col in zip(curves_fleet.items(), [C["blue"], C["green"], C["orange"], C["red"]]):
        ax[1].step(sf["days"] / 365.25, sf["S"], where="post", color=col, label=f"{name} vehicle(s) at start")
    ax[1].set_xlabel("years since signing")
    ax[1].set_title("By fleet size at signing (all cohorts)")
    ax[1].set_ylim(0, 1.02)
    ax[1].legend(fontsize=8, frameon=False)
    _save(fig, out, "02_survival_curves.png")


def fig_validation(cov: pd.DataFrame, dev: dict, out: Path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    c = cov[cov["signing_year"].between(2019, 2023)]
    x = np.arange(len(c))
    ax[0].bar(x - 0.2, c["active_in_2024_export"], width=0.4, color=C["orange"], label="active customers in the Jan-2024 export")
    ax[0].bar(x + 0.2, c["in_2026_table_any_status"], width=0.4, color=C["blue"], label="customers of that cohort in the Sep-2026 table (any status)")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(c["signing_year"].astype(int))
    ax[0].set_xlabel("signing year")
    ax[0].set_ylabel("customers")
    ax[0].set_title("The 2023 cohort: fewer rows today than active customers in 2024")
    ax[0].set_ylim(0, c[["active_in_2024_export", "in_2026_table_any_status"]].max().max() * 1.3)
    ax[0].legend(fontsize=8, frameon=False, loc="upper left")
    labels = ["unchanged", "moved +1 year", "moved +2 years", "other"]
    vals = [dev["activation_unchanged"], dev["activation_moved_one_year"], dev["activation_moved_two_years"], dev["activation_moved_other"]]
    ax[1].barh(labels, vals, color=[C["grey"], C["blue"], C["blue"], C["sky"]])
    for i, v in enumerate(vals):
        ax[1].text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=9)
    ax[1].set_xlim(0, 0.6)
    ax[1].set_title(f"activation_date of the same device, 2024 vs 2026 export (n={dev['devices_found_in_2026_table']:,})")
    ax[1].invert_yaxis()
    _save(fig, out, "03_export_comparison.png")


def fig_exit_and_signals(exit_t: pd.DataFrame, days_q: dict, ors: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.3]})
    e = exit_t.set_index("exit_mode").reindex(["non_renewal", "early_return", "purged_after_2024"]).fillna(0)
    ax[0].bar(["let the term expire", "returned device(s)\nduring the term", "purged after 2024\n(outcome unknown)"], e["share"], color=[C["blue"], C["red"], C["grey"]])
    for i, v in enumerate(e["share"]):
        ax[0].text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=9)
    ax[0].set_ylim(0, 0.7)
    ax[0].set_title("How churned customers left")
    ax[0].set_xlabel(f"early returns happen at a median of day {days_q['p50']} of the 365-day term", fontsize=8.5)
    o = ors.copy()
    names = {"k_2": "2nd anniversary (vs 1st)", "k_3": "3rd anniversary", "k_4": "4th anniversary", "k_5": "5th anniversary", "year_2024": "anniversary in 2024 (vs 2023)", "year_2025": "anniversary in 2025", "year_2026": "anniversary in 2026",
             "log_fleet_at_decision": "fleet size (per log unit)", "fleet_shrank_by_decision": "fleet smaller than at start", "returned_in_term": "returned a device this term", "outside_ca": "outside California",
             "channel_other_rep": "sold by a named rep", "channel_unknown": "channel unknown", "free_email_domain": "free e-mail domain"}
    o["label"] = o["covariate"].map(names).fillna(o["covariate"])
    o = o.iloc[::-1]
    ypos = np.arange(len(o))
    ax[1].errorbar(o["odds_ratio"], ypos, xerr=[o["odds_ratio"] - o["or_lo"], o["or_hi"] - o["odds_ratio"]], fmt="o", color=C["blue"], ecolor=C["grey"], capsize=2)
    ax[1].axvline(1, color="black", lw=0.8)
    ax[1].set_yticks(ypos)
    ax[1].set_yticklabels(o["label"], fontsize=8.5)
    ax[1].set_xscale("log")
    ax[1].set_xticks([0.3, 0.5, 0.7, 1, 1.5, 2, 3])
    ax[1].set_xticklabels(["0.3", "0.5", "0.7", "1", "1.5", "2", "3"])
    ax[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[1].set_xlabel("odds ratio of not renewing (log scale, 95% cluster-bootstrap interval)")
    ax[1].set_title("Discrete-time hazard, scored 60 days before the anniversary")
    _save(fig, out, "04_exit_mode_and_hazard_odds.png")


def fig_models(met: pd.DataFrame, cal: pd.DataFrame, hold: dict, hold_cal: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    m = met.set_index("model")
    labels = {"base_rate": "base rate", "logistic_signing_only": "logistic:\nsigning attributes", "logistic_signing_plus_decision": "logistic:\n+ 60-day signals", "boosting_signing_plus_decision": "boosting:\n+ 60-day signals"}
    x = np.arange(len(m))
    ax[0].bar(x - 0.2, m["recall_top20"], width=0.4, color=C["blue"], label="share of churners in the top 20 % of accounts")
    ax[0].bar(x + 0.2, m["precision_top20"], width=0.4, color=C["orange"], label="churn rate inside that top 20 %")
    ax[0].axhline(0.2, color="black", lw=0.8, ls="--")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([labels[i] for i in m.index], fontsize=8.5)
    ax[0].set_ylim(0, 0.5)
    ax[0].set_title(f"First-anniversary holdout (2024 signings, n={int(m['n'].iloc[0])}): calling the top fifth")
    ax[0].legend(fontsize=8, frameon=False)
    for name, col in [("logistic_signing_plus_decision", C["blue"]), ("boosting_signing_plus_decision", C["orange"])]:
        d = cal[cal["model"] == name]
        ax[1].plot(d["predicted"], d["observed"], marker="o", color=col, label=f"{labels[name].replace(chr(10), ' ')} (2024 signings)")
    ax[1].plot(hold_cal["predicted"], hold_cal["observed"], marker="s", color=C["green"], label=f"hazard model fitted 2023-25, scored on 2026 anniversaries\n(AUC {hold['roc_auc']:.2f}, predicted mean {hold['predicted_mean']:.0%} vs observed {hold['base_rate']:.0%})")
    lim = max(cal["predicted"].max(), hold_cal["predicted"].max(), cal["observed"].max()) + 0.05
    ax[1].plot([0, lim], [0, lim], color="black", lw=0.8, ls="--")
    ax[1].set_xlabel("predicted share not renewing (quintiles)")
    ax[1].set_ylabel("observed")
    ax[1].set_title("Calibration on time-ordered holdouts")
    ax[1].legend(fontsize=7.5, frameon=False)
    _save(fig, out, "05_model_holdouts.png")


def fig_demand(cal: pd.DataFrame, sign: pd.DataFrame, fc: pd.DataFrame, best: str, out: Path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    c = cal[cal["month"] > cal["month"].min()] if len(cal) > 12 else cal
    x = c["month"].dt.strftime("%Y-%m")
    ax[0].bar(x, c["expected_renewals_recent"], color=C["blue"], label="expected renewals")
    ax[0].bar(x, c["expected_non_renewals_recent"], bottom=c["expected_renewals_recent"], color=C["red"], label="expected non-renewals (recent-year rate)")
    ax[0].bar(x, c["expected_non_renewals_peak"] - c["expected_non_renewals_recent"], bottom=c["slots_due"], color=C["red"], alpha=0.3, label="additional at the 2025 peak rate")
    ax[0].set_ylabel("vehicle subscriptions with a term ending")
    ax[0].set_title("Renewal calendar, next 12 months")
    ax[0].tick_params(axis="x", rotation=60, labelsize=8)
    ax[0].legend(fontsize=8, frameon=False)
    s = sign[sign["month"] >= "2023-01-01"]
    ax[1].plot(s["month"], s["new_customers"], color=C["blue"], label="new customers per month (first contract date)")
    ax[1].plot(fc["month"], fc["forecast"], color=C["orange"], label=f"forecast ({best.replace('_', ' ')})")
    ax[1].fill_between(fc["month"], fc["lo"], fc["hi"], color=C["orange"], alpha=0.2, label="+-1.5 x backtest MAE")
    ax[1].set_ylim(0, None)
    ax[1].set_title("New-customer signings")
    ax[1].legend(fontsize=8, frameon=False)
    _save(fig, out, "06_renewal_calendar_and_signings.png")


def fig_migration(mix: pd.DataFrame, fun: pd.DataFrame, outc: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4), gridspec_kw={"width_ratios": [0.9, 1.1, 1.2]})
    m = mix.set_index("generation")
    gens = [g for g in ["rtps47", "rtel47", "rtoa47", "rtps12", "rtha55", "unknown"] if g in m.index]
    x = np.arange(len(gens))
    ax[0].bar(x - 0.2, [m.loc[g, "share_2024_01"] for g in gens], width=0.4, color=C["grey"], label="Jan 2024")
    ax[0].bar(x + 0.2, [m.loc[g, "share_2026_09"] for g in gens], width=0.4, color=C["blue"], label="Sep 2026")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(gens, fontsize=8, rotation=35)
    ax[0].set_ylabel("share of active subscriptions")
    ax[0].set_title("Hardware generation of active subscriptions")
    ax[0].legend(fontsize=8, frameon=False)
    f = fun.set_index("round")
    x = np.arange(len(f))
    ax[1].bar(x - 0.27, f["contacted_rows"], width=0.27, color=C["grey"], label="calls made")
    ax[1].bar(x, f["reached_rows"], width=0.27, color=C["blue"], label="reached a person")
    ax[1].bar(x + 0.27, f["agreed_rows"], width=0.27, color=C["green"], label="agreed to exchange")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([f"round {i}\n{p}" for i, p in zip(f.index, f["campaign_phase"].str.replace(" outreach", ""))], fontsize=8)
    ax[1].set_title("Migration call rounds")
    ax[1].legend(fontsize=8, frameon=False)
    o = outc.set_index("result_in_2025")
    order = [g for g in ["agreed to exchange", "already migrated", "no answer (text sent)", "listed, never contacted", "cancel / closed / on hold"] if g in o.index]
    y = np.arange(len(order))
    ax[2].barh(y - 0.2, [o.loc[g, "migrated_share"] for g in order], height=0.4, color=C["green"], label="migrated by Sep 2026")
    ax[2].barh(y + 0.2, [o.loc[g, "left_share"] for g in order], height=0.4, color=C["red"], label="no longer a customer")
    ax[2].set_yticks(y)
    ax[2].set_yticklabels([f"{g}\n(n={int(o.loc[g, 'customers'])})" for g in order], fontsize=8)
    ax[2].invert_yaxis()
    ax[2].set_xlim(0, 1)
    ax[2].set_title("2025 rounds: state in Sep 2026 by what the customer said")
    ax[2].legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    _save(fig, out, "07_migration_and_calls.png")


def pct(x, d=0):
    return f"{x * 100:.{d}f} %"


def write_report(path: Path, ctx: dict):
    s = ctx["summary"]
    val = ctx["validation"]
    dev = ctx["device_validation"]
    km = ctx["km_all"].iloc[0]
    ors = ctx["odds"].set_index("covariate")
    met = ctx["model_metrics"].set_index("model")
    hold = ctx["hazard_holdout"]
    cal = ctx["renewal_calendar"]
    fun = ctx["funnel"].set_index("round")
    outc = ctx["call_outcomes"].set_index("result_in_2025")
    byyear = ctx["hazard_by_year"].set_index("calendar_year")
    realized = ctx["realized_2024"]
    rl = ctx["risk_list"]
    mix = ctx["generation_mix"].set_index("generation")
    tr = ctx["transition_meta"]
    seg = ctx["segments_first_anniversary"]

    def orl(name):
        r = ors.loc[name]
        return f"{r['odds_ratio']:.2f} ({r['or_lo']:.2f}-{r['or_hi']:.2f})"

    def seg_line(by, a, b):
        d = seg[seg["segment"] == by].set_index("value")
        return f"{pct(d.loc[a, 'churn_rate'])} vs {pct(d.loc[b, 'churn_rate'])}"

    lines = []
    lines.append("# Seven years of one-year ELD subscriptions: churn, the renewal calendar and the migration campaign\n")
    lines.append(f"Data: the vendor's CRM contract list as of {s['snapshot_date']} ({s['contract_rows']:,} vehicle-subscription rows, {s['organizations']:,} customer organizations with usable dates), "
                 f"an export of the same list from 5 January 2024 ({s['snapshot_2024_rows']:,} active rows), and the ELD 2.0 migration call log ({s['call_rows']:,} list entries, {s['calls_made']:,} calls). "
                 "All identifiers are keyed pseudonyms; see `data/input/SOURCE.md`.\n")

    lines.append("## 1. What the dates mean, and what the table forgets\n")
    lines.append(f"- The CRM keeps one row per vehicle slot and **overwrites the term dates at every renewal**: {pct(s['gap_whole_years'])} of the gaps between `contract_date` and `activation_date` are whole multiples of 365 days, "
                 f"and for the {dev['devices_found_in_2026_table']:,} units present in both exports, `activation_date` is unchanged for {pct(dev['activation_unchanged'])}, moved forward by exactly one year for {pct(dev['activation_moved_one_year'])} and by other amounts for {pct(dev['activation_moved_other'])} (device swaps re-date the row). "
                 "`activation_date` is therefore the start of the *current* term, not when the device went live; `contract_date` is the original sale and survives a device swap. Anything computed from `activation_date` as if it were an activation - monthly activation counts, time-to-return - is wrong by construction.")
    lines.append(f"- **The table purges old rows.** Of the {val['orgs_active_2024_01_05']:,} customers active on 5 Jan 2024, {val['purged_from_2026_table']} ({pct(1 - val['present_in_2026_table'], 1)}) have no dated row left in the 2026 table; "
                 f"the 2023 signing cohort has {ctx['cohort_coverage'].set_index('signing_year').loc[2023, 'in_2026_table_any_status']:,} customers in today's table but had {ctx['cohort_coverage'].set_index('signing_year').loc[2023, 'active_in_2024_export']:,} *active* ones in January 2024. "
                 "Purged customers are churned customers, so churn before 2023 is under-recorded; this analysis restores the purged 2024 customers from the older export and treats anniversaries before 2023 as descriptive only.")
    lines.append(f"- Where both exports can be compared the reconstruction holds: {pct(val['reconstructed_as_covered_on_2024_01_05'], 1)} of the 2024 customers are reconstructed as covered on that date and {pct(val['start_date_within_31_days_of_2024_export'], 1)} get the same start date (within a month).\n")
    lines.append("![exports](figures/03_export_comparison.png)\n")

    lines.append("## 2. The lifecycle\n")
    lines.append(f"- Across all cohorts, {pct(km['S_365d'], 1)} of customers are still there after one year, {pct(km['S_730d'], 1)} after two, {pct(km['S_1095d'], 1)} after three and {pct(km['S_1460d'], 1)} after four (Kaplan-Meier; early cohorts flattered by purging). "
                 f"Median tenure is {km['median_tenure_days'] / 365.25:.1f} years.")
    lines.append(f"- **The first anniversary is not a cliff.** Among anniversaries reached in 2024-2026 the share not renewing is {pct(ctx['k_rates'][1])} at the first anniversary, {pct(ctx['k_rates'][2])} at the second, {pct(ctx['k_rates'][3])} at the third and {pct(ctx['k_rates'][4])} at the fourth; the tenure odds ratios in the hazard model are {orl('k_2')} for the second and {orl('k_4')} for the fourth anniversary relative to the first.")
    lines.append(f"- **Calendar time dominates.** The share of anniversaries ending in non-renewal rose from {pct(byyear.loc[2023, 'churn_rate'])} in 2023 to {pct(byyear.loc[2024, 'churn_rate'])} in 2024, {pct(byyear.loc[2025, 'churn_rate'])} in 2025 and {pct(byyear.loc[2026, 'churn_rate'])} in 2026 to date, at every tenure. "
                 f"The January-2024 export confirms the level independently: the customers active then churned at their next anniversary at {pct(realized[realized['customers'] >= 50]['churn_rate'].min())}-{pct(realized[realized['customers'] >= 50]['churn_rate'].max())} depending on signing cohort, purged customers counted as churned. "
                 f"The hazard model's odds ratio for a 2025 anniversary versus 2023 is {orl('year_2025')}.\n")
    lines.append("![hazard](figures/01_churn_by_year_and_anniversary.png)\n")
    lines.append("![km](figures/02_survival_curves.png)\n")

    lines.append("## 3. Who leaves, and how\n")
    e = ctx["exit_mode"].set_index("exit_mode")
    dq = ctx["early_return_days"]
    lines.append(f"- **Fleet size is the strongest attribute.** Among accounts still active 60 days before an anniversary (2023-2026), those running one vehicle did not renew {pct(seg.set_index(['segment', 'value']).loc[('fleet_at_decision_band', '1'), 'churn_rate'])} of the time, two vehicles {pct(seg.set_index(['segment', 'value']).loc[('fleet_at_decision_band', '2'), 'churn_rate'])}, three to five {pct(seg.set_index(['segment', 'value']).loc[('fleet_at_decision_band', '3-5'), 'churn_rate'])} and six or more {pct(seg.set_index(['segment', 'value']).loc[('fleet_at_decision_band', '6+'), 'churn_rate'])}; "
                 f"in the hazard model each log-unit of fleet size lowers the odds of not renewing by a factor of {orl('log_fleet_at_decision')}. Fleet size *at signing* separates much less ({seg_line('fleet_band', '1', '2')} for one- vs two-vehicle starts): what matters is the fleet the customer still runs when the renewal comes up.")
    lines.append(f"- **{pct(ctx['midterm_share_first_year'])} of first-year churn happens before the 60-day decision point**: the customer has already returned the device(s) by the time a renewal desk would look at the account. Early intervention has to key off the return request, not the expiry date.")
    lines.append(f"- {pct(e.loc['early_return', 'share'])} of churned customers **returned their device(s) before the term ran out**, at a median of day {dq['p50']} of the 365-day term (quartiles {dq['p25']}-{dq['p75']}); {pct(e.loc['non_renewal', 'share'])} simply let the term expire. "
                 f"A return during the current term, visible 60 days before the anniversary, multiplies the odds of non-renewal by {orl('returned_in_term')} - it is the one operational signal the CRM already holds.")
    lines.append(f"- A free e-mail domain (owner-operators rather than companies with their own domain) goes with *lower* churn, odds ratio {orl('free_email_domain')}; being outside California {orl('outside_ca')} and the sales channel {orl('channel_other_rep')} add little once fleet size and year are in the model. "
                 f"Cox on signing attributes alone (cohorts {ctx['cox_meta']['cohorts']}, n={ctx['cox_meta']['n']:,}) has a concordance of {ctx['cox_meta']['concordance']:.2f}: what a customer looks like on day one says very little about whether they will stay.\n")
    lines.append("![exit](figures/04_exit_mode_and_hazard_odds.png)\n")

    lines.append("## 4. Can the renewal desk pick the right accounts 60 days out?\n")
    b, l1, l2, g = met.loc["base_rate"], met.loc["logistic_signing_only"], met.loc["logistic_signing_plus_decision"], met.loc["boosting_signing_plus_decision"]
    lines.append(f"- First-anniversary holdout - trained on 2021-2023 signings, scored on the {int(l2['n'])} customers who signed in 2024 (non-renewal rate {pct(b['base_rate'])}): ROC-AUC {l1['roc_auc']:.2f} with signing attributes only, {l2['roc_auc']:.2f} ({l2['roc_auc_lo']:.2f}-{l2['roc_auc_hi']:.2f}) once the 60-day signals (device returned, fleet shrank, fleet size now) are added; gradient boosting {g['roc_auc']:.2f}. "
                 f"Calling the top 20 % of accounts reaches {pct(l2['recall_top20'])} of the churners (random: 20 %) with a churn rate of {pct(l2['precision_top20'])} inside the list.")
    lines.append(f"- All-anniversary hazard model fitted on 2023-2025 and scored on the {hold['n']:,} anniversaries of 2026: ROC-AUC {hold['roc_auc']:.2f} ({hold['roc_auc_lo']:.2f}-{hold['roc_auc_hi']:.2f}), top 20 % reaches {pct(hold['recall_top20'])} of churners. "
                 f"Scored at the 2025 level it predicts {pct(hold['predicted_mean'])} non-renewal against {pct(hold['base_rate'])} observed: the *level* moves from year to year by more than any customer attribute, so the model is a ranking tool, not a forecast of the rate.")
    lines.append("- Verdict: the CRM alone yields a modest ranking (about 1.5x lift at the top), carried by two things - fleet size and a device return during the term. Usage data (the telemetry the foundation tables hold) is where a better signal would have to come from.\n")
    lines.append("![models](figures/05_model_holdouts.png)\n")

    lines.append("## 5. The next 60 days and the next 12 months\n")
    lines.append(f"- **Risk list (`reports/renewal_risk_list.csv`)**: {len(rl):,} customers with {int(rl['active_slots'].sum()):,} subscriptions reach an anniversary between {s['snapshot_date']} and {s['risk_horizon_end']}. "
                 f"Scored with the hazard model, {int(rl['expected_churn_slots'].sum())} subscriptions are expected not to renew; the {int((rl['tier'] == 'A').sum())} tier-A accounts (top fifth by probability) carry {pct(rl.loc[rl['tier'] == 'A', 'expected_churn_slots'].sum() / rl['expected_churn_slots'].sum())} of that expected loss, and {int(rl['returned_device_last_12m'].sum())} of them returned a device in the last twelve months.")
    tot = cal[["slots_due", "customers_due", "expected_non_renewals_recent", "expected_non_renewals_peak"]].sum()
    peak_m = cal.loc[cal["slots_due"].idxmax()]
    lines.append(f"- **Renewal calendar**: {int(tot['slots_due']):,} subscriptions from {int(tot['customers_due']):,} customers come up for renewal in the twelve months after the export; the model expects {int(tot['expected_non_renewals_recent']):,} non-renewals at the recent-year level and {int(tot['expected_non_renewals_peak']):,} at the 2025 peak level ({pct(tot['expected_non_renewals_recent'] / tot['slots_due'])}-{pct(tot['expected_non_renewals_peak'] / tot['slots_due'])}). "
                 f"The load is seasonal: {int(peak_m['slots_due'])} terms end in {peak_m['month'].strftime('%B %Y')} against {int(cal['slots_due'].iloc[1:].min())} in the quietest month, because signings cluster in spring and summer.")
    fc = ctx["signings_forecast"]
    lines.append(f"- **New customers**: {ctx['signings_recent_mean']:.0f} per month over the last twelve months. A rolling-origin backtest over {ctx['backtest_origins']} origins puts the four candidate methods within a few customers of each other (MAE {ctx['backtest_best_mae']:.1f}-{ctx['backtest_worst_mae']:.1f} per month); the parsimony rule picks *{ctx['signings_method'].replace('_', ' ')}*: "
                 f"about {fc['forecast'].sum():.0f} new customers in the next twelve months ({fc['lo'].sum():.0f}-{fc['hi'].sum():.0f} with the backtest band), against {ctx['signings_flat_12m']:.0f} if the last twelve months simply repeat. The series is short, the last two months are a jump, and the pre-2023 years are thinned by purging, so this is a planning range, not a prediction.")
    lines.append(f"- **Hardware**: {pct(mix.loc['rtha55', 'share_2026_09'])} of active subscriptions already run the 2.0 unit; {int(ctx['pending_legacy_slots'])} legacy units at {int(ctx['pending_legacy_customers'])} customers remain to be swapped. "
                 f"Device returns run at {ctx['returns_recent_mean']:.0f} per month in 2026, of which {pct(ctx['returns_exit_share'])} were exits and the rest swaps or downsizing (the last two months cannot yet tell the two apart).\n")
    lines.append("![demand](figures/06_renewal_calendar_and_signings.png)\n")

    lines.append("## 6. The migration call campaign\n")
    lines.append(f"- Five rounds: {int(fun.loc[1, 'contacted_rows'])}, {int(fun.loc[2, 'contacted_rows']):,}, {int(fun.loc[3, 'contacted_rows'])}, {int(fun.loc[4, 'contacted_rows'])} and {int(fun.loc[5, 'contacted_rows'])} calls. "
                 f"Round 1 reached a person on {pct(fun.loc[1, 'reach_rate_of_contacted'])} of calls; round 2, the largest, on {pct(fun.loc[2, 'reach_rate_of_contacted'])} - {pct(ctx['round2_no_answer'])} of its {int(fun.loc[2, 'contacted_rows']):,} calls ended in a text message. "
                 f"Across the 2025 rounds the reach rate did not improve with repeated attempts ({', '.join(pct(v) for v in ctx['reach_by_attempt'])} on the first, second and third attempt).")
    lines.append(f"- What customers said in 2025 predicts where they are in September 2026: of those who **agreed to exchange**, {pct(outc.loc['agreed to exchange', 'migrated_share'])} are on the new unit and {pct(outc.loc['agreed to exchange', 'left_share'])} have left; "
                 f"of those who **did not answer** (text sent), {pct(outc.loc['no answer (text sent)', 'migrated_share'])} migrated and {pct(outc.loc['no answer (text sent)', 'left_share'])} left; "
                 f"of those who said **cancel / closed / on hold**, {pct(outc.loc['cancel / closed / on hold', 'left_share'])} have left. Customers listed but never called were the worst group ({pct(outc.loc['listed, never contacted', 'left_share'])} gone), which says the lists included accounts that were already on their way out.")
    lines.append(f"- List hygiene: {int(fun.loc[4, 'already_migrated_rows']) + int(fun.loc[5, 'already_migrated_rows'])} of the {int(fun.loc[4, 'contacted_rows']) + int(fun.loc[5, 'contacted_rows'])} September-2026 calls went to customers who had already swapped the device.\n")
    lines.append("![calls](figures/07_migration_and_calls.png)\n")

    lines.append("## 7. Limitations\n")
    lines.append("- One vendor, one CRM, customers concentrated in California and in one- and two-vehicle fleets; nothing here is an industry rate.")
    lines.append("- Renewal histories are reconstructed from overwritten rows; the January-2024 export validates the method but only for customers alive then. Anniversaries before 2023 are under-recorded because churned rows are purged.")
    lines.append(f"- The replacement link resolves to a known earlier generation for {pct(tr['previous_generation_known'])} of replacement rows, so generation transitions are a sample; whether swapping hardware changes renewal cannot be identified from this export (the swap is recorded at renewal).")
    lines.append("- Call rounds 1-3 carry no per-call dates; outcomes are measured on the export date, not at a fixed interval after the call. Observational throughout: the odds ratios describe who renews, not what would happen if an attribute were changed.")
    lines.append("- Organization churn is derived from contract rows, not from billing; customers on a billing hold (inactive / unpaid / suspended but not expired) are kept as customers and flagged.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
