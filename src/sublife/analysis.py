"""Small shared helpers: Wilson intervals, rate tables with small-cell suppression, exit modes."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

MIN_CELL = 5  # counts below this are published as "<5"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, max(0.0, c - h), min(1.0, c + h))


def suppress(n) -> str:
    return "<5" if (pd.notna(n) and n < MIN_CELL) else (str(int(n)) if pd.notna(n) else "")


def rate_table(df: pd.DataFrame, by: str, outcome: str = "churned", min_n: int = 20) -> pd.DataFrame:
    """Outcome rate by group with Wilson 95% CI; groups below min_n are kept but marked as a lead only."""
    rows = []
    for g, d in df.groupby(by, dropna=False):
        k = int(d[outcome].sum())
        r, lo, hi = wilson(k, len(d))
        rows.append({by: g, "n": len(d), "events": suppress(k), "rate": r, "lo": lo, "hi": hi, "lead_only": len(d) < min_n})
    return pd.DataFrame(rows)


def exit_mode_table(o: pd.DataFrame) -> pd.DataFrame:
    """How churned customers left, and how far into their final term the early returns happened."""
    ch = o[o["churned"] == 1]
    t = ch["exit_mode"].value_counts().rename_axis("exit_mode").reset_index(name="customers")
    t["share"] = t["customers"] / t["customers"].sum()
    early = ch[ch["exit_mode"] == "early_return"]
    days_in = (early["coverage_end"] - (early["last_term_end"] - pd.Timedelta(365, "D"))).dt.days
    q = days_in.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(0)
    t.attrs["early_return_days_into_term"] = {f"p{int(k * 100)}": int(v) for k, v in q.items()}
    return t


def two_proportion_test(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided p-value for a difference between two proportions (Fisher's exact test)."""
    return float(stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])
