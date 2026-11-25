"""P&L Attribution Test (PLAT) — the FRTB check that the risk model captures
the right risk factors. Flat functions comparing hypothetical P&L (truth)
against risk-theoretical P&L (the model's view).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def hypothetical_pnl(returns: pd.DataFrame, w: pd.Series) -> pd.Series:
    """Full-revaluation P&L using all risk factors (the truth, HPL)."""
    return pd.Series(
        returns[w.index].values @ w.values, index=returns.index, name="HPL"
    )


def risk_theoretical_pnl(
    returns: pd.DataFrame,
    w: pd.Series,
    omit=None,
    factor_scale: dict | None = None,
    noise: float = 0.0,
    seed: int = 0,
) -> pd.Series:
    "P&L from a risk model that may misrepresent factors (RTPL)."
    used = w.copy()
    if omit:
        used = used.drop(omit)
    if factor_scale:
        for t, s in factor_scale.items():
            if t in used.index:
                used[t] = used[t] * s
    base = returns[used.index].values @ used.values
    if noise:
        rng = np.random.default_rng(seed)
        base = base + rng.normal(0, noise * base.std(), len(base))
    return pd.Series(base, index=returns.index, name="RTPL")


def plat_metrics(hpl: pd.Series, rtpl: pd.Series):
    """The two FRTB PLAT statistics: Spearman correlation and KS statistic."""
    spearman, _ = stats.spearmanr(hpl, rtpl)
    ks_stat, _ = stats.ks_2samp(hpl, rtpl)
    return spearman, ks_stat


def plat_zone(spearman: float, ks: float) -> str:
    "FRTB-style PLAT zone (illustrative thresholds)."
    if spearman >= 0.80 and ks <= 0.09:
        return "GREEN (pass)"
    if spearman < 0.70 or ks > 0.12:
        return "RED (fail)"
    return "AMBER (watch)"


def plat_report(hpl: pd.Series, models: dict) -> pd.DataFrame:
    """Classify each candidate risk model green/amber/red."""
    rows = []
    for name, rtpl in models.items():
        sp, ks = plat_metrics(hpl, rtpl)
        rows.append(
            {"model": name, "spearman": sp, "ks_stat": ks, "zone": plat_zone(sp, ks)}
        )
    return pd.DataFrame(rows).set_index("model")
