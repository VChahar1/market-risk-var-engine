"""Regulatory backtesting: rolling VaR backtest, Basel traffic-light, Kupiec test.

Flat functions operating on the daily P&L series. No look-ahead: each day's VaR
uses only the trailing window available up to that day.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def rolling_backtest(pnl: pd.Series, confidence: float = 0.99,
                     window: int = 250, method: str = "historical") -> pd.DataFrame:
    """Walk forward: estimate VaR from the trailing window, test on the next day.

    Returns a DataFrame indexed by date with predicted VaR, realized P&L, and an
    'exception' flag (the loss exceeded VaR).
    """
    alpha = 1 - confidence
    z = stats.norm.ppf(alpha)
    records = []
    vals = pnl.values
    for t in range(window, len(pnl)):
        win = vals[t - window:t]
        if method == "historical":
            var = -np.quantile(win, alpha)
        elif method == "parametric":
            var = -(win.mean() + z * win.std())
        else:
            raise ValueError(method)
        records.append({
            "date": pnl.index[t],
            "VaR": var,
            "pnl": vals[t],
            "exception": vals[t] < -var,
        })
    return pd.DataFrame(records).set_index("date")


def basel_traffic_light(n_exceptions: int, n_days: int, confidence: float = 0.99):
    """Classify into Basel green/amber/red, scaled to a 250-day equivalent.

    NOTE: the Basel zones are defined for exactly 250 days; scaling a longer
    backtest is an approximation. Pair with the more powerful Kupiec test.
    """
    scaled = n_exceptions * (250 / n_days)
    if scaled < 5:
        zone = "GREEN"
    elif scaled < 10:
        zone = "AMBER"
    else:
        zone = "RED"
    return zone, scaled


def kupiec_pof_test(n_exceptions: int, n_days: int, confidence: float = 0.99):
    """Kupiec proportion-of-failures likelihood-ratio test.

    Returns (LR statistic, p-value). p < 0.05 => reject the VaR model.
    """
    p = 1 - confidence
    x, n = n_exceptions, n_days
    pi_hat = x / n
    if x == 0:
        lr = -2 * n * np.log(1 - p)
    else:
        lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p)
                   - (n - x) * np.log(1 - pi_hat) - x * np.log(pi_hat))
    return lr, 1 - stats.chi2.cdf(lr, df=1)


def backtest_report(pnl: pd.Series, confidence: float = 0.99,
                    window: int = 250) -> pd.DataFrame:
    """Backtest historical and parametric VaR side by side with all metrics."""
    rows = []
    for method in ("historical", "parametric"):
        bt = rolling_backtest(pnl, confidence, window, method)
        n_days = len(bt); n_exc = int(bt["exception"].sum())
        zone, _ = basel_traffic_light(n_exc, n_days, confidence)
        _, pval = kupiec_pof_test(n_exc, n_days, confidence)
        rows.append({
            "method": method, "exceptions": n_exc,
            "expected": (1 - confidence) * n_days,
            "rate": n_exc / n_days, "basel_zone": zone, "kupiec_p": pval,
        })
    return pd.DataFrame(rows).set_index("method")
