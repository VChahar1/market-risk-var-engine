"VaR and Expected Shortfall — three methods, flat functions."

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def historical_var(pnl: pd.Series, confidence: float = 0.99) -> float:
    """VaR by historical simulation: empirical percentile, no distribution assumed."""
    return -np.quantile(pnl, 1 - confidence)


def parametric_var(pnl: pd.Series, confidence: float = 0.99) -> float:
    """VaR assuming normally distributed P&L (variance-covariance method)."""
    mu, sigma = pnl.mean(), pnl.std()
    z = stats.norm.ppf(1 - confidence)
    return -(mu + z * sigma)


def monte_carlo_var(
    returns: pd.DataFrame,
    w: pd.Series,
    confidence: float = 0.99,
    n_sims: int = 100_000,
    seed: int = 42,
) -> float:
    """VaR by Monte Carlo: simulate correlated returns from a fitted MVN."""
    rng = np.random.default_rng(seed)
    mu = returns[w.index].mean().values
    cov = returns[w.index].cov().values
    sims = rng.multivariate_normal(mu, cov, size=n_sims)
    sim_pnl = sims @ w.values
    return -np.quantile(sim_pnl, 1 - confidence)


def historical_es(pnl: pd.Series, confidence: float = 0.99) -> float:
    """Expected Shortfall: mean loss in the tail beyond VaR (FRTB measure)."""
    threshold = np.quantile(pnl, 1 - confidence)
    tail = pnl[pnl <= threshold]
    return -tail.mean()


def risk_summary(pnl: pd.Series, returns: pd.DataFrame, w: pd.Series) -> pd.DataFrame:
    """All methods, both confidence levels, VaR and ES, side by side."""
    rows = []
    for c in (0.95, 0.99):
        rows.append(
            {
                "confidence": f"{c:.0%}",
                "VaR_historical": historical_var(pnl, c),
                "VaR_parametric": parametric_var(pnl, c),
                "VaR_montecarlo": monte_carlo_var(returns, w, c),
                "ES_historical": historical_es(pnl, c),
            }
        )
    return pd.DataFrame(rows).set_index("confidence")
