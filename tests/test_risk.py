"""Test suite for the market-risk engine.

These verify invariants (properties that must always hold) and analytic
benchmarks (cases with a known answer) — the checks a model-validation team
writes. Run with:  pytest tests/test_risk.py
"""
import numpy as np
import pandas as pd
from scipy import stats

from src.risk_measures import (
    historical_var, parametric_var, historical_es,
)
from src.backtesting import rolling_backtest, kupiec_pof_test
from src.plat import hypothetical_pnl, risk_theoretical_pnl, plat_metrics
from src.portfolio import signed_notionals


def test_parametric_var_matches_analytic():
    """For normal P&L, parametric 99% VaR = -(mu + z*sigma) exactly."""
    rng = np.random.default_rng(0)
    p = pd.Series(rng.normal(100, 1000, 500_000))
    expected = -(p.mean() + stats.norm.ppf(0.01) * p.std())
    assert abs(parametric_var(p, 0.99) - expected) < 1e-6


def test_es_geq_var():
    """Expected Shortfall is always >= VaR."""
    rng = np.random.default_rng(1)
    p = pd.Series(rng.normal(0, 1000, 100_000))
    assert historical_es(p, 0.99) >= historical_var(p, 0.99)


def test_var_increases_with_confidence():
    """99% VaR >= 95% VaR."""
    rng = np.random.default_rng(2)
    p = pd.Series(rng.normal(0, 1000, 100_000))
    assert historical_var(p, 0.99) >= historical_var(p, 0.95)


def test_historical_parametric_converge_for_normal():
    """For genuinely normal data, the two VaR methods nearly agree."""
    rng = np.random.default_rng(3)
    p = pd.Series(rng.normal(0, 1000, 500_000))
    h, q = historical_var(p, 0.99), parametric_var(p, 0.99)
    assert abs(h - q) / q < 0.05


def test_backtest_exception_rate_is_calibrated():
    """A backtest of well-calibrated VaR yields ~ the expected exception rate."""
    rng = np.random.default_rng(4)
    p = pd.Series(rng.normal(0, 1000, 3000))
    bt = rolling_backtest(p, 0.99, 250, "parametric")
    assert 0.005 < bt["exception"].mean() < 0.02


def test_kupiec_passes_for_correct_model():
    """Kupiec should NOT reject when exceptions match expectation (exactly 1%)."""
    _, pval = kupiec_pof_test(25, 2500, 0.99)
    assert pval > 0.05


def test_plat_detects_broken_model():
    """PLAT must rate a model missing a material factor worse than a good one."""
    rng = np.random.default_rng(5)
    w = signed_notionals()
    cols = list(w.index)
    rr = pd.DataFrame(rng.normal(0, 0.01, (2000, len(cols))), columns=cols)
    hpl = hypothetical_pnl(rr, w)
    good = risk_theoretical_pnl(rr, w, noise=0.02)
    poor = risk_theoretical_pnl(rr, w, omit=["SPY"],
                                factor_scale={"AAPL": 0.5, "JPM": 0.5}, noise=0.05)
    sp_g, ks_g = plat_metrics(hpl, good)
    sp_p, ks_p = plat_metrics(hpl, poor)
    assert sp_g > sp_p and ks_g < ks_p
